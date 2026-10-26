"""Question-only/evidence-only leakage diagnostics, outside core architecture comparisons."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .data import bank, collate, training_batch, digest
from .run import ROOT, write, inputs, evaluate, mark


class Shortcut(nn.Module):
    def __init__(self, mode):
        super().__init__(); self.mode = mode
        self.embedding = nn.Embedding(640, 128, padding_idx=0)
        if mode == 'question_only':
            self.encoder = nn.GRU(128, 128, batch_first=True)
        else:
            self.encoder = nn.Sequential(nn.Linear(128, 128), nn.Tanh())
        self.head = nn.Linear(128, 80)

    def forward(self, records, question, mask, **ignored):
        if self.mode == 'question_only':
            sequence, _ = self.encoder(self.embedding(question))
            length = (question != 0).sum(-1).clamp_min(1)
            feature = sequence[torch.arange(len(question), device=question.device), length-1]
        else:
            evidence = self.encoder(self.embedding(records).sum(-2))
            feature = (evidence*mask[..., None]).sum(1)/mask.sum(1, keepdim=True)
        return self.head(feature), None


def key(e):
    # Attribute question words can identify the answer alphabet, but never the answer.
    return (e['family'], (e.get('spec') or {}).get('attribute', 0) if e['family'] in ('retrieve', 'trace') else 0)


def identifier(mode, seed):
    return 'diagnostic_'+mode+(f'_seed{seed}' if seed != 11 else '')


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS required')
    torch.set_num_threads(4); device = torch.device('mps'); cfg = json.loads(Path('configs/v03/base.json').read_text())
    dev = bank('dev', cfg['dev_per_slice']); test = bank('test', cfg['test_per_slice'])
    modes = ('question_only', 'evidence_only')
    statuses = json.loads((ROOT/'conditions.json').read_text())
    pairs = [(mode, seed) for mode in modes for seed in cfg['seeds']]
    for mode, seed in pairs:
        statuses.setdefault(identifier(mode, seed), {'status': 'not_executed', 'reason': 'input-independent diagnostic', 'seed': seed, 'mode': mode})
    write(ROOT/'conditions.json', statuses)
    for mode, seed in pairs:
        name = identifier(mode, seed); folder = ROOT/name; folder.mkdir(exist_ok=True)
        if (folder/'result.json').exists():
            continue
        torch.manual_seed(seed); rng = np.random.default_rng(70000+seed); model = Shortcut(mode).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.0001)
        best = float('inf'); best_state = None; best_step = 0; history = []; digests = []; started = time.perf_counter()
        mark(ROOT, name, 'running')
        for step in range(2048):
            batch = training_batch(rng, 16)
            digests.append(__import__('hashlib').sha256(b''.join(batch[k].numpy().tobytes() for k in ('records', 'question', 'mask', 'target'))).hexdigest())
            for group in optimizer.param_groups:
                group['lr'] = .003*min(1., (step+1)/50)*(.2+.8*.5*(1+np.cos(np.pi*step/2048)))
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(**inputs(batch, device)); loss = F.cross_entropy(logits, batch['target'].to(device))
            assert bool(loss.isfinite()); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            if (step+1) % 256 == 0:
                score = evaluate(model, dev, device)
                history.append({'step': step+1, 'validation_loss': score['loss'], 'validation_accuracy': score['accuracy']})
                if score['loss'] < best:
                    best = score['loss']; best_step = step+1; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.mps.synchronize(); model.load_state_dict(best_state)
        result = {'status': 'completed', 'mode': mode, 'seed': seed, 'steps': 2048, 'episodes_seen': 32768,
                  'parameters': sum(p.numel() for p in model.parameters()), 'validation_loss': best,
                  'best_step': best_step, 'validation_history': history,
                  'source_hashes': {p: __import__('hashlib').sha256(Path(p).read_bytes()).hexdigest() for p in ('v03/baselines.py', 'v03/data.py', 'v03/run.py')},
                  'wall_seconds': time.perf_counter()-started, 'data_schedule_sha256': digest(digests),
                  'test': evaluate(model, test, device), 'interpretation': 'Leakage/prior diagnostic, not an exact parameter/compute-matched core control.'}
        torch.save({'model': best_state, 'mode': mode, 'step': best_step}, folder/'best.pt'); write(folder/'result.json', result)
        mark(ROOT, name, 'completed'); print(json.dumps({'mode': mode, 'seed': seed, 'test': result['test']['accuracy']}), flush=True)
    priors = {}
    for e in dev:
        priors.setdefault(key(e), np.zeros(80))[e['target']] += 1
    records = []
    for e in test:
        p = priors[key(e)]; p = p/p.sum()
        records.append({'family': e['family'], 'objects': e['objects'], 'depth': e['depth'], 'holdout': e['holdout'],
                        'majority_correct': int(int(p.argmax()) == e['target']),
                        'random_from_dev_distribution_expected_accuracy': float(p[e['target']]), 'world_hash': e['world_hash']})
    write(ROOT/'diagnostic_priors.json', {'status': 'completed', 'choice_source': 'development labels only; task/attribute inferred from legitimate question tokens',
          'majority_accuracy': float(np.mean([r['majority_correct'] for r in records])),
          'random_answer_expected_accuracy': float(np.mean([r['random_from_dev_distribution_expected_accuracy'] for r in records])),
          'episode_results': records})


if __name__ == '__main__':
    main()
