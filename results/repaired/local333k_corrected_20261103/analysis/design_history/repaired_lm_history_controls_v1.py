"""Predeclared text-history interventions on the final pretraining checkpoints."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from evaluations.repaired.data import ROOT, lm_windows
from evaluations.repaired.model import build, NAMES
from v07.data import Corpus
from v07.common import digest, write
from transformers.utils import logging


@torch.inference_mode()
def main():
    root = PROJECT/'results/repaired/local333k_corrected_20261103'
    assert json.loads((root/'registry.json').read_text())['status'] == 'completed'
    declaration = json.loads((root/'lm_history_control_protocol.json').read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == declaration['source_sha256']
    protocol = json.loads((root/'protocol.json').read_text())
    logging.set_verbosity_error(); torch.set_num_threads(4); device = torch.device('mps')
    groups = defaultdict(list); rng = np.random.default_rng(62801)
    for values, mask, doc in lm_windows(Corpus(ROOT), 'test'):
        if mask[255:511].all() and not mask[:255].any():
            groups[doc].append(values)
    for rows in groups.values():
        rng.shuffle(rows)
    selected = []
    for round_index in range(128):
        for doc in sorted(groups):
            if len(groups[doc]) > round_index and len(selected) < 128:
                selected.append((doc, groups[doc][round_index]))
    assert len(selected) == 128
    docs = np.asarray([doc for doc, _ in selected]); values = np.stack([v for _, v in selected])
    donors = []
    for i in range(128):
        j = (i+1) % 128
        while docs[j] == docs[i]:
            j = (j+1) % 128
        donors.append(j)
    folder = root/'lm_history'; folder.mkdir(exist_ok=True)
    results = {}
    for name in NAMES:
        per_seed = []
        for seed in protocol['training']['seeds']:
            target = folder/f'{name}_s{seed}.json'
            if target.exists():
                per_seed.append(json.loads(target.read_text())); continue
            model = build(name, seed).to(device).eval()
            saved = torch.load(root/f'final_{name}_s{seed}_cpt/last.pt', map_location='cpu', weights_only=True)
            assert saved['protocol_sha256'] == digest(protocol)
            model.load_state_dict(saved['model'])
            condition = dict(seed=seed, documents=docs.tolist(), conditions={})
            for mode, length in [('warm', 0), ('remote128_replaced', 128), ('prefix384_replaced', 384)]:
                changed = values.copy()
                if length:
                    changed[:, :length] = values[donors, :length]
                assert np.array_equal(changed[:, 384:512], values[:, 384:512])
                losses = []
                for start in range(0, 128, 16):
                    x = torch.as_tensor(changed[start:start+16, :-1].copy(), device=device)
                    y = torch.as_tensor(values[start:start+16, 384:512].copy(), device=device)
                    logits = model(x)[0][:, 383:511]
                    ce = F.cross_entropy(logits.float().reshape(-1, 4096), y.reshape(-1), reduction='none').reshape(len(x), 128)
                    losses.extend(ce.double().sum(-1).cpu().tolist())
                condition['conditions'][mode] = dict(nll=sum(losses)/(128*128), per_window_nll_sum=losses, targets_per_window=128)
            write(target, condition); per_seed.append(condition)
            model.to('cpu'); del model
            torch.mps.empty_cache()
            print(name, seed, {k: round(v['nll'], 6) for k, v in condition['conditions'].items()}, flush=True)
        effects = {}
        unique = sorted(set(docs.tolist())); random = np.random.default_rng(62809)
        for mode in ('remote128_replaced', 'prefix384_replaced'):
            table = []; counts = []
            for r in per_seed:
                a = np.asarray(r['conditions'][mode]['per_window_nll_sum'])
                b = np.asarray(r['conditions']['warm']['per_window_nll_sum'])
                table.append([(a-b)[docs == d].sum() for d in unique]); counts.append([int((docs == d).sum())*128 for d in unique])
            table, counts = np.asarray(table), np.asarray(counts)
            ss = random.integers(0, 3, (20000, 3)); dd = random.integers(0, len(unique), (20000, len(unique)))
            draws = table[ss[:, :, None], dd[:, None, :]].sum((1, 2))/counts[ss[:, :, None], dd[:, None, :]].sum((1, 2))
            effects[mode] = dict(nll_increase=float(table.sum()/counts.sum()), paired_seed_document_95ci=np.quantile(draws, [.025, .975]).tolist())
        results[name] = dict(status='completed', seeds=3, windows_per_seed=128, documents=len(unique), effects=effects)
    write(root/'lm_history_controls.json', dict(status='completed', conditions=results,
        score='Last128 real text targets; same target positions across modes; first128 or first384 inputs replaced by a different document',
        sample='128 nonoverlapping coverage windows, balanced round-robin over eligible documents; supplementary diagnostic, not full-test PPL',
        scope='Observed-seed/document bootstrap, exploratory intervals; checkpoints remain unchanged'))


if __name__ == '__main__':
    main()
