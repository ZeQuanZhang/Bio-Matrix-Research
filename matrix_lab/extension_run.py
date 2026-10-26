"""Separate, registered comparison of Gaussian and oscillatory extensions."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import training_batch
from .extensions import ExtendedMemory
from .models import Config, MemoryModel
from .run import banks, evaluate, save, sync

VARIANTS = ('reaction32', 'reaction64', 'reaction96', 'bayes', 'spectral_zero', 'spectral_fixed', 'spectral_learned', 'full')
ROOT = Path('results/matrix_extensions')
CONFIG = {'steps': 2000, 'batch_size': 64, 'lr': .003, 'eval_every': 500,
          'seeds': [42, 137, 2026], 'test_episodes': 256, 'validation_episodes': 128,
          'device': 'mps', 'variants': list(VARIANTS), 'noisy_training_fraction': .25,
          'noise_probability': .2, 'noisy_observations_per_key': 3,
          'selection': 'mean validation CE: normal 2key delay12, normal4key delay24, noisy2key delay12'}


def mark(batch, noisy):
    events, labels, positions = batch
    marker = torch.tensor([1, 8, int(noisy)]).view(1, 1, 3).expand(len(events), 1, 3)
    return torch.cat((marker, events), 1), labels, tuple(p+1 for p in positions)


def noisy_batch(rng, batch=64, pairs=2, delay=12, probability=.2, repeats=3):
    keys = rng.random((batch, 8)).argsort(1)[:, :pairs]
    truth = rng.integers(4, size=(batch, pairs))
    events = []
    for _ in range(repeats):
        for i in range(pairs):
            flip = rng.random(batch) < probability
            observed = (truth[:, i] + flip*rng.integers(1, 4, size=batch)) % 4
            events.append(np.stack((np.zeros(batch, dtype=int), keys[:, i], observed), 1))
    for _ in range(delay):
        events.append(np.stack((np.ones(batch, dtype=int), rng.integers(8, size=batch), rng.integers(4, size=batch)), 1))
    query = rng.integers(pairs, size=batch)
    positions = (len(events),)
    events.append(np.stack((np.full(batch, 2), keys[np.arange(batch), query], np.full(batch, 4)), 1))
    labels = truth[np.arange(batch), query, None]
    return mark((torch.tensor(np.stack(events, 1)).long(), torch.tensor(labels).long(), positions), True)


def extension_banks(count, validation=False):
    bank = {k: mark(v, False) for k, v in banks(count, validation).items()}
    if validation:
        bank.pop('one_short')
        bank['noisy_p20'] = noisy_batch(np.random.default_rng(130001), count)
    else:
        for name, probability, repeats in [('noisy_p00', 0., 3), ('noisy_p20', .2, 3), ('noisy_p35', .35, 3), ('noisy_repeat5', .2, 5)]:
            bank[name] = noisy_batch(np.random.default_rng(140001), count, probability=probability, repeats=repeats)
    return bank


def model_for(variant):
    if variant.startswith('reaction'):
        model = MemoryModel(Config(cell='reaction', readout='dynamic', state=int(variant[8:])))
        model.state_floats = model.cfg.state
        return model
    return ExtendedMemory(variant)


def train(variant, seed):
    torch.set_num_threads(4); torch.manual_seed(seed)
    device = torch.device(CONFIG['device'])
    if not torch.backends.mps.is_available():
        raise RuntimeError('Registered MPS execution unavailable')
    model = model_for(variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=.0001)
    source = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [Path(__file__), Path('matrix_lab/extensions.py'), Path('matrix_lab/models.py'), Path('matrix_lab/data.py'), Path('matrix_lab/run.py')]}
    protocol = {'config': CONFIG, 'variant': variant, 'seed': seed, 'parameters': model.parameter_count(),
                'state_floats': model.state_floats, 'source_hashes': source, 'torch': str(torch.__version__)}
    folder = ROOT / f'{variant}_seed{seed}'; folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'result.json').exists():
        assert json.loads((folder / 'result.json').read_text())['protocol'] == protocol
        return
    rng = np.random.default_rng(seed)
    step, best, seconds, count, peak, digests = 0, float('inf'), 0., 0, 0, []
    if (folder / 'last.pt').exists():
        c = torch.load(folder / 'last.pt', map_location=device, weights_only=True)
        assert c['protocol'] == protocol
        model.load_state_dict(c['model']); optimizer.load_state_dict(c['optimizer'])
        rng.bit_generator.state = c['rng']
        step, best, seconds, count, peak, digests = (c[k] for k in ('step', 'best', 'seconds', 'count', 'peak', 'digests'))
    (folder / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    valid = extension_banks(CONFIG['validation_episodes'], True)
    rolling = []
    with (folder / 'metrics.jsonl').open('a') as log:
        for i in range(step, CONFIG['steps']):
            if rng.random() < CONFIG['noisy_training_fraction']:
                batch = noisy_batch(rng, CONFIG['batch_size'], pairs=int(rng.integers(1, 4)), delay=int(rng.integers(17)))
            else:
                batch = mark(training_batch(rng, i, CONFIG['batch_size']), False)
            events, labels, positions = batch
            digests.append(hashlib.sha256(events.numpy().tobytes()+labels.numpy().tobytes()).hexdigest())
            x, y = events.to(device), labels.to(device)
            lr = CONFIG['lr']*min(1., (i+1)/50)*(.2+.4*(1+math.cos(math.pi*i/CONFIG['steps'])))
            optimizer.param_groups[0]['lr'] = lr
            sync(device); tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x, positions)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
            if not bool(loss.isfinite()):
                raise FloatingPointError('Nonfinite extension loss')
            loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not bool(norm.isfinite()):
                raise FloatingPointError('Nonfinite extension gradient')
            optimizer.step(); sync(device)
            seconds += time.perf_counter()-tick; count += events.shape[0]*events.shape[1]
            peak = max(peak, torch.mps.driver_allocated_memory()); rolling.append(float(loss.detach()))
            if (i+1) % 100 == 0:
                entry = {'step': i+1, 'train_loss': float(np.mean(rolling)), 'seconds': seconds}
                rolling.clear(); log.write(json.dumps(entry)+'\n'); log.flush(); print(json.dumps(entry), flush=True)
            if (i+1) % CONFIG['eval_every'] == 0:
                scores = evaluate(model, valid, device)
                score = float(np.mean([s['carry']['loss'] for s in scores.values()]))
                improved = score < best; best = min(best, score)
                log.write(json.dumps({'step': i+1, 'validation_loss': score})+'\n'); log.flush()
                save(folder/'last.pt', {'protocol': protocol, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'rng': rng.bit_generator.state, 'step': i+1, 'best': best, 'seconds': seconds, 'count': count, 'peak': peak, 'digests': digests})
                if improved:
                    save(folder/'best.pt', {'protocol': protocol, 'model': model.state_dict(), 'step': i+1})
    checkpoint = torch.load(folder/'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    test = evaluate(model, extension_banks(CONFIG['test_episodes']), device, ('carry', 'reset', 'swap'))
    result = {'protocol': protocol, 'best_step': checkpoint['step'], 'test': test, 'seconds': seconds,
              'events': count, 'events_per_second': count/seconds, 'metal_driver_bytes': peak,
              'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'schedule_sha256': hashlib.sha256(''.join(digests).encode()).hexdigest()}
    (folder/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'completed': variant, 'seed': seed, 'noisy_accuracy': test['noisy_p20']['carry']['accuracy']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--variant', choices=VARIANTS); parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.variant:
        train(args.variant, args.seed)
    else:
        ROOT.mkdir(parents=True, exist_ok=True); (ROOT/'config.json').write_text(json.dumps(CONFIG, indent=2))
        for seed in CONFIG['seeds']:
            for variant in VARIANTS:
                folder = ROOT/f'{variant}_seed{seed}'; folder.mkdir(parents=True, exist_ok=True)
                print(f'Start extension {variant} seed={seed}', flush=True)
                with (folder/'console.log').open('a') as output:
                    subprocess.run([sys.executable, '-u', '-m', 'matrix_lab.extension_run', '--variant', variant, '--seed', str(seed)], stdout=output, stderr=subprocess.STDOUT, check=True)
                print(f'Complete extension {variant} seed={seed}', flush=True)
