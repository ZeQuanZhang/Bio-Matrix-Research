"""Registered, resumable state-only memory experiments on MPS."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import CONDITIONS, generate, training_batch
from .models import Config, MemoryModel, VARIANTS

ROOT = Path('results/matrix')


def sync(device):
    if device.type == 'mps':
        torch.mps.synchronize()


def save(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def banks(count, validation=False):
    conditions = ({'one_short': dict(pairs=1, delay=0), 'two_delay12': dict(pairs=2, delay=12),
                   'four_delay24': dict(pairs=4, delay=24)} if validation else CONDITIONS)
    return {name: generate(np.random.default_rng((81000 if validation else 92000) + i), count, **condition)
            for i, (name, condition) in enumerate(conditions.items())}


@torch.no_grad()
def evaluate(model, bank, device, interventions=('carry',), batch=128):
    model.eval()
    result = {}
    for name, (events, targets, queries) in bank.items():
        result[name] = {}
        for intervention in interventions:
            accuracy, losses = [], []
            for start in range(0, len(events), batch):
                x, y = events[start:start + batch].to(device), targets[start:start + batch].to(device)
                logits, state = model(x, queries, intervention=intervention)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction='none').view_as(y)
                accuracy.extend((logits.argmax(-1) == y).float().mean(1).cpu().tolist())
                losses.extend(loss.mean(1).cpu().tolist())
            result[name][intervention] = {'accuracy': float(np.mean(accuracy)), 'loss': float(np.mean(losses)),
                                          'episodes': len(accuracy), 'episode_accuracy': accuracy, 'episode_loss': losses}
    model.train()
    return result


def train(variant, seed, cfg, root=ROOT):
    device = torch.device(cfg['device'])
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS was requested but is unavailable; no silent CPU fallback.')
    torch.set_num_threads(4); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cell, readout = VARIANTS[variant]
    model_cfg = Config(cell=cell, readout=readout, **{k: cfg[k] for k in ('width', 'state', 'reaction_rank', 'matrix_rank')})
    model = MemoryModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=.0001)
    source = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [Path('matrix_lab/models.py'), Path('matrix_lab/data.py'), Path(__file__)]}
    protocol = {'experiment': cfg, 'model': asdict(model_cfg), 'variant': variant, 'seed': seed,
                'parameters': model.parameter_count(), 'persistent_state_floats': cfg['state'],
                'source_hashes': source, 'torch': str(torch.__version__), 'device': str(device)}
    folder = root / f'{variant}_seed{seed}'; folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'result.json').exists():
        existing = json.loads((folder / 'result.json').read_text())
        if existing['protocol'] != protocol:
            raise ValueError('Completed results use another protocol; choose another results directory.')
        return
    (folder / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    valid = banks(cfg['validation_episodes'], validation=True)
    step, best, seconds, event_count, max_driver = 0, float('inf'), 0., 0, 0
    batch_digests = []
    if (folder / 'last.pt').exists():
        checkpoint = torch.load(folder / 'last.pt', map_location=device, weights_only=True)
        if checkpoint['protocol'] != protocol:
            raise ValueError('Resume protocol mismatch')
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        rng.bit_generator.state = checkpoint['data_rng']
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        step, best, seconds = checkpoint['step'], checkpoint['best'], checkpoint['seconds']
        event_count, batch_digests = checkpoint['events'], checkpoint['batch_digests']
        max_driver = checkpoint['max_driver']
    losses = []
    started_wall = time.perf_counter()
    with (folder / 'metrics.jsonl').open('a') as log:
        for i in range(step, cfg['steps']):
            events, targets, queries = training_batch(rng, i, cfg['batch_size'])
            digest = hashlib.sha256(events.numpy().tobytes() + targets.numpy().tobytes()).hexdigest()
            batch_digests.append(digest)
            x, y = events.to(device), targets.to(device)
            warm = min(1., (i + 1) / 50)
            rate = cfg['lr'] * warm * (.2 + .8 * .5 * (1 + math.cos(math.pi * i / cfg['steps'])))
            for group in optimizer.param_groups:
                group['lr'] = rate
            sync(device); tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits, state = model(x, queries)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
            if not bool(loss.isfinite()):
                raise FloatingPointError(f'Nonfinite loss at step {i + 1}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not bool(norm.isfinite()):
                raise FloatingPointError('Nonfinite gradient')
            optimizer.step(); sync(device)
            seconds += time.perf_counter() - tick
            event_count += events.shape[0] * events.shape[1]
            losses.append(float(loss.detach()))
            if device.type == 'mps':
                max_driver = max(max_driver, torch.mps.driver_allocated_memory())
            if (i + 1) % 100 == 0 or i == 0:
                entry = {'step': i + 1, 'train_loss': float(np.mean(losses)), 'optimization_seconds': seconds,
                         'events': event_count, 'events_per_second': event_count / seconds,
                         'gradient_norm': float(norm), 'state_rms': float(state.detach().square().mean().sqrt())}
                losses.clear(); log.write(json.dumps(entry) + '\n'); log.flush(); print(json.dumps(entry), flush=True)
            if (i + 1) % cfg['eval_every'] == 0 or i + 1 == cfg['steps']:
                scores = evaluate(model, valid, device)
                value = float(np.mean([v['carry']['loss'] for v in scores.values()]))
                entry = {'step': i + 1, 'validation_loss': value,
                         'validation_accuracy': float(np.mean([v['carry']['accuracy'] for v in scores.values()]))}
                log.write(json.dumps(entry) + '\n'); log.flush(); print(json.dumps(entry), flush=True)
                improved = value < best; best = min(best, value)
                payload = {'protocol': protocol, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                           'step': i + 1, 'best': best, 'seconds': seconds, 'events': event_count,
                           'data_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
                           'batch_digests': batch_digests, 'max_driver': max_driver}
                save(folder / 'last.pt', payload)
                if improved:
                    save(folder / 'best.pt', {'protocol': protocol, 'model': model.state_dict(), 'step': i + 1})
    best_checkpoint = torch.load(folder / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint['model'])
    test = evaluate(model, banks(cfg['test_episodes']), device, ('carry', 'reset', 'swap'))
    dense_macs = cfg['width'] ** 2 + (3 if cell == 'gru' else 2) * cfg['width'] * cfg['state']
    dense_macs += (3 if cell == 'gru' else 2) * cfg['state'] ** 2 if cell != 'linear' else 0
    dense_macs += 3 * cfg['state'] * cfg['reaction_rank'] if cell == 'reaction' else 0
    result = {'protocol': protocol, 'best_step': best_checkpoint['step'], 'test': test,
              'optimization_seconds': seconds, 'wall_seconds_this_invocation': time.perf_counter() - started_wall,
              'events_seen': event_count, 'events_per_second': event_count / seconds,
              'sampled_metal_driver_bytes': max_driver, 'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'dense_macs_per_nonquery_event_estimate': dense_macs,
              'data_schedule_sha256': hashlib.sha256(''.join(batch_digests).encode()).hexdigest()}
    (folder / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'completed': str(folder), 'test_accuracy': {k: v['carry']['accuracy'] for k, v in test.items()}}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', default='matrix_lab/config.json')
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text()); root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        cfg.update(steps=100, eval_every=100, validation_episodes=32, test_episodes=32)
    if args.variant:
        train(args.variant, args.seed, cfg, root)
        return
    (root / 'config.json').write_text(json.dumps(cfg, indent=2))
    (root / 'environment.json').write_text(json.dumps({'python': platform.python_version(), 'torch': str(torch.__version__),
        'platform': platform.platform(), 'mps_available': torch.backends.mps.is_available(),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}, indent=2))
    for seed in cfg['seeds']:
        for variant in cfg['variants']:
            folder = root / f'{variant}_seed{seed}'; folder.mkdir(parents=True, exist_ok=True)
            print(f'Start {variant} seed={seed}', flush=True)
            command = [sys.executable, '-u', '-m', 'matrix_lab.run', '--variant', variant, '--seed', str(seed),
                       '--config', args.config, '--root', str(root)]
            with (folder / 'console.log').open('a') as output:
                subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)
            print(f'Complete {variant} seed={seed}', flush=True)


if __name__ == '__main__':
    main()
