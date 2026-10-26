"""Serial, resumable v0.3 paired training with explicit per-condition status."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from matrix_lab.run import save, sync
from .data import bank, collate, digest, manifest, training_batch
from .graph import statistics
from .model import Config, Network, MODELS

ROOT = Path('results/v03')


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp'); temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def cases(stage, cfg):
    result = []
    if stage == 'pilot':
        for kind in MODELS:
            for rate in cfg['lr_candidates']:
                result.append(dict(stage=stage, model=kind, nodes=128, seed=101, lr=rate, steps=cfg['pilot_steps']))
    elif stage == 'main':
        # A full C1/C2/C3 triplet at each N precedes additional independent seeds.
        for seed in cfg['seeds']:
            for n in cfg['nodes']:
                for kind in MODELS:
                    result.append(dict(stage=stage, model=kind, nodes=n, seed=seed))
    elif stage == 'capacity':
        for n, d in ((32, 254), (128, 62), (512, 14)):
            for kind in MODELS:
                result.append(dict(stage=stage, model=kind, nodes=n, state_dim=d, seed=11))
    elif stage == 'iterations':
        for n in (128, 512):
            for j in (2, 8, 32):
                for kind in MODELS:
                    result.append(dict(stage=stage, model=kind, nodes=n, iterations=j, seed=11))
    elif stage == 'topology':
        for n in (128, 512):
            for topo in ('cut', 'rewired'):
                for kind in MODELS:
                    result.append(dict(stage=stage, model=kind, nodes=n, topology=topo, seed=11))
    elif stage == 'public':
        for n in (128, 512):
            for kind in MODELS:
                result.append(dict(stage=stage, model=kind, nodes=n, seed=11))
    else:
        raise ValueError('Unknown stage')
    for c in result:
        c.setdefault('state_dim', cfg['state_dim']); c.setdefault('iterations', cfg['iterations'])
        c.setdefault('topology', 'connected'); c.setdefault('steps', cfg['steps'])
        c['id'] = f"{stage}_{c['model']}_N{c['nodes']}_d{c['state_dim']}_J{c['iterations']}_{c['topology']}_s{c['seed']}"
        if stage == 'pilot':
            c['id'] += f"_lr{c['lr']:g}"
    return result


def register(root, cfg):
    path = root/'conditions.json'
    old = json.loads(path.read_text()) if path.exists() else {}
    for stage in ('pilot', 'main', 'capacity', 'iterations', 'topology', 'public'):
        for c in cases(stage, cfg):
            old.setdefault(c['id'], {'condition': c, 'status': 'not_executed'})
    for phase, reason in [('P4_fixed_experts', 'Subsequent fixed sparse expert routing, after core network evidence.'),
                          ('P5_dynamic_biological_extensions', 'Conditional later stage; no quadratic/Bayesian/new optimizer in core comparison.')]:
        old.setdefault(phase, {'status': 'not_executed', 'reason': reason})
    write(path, old)
    return old


def mark(root, identifier, status, **extra):
    path = root/'conditions.json'; value = json.loads(path.read_text())
    stamp = now()
    value[identifier].setdefault('history', []).append({'status': status, 'utc': stamp, **extra})
    value[identifier].update(status=status, updated_utc=stamp, **extra); write(path, value)


def inputs(batch, device):
    return {k: batch[k].to(device) for k in ('records', 'question', 'mask')}


@torch.no_grad()
def evaluate(model, examples, device, batch_size=16, iterations=None, intervention=None):
    model.eval(); rows = []; tick = time.perf_counter(); sync(device)
    for start in range(0, len(examples), batch_size):
        group = examples[start:start+batch_size]; batch = collate(group); target = batch['target'].to(device)
        logits, _ = model(**inputs(batch, device), iterations=iterations, intervention=intervention)
        losses = F.cross_entropy(logits, target, reduction='none').cpu().tolist()
        predictions = logits.argmax(-1).cpu().tolist()
        for e, prediction, loss in zip(group, predictions, losses):
            rows.append({k: e[k] for k in ('world_hash', 'family', 'objects', 'depth', 'irrelevant', 'holdout')}
                        | {'target': e['target'], 'prediction': prediction, 'correct': int(prediction == e['target']), 'loss': loss})
    sync(device); elapsed = time.perf_counter()-tick; model.train()
    return {'accuracy': float(np.mean([e['correct'] for e in rows])), 'loss': float(np.mean([e['loss'] for e in rows])),
            'episodes': len(rows), 'evaluation_wall_seconds': elapsed, 'episode_results': rows}


def train(c, cfg, root, max_seconds=0):
    folder = root/c['id']; folder.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg['device'])
    if device.type != 'mps' or not torch.backends.mps.is_available():
        raise RuntimeError('This runner requires MPS; CPU is reserved for implementation checks')
    torch.set_num_threads(4); torch.manual_seed(c['seed'])
    rng = np.random.default_rng(cfg['pilot_data_seed'] if c['stage'] == 'pilot' else 70000+c['seed'])
    model_cfg = Config(model=c['model'], nodes=c['nodes'], state_dim=c['state_dim'], width=cfg['width'],
                       bottleneck=cfg['bottleneck'], rank=cfg['rank'], iterations=c['iterations'],
                       graph_seed=cfg['graph_seed']+(1000 if c['topology'] == 'rewired' else 0),
                       role_seed=cfg['role_seed'], phase_seed=cfg['phase_seed'], topology=c['topology'])
    public = None
    if c['stage'] == 'public':
        from .public import PublicData
        public = PublicData()
        model_cfg.vocab = public.vocab_size; model_cfg.answers = len(public.answers)
    model = Network(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=.0001)
    sources = ['v03/model.py', 'v03/graph.py', 'v03/data.py', 'v03/run.py'] + (['v03/public.py'] if public else [])
    source_hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    dev = public.dev if public else bank('pilot' if c['stage'] == 'pilot' else 'dev', cfg['dev_per_slice'])
    test = public.test if public else bank('test', cfg['test_per_slice'])
    forbidden = {e['world_hash'] for e in dev+test}
    protocol = {'condition': c, 'training': cfg, 'model': asdict(model_cfg), 'source_hashes': source_hashes,
                'data_manifest': public.manifest if public else {'dev': manifest(dev), 'test': manifest(test)},
                'torch': str(torch.__version__), 'device': str(device), 'dtype': 'float32',
                'matching': 'Same per-seed data order, graph, roles, initial auxiliaries, content rule, question/evidence encoder and readout; C3 adds one active coupling scalar.'}
    if (folder/'result.json').exists():
        completed = json.loads((folder/'result.json').read_text())
        if completed['protocol'] != protocol:
            raise ValueError('Existing completed result has another protocol; never overwrite it')
        mark(root, c['id'], 'completed', result=str(folder/'result.json')); return completed
    write(folder/'config.resolved.json', protocol)
    write(folder/'parameter_state_compute_ledger.json', model.ledger())
    write(folder/'dataset_manifest.json', protocol['data_manifest'])
    write(folder/'environment.json', {'python': platform.python_version(), 'platform': platform.platform(),
                                    'torch': str(torch.__version__), 'MPS': True, 'started_utc': now()})
    step = 0; best = float('inf'); seconds = 0.; digests = []; train_worlds = []; max_current = 0; max_driver = 0
    data_seconds = 0.; eval_seconds = 0.; checkpoint_seconds = 0.
    if (folder/'last.pt').exists():
        state = torch.load(folder/'last.pt', map_location=device, weights_only=True)
        if state['protocol'] != protocol:
            raise ValueError('Resume protocol/source mismatch; preserve previous run')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state = state['rng']; torch.set_rng_state(state['torch_rng'].cpu())
        step = state['step']; best = state['best']; seconds = state['seconds']; digests = state['batch_digests']
        train_worlds = state['train_worlds']; max_current = state['max_current']; max_driver = state['max_driver']
        data_seconds = state['data_seconds']; eval_seconds = state['eval_seconds']; checkpoint_seconds = state['checkpoint_seconds']
    started = time.perf_counter(); mark(root, c['id'], 'running', resumed_step=step, started_utc=now())

    def checkpoint():
        nonlocal checkpoint_seconds
        sync(device); t = time.perf_counter()
        save(folder/'last.pt', {'protocol': protocol, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
             'rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(), 'step': step, 'best': best,
             'seconds': seconds, 'batch_digests': digests, 'train_worlds': train_worlds,
             'max_current': max_current, 'max_driver': max_driver, 'data_seconds': data_seconds,
             'eval_seconds': eval_seconds, 'checkpoint_seconds': checkpoint_seconds})
        checkpoint_seconds += time.perf_counter()-t

    try:
        if step == 0:
            checkpoint()
        with (folder/'metrics.jsonl').open('a') as log:
            while step < c['steps']:
                if max_seconds and time.perf_counter()-started >= max_seconds:
                    checkpoint(); mark(root, c['id'], 'aborted', reason='Explicit wall budget reached', step=step); return None
                t = time.perf_counter()
                if public:
                    batch, hashes = public.training_batch(rng, cfg['batch_size'])
                else:
                    batch, hashes = training_batch(rng, cfg['batch_size'], with_hashes=True)
                if set(hashes) & forbidden:
                    raise AssertionError('A complete training world overlaps dev/test')
                train_worlds.extend(hashes)
                digests.append(hashlib.sha256(b''.join(batch[k].numpy().tobytes() for k in ('records', 'question', 'mask', 'target'))).hexdigest())
                data_seconds += time.perf_counter()-t
                rate = c['lr'] * min(1., (step+1)/50) * (.2+.8*.5*(1+math.cos(math.pi*step/c['steps'])))
                for group in optimizer.param_groups:
                    group['lr'] = rate
                sync(device); t = time.perf_counter(); optimizer.zero_grad(set_to_none=True)
                total_loss = 0.
                for start in range(0, cfg['batch_size'], cfg['micro_batch_size']):
                    micro = {k: v[start:start+cfg['micro_batch_size']] for k, v in batch.items()}
                    logits, states = model(**inputs(micro, device)); target = micro['target'].to(device)
                    loss = F.cross_entropy(logits, target)*len(target)/cfg['batch_size']
                    if not bool(loss.isfinite()):
                        raise FloatingPointError('Nonfinite training loss')
                    loss.backward(); total_loss += float(loss.detach())
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
                if not bool(norm.isfinite()):
                    raise FloatingPointError('Nonfinite gradient norm')
                optimizer.step(); sync(device); seconds += time.perf_counter()-t; step += 1
                current, driver = torch.mps.current_allocated_memory(), torch.mps.driver_allocated_memory()
                max_current = max(max_current, current); max_driver = max(max_driver, driver)
                if driver > cfg['memory_stop_driver_bytes']:
                    checkpoint(); mark(root, c['id'], 'aborted', reason='Registered sampled driver memory budget reached', step=step); return None
                if step == 1 or step % 64 == 0:
                    entry = {'step': step, 'train_loss': total_loss, 'gradient_norm': float(norm), 'lr': rate,
                             'optimization_seconds': seconds, 'sampled_tensor_bytes': current, 'sampled_driver_bytes': driver,
                             'state_rms': float(states[0].detach().square().mean().sqrt())}
                    log.write(json.dumps(entry)+'\n'); log.flush(); print(json.dumps({'run': c['id'], **entry}), flush=True)
                if step % cfg['eval_every'] == 0 or step == c['steps']:
                    score = evaluate(model, dev, device, cfg['micro_batch_size']); eval_seconds += score['evaluation_wall_seconds']
                    entry = {'step': step, 'validation_loss': score['loss'], 'validation_accuracy': score['accuracy']}
                    log.write(json.dumps(entry)+'\n'); log.flush(); print(json.dumps({'run': c['id'], **entry}), flush=True)
                    if score['loss'] < best:
                        best = score['loss']; save(folder/'best.pt', {'protocol': protocol, 'model': model.state_dict(), 'step': step})
                if step % 64 == 0 or step == c['steps']:
                    checkpoint()
    except KeyboardInterrupt:
        # Resume the last complete boundary; an interrupted optimizer step is never saved.
        mark(root, c['id'], 'aborted', reason='User/process interrupt; resume last safe checkpoint', step=(step//64)*64); raise
    except Exception as error:
        write(folder/'failure.json', {'step': step, 'error': repr(error), 'traceback': traceback.format_exc()})
        mark(root, c['id'], 'failed', reason=repr(error), step=step); raise
    selected = torch.load(folder/'best.pt', map_location=device, weights_only=True); model.load_state_dict(selected['model'])
    result = {'status': 'completed', 'protocol': protocol, 'best_step': selected['step'], 'validation_loss': best,
              'steps_completed': step, 'episodes_seen': step*cfg['batch_size'], 'optimization_seconds': seconds,
              'data_seconds': data_seconds, 'evaluation_seconds': eval_seconds, 'checkpoint_seconds': checkpoint_seconds,
              'invocation_wall_seconds': time.perf_counter()-started, 'sampled_tensor_bytes_max': max_current,
              'sampled_driver_bytes_max': max_driver, 'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'data_schedule_sha256': digest(digests), 'batch_sha256': digests,
              'training_worlds_unique': len(set(train_worlds)), 'training_worlds_disjoint_from_eval': not set(train_worlds) & forbidden,
              'graph': statistics(model.neighbors.cpu()), 'ledger': model.ledger()}
    if c['stage'] != 'pilot':
        result['test'] = evaluate(model, test, device, cfg['micro_batch_size'])
    write(folder/'training_world_hashes.json', train_worlds); write(folder/'timing.json', {k: v for k, v in result.items() if 'seconds' in k})
    write(folder/'sampled_memory.json', {k: v for k, v in result.items() if 'bytes' in k})
    write(folder/'result.json', result); mark(root, c['id'], 'completed', step=step, result=str(folder/'result.json'))
    print(json.dumps({'completed': c['id'], 'accuracy': result.get('test', {}).get('accuracy'), 'validation_loss': best}), flush=True)
    del model, optimizer; torch.mps.empty_cache()
    return result


def select_rates(root, cfg):
    path = root/'selection.json'
    if path.exists():
        return json.loads(path.read_text())['learning_rates']
    rates = {}
    for kind in MODELS:
        scores = []
        for c in cases('pilot', cfg):
            if c['model'] == kind:
                path_result = root/c['id']/'result.json'
                if not path_result.exists():
                    raise RuntimeError('Complete equal-budget pilots before freezing learning rates')
                scores.append((json.loads(path_result.read_text())['validation_loss'], c['lr']))
        rates[kind] = min(scores)[1]
    write(path, {'learning_rates': rates, 'rule': 'minimum independent pilot-dev CE; three equal-budget candidates per architecture', 'frozen_utc': now()})
    return rates


def main():
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=('pilot', 'main', 'capacity', 'iterations', 'topology', 'public', 'all'), default='main')
    p.add_argument('--config', default='configs/v03/base.json'); p.add_argument('--root', default=str(ROOT))
    p.add_argument('--dry-run', action='store_true'); p.add_argument('--max-seconds', type=float, default=0.)
    p.add_argument('--only', help='Exact condition ID, useful for resuming one condition')
    args = p.parse_args(); cfg = json.loads(Path(args.config).read_text()); root = Path(args.root); root.mkdir(parents=True, exist_ok=True)
    register(root, cfg)
    stages = ('pilot', 'main', 'capacity', 'iterations', 'topology', 'public') if args.stage == 'all' else (args.stage,)
    planned = [c for stage in stages for c in cases(stage, cfg) if not args.only or c['id'] == args.only]
    if args.dry_run:
        print(json.dumps({'conditions': len(planned), 'updates': sum(c['steps'] for c in planned),
                          'episodes': sum(c['steps']*cfg['batch_size'] for c in planned), 'serial': True,
                          'dependencies': ['P0 checks/profile', 'equal-budget pilot LR selection', 'official CLEVR data for public stage'],
                          'cases': planned}, indent=2)); return
    if not (root/'P0/result.json').exists():
        raise RuntimeError('Complete P0 checks and hardware profiling first')
    for c in planned:
        if c['stage'] != 'pilot':
            c['lr'] = select_rates(root, cfg)[c['model']]
        try:
            outcome = train(c, cfg, root, args.max_seconds)
        except Exception as error:
            mark(root, c['id'], 'failed', reason=repr(error)); raise
        if outcome is None:
            break


if __name__ == '__main__':
    main()
