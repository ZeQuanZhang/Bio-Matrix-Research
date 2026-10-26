"""PyCharm/CLI runner with progress, exact protocol checking and atomic checkpoints."""
import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from matrix_lab.run import save, sync
from .data import bank, collate, digest, inputs, training_batch, validate
from .model import Config, Network, detach

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT/'results/v04/continuous'


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def select_device(requested):
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if requested == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable. In PyCharm use this project .venv interpreter; choose cpu only deliberately.')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    return torch.device(requested)


def model_config(cfg, case):
    allowed = {f.name for f in fields(Config)}
    values = {k: v for k, v in cfg.items() if k in allowed}
    values.update({k: v for k, v in case.items() if k in allowed})
    if values['variant'] == 'graph':
        # Match with active, useful factor widths; no unused padding parameters.
        with torch.random.fork_rng():
            target = sum(p.numel() for p in Network(Config(**(values | {'variant': 'full'}))).parameters())
            candidates = []
            for rank in range(1, cfg['width']+1):
                count = sum(p.numel() for p in Network(Config(**(values | {'baseline_rank': rank}))).parameters())
                candidates.append((abs(count-target), rank))
            values['baseline_rank'] = min(candidates)[1]
    return Config(**values)


def cases(cfg):
    result = []
    for seed in cfg['seeds']:
        for nodes in cfg['nodes']:
            for path in cfg.get('read_paths', [cfg['read_path']]):
                for variant in cfg['variants']:
                    case = dict(variant=variant, nodes=nodes, seed=seed, read_path=path)
                    case['id'] = f"{variant}_N{nodes}_D{cfg['matrix_size']}_J{cfg['internal_steps']}_{path}_{cfg['topology']}_s{seed}"
                    result.append(case)
    return result


def mark(root, case, status, **extra):
    path = root/'conditions.json'
    records = json.loads(path.read_text()) if path.exists() else {}
    row = records.setdefault(case['id'], dict(condition=case, status='not_executed', history=[]))
    event = dict(status=status, utc=now(), **extra)
    row['history'].append(event); row.update(event); write(path, records)


@torch.no_grad()
def evaluate(model, rows, device, batch_size=4, intervention=None, progress=True):
    model.eval(); records = []; start = time.perf_counter()
    iterator = range(0, len(rows), batch_size)
    for offset in tqdm(iterator, desc='  验证' if not intervention else f'  干预 {intervention}', leave=False,
                       disable=not progress, dynamic_ncols=True, file=sys.stdout, position=2):
        group = rows[offset:offset+batch_size]; batch = collate(group)
        logp, _, details = model(**inputs(batch, device), intervention=intervention,
                                intervention_mask=batch['query'].to(device))
        probabilities = logp.exp().cpu(); routing = details['routing'].cpu()
        for i, row in enumerate(group):
            indices = batch['query'][i].nonzero().flatten().tolist()
            for t in indices:
                target = int(batch['target'][i, t]); p = probabilities[i, t]
                records.append(dict(stream_hash=row['stream_hash'], profile=row['profile'], shift=row['shift'],
                                    time=t, final=t == indices[-1], combine=bool(batch['combine'][i, t]), target=target,
                                    prediction=int(p.argmax()), probability=p.tolist(),
                                    nll=float(-p[target].clamp_min(1e-8).log()),
                                    brier=float((p-F.one_hot(torch.tensor(target), 4)).square().sum()),
                                    context_entropy=float(-(routing[i, t]*routing[i, t].clamp_min(1e-8).log()).sum())))
    sync(device); model.train()
    confidence = np.array([max(r['probability']) for r in records])
    correctness = np.array([r['prediction'] == r['target'] for r in records])
    ece = 0.
    for low in np.arange(0, 1, .1):
        selected = (confidence >= low) & (confidence < low+.1+1e-8)
        if selected.any():
            ece += selected.mean()*abs(correctness[selected].mean()-confidence[selected].mean())
    by_profile = {}
    for profile in sorted({r['profile'] for r in records}):
        selected = [r for r in records if r['profile'] == profile and r['final']]
        by_profile[profile] = dict(final_accuracy=float(np.mean([r['prediction'] == r['target'] for r in selected])),
                                   final_nll=float(np.mean([r['nll'] for r in selected])), queries=len(selected))
    return dict(accuracy=float(correctness.mean()), nll=float(np.mean([r['nll'] for r in records])),
                brier=float(np.mean([r['brier'] for r in records])), ece=float(ece),
                queries=len(records), by_profile=by_profile, seconds=time.perf_counter()-start, records=records)


def train_update(model, optimizer, batch, cfg, device):
    optimizer.zero_grad(set_to_none=True); state = None
    n_queries = int(batch['query'].sum()); valid_count = int(batch['valid'].sum())
    if not n_queries or not valid_count:
        raise ValueError('A training batch must contain valid events and scored queries')
    nll_total = 0.; correct = 0; auxiliary = 0.
    length = batch['key'].shape[1]
    for offset in range(0, length, cfg['bptt_steps']):
        piece = {k: v[:, offset:offset+cfg['bptt_steps']] for k, v in batch.items()}
        logp, state, details = model(**inputs(piece, device), state=state)
        target = piece['target'].to(device); mask = piece['query'].to(device)
        loss = F.nll_loss(logp[mask], target[mask], reduction='sum') if bool(mask.any()) else logp.sum()*0
        predictive = details['predictive'].sum()/valid_count
        objective = loss/n_queries+cfg['predictive_loss_weight']*predictive
        if not bool(torch.isfinite(objective)):
            raise FloatingPointError('Nonfinite training objective; last safe checkpoint preserved')
        objective.backward()
        nll_total += float(loss.detach()); auxiliary += float(predictive.detach())
        correct += int((logp.argmax(-1)[mask] == target[mask]).sum())
        state = detach(state)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
    optimizer.step()
    return dict(nll=nll_total/n_queries, accuracy=correct/n_queries, predictive=auxiliary, gradient_norm=float(norm))


def source_hashes():
    names = ['v04/data.py', 'v04/model.py', 'v04/run.py', 'v04/report.py', 'v03/graph.py', 'matrix_lab/run.py']
    return {n: hashlib.sha256((PROJECT/n).read_bytes()).hexdigest() for n in names}


def train(case, cfg, root, device, stop, progress=True):
    folder = root/case['id']; folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(case['seed']); model_cfg = model_config(cfg, case)
    model = Network(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=.0001)
    rng = np.random.default_rng(cfg['data_seed']+case['seed'])
    dev = bank('dev', cfg['development_quartets'], cfg['matrix_size'])
    test = bank('test', cfg['test_quartets'], cfg['matrix_size'])
    forbidden = {r['stream_hash'] for r in dev+test}
    protocol = dict(condition=case, config=cfg, model=asdict(model_cfg), sources=source_hashes(),
                    dev_sha256=digest(dev), test_sha256=digest(test), torch=str(torch.__version__), device=str(device))
    existing = folder/'protocol.json'
    if existing.exists() and json.loads(existing.read_text()) != protocol:
        raise ValueError(f'{folder}: protocol/source/device differs. Use a NEW output folder; original results protected.')
    write(existing, protocol); write(folder/'ledger.json', model.ledger())
    if (folder/'result.json').exists():
        mark(root, case, 'completed', skipped_existing=True)
        return json.loads((folder/'result.json').read_text())
    step = 0; best = float('inf'); elapsed = 0.; batches = []; saved_step = 0
    best_model = None; best_step = 0
    if (folder/'last.pt').exists():
        checkpoint = torch.load(folder/'last.pt', map_location=device, weights_only=True)
        if checkpoint['protocol'] != protocol:
            raise ValueError('Checkpoint protocol mismatch')
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        rng.bit_generator.state = checkpoint['data_rng']; torch.set_rng_state(checkpoint['torch_rng'].cpu())
        step = saved_step = checkpoint['step']; best = checkpoint['best']; elapsed = checkpoint['elapsed']; batches = checkpoint['batches']
        best_model = checkpoint['best_model']; best_step = checkpoint['best_step']
        # Remove log records beyond the last committed optimizer checkpoint.
        path = folder/'metrics.jsonl'
        if path.exists():
            kept = []
            for line in path.read_text().splitlines():
                try:
                    if json.loads(line)['step'] <= step:
                        kept.append(line)
                except (json.JSONDecodeError, KeyError):
                    break
            path.write_text('\n'.join(kept)+('\n' if kept else ''), encoding='utf-8')
    tick = time.perf_counter()

    def checkpoint(export_best=False):
        nonlocal saved_step
        sync(device)
        payload = dict(protocol=protocol, model=model.state_dict(), optimizer=optimizer.state_dict(),
                       data_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), step=step, best=best,
                       elapsed=elapsed+time.perf_counter()-tick, batches=batches,
                       best_model=best_model, best_step=best_step)
        save(folder/'last.pt', payload); saved_step = step
        if export_best:
            save(folder/'best.pt', dict(protocol=protocol, model=best_model, step=best_step, dev_nll=best))

    mark(root, case, 'running', resumed_step=step)
    try:
        if step == 0 and best_model is None:
            initial = evaluate(model, dev, device, cfg['batch_size'], progress=progress)
            write(folder/'untrained_dev.json', initial)
            best = initial['nll']; best_step = 0
            best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save(folder/'initial.pt', dict(protocol=protocol, model=best_model, step=0))
        if not (folder/'last.pt').exists():
            checkpoint()
        description = f"{case['variant']} N={case['nodes']} {case.get('read_path', cfg['read_path'])} seed={case['seed']}"
        with tqdm(total=cfg['train_steps'], initial=step, desc=description, unit='步', leave=True,
                  dynamic_ncols=True, file=sys.stdout, disable=not progress, position=1) as bar:
            while step < cfg['train_steps'] and not stop[0]:
                generated = training_batch(rng, cfg['batch_size'], cfg['matrix_size'])
                if any(r['stream_hash'] in forbidden for r in generated):
                    raise AssertionError('Train/evaluation overlap')
                batch = collate(generated); begin = time.perf_counter()
                metrics = train_update(model, optimizer, batch, cfg, device)
                sync(device); step += 1
                batches.append(digest([r['stream_hash'] for r in generated]))
                metrics.update(step=step, update_seconds=time.perf_counter()-begin,
                               profile=generated[0]['profile'], utc=now())
                if device.type == 'mps':
                    metrics['mps_current_bytes'] = int(torch.mps.current_allocated_memory())
                    metrics['mps_driver_bytes'] = int(torch.mps.driver_allocated_memory())
                improved = False
                if step % cfg['eval_every'] == 0 or step == cfg['train_steps']:
                    evaluation = evaluate(model, dev, device, cfg['batch_size'], progress=progress)
                    metrics['dev'] = {k: v for k, v in evaluation.items() if k != 'records'}
                    improved = evaluation['nll'] < best
                    if improved:
                        best = evaluation['nll']
                        best_step = step
                        best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                with (folder/'metrics.jsonl').open('a', encoding='utf-8') as log:
                    log.write(json.dumps(metrics, ensure_ascii=False)+'\n'); log.flush()
                if step % cfg['checkpoint_every'] == 0 or improved or step == cfg['train_steps'] or stop[0]:
                    checkpoint(improved)
                if step % cfg['progress_every'] == 0:
                    bar.set_postfix(loss=f"{metrics['nll']:.4f}", acc=f"{metrics['accuracy']:.1%}",
                                    dev='--' if best == float('inf') else f'{best:.4f}')
                bar.update(1)
        if stop[0]:
            checkpoint(); mark(root, case, 'aborted', step=step, safe_checkpoint_step=saved_step, reason='User stop requested')
            return None
        # At least the final step is evaluated and selected on development data only.
        checkpoint(export_best=True)
        model.load_state_dict(best_model)
        test_result = evaluate(model, test, device, cfg['batch_size'], progress=progress)
        write(folder/'test_predictions.json', test_result.pop('records'))
        controls = {}
        for intervention in ('reset_history', 'swap_history', 'no_messages', 'no_coupling', 'fixed_external'):
            if ((intervention == 'no_coupling' and not model.fourier)
                    or (intervention == 'fixed_external' and model.cfg.variant == 'graph')):
                controls[intervention] = dict(status='not_applicable', reason='This variant does not contain the intervened mechanism')
                continue
            if stop[0]:
                mark(root, case, 'aborted', step=step, safe_checkpoint_step=saved_step, reason='Stopped during diagnostics')
                return None
            outcome = evaluate(model, test, device, cfg['batch_size'], intervention, progress)
            predictions = outcome.pop('records'); write(folder/f'{intervention}_predictions.json', predictions)
            controls[intervention] = outcome
        result = dict(condition=case, status='completed', best_step=best_step, best_dev_nll=best,
                      test=test_result, interventions=controls, ledger=model.ledger(),
                      elapsed_seconds=elapsed+time.perf_counter()-tick, steps=step, batch_hashes=batches,
                      memory_bytes=(int(torch.mps.current_allocated_memory()) if device.type == 'mps' else None))
        initial = torch.load(folder/'initial.pt', map_location=device, weights_only=True)
        model.load_state_dict(initial['model'])
        initial_test = evaluate(model, test, device, cfg['batch_size'], progress=progress)
        write(folder/'untrained_test_predictions.json', initial_test.pop('records'))
        result['untrained_test'] = initial_test
        write(folder/'result.json', result); mark(root, case, 'completed', step=step, result=str(folder/'result.json'))
        tqdm.write(f"完成 {case['id']} | test={test_result['accuracy']:.2%} | NLL={test_result['nll']:.4f}", file=sys.stdout)
        return result
    except KeyboardInterrupt:
        mark(root, case, 'aborted', step=step, safe_checkpoint_step=saved_step, reason='Interrupted; only atomic checkpoints are resumable')
        raise
    except Exception:
        write(folder/'error.json', dict(utc=now(), step=step, safe_checkpoint_step=saved_step, traceback=traceback.format_exc()))
        mark(root, case, 'failed', step=step, safe_checkpoint_step=saved_step)
        raise
    finally:
        del model, optimizer
        if device.type == 'mps':
            torch.mps.empty_cache()


class RunLock:
    def __init__(self, root):
        self.path = root/'.trainer.lock'

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                pid = int(self.path.read_text())
                os.kill(pid, 0)
            except ProcessLookupError:
                self.path.unlink()
            except (ValueError, PermissionError):
                raise RuntimeError(f'Cannot verify lock {self.path}; inspect it before deleting')
            else:
                raise RuntimeError(f'Training process {pid} already owns {self.path}')
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'w') as file:
            file.write(str(os.getpid()))
        return self

    def __exit__(self, *args):
        self.path.unlink(missing_ok=True)


def execute(cfg, root, only=None, dry_run=False, progress=True):
    for name in ('train_steps', 'batch_size', 'eval_every', 'checkpoint_every', 'bptt_steps', 'development_quartets', 'test_quartets', 'progress_every'):
        if not isinstance(cfg[name], int) or cfg[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if cfg['batch_size'] % 4 or cfg['learning_rate'] <= 0:
        raise ValueError('Need complete quartets and positive learning_rate')
    planned = [c for c in cases(cfg) if only is None or c['id'] == only]
    if not planned:
        raise ValueError('No conditions selected')
    root = Path(root).resolve()
    if dry_run:
        print(json.dumps(dict(conditions=len(planned), updates=sum(cfg['train_steps'] for _ in planned),
                              root=str(root), cases=planned), ensure_ascii=False, indent=2)); return
    device = select_device(cfg['device']); torch.set_num_threads(4)
    print(f'设备: {device} | 解释器: {sys.executable} | 条件数: {len(planned)}\n结果目录: {root}', flush=True)
    if device.type == 'cpu':
        print('当前使用 CPU；完整扫描建议在 PyCharm 的本地 MPS 环境运行。', flush=True)
    stop = [False]
    def request_stop(signum, frame):
        if stop[0]:
            raise KeyboardInterrupt
        stop[0] = True
        tqdm.write('收到停止请求：当前更新完成后保存。再次中断会立即退出，保留上次安全检查点。', file=sys.stdout)
    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with RunLock(root):
            identity = dict(config=cfg, sources=source_hashes(), device=str(device))
            manifest = root/'experiment.json'
            if manifest.exists() and json.loads(manifest.read_text()) != identity:
                raise ValueError(f'{root}: experiment/source/device changed; select a new output directory')
            write(manifest, identity)
            validate(root/'data', cfg)
            write(root/'environment.json', dict(python=platform.python_version(), torch=str(torch.__version__),
                  platform=platform.platform(), device=str(device), executable=sys.executable, utc=now()))
            path = root/'conditions.json'; records = json.loads(path.read_text()) if path.exists() else {}
            for identifier, record in records.items():
                if record['status'] == 'running':
                    record.update(status='aborted', utc=now(), reason='Previous process ended without cleanup; recovered under exclusive lock')
            for case in planned:
                records.setdefault(case['id'], dict(condition=case, status='not_executed', history=[]))
            write(path, records)
            for case in tqdm(planned, desc='实验总进度', unit='条件', position=0, dynamic_ncols=True,
                             disable=not progress, file=sys.stdout):
                if stop[0]:
                    break
                try:
                    if train(case, cfg, root, device, stop, progress) is None:
                        break
                except Exception as error:
                    mark(root, case, 'failed', reason=repr(error))
                    from .report import report
                    report(root)
                    raise
                from .report import report
                report(root)
            from .report import report
            report(root)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=PROJECT/'configs/v04/base.json')
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'))
    parser.add_argument('--steps', type=int); parser.add_argument('--nodes', nargs='+', type=int)
    parser.add_argument('--seeds', nargs='+', type=int); parser.add_argument('--variants', nargs='+')
    parser.add_argument('--read-paths', nargs='+', choices=('shared', 'separated'))
    parser.add_argument('--only'); parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--no-progress', action='store_true'); parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); cfg = json.loads(args.config.read_text())
    for name in ('device', 'nodes', 'seeds', 'variants', 'read_paths'):
        if getattr(args, name) is not None:
            cfg[name] = getattr(args, name)
    if args.steps is not None:
        cfg['train_steps'] = args.steps
    if args.smoke:
        cfg.update(train_steps=2, internal_steps=2, nodes=[32], seeds=[11], variants=['graph', 'full'],
                   read_paths=['shared'], development_quartets=1, test_quartets=1, eval_every=2, checkpoint_every=1)
        if args.root == DEFAULT_ROOT:
            args.root = PROJECT/'results/v04/smoke'
    execute(cfg, args.root, args.only, args.dry_run, not args.no_progress)


if __name__ == '__main__':
    main()
