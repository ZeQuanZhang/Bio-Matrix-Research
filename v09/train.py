"""Explicit single-condition training on the frozen public corpus; never auto-runs."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from v07.common import PROJECT, RunLock, detach, device_for, digest, file_hash, save, sync, write
from v07.data import ContinuousStream, ROOT
from v07.model import invalidate_feedback
from v07.run import move_state, recover_logs, restore_rng, rng_state
from .benchmark import Progress, append, build, frozen_corpus, initialize_model, sources


FORMAT = 'v09-continuous-training-1'
CONDITIONS = ('v09_dense', 'v09_sparse75', 'v09_sparse50', 'v08_seq', 'transformer', 'content_only')
DEFAULTS = dict(condition='v09_dense', nodes=128, tokens=12000000, batch=2, sequence_tokens=64,
                session_tokens=4096, context_limit=4096, seed=11, data_seed=7101, evaluation_seed=7191,
                validation_windows=16, eval_every=50000, checkpoint_every=100,
                group_size=16, route_mode='summary', content_lr=3e-4, control_lr=1e-4,
                time_lr=3e-5, weight_decay=.01, gradient_clip=1., warmup_fraction=.02,
                minimum_lr_fraction=.1, intrusion_tokens=64, intrusion_interval=512,
                marked_probability=.5)
MILESTONES = (250000, 1000000, 3000000, 4000000, 8000000, 12000000)


def parameter_groups(model, config):
    groups = {}
    time_parameters = {'raw_tau', 'raw_trace_tau', 'raw_slow_rate', 'raw_feedback_tau'}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        category = ('time' if name in time_parameters else
                    'content' if name.split('.')[0] in ('embedding', 'input_projection', 'content_projection',
                        'read_projection', 'norm', 'ff', 'output_norm', 'blocks') else 'control')
        decay = 0. if parameter.ndim < 2 or name.endswith('.bias') or name.startswith('raw_') else config['weight_decay']
        key = category, decay
        if key not in groups:
            rate = config[category+'_lr']
            groups[key] = dict(params=[], name=category+('_no_decay' if not decay else '_decay'),
                               lr=rate, base_lr=rate, weight_decay=decay)
        groups[key]['params'].append(parameter)
    return list(groups.values())


def learning_rate_scale(completed, total, warmup_fraction=.02, minimum=.1):
    if total <= 0 or not 0 <= completed <= total or not 0 < warmup_fraction < 1 or not 0 <= minimum <= 1:
        raise ValueError('Invalid fixed-budget learning-rate schedule')
    warmup = max(1., total*warmup_fraction)
    if completed < warmup:
        return completed/warmup
    fraction = (completed-warmup)/max(1., total-warmup)
    return minimum+(1-minimum)*.5*(1+math.cos(math.pi*fraction))


def curriculum(model, completed, total):
    if hasattr(model, 'training_progress'):
        return model.training_progress(completed, total)
    if hasattr(model, 'gate_blend'):
        model.gate_blend = min(1., completed/max(1., .1*total))
        return dict(gate_blend=model.gate_blend)
    return {}


def select_targets(mask, remaining):
    selected = np.array(mask, dtype=bool, copy=True)
    if remaining < 1:
        raise ValueError('remaining target budget must be positive')
    positions = np.flatnonzero(selected)
    selected.flat[positions[remaining:]] = False
    return selected


def update(model, optimizer, values, mask, reset, state, device, gradient_clip=1.):
    """Only text and the session-start reset reach the model; masks affect CE only."""
    values, mask, reset = np.asarray(values), np.asarray(mask), np.asarray(reset)
    if values.ndim != 2 or mask.shape != (values.shape[0], values.shape[1]-1) or reset.shape != mask.shape:
        raise ValueError('values/mask/reset must be matching [B,T+1]/[B,T] arrays')
    if mask.dtype != np.bool_ or reset.dtype != np.bool_ or bool(reset[:, 1:].any()) or not bool(mask.any()):
        raise ValueError('Need a nonempty boolean loss mask and only chunk-start session resets')
    model.train(); optimizer.zero_grad(set_to_none=True)
    x = torch.as_tensor(values[:, :-1].copy(), dtype=torch.long, device=device)
    target = torch.as_tensor(values[:, 1:].copy(), dtype=torch.long, device=device)
    selected = torch.as_tensor(mask, dtype=torch.bool, device=device)
    reset_lanes = torch.as_tensor(reset[:, 0], dtype=torch.bool, device=device)
    sync(device); started = time.perf_counter()
    logits, state, diagnostics = model(x, state, reset_mask=reset_lanes)
    losses = F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction='none')
    loss = losses[selected.flatten()].mean()
    if not bool(loss.detach().isfinite().cpu()):
        raise FloatingPointError('Nonfinite loss; optimizer was not advanced')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
    optimizer.step(); state = invalidate_feedback(detach(state))
    sync(device); seconds = time.perf_counter()-started
    valid = int(mask.sum())
    row = dict(loss=float(loss.detach().cpu()), gradient_norm=float(norm.detach().cpu()),
               targets=valid, visible_inputs=int(mask.size), masked_targets=int(mask.size-valid),
               seconds=seconds, tokens_per_second=valid/seconds,
               diagnostics={name: float(value.detach().cpu()) for name, value in diagnostics.items()})
    return row, state


def capture_rng(device):
    numpy = np.random.get_state()
    return dict(torch=rng_state(device), python=random.getstate(),
                numpy=(numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]))


def restore_random(value, device):
    restore_rng(value['torch'], device); random.setstate(value['python'])
    numpy = value['numpy']
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), numpy[2], numpy[3], numpy[4]))


def checkpoint_payload(model, optimizer, state, stream, protocol, counters, device):
    return dict(format=FORMAT, protocol=protocol, model=model.state_dict(), optimizer=optimizer.state_dict(),
                runtime_state=move_state(detach(state), 'cpu'), stream=stream.state_dict(),
                rng=capture_rng(device), counters=dict(counters),
                runtime_attributes={name: getattr(model, name) for name in ('gate_blend', 'active_fraction') if hasattr(model, name)})


def save_last(root, model, optimizer, state, stream, protocol, counters, device, milestones=()):
    if training_sources() != protocol['source_hashes']:
        raise RuntimeError('Source changed during training; keep the last safe checkpoint')
    payload = checkpoint_payload(model, optimizer, state, stream, protocol, counters, device)
    best = root/'best.pt'
    payload['best_snapshot'] = torch.load(best, map_location='cpu', weights_only=True) if best.is_file() else None
    # Milestones precede last.pt: a crash cannot advance last past an unsaved
    # snapshot. Resuming an older checkpoint archives future snapshots below.
    for nominal in milestones:
        path = root/f'milestone_{nominal}_at_{counters["completed_tokens"]}.pt'
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite milestone: {path}')
        save(path, payload | dict(milestone=dict(nominal_tokens=nominal,
                                                actual_tokens=counters['completed_tokens'])))
    save(root/'last.pt', payload)


def crossed_milestones(previous, completed, total):
    return tuple(value for value in MILESTONES if previous < value <= completed and value <= total)


def archive_future_milestones(root, completed):
    for path in root.glob('milestone_*_at_*.pt'):
        match = re.fullmatch(r'milestone_(\d+)_at_(\d+)\.pt', path.name)
        if match and int(match[2]) > completed:
            path.rename(root/f'{path.stem}.pre_resume-{time.time_ns()}.pt')


def write_live(root, status, phase, counters, config, device, metrics=None, validation=None):
    row = dict(status=status, phase=phase, completed_tokens=counters['completed_tokens'],
               total_tokens=config['tokens'], step=counters['step'],
               updated_utc=datetime.now(timezone.utc).isoformat())
    if metrics:
        row.update({name: metrics[name] for name in ('loss', 'gradient_norm', 'tokens_per_second') if name in metrics})
    if validation is not None:
        row['validation'] = validation
    if device.type == 'mps':
        row['resource'] = dict(mps_allocated_bytes=torch.mps.current_allocated_memory(),
                               mps_driver_bytes=torch.mps.driver_allocated_memory())
    write(root/'live.json', row)


def restore_checkpoint(payload, model, optimizer, stream, protocol, device):
    if payload.get('format') != FORMAT or payload['protocol'] != protocol:
        raise ValueError('Resume requires the identical budget, configuration, source, data, device and Torch version')
    model.load_state_dict(payload['model']); optimizer.load_state_dict(payload['optimizer'])
    stream.load_state_dict(payload['stream'])
    for name, value in payload['runtime_attributes'].items():
        setattr(model, name, value)
    restore_random(payload['rng'], device)
    return move_state(payload['runtime_state'], device), dict(payload['counters'])


def validation_data(corpus, config):
    values, docs, starts = corpus.sample('validation', np.random.default_rng(config['evaluation_seed']),
                                        config['validation_windows'], config['sequence_tokens'])
    return values, dict(split='validation', documents=docs, starts=starts, sha256=digest(values.tolist()))


@torch.no_grad()
def evaluate(model, values, device, batch, ui=None, progress_callback=None):
    was_training = model.training; random_state = capture_rng(device)
    loss_sum, count = 0., 0
    model.eval()
    if ui:
        ui.validation(len(values))
    try:
        for start in range(0, len(values), batch):
            tokens = torch.as_tensor(values[start:start+batch].copy(), dtype=torch.long, device=device)
            logits, _, _ = model(tokens[:, :-1], None)
            loss = F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten(), reduction='sum')
            loss_sum += float(loss.cpu()); count += int(tokens[:, 1:].numel())
            if ui:
                ui.validation_step(len(tokens))
            if progress_callback:
                progress_callback(dict(completed=min(start+batch, len(values)), total=len(values), nll=loss_sum/count))
        score = loss_sum/count
        if not math.isfinite(score):
            raise FloatingPointError('Nonfinite development loss')
        return dict(nll=score, targets=count, windows=len(values), cold_state=True)
    finally:
        model.train(was_training); restore_random(random_state, device)


class TrainingProgress:
    def __init__(self, budget, device, enabled=True, initial=0):
        self.ui = Progress([dict(id='训练'), dict(id='验证')], budget, enabled)
        self.ui.device = device
        self.ui.total.set_description_str('v0.9 有效目标 token')
        self.ui.total.reset(total=budget); self.ui.total.update(initial)
        self.ui.details['训练'].update(initial)
        self.completed = initial

    def stage(self, message):
        self.ui.stage(message)

    def training(self, completed, row):
        increment = completed-self.completed; self.completed = completed
        self.ui.total.update(increment); self.ui.details['训练'].update(increment)
        self.ui.details['训练'].set_postfix_str(f'loss={row["loss"]:.4f} 目标/s={row["tokens_per_second"]:.1f}', refresh=False)

    def validation(self, count):
        self.stage('固定 validation 窗口 · 每窗冷状态')
        self.ui.details['验证'].reset(total=count)

    def validation_step(self, count):
        self.ui.details['验证'].update(count)

    def close(self):
        self.ui.close()


class StopRequest:
    """Finish the current update before writing a consistent resumable checkpoint."""
    def __init__(self):
        self.requested = False
        self.previous = {}
    def __enter__(self):
        for name in (signal.SIGINT, signal.SIGTERM):
            self.previous[name] = signal.signal(name, self._request)
        return self
    def _request(self, *_):
        self.requested = True
    def __exit__(self, *_):
        for name, previous in self.previous.items():
            signal.signal(name, previous)


def training_sources():
    extra = PROJECT/'.run/Bio-Matrix v0.9 训练.run.xml'
    return sources() | ({str(extra.relative_to(PROJECT)): file_hash(extra)} if extra.is_file() else {})


def settings(args, payload=None):
    saved = payload['protocol']['training'] if payload is not None else DEFAULTS
    result = {}
    for name in DEFAULTS:
        given = getattr(args, name, None)
        if payload is not None and given is not None and given != saved[name]:
            raise ValueError(f'Resume cannot change {name}; start a separate run')
        result[name] = saved[name] if given is None else given
    if (result['condition'] not in CONDITIONS or result['nodes'] not in (32, 128, 512)
            or min(result[name] for name in ('tokens', 'batch', 'sequence_tokens', 'session_tokens',
                'validation_windows', 'eval_every', 'checkpoint_every', 'group_size', 'context_limit')) < 1
            or result['sequence_tokens'] % 16 or result['session_tokens'] % result['sequence_tokens']):
        raise ValueError('Need supported condition/nodes, positive counts, C16 chunks dividing sessions')
    if any(result[name] <= 0 for name in ('content_lr', 'control_lr', 'time_lr', 'gradient_clip')) or result['weight_decay'] < 0:
        raise ValueError('Invalid optimizer settings')
    if any(not math.isfinite(result[name]) for name in ('content_lr', 'control_lr', 'time_lr',
            'gradient_clip', 'weight_decay', 'warmup_fraction', 'minimum_lr_fraction', 'marked_probability')):
        raise ValueError('Training settings must be finite')
    learning_rate_scale(0, result['tokens'], result['warmup_fraction'], result['minimum_lr_fraction'])
    return result


def train(args):
    device = device_for(args.device)
    if device.type == 'mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise RuntimeError('Disable MPS fallback for native training')
    torch.set_num_threads(args.threads)
    resume = Path(args.resume).resolve() if args.resume else None
    payload = torch.load(resume, map_location='cpu', weights_only=True) if resume else None
    config = settings(args, payload)
    root = resume.parent if resume else Path(args.output or PROJECT/'results/v09'/datetime.now().strftime(
        f'train_{config["condition"]}_n{config["nodes"]}_s{config["seed"]}_%Y%m%dT%H%M%S'))
    if resume and args.output is not None and Path(args.output).resolve() != root:
        raise ValueError('Resume appends to its existing directory; --output must match that directory')
    if not resume and root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Refusing to overwrite training: {root}')
    root.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root or (payload['protocol']['data_root'] if payload else ROOT))
    corpus = frozen_corpus(data_root)
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    record = dict(condition=config['condition'], nodes=config['nodes'])
    model, model_config = build(record, corpus.manifest['vocab_size'], config['group_size'], config['route_mode'],
                                context_limit=config['context_limit'])
    initialize_model(model, config['condition'], config['seed']); model.to(device)
    optimizer = torch.optim.AdamW(parameter_groups(model, config))
    stream = ContinuousStream(corpus, config['batch'], config['sequence_tokens'], config['data_seed'], config['seed'],
               session_tokens=config['session_tokens'], intrusion_tokens=config['intrusion_tokens'],
               intrusion_interval=config['intrusion_interval'], marked_probability=config['marked_probability'],
               reset_policy='session')
    source = training_sources()
    protocol = dict(training=config, model=asdict(model_config), stream=stream.protocol,
                    source_hashes=source, data_manifest_sha256=digest(corpus.manifest),
                    data_root=str(data_root.resolve()), device=str(device), torch=str(torch.__version__), dtype='float32')
    state = None
    counters = dict(step=0, completed_tokens=0, visible_inputs=0, validated_step=0, validated_tokens=0,
                    best_nll=None, best_step=None, train_seconds=0.)
    dev_values, dev_sample = validation_data(corpus, config)
    ui = None
    with RunLock(root/'.training.lock'), StopRequest() as stop:
        if payload is not None:
            state, counters = restore_checkpoint(payload, model, optimizer, stream, protocol, device)
            recover_logs(root, counters['step'], counters['validated_step'])
            archive_future_milestones(root, counters['completed_tokens'])
            if resume.name != 'last.pt' and (root/'last.pt').is_file():
                (root/'last.pt').rename(root/f'last.pre_resume-{time.time_ns()}.pt')
            old_best = root/'best.pt'
            if old_best.is_file():
                old_best.rename(root/f'best.pre_resume-{time.time_ns()}.pt')
            best = payload.get('best_snapshot')
            if best is None and resume.name == 'best.pt':
                best = {key: value for key, value in payload.items() if key != 'best_snapshot'}
            if best is not None:
                save(old_best, best)
        else:
            write(root/'protocol.json', protocol)
            write(root/'validation_sample.json', dev_sample)
            for name in source:
                path = root/'source'/name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((PROJECT/name).read_bytes())
        write(root/'status.json', dict(status='running', **counters))
        last_metrics = {}
        write_live(root, 'running', 'preparing', counters, config, device)
        save_last(root, model, optimizer, state, stream, protocol, counters, device)
        ui = TrainingProgress(config['tokens'], device, not args.quiet, counters['completed_tokens'])

        def record_validation():
            def progress(value):
                write_live(root, 'running', 'validation', counters, config, device, last_metrics, value)
            progress(dict(completed=0, total=len(dev_values)))
            score = evaluate(model, dev_values, device, config['batch'], ui, progress_callback=progress)
            score.update(step=counters['step'], completed_tokens=counters['completed_tokens'], sample=dev_sample)
            counters['validated_step'] = counters['step']; counters['validated_tokens'] = counters['completed_tokens']
            append(root/'validation.jsonl', score)
            if counters['best_nll'] is None or score['nll'] < counters['best_nll']:
                counters['best_nll'], counters['best_step'] = score['nll'], counters['step']
                save(root/'best.pt', checkpoint_payload(model, optimizer, state, stream, protocol, counters, device))

        try:
            pending_validation = (counters['completed_tokens']//config['eval_every'] > counters['validated_tokens']//config['eval_every']
                                  or counters['completed_tokens'] == config['tokens'] != counters['validated_tokens'])
            if pending_validation and not stop.requested:
                record_validation()
                save_last(root, model, optimizer, state, stream, protocol, counters, device)
            while counters['completed_tokens'] < config['tokens'] and not stop.requested:
                ui.stage(f'{config["condition"]} N{config["nodes"]} · 连续状态训练 · step {counters["step"]+1}')
                write_live(root, 'running', 'training', counters, config, device, last_metrics)
                values, mask, reset, sample = stream.next()
                selected = select_targets(mask, config['tokens']-counters['completed_tokens'])
                rate = learning_rate_scale(counters['completed_tokens']+int(selected.sum()), config['tokens'],
                                           config['warmup_fraction'], config['minimum_lr_fraction'])
                for group in optimizer.param_groups:
                    group['lr'] = group['base_lr']*rate
                course = curriculum(model, counters['completed_tokens'], config['tokens'])
                previous_targets = counters['completed_tokens']
                metrics, state = update(model, optimizer, values, selected, reset, state, device, config['gradient_clip'])
                counters['step'] += 1; counters['completed_tokens'] += metrics['targets']
                counters['visible_inputs'] += metrics['visible_inputs']; counters['train_seconds'] += metrics['seconds']
                metrics.update(step=counters['step'], completed_tokens=counters['completed_tokens'], lr_scale=rate,
                               learning_rates={group['name']: group['lr'] for group in optimizer.param_groups},
                               curriculum=course, raw_valid_targets=int(mask.sum()), sample=sample)
                append(root/'metrics.jsonl', metrics); ui.training(counters['completed_tokens'], metrics)
                last_metrics = metrics
                write_live(root, 'running', 'training', counters, config, device, metrics)
                due = (counters['completed_tokens']//config['eval_every'] > counters['validated_tokens']//config['eval_every']
                       or counters['completed_tokens'] == config['tokens'])
                if due and not stop.requested:
                    record_validation()
                milestones = crossed_milestones(previous_targets, counters['completed_tokens'], config['tokens'])
                if counters['step'] % config['checkpoint_every'] == 0 or due or stop.requested or milestones:
                    write_live(root, 'running', 'checkpoint', counters, config, device, metrics)
                    save_last(root, model, optimizer, state, stream, protocol, counters, device, milestones)
                    write(root/'status.json', dict(status='running', **counters))
            if counters['completed_tokens'] == config['tokens'] and counters['validated_tokens'] != config['tokens'] and not stop.requested:
                record_validation()
            save_last(root, model, optimizer, state, stream, protocol, counters, device)
            status = 'completed' if counters['completed_tokens'] == config['tokens'] and counters['validated_tokens'] == config['tokens'] else 'aborted'
            write(root/'status.json', dict(status=status, **counters))
            write_live(root, status, status, counters, config, device, last_metrics)
            write(root/'result.json', dict(status=status, **counters, test_evaluated=False,
                    note='Final update can consume an unscored input tail to keep C16/session boundaries; effective target budget is exact.'))
            return root, status
        except BaseException:
            (root/'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
            write(root/'status.json', dict(status='failed', last_safe_checkpoint='last.pt', **counters))
            write_live(root, 'failed', 'failed', counters, config, device, last_metrics)
            raise
        finally:
            if ui:
                ui.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'), default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--quiet', action='store_true')
    for name, default in DEFAULTS.items():
        options = dict(default=None, type=type(default))
        if name == 'condition':
            options['choices'] = CONDITIONS
        elif name == 'route_mode':
            options['choices'] = ('summary', 'cyclic')
        parser.add_argument('--'+name.replace('_', '-'), **options)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    root, status = train(args)
    print(f'v0.9 training {status}: {root}')


if __name__ == '__main__':
    main()
