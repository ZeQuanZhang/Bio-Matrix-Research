"""Local v1.0 paired memory-adaptation pilot; no downloads or remote training."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime
import io
import json
import math
import os
from pathlib import Path
import random
import time
import traceback
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from v07.common import PROJECT, RunLock, detach, device_for, digest, file_hash, save, sync, write
from v07.data import Corpus, ROOT as DATA_ROOT, prepare
from v07.model import Config as LegacyConfig, Transformer
from v07.run import move_state
from v09.benchmark import Progress, append
from v09.train import StopRequest, capture_rng, restore_random, learning_rate_scale
from .model import Config, Network


FORMAT = 'v1-transformer-bio-pilot-1'
CONDITIONS = ('transformer', 'adapter', 'bio_no_phase', 'bio_full')
DEFAULTS = dict(backbone=str(PROJECT/'results/v09/comparison_no_v08/transformer_n32_s11/last.pt'),
                data_root=str(DATA_ROOT), tokens=65_536, batch=2, segments=8,
                segment_tokens=64, context_limit=128, memory_heads=4, memory_dim=8,
                bands=3, nodes=32, internal_steps=2, full_dynamics=True, active_fraction=1.,
                seed=11, data_seed=11001, evaluation_seed=11091,
                validation_windows=8, eval_every=16_384, checkpoint_every=10,
                learning_rate=3e-4, weight_decay=.01, gradient_clip=1.,
                ce_weight=1., kd_weight=1., kd_topk=64, consolidate_every=100,
                threads=4)


class ParallelFFN(nn.Module):
    def __init__(self, base, width, rank):
        super().__init__()
        self.base = base
        self.adapter = nn.Sequential(nn.Linear(width, rank, bias=False), nn.GELU(),
                                     nn.Linear(rank, width, bias=False))
        nn.init.zeros_(self.adapter[-1].weight)

    def forward(self, hidden):
        return self.base(hidden)+self.adapter(hidden)


def source_hashes():
    paths = [*sorted((PROJECT/'v1').glob('*.py')), PROJECT/'run_biomatrix_v1.py',
             PROJECT/'.run/Bio-Matrix v1.0.run.xml', PROJECT/'v07/common.py',
             PROJECT/'v07/data.py', PROJECT/'v07/model.py', PROJECT/'v07/run.py',
             PROJECT/'v09/benchmark.py', PROJECT/'v09/train.py']
    paths += [PROJECT/path for path in ('v03/graph.py', 'v04/model.py', 'v05/model.py',
              'v06/model.py', 'v06/common.py', 'v06/graph.py', 'v07/graph.py',
              'v07/evaluate.py', 'v08/benchmark.py', 'v08/model.py', 'v08/recurrence.py')]
    return {str(path.relative_to(PROJECT)): file_hash(path) for path in paths if path.is_file()}


def settings(args, saved=None):
    result = {}
    for name, default in DEFAULTS.items():
        if saved is not None and name not in saved:
            raise ValueError('Saved protocol predates this implementation; use its source snapshot or a fresh output')
        old = default if saved is None else saved[name]
        given = getattr(args, name, None)
        if saved is not None and given is not None and given != old:
            raise ValueError(f'Resume cannot change {name}; choose a fresh output directory')
        result[name] = old if given is None else given
    for name in ('backbone', 'data_root'):
        result[name] = str(Path(result[name]).resolve())
    counts = ('tokens', 'batch', 'segments', 'segment_tokens', 'context_limit', 'memory_heads',
              'memory_dim', 'bands', 'nodes', 'internal_steps', 'validation_windows', 'eval_every',
              'checkpoint_every', 'kd_topk', 'consolidate_every', 'threads')
    if any(isinstance(result[name], bool) or not isinstance(result[name], int) or result[name] < 1 for name in counts):
        raise ValueError('All budgets, dimensions and intervals must be positive integers')
    if result['segments'] < 2 or result['segment_tokens'] > result['context_limit']:
        raise ValueError('Need at least two differentiable memory segments and segment <= local context')
    rates = ('learning_rate', 'weight_decay', 'gradient_clip', 'ce_weight', 'kd_weight')
    if any(not math.isfinite(result[name]) or result[name] < 0 for name in rates):
        raise ValueError('Optimizer and loss weights must be finite and nonnegative')
    if min(result['learning_rate'], result['gradient_clip']) <= 0 or result['ce_weight']+result['kd_weight'] <= 0:
        raise ValueError('Need positive learning rate, clip and at least one loss objective')
    if not isinstance(result['full_dynamics'], bool):
        raise ValueError('full_dynamics must be boolean')
    if not math.isfinite(result['active_fraction']) or not 0 < result['active_fraction'] <= 1:
        raise ValueError('active_fraction must be in (0,1]')
    return result


def load_corpus(root):
    root = Path(root)
    if not (root/'manifest.json').is_file():
        raise FileNotFoundError('v1.0 requires the existing frozen public corpus; this entry does not download data')
    prepare(root, vocab_size=4096)
    return Corpus(root)


def architecture(payload, config):
    old = payload['protocol']['model']
    return Config(vocab_size=old['vocab_size'], width=old['transformer_width'],
                  layers=old['transformer_layers'], heads=old['transformer_heads'],
                  ff=old['transformer_ff'], context_limit=config['context_limit'],
                  segment_tokens=config['segment_tokens'], memory_heads=config['memory_heads'],
                  memory_dim=config['memory_dim'], bands=config['bands'],
                  nodes=config['nodes'], internal_steps=config['internal_steps'],
                  full_dynamics=config['full_dynamics'], active_fraction=config['active_fraction'])


def build_models(payload, config):
    """Pair all shared tensors, including the phase ablation, before adaptation."""
    cfg = architecture(payload, config)
    torch.manual_seed(config['seed'])
    full = Network(cfg).load_backbone(payload, freeze=True)
    shared = full.state_dict()
    no_phase = Network(replace(cfg, phase_enabled=False)).load_backbone(payload, freeze=True)
    compatible = {name: value for name, value in shared.items()
                  if name in no_phase.state_dict() and no_phase.state_dict()[name].shape == value.shape}
    no_phase.load_state_dict(compatible, strict=False)
    baseline = Network(replace(cfg, memory_enabled=False)).load_backbone(payload, freeze=True)
    adapter = Network(replace(cfg, memory_enabled=False)).load_backbone(payload, freeze=True)
    target = sum(p.numel() for p in full.parameters() if p.requires_grad)
    rank = max(1, round(target/(2*cfg.width*cfg.layers)))
    for block in adapter.blocks:
        block.ff = ParallelFFN(block.ff, cfg.width, rank)
    models = dict(transformer=baseline, adapter=adapter, bio_no_phase=no_phase, bio_full=full)
    counts = {name: dict(total_parameters=sum(p.numel() for p in model.parameters()),
                         trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
              for name, model in models.items()}
    counts['adapter'].update(rank=rank, target_trainable_parameters=target,
                             trainable_parameter_difference=counts['adapter']['trainable_parameters']-target)
    return models, cfg, counts


def build_teacher(payload, device):
    cfg = LegacyConfig(**payload['protocol']['model'])
    teacher = Transformer(cfg)
    teacher.load_state_dict(payload['model'])
    teacher.requires_grad_(False)
    return teacher.to(device).eval()


def optimizer_for(model, config):
    groups = {False: [], True: []}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            groups[parameter.ndim > 1 and not name.endswith('.bias')].append(parameter)
    if not any(groups.values()):
        raise ValueError('This condition has no trainable parameters; score it as a frozen reference')
    return torch.optim.AdamW([dict(params=parameters, weight_decay=config['weight_decay'] if decay else 0.)
                              for decay, parameters in groups.items() if parameters], lr=config['learning_rate'])


def forward_segments(model, tokens, segment_tokens, **options):
    state, outputs, diagnostics = None, [], {}
    for start in range(0, tokens.shape[1], segment_tokens):
        logits, state, diagnostics = model(tokens[:, start:start+segment_tokens], state, **options)
        outputs.append(logits)
    return torch.cat(outputs, dim=1), state, diagnostics


def objective(student, target, teacher, mask, config):
    ce = F.cross_entropy(student.flatten(0, 1), target.flatten(), reduction='none').reshape_as(target)
    top_values, top_indices = teacher.float().topk(min(config['kd_topk'], teacher.shape[-1]), dim=-1)
    q = top_values.softmax(-1)
    teacher_log = top_values.log_softmax(-1)
    student_log = student.float().log_softmax(-1).gather(-1, top_indices)
    kd = (q*(teacher_log-student_log)).sum(-1)
    ce_loss, kd_loss = ce[mask].mean(), kd[mask].mean()
    return config['ce_weight']*ce_loss+config['kd_weight']*kd_loss, ce_loss, kd_loss


def scalar_diagnostics(values):
    result = {}
    for name, value in values.items():
        if isinstance(value, torch.Tensor):
            value = float(value.detach().float().mean().cpu())
        if isinstance(value, (int, float, bool)) and math.isfinite(value):
            result[name] = value
    return result


def update(model, teacher, optimizer, values, config, device, remaining):
    """One backward spans all segments; runtime gradients stop only after the step."""
    if remaining < 1 or values.ndim != 2 or values.shape[1] < 3:
        raise ValueError('Need a positive budget and a contiguous multi-token document window')
    inputs = torch.as_tensor(values[:, :-1].copy(), dtype=torch.long, device=device)
    targets = torch.as_tensor(values[:, 1:].copy(), dtype=torch.long, device=device)
    mask = torch.arange(targets.numel(), device=device).reshape_as(targets) < remaining
    if not bool(mask.any()):
        raise ValueError('Empty supervision mask')
    model.train(); optimizer.zero_grad(set_to_none=True)
    sync(device); started = time.perf_counter()
    with torch.no_grad():
        teacher_logits, _, _ = forward_segments(teacher, inputs, teacher.cfg.context_limit)
    logits, state, diagnostics = forward_segments(model, inputs, config['segment_tokens'])
    loss, ce, kd = objective(logits, targets, teacher_logits, mask, config)
    if not bool(loss.detach().isfinite()):
        raise FloatingPointError('Nonfinite objective; optimizer was not advanced')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
    optimizer.step()
    state = detach(state)
    sync(device); seconds = time.perf_counter()-started
    row = dict(loss=float(loss.detach().cpu()), ce=float(ce.detach().cpu()), kd=float(kd.detach().cpu()),
               gradient_norm=float(norm.detach().cpu()), targets=int(mask.sum().cpu()),
               visible_inputs=int(inputs.numel()), seconds=seconds,
               targets_per_second=int(mask.sum().cpu())/max(seconds, 1e-9),
               diagnostics=scalar_diagnostics(diagnostics))
    return row, state


@torch.no_grad()
def evaluate(model, values, config, device, progress=None):
    """Cold score plus paired interventions that preserve the recent local KV."""
    from .evaluate import evaluate_history
    was_training = model.training
    model.eval()
    segment = config['segment_tokens']
    prefix = max(1, config['segments']-2)*segment
    if values.ndim != 2 or values.shape[1] <= prefix+1:
        raise ValueError('Validation needs history and a complete scoring segment')
    try:
        result = evaluate_history(model, values, device, prefix, batch=config['batch'],
                                  seed=config['evaluation_seed'], progress=progress)
        cold = []
        for start in range(0, len(values), config['batch']):
            end = min(start+config['batch'], len(values))
            inputs = torch.as_tensor(values[start:end, prefix:-1].copy(), dtype=torch.long, device=device)
            targets = torch.as_tensor(values[start:end, prefix+1:].copy(), dtype=torch.long, device=device)
            logits, _, _ = model(inputs, None)
            losses = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), reduction='none')
            cold.extend(losses.reshape(len(inputs), -1).mean(1).cpu().tolist())
            if progress:
                progress('cold', end, len(values))
        if not all(math.isfinite(value) for value in cold):
            raise FloatingPointError('Nonfinite validation score')
        result['conditions']['cold'] = dict(status='completed', nll=float(np.mean(cold)),
                                            per_window_nll=cold, windows=len(cold))
        result['cold_nll'] = float(np.mean(cold))
        return result
    finally:
        model.train(was_training)


def verify(root=None):
    suite = unittest.defaultTestLoader.discover(str(PROJECT/'v1'), pattern='test_*.py', top_level_dir=str(PROJECT))
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    record = dict(status='completed' if result.wasSuccessful() else 'failed', tests=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped))
    if root is not None:
        (Path(root)/'verification.txt').write_text(output.getvalue())
        write(Path(root)/'verification.json', record)
    if not result.wasSuccessful():
        raise RuntimeError('v1.0 self-check failed; see verification.txt. Training was not started.')
    return record


def validation_data(corpus, config):
    length = config['segments']*config['segment_tokens']
    values, docs, starts = corpus.sample('validation', np.random.default_rng(config['evaluation_seed']),
                                        config['validation_windows'], length)
    return values, dict(documents=docs, starts=starts, sha256=digest(values.tolist()),
                        scoring='Only the fixed suffix after max(1, segments-2) segments is scored; normally the last two segments. Prefix is unscored.')


def save_checkpoint(path, model, optimizer, state, counters, data_rng, protocol, device):
    if source_hashes() != protocol['source_hashes']:
        raise RuntimeError('Sources changed; retain the last safe checkpoint and start a new run')
    save(path, dict(format=FORMAT, protocol=protocol, model=model.state_dict(),
                    optimizer=optimizer.state_dict(), runtime_state=move_state(detach(state), 'cpu'),
                    counters=counters.copy(), rng=capture_rng(device), data_rng=data_rng.bit_generator.state))


def restore_checkpoint(path, model, optimizer, data_rng, protocol, device):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format') != FORMAT or payload.get('protocol') != protocol:
        raise ValueError('Resume requires identical source, data, budget, model, device and Torch version')
    model.load_state_dict(payload['model'])
    optimizer.load_state_dict(payload['optimizer'])
    data_rng.bit_generator.state = payload['data_rng']
    restore_random(payload['rng'], device)
    return dict(payload['counters'])


def train(args, verify_first=True):
    device = device_for(getattr(args, 'device', 'auto'))
    if device.type == 'mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise RuntimeError('Disable MPS fallback for native v1.0 execution')
    resume = Path(args.resume).resolve() if getattr(args, 'resume', None) else None
    if resume and resume.suffix == '.pt':
        if resume.name != 'last.pt':
            raise ValueError('Resume uses only the last safe optimizer boundary: last.pt')
        resume = resume.parent.parent
    previous = json.loads((resume/'protocol.json').read_text()) if resume else None
    config = settings(args, previous['training'] if previous else None)
    if resume and getattr(args, 'output', None) and Path(args.output).resolve() != resume:
        raise ValueError('Resume output must be its existing experiment directory')
    root = (resume or Path(getattr(args, 'output', None) or PROJECT/'results/v1'/datetime.now().strftime('pilot_%Y%m%dT%H%M%S_%f'))).resolve()
    if not resume and root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Refusing to overwrite previous results: {root}')
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('TQDM_NROWS', '24')
    ui = Progress([dict(id=name) for name in CONDITIONS], config['tokens'], not getattr(args, 'quiet', False))
    ui.total.set_description_str('v1.0 已结束条件')
    ui.device = device
    registry = {name: dict(status='not_executed', completed_tokens=0) for name in CONDITIONS}
    if resume:
        registry = json.loads((root/'registry.json').read_text())['conditions']
    owns_run = False
    try:
        ui.stage('v1.0 代码自检 · 尚未启动适配训练')
        if verify_first:
            checks = root if resume is None else root/'resume_attempts'/datetime.now().strftime('%Y%m%dT%H%M%S_%f')
            checks.mkdir(parents=True, exist_ok=True)
            verify(checks)
        with RunLock(PROJECT/'results/v1/.pilot.lock'), StopRequest() as stop:
            torch.set_num_threads(config['threads'])
            ui.stage('v1.0 固定数据校验与本地主干加载')
            corpus = load_corpus(config['data_root'])
            payload = torch.load(config['backbone'], map_location='cpu', weights_only=True)
            if payload['protocol']['model']['vocab_size'] != corpus.manifest['vocab_size']:
                raise ValueError('Backbone and frozen tokenizer vocabulary mismatch')
            old_data = payload['protocol'].get('data_manifest_sha256')
            fixture = payload['protocol'].get('fixture') is True and 'fixture' in corpus.manifest
            if old_data is None and not fixture:
                raise ValueError('Backbone lacks tokenizer/data identity; vocabulary size alone is insufficient')
            if old_data is not None and old_data != digest(corpus.manifest):
                raise ValueError('Backbone was trained with a different tokenizer/data manifest')
            models, cfg, counts = build_models(payload, config)
            protocol = dict(format=FORMAT, training=config, model=asdict(cfg), parameter_counts=counts,
                            source_hashes=source_hashes(), data_manifest_sha256=digest(corpus.manifest),
                            backbone_sha256=file_hash(config['backbone']), device=str(device), torch=str(torch.__version__),
                            dtype='float32', current_text_context=config['context_limit'],
                            teacher_text_context=payload['protocol']['model']['context_limit'],
                            window_contract='Independent contiguous document windows; all segments share one backward; reset at next window.',
                            field_contract='Complete BioField segment-summary commits; matrix core has token-level delta writes.',
                            consolidation='Only complete, fully supervised, detached optimizer boundaries mutate node parameters; incomplete masked windows never consolidate.',
                            consolidation_batch_contract='Explicit allow_multilane=True: mean proposals from independent supervised train windows into shared parameters; not evidence of handling unknown independent online sessions.')
            if previous is not None and previous != protocol:
                raise ValueError('Experiment protocol changed; start a separate result directory')
            if previous is None:
                write(root/'protocol.json', protocol)
                for relative in protocol['source_hashes']:
                    destination = root/'source'/relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes((PROJECT/relative).read_bytes())
            dev_values, dev_sample = validation_data(corpus, config)
            teacher = build_teacher(payload, device)
            owns_run = True
            write(root/'validation_sample.json', dev_sample)
            write(root/'registry.json', dict(status='running', conditions=registry))
            for condition in CONDITIONS:
                if stop.requested:
                    break
                if registry[condition]['status'] == 'completed':
                    ui.total.update(1)
                    ui.details[condition].update(registry[condition].get('completed_tokens', 0))
                    continue
                model = models[condition].to(device)
                folder = root/condition
                folder.mkdir(exist_ok=True)
                ui.stage(f'v1.0 {condition} · '+('固定主干参考评分' if condition == 'transformer' else '同主干记忆/适配器训练'))
                def evaluation_progress(stage, done, total):
                    ui.stage(f'v1.0 {condition} · 验证 {stage} {done}/{total}')
                    if condition == 'transformer' and stage == 'cold':
                        ui.details[condition].update(done-ui.details[condition].n)
                if condition == 'transformer':
                    ui.details[condition].reset(total=len(dev_values))
                    score = evaluate(model, dev_values, config, device, evaluation_progress)
                    result = dict(status='completed', role='frozen_reference', training_performed=False,
                                  completed_tokens=0, updates=0, validation=score, **counts[condition])
                    write(folder/'result.json', result)
                    registry[condition] = result
                    ui.total.update(1)
                    model.to('cpu')
                    write(root/'registry.json', dict(status='running', conditions=registry))
                    continue
                optimizer = optimizer_for(model, config)
                data_rng = np.random.default_rng(config['data_seed'])
                random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
                counters = dict(updates=0, completed_tokens=0, visible_inputs=0, train_seconds=0.,
                                consolidation_attempts=0, consolidations=0)
                last = folder/'last.pt'
                if resume and last.is_file():
                    counters = restore_checkpoint(last, model, optimizer, data_rng, protocol, device)
                    # Logs beyond the last durable boundary are retained separately.
                    for name in ('metrics.jsonl', 'validation.jsonl', 'consolidation.jsonl'):
                        log = folder/name
                        if log.exists():
                            rows = [json.loads(line) for line in log.read_text().splitlines()]
                            kept = [row for row in rows if row.get('updates', 0) <= counters['updates']]
                            if len(kept) != len(rows):
                                log.rename(folder/f'{name}.pre_resume-{time.time_ns()}')
                                log.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in kept))
                ui.details[condition].update(counters['completed_tokens'])
                registry[condition] = dict(status='running', **counters)
                write(root/'registry.json', dict(status='running', conditions=registry))
                state = None
                save_checkpoint(last, model, optimizer, state, counters, data_rng, protocol, device)
                last_row, last_validation, validation_update = {}, None, -1
                while counters['completed_tokens'] < config['tokens'] and not stop.requested:
                    length = config['segments']*config['segment_tokens']
                    values, docs, starts = corpus.sample('train', data_rng, config['batch'], length)
                    rate = config['learning_rate']*learning_rate_scale(
                        min(config['tokens'], counters['completed_tokens']+config['batch']*length), config['tokens'])
                    for group in optimizer.param_groups:
                        group['lr'] = rate
                    before = counters['completed_tokens']
                    row, state = update(model, teacher, optimizer, values, config, device, config['tokens']-before)
                    counters['updates'] += 1; counters['completed_tokens'] += row['targets']
                    counters['visible_inputs'] += row['visible_inputs']; counters['train_seconds'] += row['seconds']
                    final = counters['completed_tokens'] == config['tokens']
                    consolidate_due = counters['updates'] % config['consolidate_every'] == 0 or final
                    if (condition.startswith('bio_') and config['full_dynamics'] and consolidate_due
                            and row['targets'] == row['visible_inputs']):
                        state, consolidation = model.consolidate(state, optimizer=optimizer, allow_multilane=True)
                        counters['consolidation_attempts'] += 1
                        counters['consolidations'] += int(consolidation['consolidation_layers_changed'] > 0)
                        append(folder/'consolidation.jsonl', dict(updates=counters['updates'],
                               completed_tokens=counters['completed_tokens'], **scalar_diagnostics(consolidation)))
                    row.update(counters, learning_rate=rate, documents=docs, starts=starts, sample_sha256=digest(values.tolist()))
                    append(folder/'metrics.jsonl', row)
                    ui.details[condition].update(row['targets'])
                    ui.details[condition].set_postfix_str(f'CE={row["ce"]:.4f} KD={row["kd"]:.4f} 目标/s={row["targets_per_second"]:.1f}', refresh=False)
                    due = before//config['eval_every'] < counters['completed_tokens']//config['eval_every'] or final
                    if due and not stop.requested:
                        ui.stage(f'v1.0 {condition} · 固定后缀冷/暖/错配/关闭记忆验证')
                        score = evaluate(model, dev_values, config, device, evaluation_progress)
                        append(folder/'validation.jsonl', dict(**counters, **score))
                        last_validation, validation_update = score, counters['updates']
                    last_row = row
                    registry[condition] = dict(status='running', **counters, ce=row['ce'], kd=row['kd'])
                    write(root/'live.json', dict(condition=condition, **registry[condition]))
                    if counters['updates'] % config['checkpoint_every'] == 0 or due or stop.requested:
                        save_checkpoint(last, model, optimizer, state, counters, data_rng, protocol, device)
                        write(root/'registry.json', dict(status='running', conditions=registry))
                save_checkpoint(last, model, optimizer, state, counters, data_rng, protocol, device)
                finished = counters['completed_tokens'] == config['tokens'] and not stop.requested
                result = dict(status='completed' if finished else 'aborted', role='adapted_model',
                              training_performed=True, **counters, **counts[condition],
                              last_metrics=last_row, unscored_visible_tail=counters['visible_inputs']-counters['completed_tokens'])
                if finished:
                    result['validation'] = (last_validation if validation_update == counters['updates'] else
                                            evaluate(model, dev_values, config, device, evaluation_progress))
                write(folder/'result.json', result)
                registry[condition] = result
                ui.total.update(1)
                model.to('cpu')
                write(root/'registry.json', dict(status='running', conditions=registry))
                if not finished:
                    break
            status = 'completed' if all(row['status'] == 'completed' for row in registry.values()) else 'aborted'
            summary = dict(status=status, conditions=registry, protocol='protocol.json',
                           test_evaluated=False, engineering_pilot=True,
                           note='Fixed last budget; one seed; validation has been reused during earlier research.')
            write(root/'registry.json', summary); write(root/'summary.json', summary)
            return root, status
    except BaseException:
        if resume and not owns_run:
            attempts = root/'resume_attempts'
            attempts.mkdir(exist_ok=True)
            (attempts/datetime.now().strftime('failure_%Y%m%dT%H%M%S_%f.txt')).write_text(traceback.format_exc())
        else:
            (root/'failure.txt').write_text(traceback.format_exc())
            for condition, record in registry.items():
                if record['status'] == 'running':
                    record['status'] = 'failed'
            write(root/'registry.json', dict(status='failed', conditions=registry))
        raise
    finally:
        ui.close()


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('train', 'verify'), default='train')
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'), default='auto')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--quiet', action='store_true')
    for name, default in DEFAULTS.items():
        if isinstance(default, bool):
            parser.add_argument('--'+name.replace('_', '-'), action=argparse.BooleanOptionalAction, default=None)
        else:
            parser.add_argument('--'+name.replace('_', '-'), type=type(default), default=None)
    args = parser.parse_args(arguments)
    if args.mode == 'verify':
        root = Path(args.output or PROJECT/'results/v1'/datetime.now().strftime('verify_%Y%m%dT%H%M%S_%f'))
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f'Refusing to overwrite verification: {root}')
        root.mkdir(parents=True, exist_ok=True)
        result = verify(root)
        print(f'v1.0 自检 {result["status"]}: {root}')
    else:
        root, status = train(args)
        print(f'v1.0 本地工程实验 {status}: {root}')


if __name__ == '__main__':
    main()
