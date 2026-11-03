"""v1.1: paired frozen-backbone CPT and history-dependent answer training."""
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
from torch.nn import functional as F

from v07.common import PROJECT, RunLock, detach, device_for, digest, file_hash, save, sync, write
from v07.run import move_state
from v09.benchmark import Progress, append
from v09.train import StopRequest, capture_rng, restore_random, learning_rate_scale
from v1.train import (ParallelFFN, build_teacher, optimizer_for, forward_segments,
                      scalar_diagnostics, load_corpus)
from .model import Config, Network
from .data import prepare_qa
from .evaluate import evaluate_history


FORMAT = 'v11-history-training-1'
CONDITIONS = ('transformer', 'adapter', 'bio_summary', 'bio_ordered')
DEFAULTS = dict(backbone=str(PROJECT/'results/v09/comparison_no_v08/transformer_n32_s11/last.pt'),
                data_root=str(PROJECT/'data/v06/wikitext2'), qa_root=str(PROJECT/'data/v11/squad_remote256'),
                cpt_tokens=65_536, pt_updates=128, batch=2, segments=8, segment_tokens=64,
                context_limit=128, bands=3, nodes=32, internal_steps=3, active_fraction=1.,
                seed=11, data_seed=11001, evaluation_seed=11091, validation_windows=8,
                eval_every=16_384, pt_eval_every=32, checkpoint_every=10,
                learning_rate=3e-4, weight_decay=.01, gradient_clip=1., ce_weight=1.,
                cpt_kd_weight=0., pt_kd_weight=.1, kd_topk=64, consolidate_every=16, threads=4)


def source_hashes():
    paths = [*sorted((PROJECT/'v11').glob('*.py')), *sorted((PROJECT/'v1').glob('*.py')),
             PROJECT/'run_biomatrix_v11.py', PROJECT/'.run/Bio-Matrix v1.1.run.xml']
    paths += [PROJECT/p for p in ('v03/graph.py', 'v04/model.py', 'v05/model.py',
              'v06/model.py', 'v06/common.py', 'v06/graph.py', 'v07/common.py', 'v07/data.py',
              'v07/model.py', 'v07/run.py', 'v07/graph.py', 'v07/evaluate.py',
              'v08/benchmark.py', 'v08/model.py', 'v08/recurrence.py',
              'v09/benchmark.py', 'v09/train.py')]
    return {str(p.relative_to(PROJECT)): file_hash(p) for p in paths if p.is_file()}


def settings(args, saved=None):
    result = {}
    for name, default in DEFAULTS.items():
        if saved is not None and name not in saved:
            raise ValueError('Saved protocol has a different format')
        old, given = default if saved is None else saved[name], getattr(args, name, None)
        if saved is not None and given is not None and old != given:
            raise ValueError(f'Resume cannot change {name}; choose a fresh output directory')
        result[name] = old if given is None else given
    for name in ('backbone', 'data_root', 'qa_root'):
        result[name] = str(Path(result[name]).resolve())
    integers = ('cpt_tokens', 'pt_updates', 'batch', 'segments', 'segment_tokens', 'context_limit',
                'bands', 'nodes', 'internal_steps', 'seed', 'data_seed', 'evaluation_seed',
                'validation_windows', 'eval_every', 'pt_eval_every', 'checkpoint_every',
                'kd_topk', 'consolidate_every', 'threads')
    if any(type(result[k]) is not int or result[k] < 1 for k in integers):
        raise ValueError('Counts and seeds must be positive integers')
    if result['segments'] < 4 or result['segment_tokens'] > result['context_limit']:
        raise ValueError('Need >=4 segments and segment_tokens <= local context')
    if (result['segments']-2)*result['segment_tokens'] <= result['context_limit']:
        raise ValueError('QA history must extend beyond the local attention context')
    for name in ('learning_rate', 'weight_decay', 'gradient_clip', 'ce_weight',
                 'cpt_kd_weight', 'pt_kd_weight', 'active_fraction'):
        if not math.isfinite(result[name]) or result[name] < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if min(result['learning_rate'], result['gradient_clip'], result['ce_weight']) <= 0:
        raise ValueError('Learning rate, clip and CE weight must be positive')
    if not 0 < result['active_fraction'] <= 1:
        raise ValueError('active_fraction must be in (0,1]')
    return result


def architecture(payload, config):
    old = payload['protocol']['model']
    return Config(vocab_size=old['vocab_size'], width=old['transformer_width'],
                  layers=old['transformer_layers'], heads=old['transformer_heads'],
                  ff=old['transformer_ff'], context_limit=config['context_limit'],
                  segment_tokens=config['segment_tokens'], bands=config['bands'],
                  nodes=config['nodes'], internal_steps=config['internal_steps'],
                  full_dynamics=True, active_fraction=config['active_fraction'],
                  memory_init='backbone', field_write_mode='ordered', phase_enabled=True)


def build_models(payload, config):
    cfg = architecture(payload, config)
    torch.manual_seed(config['seed'])
    ordered = Network(cfg).load_backbone(payload, freeze=True)
    summary = Network(replace(cfg, field_write_mode='summary')).load_backbone(payload, freeze=True)
    summary.load_state_dict(ordered.state_dict(), strict=True)
    baseline = Network(replace(cfg, memory_enabled=False)).load_backbone(payload, freeze=True)
    adapter = Network(replace(cfg, memory_enabled=False)).load_backbone(payload, freeze=True)
    target = sum(p.numel() for p in ordered.parameters() if p.requires_grad)
    rank = max(1, round(target/(2*cfg.width*cfg.layers)))
    for block in adapter.blocks:
        block.ff = ParallelFFN(block.ff, cfg.width, rank)
    models = dict(transformer=baseline, adapter=adapter, bio_summary=summary, bio_ordered=ordered)
    counts = {name: dict(total_parameters=sum(p.numel() for p in m.parameters()),
                         trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad))
              for name, m in models.items()}
    counts['adapter'].update(rank=rank, target_trainable_parameters=target,
                            trainable_parameter_difference=counts['adapter']['trainable_parameters']-target)
    return models, cfg, counts


def objective(student, target, teacher, mask, config, stage):
    if mask.shape != target.shape or mask.dtype != torch.bool or not bool(mask.any()):
        raise ValueError('Nonempty boolean target mask required')
    ce = F.cross_entropy(student.float().flatten(0, 1), target.flatten(), reduction='none').reshape_as(target)[mask].mean()
    kd_weight = config[f'{stage}_kd_weight']
    kd = ce.new_zeros(())
    if kd_weight:
        if teacher is None:
            raise ValueError('A teacher is required for nonzero KD weight')
        values, indices = teacher.float()[mask].topk(min(config['kd_topk'], teacher.shape[-1]), dim=-1)
        q, teacher_log = values.softmax(-1), values.log_softmax(-1)
        student_log = student.float()[mask].log_softmax(-1).gather(-1, indices)
        kd = (q*(teacher_log-student_log)).sum(-1).mean()
    return config['ce_weight']*ce+kd_weight*kd, ce, kd


def update(model, teacher, optimizer, values, mask, config, device, stage):
    inputs = torch.as_tensor(values[:, :-1].copy(), dtype=torch.long, device=device)
    targets = torch.as_tensor(values[:, 1:].copy(), dtype=torch.long, device=device)
    selected = torch.as_tensor(mask.copy(), dtype=torch.bool, device=device)
    model.train(); optimizer.zero_grad(set_to_none=True)
    sync(device); started = time.perf_counter()
    teacher_logits = None
    if config[f'{stage}_kd_weight']:
        with torch.no_grad():
            teacher_logits, _, _ = forward_segments(teacher, inputs, teacher.cfg.context_limit)
    logits, state, diagnostics = forward_segments(model, inputs, config['segment_tokens'])
    loss, ce, kd = objective(logits, targets, teacher_logits, selected, config, stage)
    if not bool(loss.detach().isfinite()):
        raise FloatingPointError('Nonfinite objective; optimizer was not advanced')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
    optimizer.step()
    state = detach(state)
    sync(device); seconds = time.perf_counter()-started
    n = int(selected.sum().cpu())
    return dict(stage=stage, loss=float(loss.detach().cpu()), ce=float(ce.detach().cpu()),
                kd=float(kd.detach().cpu()), gradient_norm=float(norm.detach().cpu()), targets=n,
                visible_inputs=inputs.numel(), seconds=seconds, targets_per_second=n/max(seconds, 1e-9),
                diagnostics=scalar_diagnostics(diagnostics)), state


def validation_data(corpus, qa, config):
    length = config['segments']*config['segment_tokens']
    values, docs, starts = corpus.sample('validation', np.random.default_rng(config['evaluation_seed']),
                                        config['validation_windows'], length)
    qvalues, mask, meta = qa.sample('validation', np.random.default_rng(config['evaluation_seed']),
                                   config['validation_windows'])
    samples = dict(lm=dict(documents=docs, starts=starts, sha256=digest(values.tolist())),
                   qa={k: v for k, v in meta.items() if k != 'first_answer_mask'})
    samples['qa']['sha256'] = digest(dict(values=qvalues.tolist(), mask=mask.tolist()))
    return (values, qvalues, mask, meta['first_answer_mask']), samples


@torch.no_grad()
def evaluate(model, datasets, config, device, progress=None):
    values, qa_values, mask, first = datasets
    prefix = (config['segments']-2)*config['segment_tokens']
    result = {}
    for task, data, selected, first_mask in (('lm', values, None, None), ('qa', qa_values, mask, first)):
        callback = ((lambda stage, done, total, t=task: progress(f'{t}/{stage}', done, total))
                    if progress else None)
        result[task] = evaluate_history(model, data, device, prefix, batch=config['batch'],
                                       seed=config['evaluation_seed'], progress=callback,
                                       target_mask=selected, first_answer_mask=first_mask)
    return result


@torch.no_grad()
def teacher_quality(teacher, datasets, config, device):
    _, values, mask, first = datasets
    sums, first_sums, count, first_count, correct = 0., 0., 0, 0, 0
    for start in range(0, len(values), config['batch']):
        v = values[start:start+config['batch']]
        inputs = torch.as_tensor(v[:, :-1].copy(), device=device, dtype=torch.long)
        targets = torch.as_tensor(v[:, 1:].copy(), device=device, dtype=torch.long)
        m = torch.as_tensor(mask[start:start+len(v)], device=device, dtype=torch.bool)
        f = torch.as_tensor(first[start:start+len(v)], device=device, dtype=torch.bool)
        logits, _, _ = forward_segments(teacher, inputs, teacher.cfg.context_limit)
        losses = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), reduction='none').reshape_as(targets)
        sums += float(losses[m].sum().cpu()); first_sums += float(losses[f].sum().cpu())
        count += int(m.sum().cpu()); first_count += int(f.sum().cpu())
        correct += int(((logits.argmax(-1) == targets) & f).sum().cpu())
    return dict(answer_nll=sums/count, first_answer_nll=first_sums/first_count,
                first_answer_accuracy=correct/first_count, targets=count, first_targets=first_count,
                note='Tiny WikiText backbone is not a QA oracle; answer CE remains the primary PT objective.')


def state_bytes(value):
    if isinstance(value, torch.Tensor): return value.numel()*value.element_size()
    if isinstance(value, dict): return sum(state_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)): return sum(state_bytes(v) for v in value)
    return 0


@torch.no_grad()
def calibrate(model, values, config, device):
    """Use frozen train inputs to check actual fresh reach and state updates at each shell."""
    model.to(device).eval()
    inputs = torch.as_tensor(values[:, :config['segment_tokens']].copy(), dtype=torch.long, device=device)
    initial = model.initial_state(len(inputs))
    sync(device); started = time.perf_counter()
    logits, state, diagnostics = model(inputs, initial)
    sync(device); elapsed = time.perf_counter()-started
    rows = []
    for layer, (memory, old, new) in enumerate(zip(model.memories, initial['field'], state['field'])):
        field = memory.field
        shells = []
        for level in range(field.shell_count):
            selected = field.shell == level
            mass = new['fresh_reached'][:, selected]
            shells.append(dict(shell=level, nodes=int(selected.sum().cpu()),
                               fresh_fraction=float((mass > 0).float().mean().cpu()),
                               min_mass=float(mass.min().cpu()), mean_mass=float(mass.mean().cpu()),
                               joseph_update_fraction=float((mass > 1e-6).float().mean().cpu()),
                               m_delta_rms=float((new['m'][:, selected]-old['m'][:, selected]).square().mean().sqrt().cpu()),
                               p_delta_rms=float((new['p'][:, selected]-old['p'][:, selected]).square().mean().sqrt().cpu()),
                               association_rms=float(new['association'][:, selected].square().mean().sqrt().cpu()),
                               fast_rms=float(new['fast'][:, selected].square().mean().sqrt().cpu())))
        if any(r['fresh_fraction'] < 1 or r['joseph_update_fraction'] < 1 for r in shells):
            raise RuntimeError(f'Fresh train observations fail shell coverage at layer {layer}: {shells}')
        rows.append(dict(layer=layer, internal_steps=field.internal_steps, shells=shells))
    if not bool(logits.isfinite().all()): raise FloatingPointError('Preflight logits are nonfinite')
    record = dict(status='completed', layers=rows, first_segment_seconds=elapsed,
                  runtime_bytes=state_bytes(state), train_inputs_sha256=digest(inputs.cpu().tolist()),
                  diagnostics=scalar_diagnostics(diagnostics))
    model.to('cpu')
    return record


def verify(root=None):
    suite = unittest.defaultTestLoader.discover(str(PROJECT/'v11'), pattern='test_*.py', top_level_dir=str(PROJECT))
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    record = dict(status='completed' if result.wasSuccessful() else 'failed', tests=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped))
    if root is not None:
        (Path(root)/'verification.txt').write_text(output.getvalue())
        write(Path(root)/'verification.json', record)
    if not result.wasSuccessful():
        raise RuntimeError('v1.1 self-check failed; training was not started. See verification.txt')
    return record


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
    model.load_state_dict(payload['model']); optimizer.load_state_dict(payload['optimizer'])
    data_rng.bit_generator.state = payload['data_rng']; restore_random(payload['rng'], device)
    return dict(payload['counters'])


def train(args, verify_first=True):
    device = device_for(getattr(args, 'device', 'auto'))
    if device.type == 'mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise RuntimeError('Disable MPS fallback for native v1.1 execution')
    resume = Path(args.resume).resolve() if getattr(args, 'resume', None) else None
    if resume and resume.suffix == '.pt':
        if resume.name != 'last.pt':
            raise ValueError('Resume uses the last safe optimizer boundary: last.pt')
        resume = resume.parent.parent
    previous = json.loads((resume/'protocol.json').read_text()) if resume else None
    config = settings(args, previous['training'] if previous else None)
    if resume and getattr(args, 'output', None) and Path(args.output).resolve() != resume:
        raise ValueError('Resume output must be its existing experiment directory')
    root = (resume or Path(getattr(args, 'output', None) or PROJECT/'results/v11'/datetime.now().strftime('pilot_%Y%m%dT%H%M%S_%f'))).resolve()
    if not resume and root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Refusing to overwrite previous results: {root}')
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('TQDM_NROWS', '24')
    length = config['segments']*config['segment_tokens']
    total_work = config['cpt_tokens']+config['pt_updates']*config['batch']*length
    ui = Progress([dict(id=n) for n in CONDITIONS], total_work, not getattr(args, 'quiet', False))
    ui.total.set_description_str('v1.1 已结束条件'); ui.device = device
    registry = {n: dict(status='not_executed', completed_work=0) for n in CONDITIONS}
    if resume:
        registry = json.loads((root/'registry.json').read_text())['conditions']
    owns_run = False
    try:
        ui.stage('v1.1 代码自检')
        if verify_first:
            checks = root if resume is None else root/'resume_attempts'/datetime.now().strftime('%Y%m%dT%H%M%S_%f')
            checks.mkdir(parents=True, exist_ok=True); verify(checks)
        with RunLock(PROJECT/'results/v11/.pilot.lock'), StopRequest() as stop:
            torch.set_num_threads(config['threads'])
            ui.stage('v1.1 WikiText校验 / 官方SQuAD下载与标注校验')
            corpus = load_corpus(config['data_root'])
            payload = torch.load(config['backbone'], map_location='cpu', weights_only=True)
            fixture = payload['protocol'].get('fixture') is True and 'fixture' in corpus.manifest
            if payload['protocol']['model']['vocab_size'] != corpus.manifest['vocab_size']:
                raise ValueError('Backbone and tokenizer vocabulary mismatch')
            if not fixture and payload['protocol'].get('data_manifest_sha256') != digest(corpus.manifest):
                raise ValueError('Backbone and frozen tokenizer/data manifest mismatch')
            qa = prepare_qa(config['qa_root'], corpus, length=length,
                            prefix=(config['segments']-2)*config['segment_tokens'],
                            context_limit=config['context_limit'],
                            backbone_layers=payload['protocol']['model']['transformer_layers'],
                            allow_download=not bool(resume))
            models, cfg, counts = build_models(payload, config)
            protocol = dict(format=FORMAT, training=config, model=asdict(cfg), parameter_counts=counts,
                            source_hashes=source_hashes(), data_manifest_sha256=digest(corpus.manifest),
                            qa_manifest_sha256=digest(qa.manifest), backbone_sha256=file_hash(config['backbone']),
                            device=str(device), torch=str(torch.__version__), dtype='float32',
                            memory_initialization=models['bio_ordered'].memory_initialization,
                            field_contract='Same fixed buffers/state/readout; ordered association gets sequential token delta updates, summary gets one mean update. P/phase/slow/credit commit once per segment.',
                            stages='CPT natural-language CE, then SQuAD answer-only CE with modest full-context teacher KL.',
                            window_contract='Independent windows; full differentiable history-to-answer path; state resets next window.',
                            consolidation='Bounded fully supervised CPT-window proposals every interval and CPT stage end. Multi-lane mean explicitly enabled; never validation/test or padded answer-only PT windows.',
                            progress_contract='CPT supervised targets plus PT visible input tokens; answer target count recorded separately.')
            if previous is not None and previous != protocol:
                raise ValueError('Experiment protocol changed; use a separate result directory')
            if previous is None:
                write(root/'protocol.json', protocol)
                for relative in protocol['source_hashes']:
                    dst = root/'source'/relative; dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes((PROJECT/relative).read_bytes())
            datasets, sample = validation_data(corpus, qa, config)
            teacher = build_teacher(payload, device)
            if resume:
                for name, record in registry.items():
                    folder = root/name
                    progressed = record.get('completed_work', 0) > 0 or (folder/'metrics.jsonl').exists()
                    if record['status'] != 'completed' and progressed and not (folder/'last.pt').is_file():
                        raise FileNotFoundError(f'Cannot resume {name}: its last safe checkpoint is missing')
                    if record['status'] != 'completed' and progressed and not (folder/'initial_validation.json').is_file():
                        raise FileNotFoundError(f'Cannot reconstruct t=0 validation for progressed condition {name}')
            owns_run = True
            write(root/'validation_sample.json', sample)
            if not resume:
                ui.stage('v1.1 自然训练输入球壳覆盖与状态容量校准')
                calibration_values = corpus.sample('train', np.random.default_rng(config['data_seed']+1),
                                                   config['batch'], length)[0]
                write(root/'calibration.json', {n: calibrate(models[n], calibration_values, config, device)
                                               for n in ('bio_summary', 'bio_ordered')})
            teacher_scores = json.loads((root/'teacher_quality.json').read_text()) if resume and (root/'teacher_quality.json').is_file() else {}
            teacher_scores.update(teacher_quality(teacher, datasets, config, device))
            write(root/'teacher_quality.json', teacher_scores)
            write(root/'registry.json', dict(status='running', conditions=registry))
            for condition in CONDITIONS:
                if stop.requested: break
                if registry[condition]['status'] == 'completed':
                    ui.total.update(1); ui.details[condition].update(registry[condition].get('completed_work', 0)); continue
                model = models[condition].to(device)
                folder = root/condition; folder.mkdir(exist_ok=True)
                def evaluation_progress(stage, done, count):
                    ui.stage(f'v1.1 {condition} · 验证 {stage} {done}/{count}')
                if condition == 'transformer':
                    ui.details[condition].reset(total=1)
                    score = evaluate(model, datasets, config, device, evaluation_progress)
                    local = score['qa']['conditions']['warm']
                    teacher_scores.update(short_context_answer_nll=local['nll'],
                                          short_context_first_answer_nll=local['first_answer_nll'],
                                          full_history_answer_nll_gain=local['nll']-teacher_scores['answer_nll'],
                                          full_history_first_answer_nll_gain=local['first_answer_nll']-teacher_scores['first_answer_nll'])
                    write(root/'teacher_quality.json', teacher_scores)
                    record = dict(status='completed', role='frozen_reference', training_performed=False,
                                  completed_work=0, validation=score, **counts[condition])
                    write(folder/'result.json', record); registry[condition] = record
                    ui.total.update(1); ui.details[condition].update(1)
                    ui.details[condition].set_postfix_str('固定主干评分完成')
                    model.to('cpu'); write(root/'registry.json', dict(status='running', conditions=registry)); continue
                optimizer = optimizer_for(model, config)
                data_rng = np.random.default_rng(config['data_seed'])
                random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
                counters = dict(updates=0, cpt_tokens=0, pt_updates=0, answer_targets=0,
                                completed_work=0, visible_inputs=0, train_seconds=0.,
                                consolidation_attempts=0, consolidations=0)
                last = folder/'last.pt'
                if resume and last.is_file():
                    counters = restore_checkpoint(last, model, optimizer, data_rng, protocol, device)
                    for name in ('metrics.jsonl', 'validation.jsonl', 'consolidation.jsonl'):
                        log = folder/name
                        if log.exists():
                            rows = [json.loads(line) for line in log.read_text().splitlines()]
                            kept = [r for r in rows if r.get('updates', 0) <= counters['updates']]
                            if len(kept) != len(rows):
                                log.rename(folder/f'{name}.pre_resume-{time.time_ns()}')
                                log.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in kept))
                registry[condition] = dict(status='running', **counters)
                write(root/'registry.json', dict(status='running', conditions=registry))
                ui.details[condition].update(counters['completed_work'])
                state = None
                save_checkpoint(last, model, optimizer, state, counters, data_rng, protocol, device)
                if not (folder/'initial_validation.json').exists():
                    ui.stage(f'v1.1 {condition} · t=0固定验证')
                    score = evaluate(model, datasets, config, device, evaluation_progress)
                    write(folder/'initial_validation.json', score)
                    append(folder/'validation.jsonl', dict(**counters, stage='initial', **score))
                last_row = {}
                while counters['completed_work'] < total_work and not stop.requested:
                    stage = 'cpt' if counters['cpt_tokens'] < config['cpt_tokens'] else 'pt'
                    if stage == 'pt' and counters['pt_updates'] == 0 and not (folder/'cpt_validation.json').exists():
                        score = evaluate(model, datasets, config, device, evaluation_progress)
                        write(folder/'cpt_validation.json', score)
                        append(folder/'validation.jsonl', dict(**counters, stage='cpt', **score))
                    ui.stage(f'v1.1 {condition} · {"自然语言CPT" if stage == "cpt" else "历史问答PT"}')
                    if stage == 'cpt':
                        values, docs, starts = corpus.sample('train', data_rng, config['batch'], length)
                        mask = np.arange(config['batch']*length).reshape(config['batch'], length) < config['cpt_tokens']-counters['cpt_tokens']
                        metadata = dict(documents=docs, starts=starts)
                        fraction_done = counters['cpt_tokens']+int(mask.sum())
                        stage_budget = config['cpt_tokens']
                    else:
                        values, mask, metadata = qa.sample('train', data_rng, config['batch'])
                        metadata = {k: v for k, v in metadata.items() if k != 'first_answer_mask'}
                        fraction_done = counters['pt_updates']+1; stage_budget = config['pt_updates']
                    rate = config['learning_rate']*learning_rate_scale(fraction_done, stage_budget)
                    for group in optimizer.param_groups: group['lr'] = rate
                    before = counters.copy()
                    row, state = update(model, teacher, optimizer, values, mask, config, device, stage)
                    counters['updates'] += 1; counters['visible_inputs'] += row['visible_inputs']
                    counters['train_seconds'] += row['seconds']
                    if stage == 'cpt':
                        counters['cpt_tokens'] += row['targets']; work = row['targets']
                        stage_end = counters['cpt_tokens'] == config['cpt_tokens']
                    else:
                        counters['pt_updates'] += 1; counters['answer_targets'] += row['targets']
                        work = row['visible_inputs']; stage_end = counters['pt_updates'] == config['pt_updates']
                    counters['completed_work'] += work
                    if condition.startswith('bio_') and stage == 'cpt' and (counters['updates'] % config['consolidate_every'] == 0 or stage_end):
                        if row['targets'] == row['visible_inputs']:
                            state, d = model.consolidate(state, optimizer=optimizer, allow_multilane=True)
                            counters['consolidation_attempts'] += 1
                            counters['consolidations'] += int(d['consolidation_layers_changed'] > 0)
                            append(folder/'consolidation.jsonl', dict(updates=counters['updates'], stage=stage, **scalar_diagnostics(d)))
                    row.update(counters, learning_rate=rate, sample=metadata,
                               sample_sha256=digest(dict(values=values.tolist(), mask=mask.tolist())))
                    append(folder/'metrics.jsonl', row)
                    ui.details[condition].update(work)
                    ui.details[condition].set_postfix_str(f'{stage.upper()} CE={row["ce"]:.4f} KD={row["kd"]:.4f} 答案目标={counters["answer_targets"]}', refresh=False)
                    due = stage_end or (stage == 'cpt' and before['cpt_tokens']//config['eval_every'] < counters['cpt_tokens']//config['eval_every']) or (stage == 'pt' and counters['pt_updates'] % config['pt_eval_every'] == 0)
                    if due and not stop.requested:
                        score = evaluate(model, datasets, config, device, evaluation_progress)
                        append(folder/'validation.jsonl', dict(**counters, stage=stage, **score))
                        if stage == 'cpt' and stage_end:
                            write(folder/'cpt_validation.json', score)
                    registry[condition] = dict(status='running', **counters, stage=stage, ce=row['ce'], kd=row['kd'])
                    write(root/'live.json', dict(condition=condition, **registry[condition]))
                    if counters['updates'] % config['checkpoint_every'] == 0 or due or stop.requested:
                        save_checkpoint(last, model, optimizer, state, counters, data_rng, protocol, device)
                        write(root/'registry.json', dict(status='running', conditions=registry))
                    last_row = row
                save_checkpoint(last, model, optimizer, state, counters, data_rng, protocol, device)
                finished = counters['completed_work'] == total_work and not stop.requested
                result = dict(status='completed' if finished else 'aborted', training_performed=True,
                              **counters, **counts[condition], last_metrics=last_row)
                if finished:
                    result['validation'] = evaluate(model, datasets, config, device, evaluation_progress)
                write(folder/'result.json', result); registry[condition] = result
                ui.total.update(1); model.to('cpu')
                write(root/'registry.json', dict(status='running', conditions=registry))
                if not finished: break
            status = 'completed' if all(r['status'] == 'completed' for r in registry.values()) else 'aborted'
            summary = dict(status=status, conditions=registry, protocol='protocol.json', test_evaluated=False,
                           note='One seed pilot, validation only. Frozen Transformer is scored, never timed as a training baseline.')
            write(root/'registry.json', summary); write(root/'summary.json', summary)
            return root, status
    except BaseException:
        if resume and not owns_run:
            attempts = root/'resume_attempts'; attempts.mkdir(exist_ok=True)
            (attempts/datetime.now().strftime('failure_%Y%m%dT%H%M%S_%f.txt')).write_text(traceback.format_exc())
        else:
            (root/'failure.txt').write_text(traceback.format_exc())
            for record in registry.values():
                if record['status'] == 'running': record['status'] = 'failed'
            write(root/'registry.json', dict(status='failed', conditions=registry))
        raise
    finally:
        ui.close()


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('train', 'verify'), default='train')
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'), default='auto')
    parser.add_argument('--output', type=Path); parser.add_argument('--resume', type=Path)
    parser.add_argument('--quiet', action='store_true')
    for name, default in DEFAULTS.items():
        parser.add_argument('--'+name.replace('_', '-'), type=type(default), default=None)
    args = parser.parse_args(arguments)
    if args.mode == 'verify':
        root = Path(args.output or PROJECT/'results/v11'/datetime.now().strftime('verify_%Y%m%dT%H%M%S_%f'))
        if root.exists() and any(root.iterdir()): raise FileExistsError(f'Refusing to overwrite {root}')
        root.mkdir(parents=True, exist_ok=True); result = verify(root)
        print(f'v1.1 自检 {result["status"]}: {root}')
    else:
        root, status = train(args)
        print(f'v1.1 本地实验 {status}: {root}')


if __name__ == '__main__':
    main()
