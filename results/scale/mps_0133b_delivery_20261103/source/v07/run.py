"""Paired public LM training with bounded TBPTT, exact resume and quiet stage progress."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict, fields, replace
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from .common import Console, PROJECT, RunLock, detach, device_for, digest, save, source_hashes, state_bytes, sync, write
from .data import CHANGE_CUES, ContinuousStream, Corpus, ROOT as DATA_ROOT, prepare
from .evaluate import evaluate
from .model import Config, Network, Stateless, Transformer, initialize, invalidate_feedback, parameter_count, transformer_shape


def cases(cfg):
    result = []
    def add(stage, seed, nodes, variant, **extra):
        case = dict(stage=stage, seed=seed, nodes=nodes, variant=variant, **extra)
        case['id'] = f'{stage}_n{nodes}_{variant}_s{seed}'+''.join(f'_{k}{v}' for k, v in extra.items())
        result.append(case)
    # Finish every model at one scale before the next scale, then repeat seeds.
    if cfg['stage'] in ('core', 'all'):
        for seed in cfg['seeds']:
            for nodes in cfg['nodes']:
                for variant in cfg['variants']:
                    add('core', seed, nodes, variant)
    if cfg['stage'] in ('mechanism', 'all'):
        for variant in ('bypass', 'no_local', 'no_slow', 'no_probability', 'no_dynamic', 'single_timescale', 'no_threshold'):
            add('mechanism', cfg['seeds'][0], 128, variant)
        add('mechanism', cfg['seeds'][0], 128, 'full', reset_policy='document')
    if cfg['stage'] in ('write_scale', 'all'):
        for nodes in cfg['nodes']:
            add('write_scale', cfg['seeds'][0], nodes, 'full', local_write_normalization='none')
    if cfg['stage'] in ('iterations', 'all'):
        for steps in (2, 8):
            for variant in ('independent', 'full'):
                add('iterations', cfg['seeds'][0], 128, variant, internal_steps=steps)
    if cfg['stage'] in ('topology', 'all'):
        for variant in ('message', 'independent', 'full'):
            add('topology', cfg['seeds'][0], 128, variant, topology='cut')
    if cfg['stage'] in ('capacity', 'all'):
        # Equal N*d² matrix entries; complete runtime state is separately disclosed.
        for nodes, size in ((32, 32), (128, 16), (512, 8)):
            for variant in ('message', 'independent', 'full'):
                add('capacity', cfg['seeds'][0], nodes, variant, matrix_size=size)
    if cfg['stage'] in ('credit', 'all'):
        for truncation in (16, 64, 128):
            for variant in ('full', 'no_local'):
                add('credit', cfg['seeds'][0], 32, variant, bptt_tokens=truncation)
    if not result:
        raise ValueError('Empty or unknown experiment stage')
    return result


def model_config(cfg, case):
    allowed = {f.name for f in fields(Config)}
    values = {k: v for k, v in (cfg | case).items() if k in allowed}
    if case['variant'] == 'transformer':
        # Match the full network at this scale, including embedding/output and node shapes.
        with torch.random.fork_rng():
            target = parameter_count(Network(Config(**(values | {'variant': 'full'}))))
        shape, _ = transformer_shape(target, values['vocab_size'])
        values.update(shape)
    return Config(**values)


def build(cfg):
    if cfg.variant == 'transformer':
        return Transformer(cfg)
    return Stateless(cfg) if cfg.variant == 'stateless' else Network(cfg)


def validate_config(cfg):
    positive = ('train_steps', 'batch_size', 'sequence_tokens', 'bptt_tokens', 'eval_every', 'checkpoint_every',
                'validation_windows', 'test_windows', 'prefix_tokens', 'target_tokens', 'cpu_threads')
    if any(not isinstance(cfg[k], int) or cfg[k] < 1 for k in positive):
        raise ValueError('Training counts and dimensions must be positive integers')
    if cfg['bptt_tokens'] > cfg['sequence_tokens'] or cfg['context_limit'] < max(cfg['bptt_tokens'], 2*cfg['prefix_tokens']+cfg['target_tokens']):
        raise ValueError('Invalid truncation/context budget')
    if (not cfg['seeds'] or not cfg['nodes'] or len(set(cfg['seeds'])) != len(cfg['seeds']) or
            len(set(cfg['nodes'])) != len(cfg['nodes']) or not cfg['variants'] or
            len(set(cfg['variants'])) != len(cfg['variants'])):
        raise ValueError('Need unique nonempty seeds and scales')
    if cfg['learning_rate'] <= 0 or cfg['gradient_clip'] <= 0 or cfg['weight_decay'] < 0:
        raise ValueError('Invalid optimizer configuration')
    if (not isinstance(cfg['session_tokens'], int) or cfg['session_tokens'] < max(cfg['sequence_tokens'], 2*cfg['prefix_tokens']+cfg['target_tokens']) or
            cfg['session_tokens'] % cfg['sequence_tokens'] or cfg['reset_policy'] not in ('session', 'document') or
            min(cfg['intrusion_tokens'], cfg['intrusion_interval']) < 0 or not 0 <= cfg['marked_probability'] <= 1):
        raise ValueError('Invalid finite-session/context-switch protocol')


def training_batch(corpus, cfg, seed, step):
    """Deterministic independent probe window; formal training uses ContinuousStream."""
    rng = np.random.default_rng(np.random.SeedSequence([cfg['data_seed'], seed, step]))
    values, docs, starts = corpus.sample('train', rng, cfg['batch_size'], cfg['sequence_tokens'])
    return values, dict(documents=docs, starts=starts, sha256=digest(values.tolist()))


def train_update(model, optimizer, values, cfg, device, step, state=None, reset_mask=None, loss_mask=None):
    model.train(); optimizer.zero_grad(set_to_none=True)
    model.gate_blend = min(1., (step+1)/max(1, cfg['gate_warmup_steps']))
    x = torch.as_tensor(values[:, :-1].copy(), dtype=torch.long, device=device)
    y = torch.as_tensor(values[:, 1:].copy(), dtype=torch.long, device=device)
    mask = torch.ones_like(y, dtype=torch.bool) if loss_mask is None else torch.as_tensor(loss_mask, dtype=torch.bool, device=device)
    if mask.shape != y.shape or not bool(mask.any()):
        raise ValueError('Each update requires a matching nonempty valid-target mask')
    reset = torch.zeros_like(mask)
    if reset_mask is not None:
        given = torch.as_tensor(reset_mask, dtype=torch.bool, device=device)
        if given.shape == (len(x),):
            reset[:, 0] = given
        elif given.shape == x.shape:
            reset = given
        else:
            raise ValueError('reset_mask must be [B] or [B,T]')
    total_loss = torch.zeros((), device=device); valid = mask.sum()
    truncation = cfg['bptt_tokens']; statistics = {}; statistic_weights = {}
    reset_positions = reset.any(0).nonzero().flatten().cpu().tolist()
    for start in range(0, x.shape[1], truncation):
        end = min(start+truncation, x.shape[1]); chunk_loss = None
        cuts = [start]+[t for t in reset_positions if start < t < end]+[end]
        for begin, finish in zip(cuts, cuts[1:]):
            selected = mask[:, begin:finish]
            logits, state, diag = model(x[:, begin:finish], state, reset_mask=reset[:, begin],
                                       diagnostic_mask=selected)
            losses = F.cross_entropy(logits.flatten(0, 1), y[:, begin:finish].flatten(), reduction='none')
            part = (losses*selected.flatten()).sum()/valid
            chunk_loss = part if chunk_loss is None else chunk_loss+part
            count = selected.sum().detach()
            if bool(count > 0):
                for name, value in diag.items():
                    value = value.detach()
                    if '_min_' in name or name.endswith('_min') or name.startswith('min_'):
                        statistics[name] = torch.minimum(statistics[name], value) if name in statistics else value
                    elif '_max_' in name or name.endswith('_max') or name.startswith('max_'):
                        statistics[name] = torch.maximum(statistics[name], value) if name in statistics else value
                    else:
                        statistics[name] = statistics.get(name, torch.zeros_like(value))+value*count
                        statistic_weights[name] = statistic_weights.get(name, torch.zeros_like(count))+count
        chunk_loss.backward(); total_loss += chunk_loss.detach(); state = detach(state)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
    if not torch.isfinite(total_loss):
        raise FloatingPointError('Nonfinite loss; optimizer was not advanced')
    optimizer.step(); state = invalidate_feedback(state)
    diagnostics = {name: float((value/statistic_weights[name] if name in statistic_weights else value).cpu())
                   for name, value in statistics.items()}
    return dict(loss=float(total_loss.cpu()), gradient_norm=float(norm.cpu()),
                valid_targets=int(valid.cpu()), masked_targets=int(y.numel()-valid.cpu()),
                padding_targets=int(((~mask)&(y == 0)).sum().cpu()), diagnostic_samples=int(valid.cpu()),
                **diagnostics), state


def make_training_stream(corpus, cfg, seed):
    return ContinuousStream(corpus, cfg['batch_size'], cfg['sequence_tokens'], cfg['data_seed'], seed,
                            **{key: cfg[key] for key in ('session_tokens', 'intrusion_tokens', 'intrusion_interval',
                                                        'marked_probability', 'reset_policy')})


def move_state(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_state(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_state(v, device) for v in value)
    return value


def rng_state(device):
    result = {'cpu': torch.get_rng_state()}
    if device.type == 'mps':
        result['mps'] = torch.mps.get_rng_state()
    if device.type == 'cuda':
        result['cuda'] = torch.cuda.get_rng_state(device)
    return result


def restore_rng(value, device):
    torch.set_rng_state(value['cpu'].cpu())
    if device.type == 'mps':
        torch.mps.set_rng_state(value['mps'].cpu())
    elif device.type == 'cuda':
        torch.cuda.set_rng_state(value['cuda'].cpu(), device)


def checkpoint(path, model, optimizer, protocol, step, best, seconds, device, runtime_state, stream, validation_step=0):
    best_path = path.parent/'best.pt'
    best_checkpoint = torch.load(best_path, map_location='cpu', weights_only=True) if best_path.exists() else None
    save(path, dict(protocol=protocol, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    step=step, validation_step=validation_step, best=best, train_seconds=seconds,
                    rng=rng_state(device), best_checkpoint=best_checkpoint,
                    runtime_state=move_state(detach(runtime_state), 'cpu'), stream=stream.state_dict()))


def recover_logs(folder, step, validation_step):
    """Preserve uncommitted tail separately so resumed metrics count each update once."""
    for name in ('metrics.jsonl', 'validation.jsonl'):
        committed_step = validation_step if name == 'validation.jsonl' else step
        path = folder/name
        if not path.exists():
            continue
        committed, tail = [], []
        for line in path.read_text().splitlines():
            try:
                (committed if json.loads(line)['step'] <= committed_step else tail).append(line)
            except (json.JSONDecodeError, KeyError):
                tail.append(line)
        if tail:
            (folder/f'{name}.uncommitted-{time.time_ns()}').write_text('\n'.join(tail)+'\n')
            path.write_text('\n'.join(committed)+('\n' if committed else ''))


def train(cfg, case, corpus, folder, device, ui, stop):
    folder.mkdir(parents=True, exist_ok=True)
    seed = case['seed']; torch.manual_seed(seed)
    effective = cfg | {k: v for k, v in case.items() if k in cfg}
    mc = model_config(cfg, case); model = build(mc); initialize(model, seed); model = model.to(device)
    source = source_hashes()
    protocol = dict(config=effective, case=case, model=asdict(mc), data_manifest_sha256=digest(corpus.manifest),
                    source_hashes=source, torch=str(torch.__version__), device=str(device), dtype='float32')
    existing = folder/'protocol.json'
    if existing.exists() and json.loads(existing.read_text()) != protocol:
        raise ValueError(f'{folder}: 源码/配置/数据/设备已变；请更换输出目录，避免混合结果。')
    write(existing, protocol)
    if (folder/'result.json').exists():
        return json.loads((folder/'result.json').read_text())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    step = 0; validation_step = 0; best = float('inf'); seconds = 0.; state = None
    stream = make_training_stream(corpus, effective, seed)
    if (folder/'last.pt').exists():
        cp = torch.load(folder/'last.pt', map_location='cpu', weights_only=True)
        if cp['protocol'] != protocol:
            raise ValueError('Checkpoint protocol mismatch')
        model.load_state_dict(cp['model']); optimizer.load_state_dict(cp['optimizer'])
        step, best, seconds = cp['step'], cp['best'], cp['train_seconds']; restore_rng(cp['rng'], device)
        validation_step = cp['validation_step']
        stream.load_state_dict(cp['stream']); state = move_state(cp['runtime_state'], device)
        recover_logs(folder, step, validation_step)
        if cp['best_checkpoint'] is not None:
            save(folder/'best.pt', cp['best_checkpoint'])
        elif (folder/'best.pt').exists():
            (folder/'best.pt').replace(folder/f'best-uncommitted-{time.time_ns()}.pt')
    else:
        recover_logs(folder, 0, 0)
        if (folder/'best.pt').exists():
            (folder/'best.pt').replace(folder/f'best-uncommitted-{time.time_ns()}.pt')
        save(folder/'initial.pt', model.state_dict())
    ledger = dict(parameters=parameter_count(model), runtime_state_bytes_initial=state_bytes(model.initial_state(cfg['batch_size'])),
                  matrix_entries=case['nodes']*mc.matrix_size**2 if mc.variant not in ('transformer', 'stateless') else None,
                  batch=cfg['batch_size'], sequence_tokens=cfg['sequence_tokens'], gradient_horizon=effective['bptt_tokens'],
                  internal_steps=mc.internal_steps, graph_in_degree=mc.neighbors,
                  message='d*d carrier + (2*d+1) fresh-evidence packet, fixed modes/metadata; no path history',
                  state_note='Transformer initial KV is empty; populated bytes measured after training/evaluation',
                  training_stream='Full A documents + B intrusions + unseen A returns; mixed marked/unmarked switches; runtime state persists until finite-session reset',
                  reset_policy=effective['reset_policy'], session_tokens=effective['session_tokens'],
                  stream_mask_note='Only real within-source adjacent targets are scored; hidden source continuity never suppresses local feedback',
                  target_budget_upper_bound=cfg['train_steps']*cfg['batch_size']*cfg['sequence_tokens'],
                  optimizer_state_boundary='Runtime state detached at TBPTT/update boundaries; cached local feedback invalidated after parameter changes',
                  compute_matching='Same training tokens/updates; parameter matched Transformer; FLOPs/wall time NOT forced equal')
    write(folder/'ledger.json', ledger)
    prefix = f'{case["variant"]} N={case["nodes"]} seed={seed}'
    ui.stage('训练', f'{prefix}；从 {step}/{cfg["train_steps"]} 恢复；{ledger["parameters"]:,} 参数', total=cfg['train_steps'], initial=step)
    while step < cfg['train_steps']:
        if stop['requested']:
            break
        warmup = min(1., (step+1)/max(1, cfg['warmup_steps']))
        cosine = .1+.9*.5*(1+math.cos(math.pi*step/cfg['train_steps']))
        lr = cfg['learning_rate']*warmup*cosine
        for group in optimizer.param_groups:
            group['lr'] = lr
        values, loss_mask, reset_mask, sample = stream.next()
        sync(device); started = time.perf_counter()
        metrics, state = train_update(model, optimizer, values, effective, device, step,
                                      state=state, reset_mask=reset_mask, loss_mask=loss_mask)
        sync(device); elapsed = time.perf_counter()-started; seconds += elapsed; step += 1
        metrics.update(step=step, lr=lr, seconds=elapsed, tokens_per_second=metrics['valid_targets']/elapsed,
                       gate_blend=model.gate_blend, data=sample,
                       history_age_max=max(sample['age_end']), reset_lanes=int(reset_mask.any(1).sum()),
                       reset_events=int(reset_mask.sum()),
                       cumulative_valid_targets=stream.targets,
                       cumulative_visible_inputs=stream.inputs, cumulative_padding_targets=0,
                       cumulative_masked_targets=stream.inputs-stream.targets)
        with (folder/'metrics.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(metrics, ensure_ascii=False, allow_nan=False)+'\n')
        display = dict(loss=f'{metrics["loss"]:.3f}', lr=f'{lr:.1e}', tok_s=f'{metrics["tokens_per_second"]:.1f}',
                       age=metrics['history_age_max'], reset=metrics['reset_lanes'])
        if 'reached' in metrics:
            display['reached'] = f'{metrics["reached"]:.0%}'
        ui.update(**display)
        if step % cfg['eval_every'] == 0 or step == cfg['train_steps']:
            try:
                dev = evaluate(model, corpus, 'validation', effective, device, ui, stop=stop)
            except KeyboardInterrupt:
                checkpoint(folder/'last.pt', model, optimizer, protocol, step, best, seconds, device, state, stream, validation_step)
                return None
            score = dev['conditions']['carry']['nll']
            ui.record(dict(stage='validation_result', case=case['id'], step=step, metrics=dev))
            with (folder/'validation.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(step=step, **dev), allow_nan=False)+'\n')
            if score < best:
                best = score; save(folder/'best.pt', dict(model=model.state_dict(), step=step, validation_nll=best))
            validation_step = step
            ui.stage('训练', f'{prefix}；验证 NLL={score:.4f}；best={best:.4f}', total=cfg['train_steps'], initial=step)
        if step % cfg['checkpoint_every'] == 0 or step == cfg['train_steps']:
            ui.update(0, 状态='保存检查点')
            checkpoint(folder/'last.pt', model, optimizer, protocol, step, best, seconds, device, state, stream, validation_step)
    checkpoint(folder/'last.pt', model, optimizer, protocol, step, best, seconds, device, state, stream, validation_step)
    ledger.update(runtime_state_bytes_populated=state_bytes(state), train_seconds=seconds,
                  actual_training_targets=stream.targets, actual_visible_inputs=stream.inputs,
                  padding_targets=0, masked_boundary_targets=stream.inputs-stream.targets,
                  train_target_equivalent_exposures=stream.targets/sum(np.diff(corpus.offsets['train'])-1),
                  stream_resets=stream.resets, stream_epoch=stream.epoch, stream_scheduled_switches=stream.switches,
                  natural_change_proxy_exposures=stream.change_proxy_hits,
                  process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system() == 'Darwin' else 1024))
    if device.type == 'mps':
        ledger.update(mps_current_allocated=torch.mps.current_allocated_memory(),
                      mps_driver_allocated=torch.mps.driver_allocated_memory(),
                      mps_recommended_max=torch.mps.recommended_max_memory())
    write(folder/'ledger.json', ledger)
    if step < cfg['train_steps']:
        ui.stage('已中止并保存', f'{prefix}；安全恢复点 {step}')
        return None
    validation_path = folder/'validation.jsonl'
    if validation_step != step:
        # A stop during the final validation must resume evaluation without another optimizer step.
        try:
            dev = evaluate(model, corpus, 'validation', effective, device, ui, stop=stop)
        except KeyboardInterrupt:
            return None
        score = dev['conditions']['carry']['nll']
        with validation_path.open('a') as handle:
            handle.write(json.dumps(dict(step=step, **dev), allow_nan=False)+'\n')
        if score < best:
            best = score; save(folder/'best.pt', dict(model=model.state_dict(), step=step, validation_nll=best))
        validation_step = step
        checkpoint(folder/'last.pt', model, optimizer, protocol, step, best, seconds, device, state, stream, validation_step)
    best_cp = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(best_cp['model'])
    try:
        test = evaluate(model, corpus, 'test', effective, device, ui, folder, interventions=True, stop=stop)
    except KeyboardInterrupt:
        return None
    # One bounded, reproducible continuation state from public text; never used as a hidden history cache.
    example_values, _ = training_batch(corpus, cfg, seed, max(0, step-1))
    model.eval()
    with torch.no_grad():
        _, example_state, _ = model(torch.tensor(example_values[:1, :-1].copy(), device=device, dtype=torch.long))
    save(folder/'runtime_example.pt', dict(model=model.state_dict(), runtime_state=detach(example_state),
                                          tokens=torch.tensor(example_values[:1, :-1].copy()), model_config=asdict(mc)))
    ledger['runtime_example_per_stream_bytes'] = state_bytes(example_state)
    write(folder/'ledger.json', ledger)
    from .report import plot_training
    plot_training(folder, cfg['gate_warmup_steps'] if case['variant'] != 'transformer' else 0)
    result = dict(case=case, best_step=best_cp['step'], validation_nll=best_cp['validation_nll'], test=test,
                  ledger=ledger, status='completed', protocol_sha256=digest(protocol))
    write(folder/'result.json', result)
    ui.stage('本组完成', f'{prefix}；test BPE-PPL={test["conditions"]["carry"]["bpe_perplexity"]:.2f}')
    return result


def report(root, records):
    completed = [json.loads((root/r['id']/'result.json').read_text()) for r in records
                 if r['status'] == 'completed' and (root/r['id']/'result.json').exists()]
    write(root/'summary.json', dict(conditions=records, completed=completed))
    lines = ['# v0.7 实验状态', '', '状态登记包括未执行、运行中、已完成、失败、中止。', '',
             '| 条件 | 状态 | test BPE PPL | 历史收益 ΔNLL | 无关历史代价 ΔNLL |', '|---|---|---:|---:|---:|']
    by_id = {r['case']['id']: r for r in completed}
    for record in records:
        result = by_id.get(record['id'])
        values = ('—', '—', '—') if result is None else (
            f'{result["test"]["conditions"]["carry"]["bpe_perplexity"]:.3f}',
            f'{result["test"]["contrasts"]["relevant_history_gain"]:.4f}',
            f'{result["test"]["contrasts"]["irrelevant_history_cost"]:.4f}')
        lines.append(f'| {record["id"]} | {record["status"]} | '+ ' | '.join(values)+' |')
    # Report paired per-window differences, then independent seed means. Windows are not extra seeds.
    pairs = []
    for full in completed:
        if full['case']['variant'] != 'full' or full['case']['stage'] != 'core':
            continue
        for baseline in completed:
            a, b = full['case'], baseline['case']
            if b['stage'] == 'core' and b['nodes'] == a['nodes'] and b['seed'] == a['seed'] and b['variant'] != 'full':
                if full['test']['evaluation_sha256'] != baseline['test']['evaluation_sha256']:
                    raise ValueError('Paired evaluation inputs, targets or intervention protocol differ')
                difference = np.array(full['test']['conditions']['carry']['per_window_nll'])-np.array(baseline['test']['conditions']['carry']['per_window_nll'])
                pairs.append(dict(nodes=a['nodes'], seed=a['seed'], baseline=b['variant'],
                                  full_minus_baseline_nll=float(difference.mean()), per_window_difference=difference.tolist()))
    write(root/'paired_differences.json', pairs)
    grouped = {}
    for pair in pairs:
        grouped.setdefault((pair['nodes'], pair['baseline']), []).append(pair['full_minus_baseline_nll'])
    seed_summary = [dict(nodes=n, baseline=baseline, seeds=len(values), mean_delta_nll=float(np.mean(values)),
                         seed_std=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                         all_seed_deltas=values) for (n, baseline), values in grouped.items()]
    write(root/'seed_summary.json', seed_summary)
    lines.extend(['', '负的 full_minus_baseline_nll 表示 full 较好。原始窗口差值见 paired_differences.json；重复种子单独计数。',
                  'PPL 为自训练 byte-BPE 的窗口指标；计算预算和状态容量见每组 ledger。',
                  '同步程度只作诊断，不作为成功标准。短程 smoke/preflight 不支持能力结论。'])
    (root/'FINDINGS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def execute(cfg, root, data_root=DATA_ROOT, dry_run=False, progress=True):
    validate_config(cfg); root = Path(root); root.mkdir(parents=True, exist_ok=True)
    ui = Console(root/'events.jsonl', enabled=progress)
    planned = cases(cfg)
    if dry_run:
        records = [c | dict(status='not_executed') for c in planned]
        write(root/'plan.json', dict(config=cfg, conditions=records))
        if not (root/'conditions.json').exists():
            write(root/'conditions.json', dict(config=cfg, conditions=records))
            report(root, records)
        ui.stage('计划就绪', f'{len(records)} 组；未启动训练'); return records
    device = device_for(cfg['device']); torch.set_num_threads(cfg['cpu_threads'])
    stop = {'requested': False}
    def request_stop(*_):
        if stop['requested']:
            raise KeyboardInterrupt
        stop['requested'] = True
        ui.record(dict(stage='stop_requested', note='Save after current complete optimizer update'))
    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with RunLock(PROJECT/'results/v07/.queue.lock'), (RunLock(PROJECT/'results/v06/.queue.lock', record_owner=False) if device.type != 'cpu' else nullcontext()):
            # Respect old-version queues as well; never remove their locks.
            from v05.run import check_other_training
            check_other_training(device, root)
            ui.stage('设备与协议检查', f'{device} / FP32 / TBPTT={cfg["bptt_tokens"]}；输出 {root}')
            if device.type == 'mps':
                ui.record(dict(stage='mps_memory', recommended_bytes=torch.mps.recommended_max_memory()))
            prepare(data_root, cfg['vocab_size'], ui); corpus = Corpus(data_root)
            write(root/'training_change_proxy_sources.json', dict(split='train', cues=list(CHANGE_CUES),
                  interpretation='Uncurated lexical state-change cues only; no semantic rule-reversal labels; used for exposure audit, never model input or routing',
                  data_manifest_sha256=digest(corpus.manifest), spans=corpus.change_proxy_spans('train')))
            cfg = cfg | {'vocab_size': corpus.manifest['vocab_size']}
            existing = root/'conditions.json'
            records = [c | dict(status='not_executed') for c in planned]
            if existing.exists():
                old = json.loads(existing.read_text())
                if old['config'] != cfg or [r['id'] for r in old['conditions']] != [r['id'] for r in records]:
                    raise ValueError('Queue configuration changed; choose a new output directory')
                records = old['conditions']
                for r in records:
                    if r['status'] == 'running':
                        r['status'] = 'aborted'; r['reason'] = 'Previous process stopped without cleanup; resume from last atomic checkpoint'
            def register():
                write(existing, dict(config=cfg, conditions=records)); report(root, records)
            register()
            for index, record in enumerate(records):
                if stop['requested']:
                    break
                ui.stage('构建配对模型', f'总条件 {index+1}/{len(records)}；{record["id"]}')
                record.update(status='running', started_at=time.time()); register()
                try:
                    case = {k: v for k, v in record.items() if k not in ('status', 'started_at', 'finished_at', 'error', 'reason')}
                    result = train(cfg, case, corpus, root/record['id'], device, ui, stop)
                    record.update(status='completed' if result else 'aborted', finished_at=time.time())
                    register()
                except KeyboardInterrupt:
                    record.update(status='aborted', reason='Interrupted; latest completed checkpoint retained'); register(); break
                except Exception:
                    record.update(status='failed', error=traceback.format_exc()); register()
                    (root/record['id']/'error.txt').write_text(record['error'])
                    raise
                if device.type == 'mps':
                    torch.mps.empty_cache()
            ui.stage('队列已停止' if stop['requested'] else '队列完成',
                     f'{sum(r["status"] == "completed" for r in records)}/{len(records)} 已完成；详细结果见 FINDINGS.md')
            return records
    finally:
        ui.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def load_config(path=None):
    return json.loads(Path(path or PROJECT/'configs/v07/base.json').read_text())


def smoke_config(cfg):
    return cfg | dict(nodes=[32], seeds=[11], variants=['message', 'independent', 'full', 'transformer', 'stateless'],
                      train_steps=2, batch_size=1, sequence_tokens=8, bptt_tokens=4, internal_steps=4,
                      validation_windows=1, test_windows=1, prefix_tokens=8, target_tokens=4,
                      eval_every=2, checkpoint_every=1, checkpoint_events=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path); parser.add_argument('--mode', choices=['paired', 'single', 'smoke', 'plan', 'preflight', 'prepare'], default='paired')
    parser.add_argument('--root', type=Path); parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--device', choices=['auto', 'mps', 'cuda', 'cpu'])
    args = parser.parse_args(); cfg = load_config(args.config)
    if args.device:
        cfg['device'] = args.device
    if args.mode == 'prepare':
        prepare(args.data_root, cfg['vocab_size']); return
    if args.mode == 'preflight':
        from .profile import preflight
        preflight(cfg, args.root or PROJECT/'results/v07/aligned_preflight', args.data_root); return
    if args.mode == 'smoke':
        cfg = smoke_config(cfg)
    elif args.mode == 'single':
        cfg.update(nodes=[cfg['nodes'][0]], seeds=[cfg['seeds'][0]], variants=['full'], stage='core')
    root = args.root or PROJECT/('results/v07/aligned_smoke' if args.mode == 'smoke' else 'results/v07/aligned_public_lm')
    execute(cfg, root, args.data_root, dry_run=args.mode == 'plan')


if __name__ == '__main__':
    main()
