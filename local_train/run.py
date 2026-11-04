"""Independent Mac queue; reuse frozen V3 operators and the official evaluator."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import gc
import hashlib
from importlib.metadata import version
import json
import logging
import math
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch

from scale import engine
from scale.data import Tokens
from scale.model import CONDITIONS, build, inspect
from scale.run import append, clean_log, lm_batch, sources as scale_sources
from v07.common import PROJECT, RunLock, device_for, digest, file_hash, save, write
from v09.train import StopRequest, capture_rng, restore_random
from evaluations.standard.run import LOCK as EVALUATION_LOCK
from . import data, evaluate
from .console import Progress

CONFIG = PROJECT/'configs/local/v3_31m.json'
OUTPUT = PROJECT/'results/local_train/v3_31m_100m_s11'
CHECKPOINT = 'bio-v3-local-checkpoint-1'


def settings(path, smoke=False):
    c = json.loads(path.read_text())
    if c['format'] != 'bio-v3-local-train-1':
        raise ValueError('Unsupported local training protocol')
    if any(type(c[k]) is not int or c[k] < 0 for k in ('seed', 'data_seed')) or type(c['activation_checkpoint']) is not bool:
        raise ValueError('Invalid seed or checkpoint setting')
    for key in ('pretrain_tokens', 'sequence', 'microbatch', 'accumulation', 'warmup_tokens',
                'checkpoint_every', 'eval_every', 'validation_sequences', 'eval_batch', 'threads'):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f'Positive integer required: {key}')
    if c['milestones'] != sorted(set(c['milestones'])) or c['milestones'][-1] != c['pretrain_tokens'] or min(c['milestones']) < 1:
        raise ValueError('Invalid paired milestones')
    if c['sequence'] != c['model']['context_limit'] or c['sequence'] % c['model']['segment_tokens']:
        raise ValueError('Sequence must cover complete Bio blocks and match context')
    if not 0 < c['warmup_tokens'] < c['pretrain_tokens']:
        raise ValueError('Warmup exceeds the training budget')
    for key in ('learning_rate', 'gradient_clip', 'mps_limit_gib'):
        if not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f'Invalid {key}')
    if not 0 <= c['minimum_lr_fraction'] <= 1 or not 0 <= c['weight_decay'] < 1:
        raise ValueError('Invalid decay settings')
    if smoke:
        per_update = c['sequence']*c['microbatch']*c['accumulation']
        c.update(pretrain_tokens=per_update*2, milestones=[per_update, per_update*2],
                 warmup_tokens=per_update, checkpoint_every=1, eval_every=2, validation_sequences=2)
    return c


def sources():
    paths = list((PROJECT/'local_train').glob('*.py'))
    paths += list((PROJECT/'evaluations/standard').glob('*.py'))
    paths += [EVALUATION_LOCK]
    return scale_sources() | {str(p.relative_to(PROJECT)): file_hash(p) for p in paths}


def release(device):
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()


def train(root, data_root, c, ui, stop, device, smoke=False, max_updates=None):
    manifest = data.prepare(data_root, c, ui, stop)
    p = dict(format=c['format'], config=c, device=str(device), precision='fp32', engineering_smoke=smoke,
             packages={k: version(k) for k in ('torch', 'lm_eval', 'datasets', 'transformers', 'tokenizers', 'numpy', 'pyarrow')},
             data_manifest_sha256=digest(manifest), sources=sources(), capacity=inspect(c['model']),
             standard_evaluation=evaluate.protocol(),
             comparison='Same ordered token batches, masks, updates and token schedule; matched parameters. Compute time is measured, not matched. One training seed.',
             state='Reset each 512-token sequence; complete gradients across four 128-token Bio blocks. No persistent state across unrelated packed samples.')
    if (root/'protocol.json').exists():
        if json.loads((root/'protocol.json').read_text()) != p:
            raise ValueError('恢复被拒绝：配置、代码、数据、设备或运行环境已改变，请使用新的结果目录')
    else:
        write(root/'protocol.json', p)
        for name in p['sources']:
            destination = root/'source'/name; destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((PROJECT/name).read_bytes())
    registry_path = root/'registry.json'
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else dict(conditions={n: dict(training='not_executed', evaluation='not_executed') for n in CONDITIONS})
    registry['status'] = 'running'; write(registry_path, registry)
    ui.total.total = 1+3*len(c['milestones'])+15; ui.total.n = 1
    for name in CONDITIONS:
        ui.details[name].total = c['pretrain_tokens']
    stream = Tokens(data_root, manifest, 'train'); val = Tokens(data_root, manifest, 'validation')
    order = np.random.default_rng(c['data_seed']).permutation((stream.total-1)//c['sequence'])
    batch = c['microbatch']*c['accumulation']; active = None; phase = 'training'; executed = 0
    model = opt = payload = None
    try:
        for target in c['milestones']:
            traces = []
            for name in CONDITIONS:
                active = name; phase = 'training'; folder = root/name; folder.mkdir(exist_ok=True)
                last = folder/'last.pt'; artifact = folder/f'tokens_{target}.json'
                counter = dict(tokens=0, blocks=0, updates=0, train_seconds=0., batch_chain='')
                if last.exists():
                    payload = torch.load(last, map_location='cpu', weights_only=True)
                    if payload['format'] != CHECKPOINT or payload['protocol_sha256'] != digest(p):
                        raise ValueError('Checkpoint identity mismatch')
                    counter = payload['counters']
                bar = ui.details[name]; bar.n = counter['tokens']
                if counter['tokens'] >= target and artifact.exists() and artifact.with_suffix('.pt').exists():
                    traces.append(json.loads(artifact.read_text())['batch_chain']); payload = None
                    ui.total.update()
                    continue
                if counter['tokens'] > target:
                    raise ValueError('Earlier paired milestone is missing; preserve this run for inspection')
                ui.stage(f'{name} · 初始化/恢复 · 目标 {target:,} token')
                registry['conditions'][name]['training'] = 'running'; write(registry_path, registry)
                model = build(name, c['model'], c['seed'], c['activation_checkpoint']).to(device)
                opt = engine.optimizer(model, c['learning_rate'], c['weight_decay'], device)
                if payload is not None:
                    model.load_state_dict(payload['model']); opt.load_state_dict(payload['optimizer'])
                    restore_random(payload['rng'], device); payload = None
                clean_log(folder/'metrics.jsonl', counter['updates'])
                clean_log(folder/'validation.jsonl', counter['updates'])

                def commit():
                    if sources() != p['sources']:
                        raise ValueError('Frozen source changed; checkpoint refused')
                    save(last, dict(format=CHECKPOINT, protocol_sha256=digest(p), model=model.state_dict(),
                         optimizer=opt.state_dict(), rng=capture_rng(device), counters=counter.copy()))

                while counter['tokens'] < target:
                    if stop.requested or max_updates is not None and executed >= max_updates:
                        commit(); raise KeyboardInterrupt
                    values, mask = lm_batch(stream, order, counter['blocks'], batch, c['sequence'], target-counter['tokens'])
                    completed = counter['tokens']+int(mask.sum())
                    rate = c['learning_rate']*engine.rate_at(completed, c['pretrain_tokens'], c['warmup_tokens'], c['minimum_lr_fraction'])
                    for group in opt.param_groups:
                        group['lr'] = rate
                    ui.stage(f'{name} · 训练第 {counter["updates"]+1:,} 步 · {counter["tokens"]:,}/{target:,} token · {bar.postfix}')
                    row = engine.update(model, opt, values, mask, c['microbatch'], device, 'fp32', c['gradient_clip'])
                    counter.update(tokens=completed, blocks=counter['blocks']+len(values), updates=counter['updates']+1,
                                   train_seconds=counter['train_seconds']+row['seconds'],
                                   batch_chain=hashlib.sha256(counter['batch_chain'].encode()+values.tobytes()+mask.tobytes()).hexdigest())
                    executed += 1; row.update(counter, lr=rate)
                    append(folder/'metrics.jsonl', row)
                    bar.n = completed; bar.postfix = f'CE={row["ce"]:.4f} {row["targets_per_second"]:.0f} tok/s'
                    seconds_left = (c['pretrain_tokens']-completed)/max(1, completed)*counter['train_seconds']
                    ui.stage(f'{name} · {completed:,}/{target:,} token · {bar.postfix} · LR={rate:.2g} · 本模型约剩 {seconds_left/3600:.1f}h')
                    write(folder/'progress.json', dict(status='running', **row))
                    if counter['updates'] % c['checkpoint_every'] == 0 or completed == target or stop.requested:
                        ui.stage(f'{name} · 保存断点 · {bar.postfix}'); commit()
                    if stop.requested:
                        raise KeyboardInterrupt
                    if counter['updates'] % c['eval_every'] == 0 or completed == target:
                        ui.stage(f'{name} · 固定留出语料验证 · {bar.postfix}')
                        metrics = engine.evaluate_lm(model, val, c['sequence'], c['validation_sequences'], device, 'fp32')
                        append(folder/'validation.jsonl', dict(updates=counter['updates'], tokens=completed, **metrics))
                ui.stage(f'{name} · 保存 {target:,} token 里程碑')
                save(artifact.with_suffix('.pt'), dict(format=CHECKPOINT, protocol_sha256=digest(p), model=model.state_dict(), counters=counter.copy()))
                write(artifact, dict(status='completed', **counter, checkpoint_sha256=file_hash(artifact.with_suffix('.pt'))))
                traces.append(counter['batch_chain'])
                ui.total.update()
                registry['conditions'][name]['training'] = 'completed' if target == c['pretrain_tokens'] else 'paused_at_milestone'
                registry['conditions'][name]['tokens'] = target; write(registry_path, registry)
                model = opt = None; release(device)
            if len(set(traces)) != 1:
                raise ValueError('Paired models received different ordered token batches or masks')
            write(root/f'paired_{target}.json', dict(status='completed', tokens_per_model=target, identical_batch_chain=traces[0]))
        for name in CONDITIONS:
            active = name; phase = 'evaluation'; folder = root/name
            if registry['conditions'][name]['evaluation'] == 'completed':
                ui.total.update(5)
                for task in evaluate.protocol()['tasks']:
                    ui.details[task].n = ui.details[task].total
                continue
            if stop.requested:
                raise KeyboardInterrupt
            registry['conditions'][name]['evaluation'] = 'running'; write(registry_path, registry)
            model = build(name, c['model'], c['seed'], c['activation_checkpoint']).to(device)
            checkpoint = folder/f'tokens_{c["pretrain_tokens"]}.pt'
            if file_hash(checkpoint) != json.loads(checkpoint.with_suffix('.json').read_text())['checkpoint_sha256']:
                raise ValueError('Final checkpoint changed')
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            model.load_state_dict(payload['model']); payload = None
            for task in evaluate.protocol()['tasks']:
                ui.details[task].reset(1); ui.details[task].postfix = '等待'
            evaluate.run(model, data_root, folder/'standard_eval', c, ui, stop, smoke)
            registry['conditions'][name]['evaluation'] = 'completed'; write(registry_path, registry)
            model = None; release(device)
        registry['status'] = 'completed'; ui.total.n = ui.total.total
        ui.stage('三模型训练与公开评测已完成，结果已保存')
    except BaseException as error:
        registry['status'] = 'aborted' if isinstance(error, KeyboardInterrupt) else 'failed'
        if active:
            registry['conditions'][active][phase] = registry['status']
        raise
    finally:
        write(registry_path, registry)
        model = opt = payload = None; release(device)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('train', 'prepare', 'smoke'), default='train')
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)
    smoke = args.mode == 'smoke'; c = settings(args.config, smoke)
    root = args.output or (PROJECT/'results/local_train'/datetime.now().strftime('smoke_%Y%m%dT%H%M%S_%f') if smoke else OUTPUT)
    data_root = args.data_root or PROJECT/'data/local31m'/('smoke' if smoke else 'fineweb_100m')
    root.mkdir(parents=True, exist_ok=True)
    with RunLock(PROJECT/'results/local_train/queue.lock'), StopRequest() as stop:
        ui = Progress(not args.quiet)
        status = dict(status='running', mode=args.mode)
        previous = root/'execution.json'
        if previous.exists() and json.loads(previous.read_text())['status'] == 'running':
            append(root/'attempts.jsonl', dict(status='aborted', reason='previous_process_ended_without_final_status', time=datetime.now().isoformat()))
        write(previous, status)
        append(root/'attempts.jsonl', dict(**status, time=datetime.now().isoformat()))
        try:
            with (root/'runtime.private.log').open('a', encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
                logging.basicConfig(stream=log, level=logging.INFO, force=True)
                for name in ('httpx', 'httpx2', 'httpcore'):
                    logging.getLogger(name).setLevel(logging.WARNING)
                device = device_for(args.device); ui.device = device; torch.set_num_threads(c['threads'])
                if device.type == 'mps':
                    torch.mps.set_per_process_memory_fraction(min(1., c['mps_limit_gib']*2**30/torch.mps.recommended_max_memory()))
                if shutil.disk_usage(root).free < 10*2**30:
                    raise OSError('至少保留 10 GiB 数据与检查点空间')
                if args.mode == 'prepare':
                    data.prepare(data_root, c, ui, stop)
                else:
                    train(root, data_root, c, ui, stop, device, smoke)
                status['status'] = 'completed'
        except KeyboardInterrupt:
            status['status'] = 'aborted'; ui.stage('已停止，已提交断点保留；再次 Run 自动续跑')
        except BaseException as error:
            status.update(status='failed', error=str(error)); ui.stage(f'失败：{error}')
            with (root/'failure.private.log').open('a') as log:
                traceback.print_exc(file=log)
            raise
        finally:
            append(root/'attempts.jsonl', dict(**status, time=datetime.now().isoformat()))
            write(root/'execution.json', status); ui.close()
        print(f'本地任务 {status["status"]} · 结果：{root}', flush=True)


if __name__ == '__main__':
    main()
