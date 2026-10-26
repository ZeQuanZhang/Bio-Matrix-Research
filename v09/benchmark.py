"""Paired local v0.9 engineering experiments, with immutable source/data records."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import io
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import statistics
import threading
import time
import traceback
import unittest

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from v07.common import (PROJECT, RunLock, detach, device_for, digest, file_hash,
                        save, source_hashes, state_bytes, sync, write)
from v07.data import Corpus, ROOT, prepare
from v07.model import Config as TransformerConfig, Transformer, parameter_count
from v08.benchmark import append, matched_transformer, training_update
from v08.model import Config as PreviousConfig, Network as PreviousNetwork, initialize as initialize_previous


CONDITIONS = ('v08_seq', 'v09_E', 'v09_dense', 'v09_sparse75', 'v09_sparse50', 'transformer')


def plan(nodes):
    return [dict(id=f'{condition}_n{n}', condition=condition, nodes=n, status='not_executed')
            for n in nodes for condition in CONDITIONS]


class Progress:
    """Keep the total, every condition, stage and resources visible in one screen."""
    def __init__(self, records, updates, enabled=True):
        columns = max(100, shutil.get_terminal_size((120, 30)).columns)
        options = dict(ncols=columns, mininterval=1., disable=not enabled, leave=True)
        self.total = tqdm(total=len(records), desc='v0.9 总条件', position=0, **options)
        self.details = {r['id']: tqdm(total=updates, desc=r['id'], position=i+1, **options)
                        for i, r in enumerate(records)}
        self.stage_bar = tqdm(total=0, desc='准备与校验', position=len(records)+1, **options)
        self.resource_bar = tqdm(total=0, desc='资源', position=len(records)+2, **options)
        self.device = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._refresh, daemon=True) if enabled else None
        if self.thread:
            self.thread.start()

    def _refresh(self):
        previous_cpu, previous_wall = time.process_time(), time.perf_counter()
        while not self.stop.wait(1.):
            cpu, wall = time.process_time(), time.perf_counter()
            used = 100*(cpu-previous_cpu)/max(wall-previous_wall, 1e-9)
            previous_cpu, previous_wall = cpu, wall
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rss = rss if platform.system() == 'Darwin' else rss*1024
            description = f'进程 CPU {used:.0f}% | 峰值 RSS {rss/2**30:.2f} GiB'
            if self.device is not None:
                if self.device.type == 'mps':
                    description += f' | Metal 分配 {torch.mps.current_allocated_memory()/2**30:.2f} GiB'
                elif self.device.type == 'cuda':
                    description += f' | 显存分配 {torch.cuda.memory_allocated(self.device)/2**30:.2f} GiB'
            self.resource_bar.set_description_str(description, refresh=False)
            for bar in (self.total, *self.details.values(), self.stage_bar, self.resource_bar):
                bar.refresh()

    def stage(self, message):
        self.stage_bar.set_description_str(message)

    def update(self, record, row):
        bar = self.details[record['id']]
        bar.set_postfix_str(f'{row["phase"]} loss={row["loss"]:.3f} 目标/s={row["tokens_per_second"]:.1f}', refresh=False)
        bar.update()

    def finish(self, record):
        self.details[record['id']].set_postfix_str(record['status'])
        self.total.update()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join()
        for bar in (self.resource_bar, self.stage_bar, *reversed(tuple(self.details.values())), self.total):
            bar.close()


def sources():
    paths = [*sorted((PROJECT/'v08').glob('*.py')), *sorted((PROJECT/'v09').glob('*.py')),
             PROJECT/'run_biomatrix_v09.py', PROJECT/'.run/Bio-Matrix v0.9 验证与测速.run.xml']
    return source_hashes() | {str(path.relative_to(PROJECT)): file_hash(path) for path in paths if path.is_file()}


def frozen_corpus(path):
    path = Path(path)
    if not (path/'manifest.json').is_file():
        raise FileNotFoundError('需要已准备的冻结 WikiText-2；此入口不下载数据或拟合分词器')
    prepare(path, vocab_size=4096)
    return Corpus(path)


def verify(root):
    suite = unittest.TestSuite()
    for module in ('v09.test_core', 'v09.test_feedback', 'v09.test_routing_stream',
                   'v09.test_benchmark', 'v09.test_diagnostics', 'v09.test_train',
                   'v09.test_baselines', 'v09.test_experiment', 'v09.test_evaluate'):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(module))
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    (root/'verification.txt').write_text(output.getvalue(), encoding='utf-8')
    record = dict(status='completed' if result.wasSuccessful() else 'failed', tests=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped))
    write(root/'verification.json', record)
    if not result.wasSuccessful():
        raise RuntimeError('v0.9 校验失败；未启动计时，见 verification.txt')
    return record


def build(record, vocab, group_size=16, route_mode='summary', context_limit=256):
    from .model import Config, Network
    condition = record['condition']
    if condition == 'content_only':
        from .baselines import Config as ContentConfig, ContentOnly
        cfg = ContentConfig(vocab_size=vocab, context_limit=context_limit)
        return ContentOnly(cfg), cfg
    base = dict(vocab_size=vocab, nodes=record['nodes'], matrix_size=8, width=64,
                internal_steps=4, bands=3, modes=3, neighbors=8, credit_horizon=32,
                block_size=16, implementation='sequential')
    if condition == 'v08_seq':
        cfg = PreviousConfig(**base)
        return PreviousNetwork(cfg), cfg
    options = dict(group_size=group_size, route_mode=route_mode, reuse_controls=True, batch_feedback=True)
    if condition == 'v09_E':
        options.update(content_base=False, address_size=0, active_fraction=1.)
    elif condition == 'v09_sparse75':
        options['active_fraction'] = .75
    elif condition == 'v09_sparse50':
        options['active_fraction'] = .5
    elif condition not in ('v09_dense', 'transformer'):
        raise ValueError(f'Unknown condition: {condition}')
    cfg = Config(**base, **options)
    model = Network(cfg)
    if condition == 'transformer':
        target = parameter_count(model)
        cfg = matched_transformer(TransformerConfig(vocab_size=vocab, nodes=record['nodes'],
                                                   context_limit=context_limit), target)
        return Transformer(cfg), cfg
    return model, cfg


def initialize_model(model, condition, seed):
    if condition == 'content_only':
        from .baselines import initialize
    elif condition.startswith('v09'):
        from .model import initialize
    else:
        initialize = initialize_previous
    initialize(model, seed)


def compare_pair(left, right, values, device):
    """Compare every reference state/parameter; report new routing metadata explicitly."""
    outputs = []
    for model in (left, right):
        model.to(device).train()
        logits, state, _ = model(values[:, :-1])
        loss = F.cross_entropy(logits.flatten(0, 1), values[:, 1:].flatten())
        loss.backward(); sync(device)
        outputs.append((logits.detach(), detach(state), float(loss.detach().cpu())))
    norm_rtol = 2e-3 if device.type == 'mps' else 1e-4
    right_parameters = dict(right.named_parameters())
    gradient_rows, state_rows, passed = {}, {}, True
    for name, parameter in left.named_parameters():
        other = right_parameters.get(name)
        if other is None or parameter.shape != other.shape:
            row = dict(passed=False, reason='missing or differently shaped reference parameter')
        elif parameter.grad is None or other.grad is None:
            row = dict(passed=parameter.grad is None and other.grad is None, missing=True)
        else:
            reference = float(parameter.grad.norm().detach().cpu())
            error = float((parameter.grad-other.grad).norm().detach().cpu())
            finite = bool(parameter.grad.isfinite().all().cpu()) and bool(other.grad.isfinite().all().cpu())
            row = dict(reference_norm=reference, error_norm=error,
                       passed=finite and error <= 1e-6+norm_rtol*reference)
        gradient_rows[name] = row; passed &= row['passed']
    for name, value in outputs[0][1].items():
        other = outputs[1][1].get(name)
        if other is None or value.shape != other.shape:
            row = dict(passed=False, reason='missing or differently shaped reference state')
        else:
            good = torch.allclose(value, other, atol=3e-5, rtol=3e-4) if value.is_floating_point() else torch.equal(value, other)
            error = float((value-other).abs().max().cpu()) if value.is_floating_point() and value.numel() else 0.
            row = dict(passed=bool(good), max_abs_error=error)
        state_rows[name] = row; passed &= row['passed']
    extras = {}
    for name in outputs[1][1].keys()-outputs[0][1].keys():
        value = outputs[1][1][name]
        finite = bool(value.isfinite().all().cpu())
        extras[name] = dict(shape=list(value.shape), finite=finite)
        passed &= finite
    extra_parameters = {name: dict(gradient_missing=parameter.grad is None)
                        for name, parameter in right.named_parameters()
                        if name not in dict(left.named_parameters())}
    for name in extra_parameters:
        gradient = right_parameters[name].grad
        passed &= gradient is None or bool(gradient.isfinite().all().cpu())
    logit_error = float((outputs[0][0]-outputs[1][0]).abs().max().cpu())
    passed &= bool(torch.allclose(outputs[0][0], outputs[1][0], atol=3e-5, rtol=3e-4))
    passed &= all(math.isfinite(row[2]) for row in outputs)
    return dict(status='completed' if passed else 'failed', losses=[row[2] for row in outputs],
                logit_max_abs_error=logit_error, gradients=gradient_rows, states=state_rows,
                extra_runtime_state=extras, extra_parameters=extra_parameters,
                gradient_norm_rtol=norm_rtol, gradient_norm_atol=1e-6)


def native_equivalence(root, corpus, device, seed):
    from .model import Config, Network, initialize
    values, _, _ = corpus.sample('train', np.random.default_rng(seed), 2, 32)
    tensor = torch.tensor(values, dtype=torch.long, device=device)
    old, _ = build(dict(condition='v08_seq', nodes=32), corpus.manifest['vocab_size'])
    equivalent, cfg = build(dict(condition='v09_E', nodes=32), corpus.manifest['vocab_size'])
    initialize_previous(old, seed); initialize(equivalent, seed)
    first = compare_pair(old, equivalent, tensor, device)
    del old, equivalent
    cfg = Config(vocab_size=corpus.manifest['vocab_size'], nodes=32, matrix_size=8,
                 width=64, bands=3, modes=3, internal_steps=4, block_size=16,
                 credit_horizon=32, active_fraction=.5, group_size=8, implementation='sequential')
    sequential = Network(cfg); triangular = Network(replace(cfg, implementation='triangular'))
    initialize(sequential, seed); triangular.load_state_dict(sequential.state_dict())
    second = compare_pair(sequential, triangular, tensor, device)
    record = dict(status='completed' if first['status'] == second['status'] == 'completed' else 'failed',
                  device=str(device), dtype='float32', sample_sha256=digest(values.tolist()),
                  v08_vs_v09_E=first, sparse_content_sequential_vs_triangular=second,
                  sparse_config=asdict(cfg), note='All v0.8 state keys/parameters are checked; new E route metadata and unused route parameters are disclosed.')
    write(root/'native_equivalence.json', record)
    if record['status'] != 'completed':
        raise RuntimeError('原生设备输出/完整状态/梯度不一致，见 native_equivalence.json')
    return record


def route_snapshot(model):
    return [[dict(groups=list(row['groups']), active_ids=row['active_ids'].detach().cpu().tolist())
             for row in block] for block in getattr(model, 'last_routes', [])]


def run_case(record, args, corpus, device, folder, ui):
    torch.manual_seed(args.seed)
    model, cfg = build(record, corpus.manifest['vocab_size'], args.group_size, args.route_mode)
    initialize_model(model, record['condition'], args.seed); model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    count = args.warmup+args.repeats+1
    values, docs, starts = corpus.sample('train', np.random.default_rng(args.seed), args.batch,
                                        count*args.sequence_tokens)
    tensor = torch.tensor(values, dtype=torch.long, device=device)
    state, warmups, trials, profile_row = None, [], [], None
    ui.stage(f'{record["id"]} · 连续文档块训练更新')
    for step in range(count):
        begin = step*args.sequence_tokens
        profile = step == count-1
        if hasattr(model, 'profile_enabled'):
            model.profile_enabled = profile
            model.profile_records = []
        row, state = training_update(model, optimizer, tensor[:, begin:begin+args.sequence_tokens+1], state, device, profile)
        row.update(step=step, phase='profile' if profile else 'warmup' if step < args.warmup else 'timed')
        row['routes'] = route_snapshot(model)
        if profile:
            row['regions'] = getattr(model, 'profile_records', [])
            profile_row = row
        elif step < args.warmup:
            warmups.append(row)
        else:
            trials.append(row)
        append(folder/'raw.jsonl', row); ui.update(record, row)
    for parameter in model.parameters():
        if not bool(parameter.detach().isfinite().all().cpu()):
            raise FloatingPointError('Nonfinite final parameter')
    save(folder/'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
         runtime_state=state, config=asdict(cfg), record=record, step=count, formal_training=False,
         note='Engineering snapshot; no exact training-resume protocol.'))
    result = dict(status='completed', config=asdict(cfg), parameters=parameter_count(model),
                  state_bytes=state_bytes(state), warmup=warmups, trials=trials, profile=profile_row,
                  median_seconds=statistics.median(row['seconds'] for row in trials),
                  targets=sum(row['targets'] for row in trials),
                  tokens_per_second=sum(row['targets'] for row in trials)/sum(row['seconds'] for row in trials),
                  sample=dict(split='train', documents=docs, starts=starts, sha256=digest(values.tolist())),
                  actual_active_fraction=[row['diagnostics'].get('active_fraction') for row in trials])
    if device.type == 'mps':
        result['memory'] = dict(allocated_snapshot=torch.mps.current_allocated_memory(),
                                driver_snapshot=torch.mps.driver_allocated_memory(), isolated_peak=False)
    elif device.type == 'cuda':
        result['memory'] = dict(peak_allocated=torch.cuda.max_memory_allocated(device),
                                peak_reserved=torch.cuda.max_memory_reserved(device))
    return result


def report(root, records):
    lines = ['# Bio-Matrix v0.9 配对工程实测', '',
             '本地设备；冻结 WikiText-2/4096 BPE；工程正确性与训练吞吐，不承担长期学习质量结论。', '',
             '| 条件 | 状态 | 中位 ms | 合计有效目标/秒 | 参数 | 状态字节 | 实际活跃比例 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in records:
        result = row.get('result', {})
        ms = f'{1000*result["median_seconds"]:.2f}' if 'median_seconds' in result else '—'
        speed = f'{result["tokens_per_second"]:.1f}' if 'tokens_per_second' in result else '—'
        active = [value for value in result.get('actual_active_fraction', []) if value is not None]
        fraction = f'{statistics.mean(active):.3f}' if active else '—'
        lines.append(f'| {row["id"]} | {row["status"]} | {ms} | {speed} | {result.get("parameters", "—")} | {result.get("state_bytes", "—")} | {fraction} |')
    lines += ['', '## 比较边界', '',
              'v09_E 关闭新内容主干与低维地址，满节点执行；其对 v08_seq 是代数复用/反馈批处理对照。',
              'v09_dense 启用新内容和地址；稀疏条件还改变活跃节点与空闲冻结规则，均是新动力学。',
              '75%/50% 是目标预算；实际预算包含必选通路、整数分组和独立 batch 路由，表中如实报告。',
              'Transformer 按 v09_dense 参数量匹配；参数接近不等于状态、FLOPs或历史容量相同。',
              '同一规模各模型共享连续文档、FP32、batch、序列、AdamW 与有效目标；完整训练更新含前向、CE、反向、clip、优化器、detach和信用失效。',
              '数据采样、保存、路由记录转CPU和控制台在计时外；模型内路由/收集/回填在计时内。',
              '额外 profile 使用同步记录整网阶段及 v09 区域；不混入三个主计时样本。区域和不是反向耗时归因。',
              'MPS 内存为分配快照，不是独占峰值；进程RSS行为峰值。固定执行顺序与少量样本可能受到热状态影响。',
              '逐token流式接口采用前缀重放以保持块时序；本次没有测生成延迟，也不宣称高速decode。',
              '本入口不启动正式长训、封存测试或远程任务；last.pt为工程快照，未提供精确恢复。', '',
              f'完成 {sum(row["status"] == "completed" for row in records)}/{len(records)}；所有失败、中止和未执行保存在 registry.json。', '']
    (root/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def validate(args):
    if args.repeats < 3 or args.warmup < 1 or min(args.batch, args.threads, args.group_size) < 1:
        raise ValueError('Need repeats>=3, warmup>=1 and positive batch/threads/group-size')
    if args.sequence_tokens < 32 or args.sequence_tokens % 16:
        raise ValueError('sequence-tokens must be a multiple of C16 and >=32')
    if not args.nodes or len(set(args.nodes)) != len(args.nodes) or any(n not in (32, 128, 512) for n in args.nodes):
        raise ValueError('nodes must be unique values from 32,128,512')


def execute(args):
    validate(args)
    root = Path(args.output or PROJECT/'results/v09'/datetime.now().strftime('benchmark_%Y%m%dT%H%M%S'))
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Refusing to overwrite results: {root}')
    root.mkdir(parents=True, exist_ok=True)
    records = plan(args.nodes); write(root/'registry.json', records)
    source = sources()
    manifest = dict(status='running', created_utc=datetime.now(timezone.utc).isoformat(),
          args={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
          platform=platform.platform(), torch=str(torch.__version__), source_hashes=source,
          formal_training=False, timing_dtype='float32', concurrency=1)
    write(root/'manifest.json', manifest)
    for name in source:
        path = root/'source'/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((PROJECT/name).read_bytes())
    ui = Progress(records, args.warmup+args.repeats+1, enabled=not args.quiet)
    current = None
    try:
        with RunLock(PROJECT/'results/v09/.benchmark.lock'):
            ui.stage('CPU 数学、因果、信用、流式与路由验证')
            verification = verify(root)
            torch.set_num_threads(args.threads)
            device = device_for(args.device); ui.device = device
            if device.type == 'mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
                raise RuntimeError('Disable PYTORCH_ENABLE_MPS_FALLBACK for native GPU timing')
            ui.stage('冻结数据 SHA-256 校验')
            corpus = frozen_corpus(args.data_root)
            ui.stage('原生 FP32 等价 E 与稀疏内容 seq/tri 前反向验收')
            native = native_equivalence(root, corpus, device, args.seed)
            write(root/'environment.json', dict(device=str(device), threads=torch.get_num_threads(),
                  data_manifest_sha256=digest(corpus.manifest), data_manifest=corpus.manifest,
                  verification=verification, native_status=native['status'], mps_fallback=False))
            for current in records:
                if current['condition'] not in args.conditions:
                    current['reason'] = 'excluded by --conditions'; ui.finish(current)
                    continue
                current['status'] = 'running'; write(root/'registry.json', records)
                folder = root/current['id']; folder.mkdir()
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                try:
                    current['result'] = run_case(current, args, corpus, device, folder, ui)
                    current['status'] = 'completed'
                except Exception:
                    current['status'] = 'failed'; current['error'] = traceback.format_exc()
                    (folder/'error.txt').write_text(current['error'], encoding='utf-8')
                write(folder/'result.json', current); write(root/'registry.json', records)
                report(root, records); ui.finish(current)
                gc.collect()
                if device.type == 'mps':
                    torch.mps.empty_cache()
                elif device.type == 'cuda':
                    torch.cuda.empty_cache()
            if sources() != source:
                raise RuntimeError('Source changed during experiment; measurements must not be pooled')
            status = 'failed' if any(row['status'] == 'failed' for row in records) else 'completed'
            write(root/'summary.json', dict(status=status, conditions=records, formal_training=False))
            write(root/'manifest.json', manifest | dict(status=status))
            return root, status
    except KeyboardInterrupt:
        if current is not None and current['status'] == 'running':
            current['status'] = 'aborted'
        write(root/'registry.json', records); report(root, records)
        write(root/'summary.json', dict(status='aborted', conditions=records, formal_training=False))
        write(root/'manifest.json', manifest | dict(status='aborted'))
        raise
    except Exception:
        (root/'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        write(root/'summary.json', dict(status='failed', conditions=records, formal_training=False))
        write(root/'manifest.json', manifest | dict(status='failed'))
        raise
    finally:
        ui.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'), default='auto')
    parser.add_argument('--nodes', nargs='+', type=int, default=[32, 128, 512])
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument('--sequence-tokens', type=int, default=64)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--warmup', type=int, default=6)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--group-size', type=int, default=16)
    parser.add_argument('--route-mode', choices=('summary', 'cyclic'), default='summary')
    parser.add_argument('--data-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    root, status = execute(args)
    print(f'v0.9 {status}: {root}/REPORT.md')
    if status != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
