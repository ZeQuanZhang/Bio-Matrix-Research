"""Reproducible local compute experiments; never resumes the retired A800 host."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import io
import json
import math
from pathlib import Path
import platform
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
from v07.model import (Config as OldConfig, Network as OldNetwork, Transformer,
                       initialize, invalidate_feedback, parameter_count)
from .recurrence import parallel_delta, sequential_delta, triangular_delta


class Progress:
    """Two stationary bars, refreshed once a second even during a long update."""
    def __init__(self, total, enabled=True):
        self.total = tqdm(total=total, desc='v0.8 条件', position=0, dynamic_ncols=True,
                          mininterval=1, disable=not enabled)
        self.detail = tqdm(total=1, desc='准备', position=1, dynamic_ncols=True,
                           mininterval=1, leave=False, disable=not enabled)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.refresh, daemon=True)
        self.thread.start()

    def refresh(self):
        while not self.stop.wait(1):
            self.total.refresh(); self.detail.refresh()

    def stage(self, name, count):
        self.detail.reset(total=count); self.detail.set_description_str(name)

    def update(self, **values):
        self.detail.set_postfix(values, refresh=False); self.detail.update()

    def close(self):
        self.stop.set(); self.thread.join(); self.detail.close(); self.total.close()


def append(path, value):
    with Path(path).open('a', encoding='utf-8') as output:
        output.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')


def sources():
    paths = list((PROJECT/'v08').glob('*.py'))+[PROJECT/'run_biomatrix_v08.py']
    return source_hashes() | {str(p.relative_to(PROJECT)): file_hash(p) for p in paths if p.exists()}


def verify(root):
    suite = unittest.TestSuite()
    for name in ('v08.test_recurrence', 'v08.test_model', 'v08.test_benchmark'):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name))
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    (root/'verification.txt').write_text(log.getvalue(), encoding='utf-8')
    record = dict(status='completed' if result.wasSuccessful() else 'failed',
                  tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  skipped=len(result.skipped))
    write(root/'verification.json', record)
    if not result.wasSuccessful():
        raise RuntimeError('数学校验失败；见 verification.txt，未启动计时')
    return record


def plan(nodes, sequence):
    records = []
    for n in nodes:
        for length in (16, 32):
            for method in ('sequential', 'parallel', 'triangular'):
                records.append(dict(id=f'kernel_n{n}_t{length}_{method}', kind='kernel',
                                    nodes=n, sequence=length, method=method, status='not_executed'))
        for family, method, block in (('v07', 'sequential', 1),
                                      ('v08', 'sequential', 16), ('v08', 'parallel', 16),
                                      ('v08', 'triangular', 16), ('v08', 'triangular', 32),
                                      ('transformer', 'batched', 1)):
            records.append(dict(id=f'{family}_n{n}_{method}_c{block}', kind='model', nodes=n,
                                sequence=sequence, family=family, method=method, block=block,
                                status='not_executed'))
    return records


def matched_transformer(cfg, target):
    choices = []
    for width in range(16, 129, 8):
        for ff in range(2*width, 6*width+1, 8):
            candidate = replace(cfg, variant='transformer', transformer_width=width,
                                transformer_ff=ff, transformer_heads=4)
            # Count analytically to avoid consuming model initialization randomness.
            count = cfg.vocab_size*width + cfg.transformer_layers*(4*width*width+2*width*ff+ff+5*width)+2*width
            choices.append((abs(count-target), candidate))
    return min(choices, key=lambda item: item[0])[1]


def build(record, vocab):
    from .model import Config, Network
    options = dict(vocab_size=vocab, nodes=record['nodes'], matrix_size=8, width=64,
                   internal_steps=4, bands=3, modes=3, neighbors=8, credit_horizon=32)
    if record['family'] == 'v08':
        cfg = Config(**options, block_size=record['block'], implementation=record['method'])
        return Network(cfg), cfg
    if record['family'] == 'v07':
        cfg = OldConfig(**options)
        return OldNetwork(cfg), cfg
    target = parameter_count(Network(Config(**options)))
    cfg = matched_transformer(OldConfig(**options), target)
    return Transformer(cfg), cfg


def native_equivalence(root, corpus, device, seed):
    """Check production d8 FP32 hard-gate execution before trusting GPU timing."""
    from .model import Config, Network, initialize as initialize_v08
    cfg = Config(vocab_size=corpus.manifest['vocab_size'], nodes=32, matrix_size=8,
                 width=64, bands=3, modes=3, internal_steps=4, block_size=16,
                 implementation='sequential', credit_horizon=32)
    values, _, _ = corpus.sample('train', np.random.default_rng(seed), 2, 32)
    tensor = torch.tensor(values, dtype=torch.long, device=device)
    models = [Network(cfg), Network(replace(cfg, implementation='triangular'))]
    initialize_v08(models[0], seed); models[1].load_state_dict(models[0].state_dict())
    results = []
    for model in models:
        model.to(device).train()
        logits, state, _ = model(tensor[:, :-1])
        loss = F.cross_entropy(logits.flatten(0, 1), tensor[:, 1:].flatten())
        loss.backward(); sync(device)
        results.append((logits.detach(), detach(state), float(loss.detach().cpu())))
    norm_rtol = 2e-3 if device.type == 'mps' else 1e-4
    gradients = {}; passed = True
    for (name, left), (other, right) in zip(models[0].named_parameters(), models[1].named_parameters()):
        assert name == other
        if left.grad is None or right.grad is None:
            gradients[name] = dict(passed=left.grad is None and right.grad is None, missing=True)
        else:
            reference = float(left.grad.norm().detach().cpu())
            error = float((left.grad-right.grad).norm().detach().cpu())
            finite = bool(left.grad.isfinite().all().cpu()) and bool(right.grad.isfinite().all().cpu())
            gradients[name] = dict(reference_norm=reference, error_norm=error,
                 relative_error=error/max(reference,1e-12), passed=finite and error <= 1e-6+norm_rtol*reference)
        passed &= gradients[name]['passed']
    state_errors = {}
    for name, left in results[0][1].items():
        right = results[1][1][name]
        if left.is_floating_point():
            state_errors[name] = float((left-right).abs().max().cpu()) if left.numel() else 0.
            passed &= torch.allclose(left, right, atol=3e-5, rtol=3e-4)
        else:
            passed &= torch.equal(left,right)
    logit_error = float((results[0][0]-results[1][0]).abs().max().cpu())
    passed &= torch.allclose(results[0][0],results[1][0],atol=3e-5,rtol=3e-4)
    record = dict(status='completed' if passed else 'failed', device=str(device), config=asdict(cfg),
                  sample_sha256=digest(values.tolist()), losses=[x[2] for x in results],
                  logit_max_abs_error=logit_error, state_max_abs_errors=state_errors,
                  gradients=gradients, gradient_norm_rtol=norm_rtol, gradient_norm_atol=1e-6,
                  global_gradient_norm=float(torch.stack([p.grad.norm().square() for p in models[0].parameters() if p.grad is not None]).sum().sqrt().cpu()))
    write(root/'native_equivalence.json', record)
    if not passed:
        raise RuntimeError('Native FP32 implementation mismatch; see native_equivalence.json')
    del models, results
    return record


def training_update(model, optimizer, values, state, device, profile=False):
    """Time device-resident forward/CE/backward/clip/AdamW, with common boundaries."""
    times = {}; sync(device); started = last = time.perf_counter()

    def mark(name):
        nonlocal last
        if profile:
            sync(device); now = time.perf_counter(); times[name] = now-last; last = now

    optimizer.zero_grad(set_to_none=True)
    logits, state, diagnostics = model(values[:, :-1], state)
    mark('forward')
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), values[:, 1:].reshape(-1))
    mark('loss')
    loss.backward(); mark('backward')
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    mark('clip')
    optimizer.step(); mark('optimizer')
    state = invalidate_feedback(detach(state)); mark('state_bookkeeping')
    sync(device); seconds = time.perf_counter()-started
    metrics = dict(seconds=seconds, loss=float(loss.detach().cpu()), gradient_norm=float(norm.detach().cpu()),
                   targets=values.shape[0]*(values.shape[1]-1), phase_seconds=times,
                   diagnostics={name: float(value.detach().cpu()) for name,value in diagnostics.items()})
    if not all(math.isfinite(metrics[key]) for key in ('seconds', 'loss', 'gradient_norm')):
        raise FloatingPointError('Nonfinite loss/gradient/timing')
    metrics['tokens_per_second'] = metrics['targets']/seconds
    return metrics, state


def model_case(record, args, corpus, device, folder, ui):
    torch.manual_seed(args.seed)
    model, cfg = build(record, corpus.manifest['vocab_size'])
    if record['family'] == 'v08':
        from .model import initialize as initialize_v08
        initialize_v08(model, args.seed)
    else:
        initialize(model, args.seed)
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    count = args.warmup+args.repeats+1
    # One document per lane, contiguous windows across optimizer updates.
    values, docs, starts = corpus.sample('train', np.random.default_rng(args.seed), args.batch,
                                        count*args.sequence_tokens)
    tensor = torch.tensor(values, dtype=torch.long, device=device)
    sample_hash = digest(values.tolist())
    state = None; trials = []; warmups = []
    ui.stage(record['id'], count)
    for step in range(count):
        begin = step*args.sequence_tokens
        batch = tensor[:, begin:begin+args.sequence_tokens+1]
        profile = step == count-1
        metrics, state = training_update(model, optimizer, batch, state, device, profile)
        metrics.update(step=step, phase='profile' if profile else 'warmup' if step < args.warmup else 'timed')
        append(folder/'raw.jsonl', metrics)
        if profile:
            phases = metrics
        elif step < args.warmup:
            warmups.append(metrics)
        else:
            trials.append(metrics)
        ui.update(loss=f'{metrics["loss"]:.3f}', tps=f'{metrics["tokens_per_second"]:.0f}')
    for value in model.parameters():
        if not bool(value.detach().isfinite().all().cpu()):
            raise FloatingPointError('Nonfinite trained parameter')
    save(folder/'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                               runtime_state=state, config=asdict(cfg), record=record,
                               step=count, formal_training=False))
    result = dict(status='completed', config=asdict(cfg), parameters=parameter_count(model),
                  state_bytes=state_bytes(state), trials=trials, warmup=warmups, profile=phases,
                  median_seconds=statistics.median(x['seconds'] for x in trials),
                  targets=sum(x['targets'] for x in trials),
                  tokens_per_second=sum(x['targets'] for x in trials)/sum(x['seconds'] for x in trials),
                  sample=dict(split='train', documents=docs, starts=starts, sha256=sample_hash),
                  diagnostics='v07 retains its built-in per-token diagnostics; no console logging inside update',
                  bptt='one full sequence per backward, all families; detach and invalidate once per optimizer update')
    if device.type == 'mps':
        result['memory'] = dict(allocated_snapshot=torch.mps.current_allocated_memory(),
                                driver_snapshot=torch.mps.driver_allocated_memory(),
                                recommended_max=torch.mps.recommended_max_memory(), isolated_peak=False)
    elif device.type == 'cuda':
        result['memory'] = dict(peak_allocated=torch.cuda.max_memory_allocated(device),
                                peak_reserved=torch.cuda.max_memory_reserved(device))
    del optimizer, model, state
    return result


def kernel_case(record, args, device, folder, ui):
    torch.manual_seed(args.seed)
    n, length, d = record['nodes'], record['sequence'], 8
    shape = (args.batch, n, 3)
    initial = (.02*torch.randn(*shape, d, d)).to(device).requires_grad_()
    key = F.normalize(torch.randn(*shape, length, d), dim=-1).to(device).requires_grad_()
    value = (.1*torch.randn(*shape, length, d)).to(device).requires_grad_()
    alpha = torch.full((*shape, length), .98, device=device, requires_grad=True)
    beta = torch.full((*shape, length), .1, device=device, requires_grad=True)
    inputs = (initial, key, value, alpha, beta)
    method = dict(sequential=sequential_delta, parallel=parallel_delta, triangular=triangular_delta)[record['method']]
    trials = []; warmups = []; ui.stage(record['id'], args.warmup+args.repeats)
    for step in range(args.warmup+args.repeats):
        for item in inputs:
            item.grad = None
        sync(device); begin = time.perf_counter()
        states, final = method(*inputs)
        loss = states.square().mean()+final.square().mean()
        loss.backward(); sync(device); elapsed = time.perf_counter()-begin
        finite = bool(loss.detach().isfinite().cpu()) and all(bool(x.grad.isfinite().all().cpu()) for x in inputs)
        if not finite:
            raise FloatingPointError('Nonfinite recurrence output/gradient')
        row = dict(step=step, seconds=elapsed, loss=float(loss.detach().cpu()), finite=finite,
                   phase='warmup' if step < args.warmup else 'timed')
        append(folder/'raw.jsonl', row)
        (warmups if step < args.warmup else trials).append(row)
        ui.update(ms=f'{1000*elapsed:.1f}')
    return dict(status='completed', matrix_size=d, bands=3, batch=args.batch, trials=trials,
                warmup=warmups, median_seconds=statistics.median(x['seconds'] for x in trials),
                scope='recurrence forward+backward only; synthetic algebra inputs, not a language-model result')


def report(root, records):
    complete = [r for r in records if r['status'] == 'completed']
    lines = ['# v0.8 本机工程实测', '',
             '真实设备计算；无远程任务。本报告检验正确性和吞吐，不评价长期语言能力。', '',
             '| 条件 | 状态 | 中位耗时 ms | 合计有效目标/秒 | 参数 | 状态字节 |',
             '|---|---|---:|---:|---:|---:|']
    for r in records:
        result = r.get('result', {})
        millis = f'{1000*result["median_seconds"]:.2f}' if 'median_seconds' in result else '—'
        speed = f'{result["tokens_per_second"]:.1f}' if 'tokens_per_second' in result else '—'
        lines.append(f'| {r["id"]} | {r["status"]} | {millis} | {speed} | {result.get("parameters", "—")} | {result.get("state_bytes", "—")} |')
    lines += ['', '## 可比较范围', '',
              'v0.8 同 C 的 sequential/parallel/triangular 是同一动力学的计算实现对照。',
              'C16 与 C32 的反馈时序不同；v0.7 与 v0.8 也属于不同模型，吞吐比不能充当保函数提速或质量结论。',
              '各模型使用相同数据、batch、序列长度、FP32、优化器和完整训练更新计时。参数及状态容量照实报告，未声称状态相等。',
              '预热独立保存；phase_seconds 是额外同步的单独剖析，不混入主计时。MPS 内存是快照，不是独占峰值。',
              '设备驻留输入的计时包括前向、CE、反向、裁剪、AdamW、状态 detach/信用失效，不含下载、数据采样、保存和进度显示。',
              'v0.7 原有诊断保留，故另提供纯递推核对照。未关闭旧诊断的整网比不能精确归因于架构本身。', '',
              '本次 v0.7 对照使用 H32 信用槽，与旧正式 H8 不同；每个条件的完整配置见 result.json。',
              '尚未实现保持同一块边界的逐 token 生成 API；不足 C 的 forward 会提前提交边界。',
              '三个计时样本是探索实测，固定执行顺序可能受 Mac 热状态影响。', '',
              f'完成 {len(complete)}/{len(records)}；未执行、失败、中止见 registry.json。', '']
    (root/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def execute(args):
    if args.repeats < 3 or args.warmup < 1 or args.batch < 1 or args.threads < 1:
        raise ValueError('Need repeats>=3, warmup>=1 and positive batch/threads')
    if args.sequence_tokens < 64 or args.sequence_tokens % 32:
        raise ValueError('sequence-tokens must be a multiple of 32 and >=64 (at least two C32 blocks)')
    root = args.output or PROJECT/'results/v08'/datetime.now().strftime('benchmark_%Y%m%dT%H%M%S')
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Refusing to overwrite results: {root}')
    root.mkdir(parents=True, exist_ok=True)
    records = plan(args.nodes, args.sequence_tokens)
    write(root/'registry.json', records)
    original_sources = sources()
    write(root/'manifest.json', dict(status='running', created_utc=datetime.now(timezone.utc).isoformat(),
          args={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
          platform=platform.platform(), torch=str(torch.__version__), source_hashes=original_sources,
          formal_training=False, timing_dtype='float32', concurrency=1))
    for name in original_sources:
        target = root/'source'/name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((PROJECT/name).read_bytes())
    ui = Progress(len(records), enabled=not args.quiet)
    current = None
    try:
        with RunLock(PROJECT/'results/v08/.benchmark.lock'):
            ui.stage('数学、梯度、因果与状态校验', 1)
            verification = verify(root); ui.update()
            # Unit tests explicitly use one CPU thread; restore requested benchmark setting.
            torch.set_num_threads(args.threads)
            device = device_for(args.device)
            if device.type == 'mps' and __import__('os').environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
                raise RuntimeError('MPS fallback is enabled; disable it for a native GPU timing')
            prepare(args.data_root); corpus = Corpus(args.data_root)
            ui.stage('原生设备 FP32 硬门一致性', 1)
            native_equivalence(root, corpus, device, args.seed); ui.update()
            write(root/'environment.json', dict(device=str(device), threads=torch.get_num_threads(),
                  data_manifest_sha256=digest(corpus.manifest), data_manifest=corpus.manifest,
                  verification=verification, mps_fallback=False))
            for current in records:
                if args.scope != 'all' and current['kind'] != args.scope:
                    current['reason'] = f'excluded by --scope {args.scope}'
                    continue
                current['status'] = 'running'; write(root/'registry.json', records)
                folder = root/current['id']; folder.mkdir()
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                try:
                    current['result'] = (kernel_case(current, args, device, folder, ui)
                                         if current['kind'] == 'kernel' else model_case(current, args, corpus, device, folder, ui))
                    current['status'] = 'completed'
                except Exception:
                    current['status'] = 'failed'; current['error'] = traceback.format_exc()
                    (folder/'error.txt').write_text(current['error'], encoding='utf-8')
                write(folder/'result.json', current); write(root/'registry.json', records)
                report(root, records); ui.total.update()
                gc.collect()
                if device.type == 'mps':
                    torch.mps.empty_cache()
                elif device.type == 'cuda':
                    torch.cuda.empty_cache()
            if sources() != original_sources:
                raise RuntimeError('Source changed during experiment; do not pool measurements')
            status = 'failed' if any(r['status'] == 'failed' for r in records) else 'completed'
            write(root/'summary.json', dict(status=status, conditions=records, formal_training=False))
            return root, status
    except KeyboardInterrupt:
        if current is not None and current['status'] == 'running':
            current['status'] = 'aborted'
        write(root/'registry.json', records); report(root, records)
        write(root/'summary.json', dict(status='aborted', conditions=records, formal_training=False))
        raise
    except Exception:
        (root/'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        write(root/'summary.json', dict(status='failed', conditions=records, formal_training=False))
        raise
    finally:
        ui.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('auto','cpu','mps','cuda'), default='auto')
    parser.add_argument('--nodes', type=int, nargs='+', default=[32,128,512])
    parser.add_argument('--sequence-tokens', type=int, default=64)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--warmup', type=int, default=6)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--data-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--scope', choices=('all','kernel','model'), default='all')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    root, status = execute(args)
    print(f'v0.8 {status}: {root}/REPORT.md')
    if status != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
