"""Single-GPU independent-job concurrency probe; never starts the formal queue.

Each fresh Python worker owns a full model/AdamW/stream/state, finishes warmup,
then waits for the parent's file barrier. Run on an otherwise available GPU.
Use --self-test for a short CPU-only mock of process/barrier/failure accounting.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def wait_go(path, timeout):
    started = time.perf_counter()
    while not path.exists():
        if time.perf_counter()-started > timeout:
            raise TimeoutError('Timed out waiting for synchronized go barrier')
        time.sleep(.02)
    return read_json(path)


def worker(spec_path):
    spec = read_json(spec_path); folder = Path(spec['folder']); result = dict(status='failed', worker=spec['index'], seed=spec['seed'])
    result['benchmark_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['compile_step'] = spec.get('compile_step',False)
    try:
        if spec['mock']:
            if spec.get('inject_failure'):
                raise RuntimeError('Intentional CPU mock failure before ready')
            time.sleep(.005)
            ready = dict(status='ready', warmup_updates=spec['warmup_updates'], mock=True)
            write_json(folder/'ready.json', ready)
            go = wait_go(Path(spec['go']), spec['timeout'])
            started = time.perf_counter(); updates = []
            for _ in range(spec['probe_updates']):
                time.sleep(.01+.001*spec['index'])
                updates.append(dict(valid_targets=256, loss=1., gradient_norm=.5, finite=True))
            ended = time.perf_counter(); valid = sum(row['valid_targets'] for row in updates)
            result.update(status='completed', mock=True, model_constructed=False, updates=updates,
                          valid_targets=valid, warmup_targets=256*spec['warmup_updates'],
                          timed_start=started, timed_end=ended, timed_seconds=ended-started,
                          tokens_per_second=valid/(ended-started), go_perf_counter=go['go_perf_counter'],
                          cuda_peak_allocated_bytes=None, cuda_peak_reserved_bytes=None,
                          finite=True, source_hashes={'mock':'no-model-or-GPU'})
        else:
            sys.path.insert(0, str(PROJECT))
            import torch
            from dataclasses import asdict
            from v07.common import digest, source_hashes, state_bytes
            from v07.data import Corpus, prepare
            from v07.model import initialize, parameter_count
            from v07.run import build, load_config, make_training_stream, model_config, train_update, validate_config

            torch.set_num_threads(1); torch.set_num_interop_threads(1)
            device = torch.device(spec['device'])
            if device.type != 'cuda' or not torch.cuda.is_available():
                raise RuntimeError('CUDA probe requires an available CUDA GPU; no CPU fallback')
            torch.cuda.set_device(device)
            cfg = load_config(spec['config']) | dict(device='cuda', cpu_threads=1, batch_size=2,
                     sequence_tokens=128, bptt_tokens=128, nodes=[spec['nodes']], variants=['full'], seeds=[spec['seed']])
            validate_config(cfg)
            source = source_hashes()
            if not (Path(spec['data_root'])/'manifest.json').is_file():
                raise FileNotFoundError('Benchmark requires an already prepared frozen dataset; no download or tokenizer fit')
            prepare(Path(spec['data_root']), cfg['vocab_size'])
            corpus = Corpus(Path(spec['data_root']))
            cfg['vocab_size'] = corpus.manifest['vocab_size']
            case = dict(stage='benchmark', nodes=spec['nodes'], variant='full', seed=spec['seed'])
            mc = model_config(cfg, case)
            torch.manual_seed(spec['seed']); model = build(mc); initialize(model, spec['seed']); model.to(device)
            acceleration = None
            adapter_path = PROJECT/'scripts/v07_cuda_acceleration.py'
            if spec.get('compile_step'):
                from scripts.v07_cuda_acceleration import attach_compiled_step
                acceleration = attach_compiled_step(model, device=device, mode='default', fullgraph=True,
                                                   dynamic=False, record_path=folder/'acceleration.json')
                result['adapter_script_sha256'] = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
                from torch._dynamo.utils import counters
                def unique_graphs():
                    return int(counters.get('stats',{}).get('unique_graphs',0))
            else:
                def unique_graphs():
                    return 0
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
            stream = make_training_stream(corpus, cfg, spec['seed']); state = None; warmups = []
            warmup_limit = spec['warmup_updates']+(3 if spec.get('compile_step') else 0)
            stable = False
            for step in range(warmup_limit):
                values, mask, reset, sample = stream.next()
                graphs_before = unique_graphs()
                torch.cuda.synchronize(device); update_start = time.perf_counter()
                metrics, state = train_update(model, optimizer, values, cfg, device, step,
                                               state=state, reset_mask=reset, loss_mask=mask)
                torch.cuda.synchronize(device)
                graphs_after = unique_graphs()
                warmups.append(dict(seconds=time.perf_counter()-update_start, metrics=metrics,
                                    content_sha256=sample['content_sha256'],
                                    new_unique_graphs=graphs_after-graphs_before))
                stable = graphs_after == graphs_before
                if len(warmups) >= spec['warmup_updates'] and (not spec.get('compile_step') or stable):
                    break
            if spec.get('compile_step') and (not stable or not any(row['new_unique_graphs']>0 for row in warmups)):
                raise RuntimeError('Compiled step did not demonstrate a stable complete warmup update before ready')
            ready = dict(status='ready', config=cfg, config_sha256=digest(cfg), source_hashes=source,
                         data_manifest_sha256=digest(corpus.manifest), model=asdict(mc),
                         parameters=parameter_count(model), warmup=warmups, device=str(device),
                         cuda_device=torch.cuda.get_device_name(device), torch=str(torch.__version__), dtype='float32',
                         cpu_threads=torch.get_num_threads(), tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                         tf32_cudnn=torch.backends.cudnn.allow_tf32,
                         compile_step=spec.get('compile_step',False), acceleration=acceleration,
                         adapter_script_sha256=result.get('adapter_script_sha256'),
                         warmup_updates_completed=len(warmups), unique_graphs_at_ready=unique_graphs())
            write_json(folder/'ready.json', ready)
            go = wait_go(Path(spec['go']), spec['timeout'])
            torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter(); updates = []
            result['updates'] = updates
            for probe_step in range(spec['probe_updates']):
                update_start = time.perf_counter(); values, mask, reset, sample = stream.next()
                graphs_before = unique_graphs()
                metrics, state = train_update(model, optimizer, values, cfg, device,
                        len(warmups)+probe_step, state=state, reset_mask=reset, loss_mask=mask)
                torch.cuda.synchronize(device)
                finite = math.isfinite(metrics['loss']) and math.isfinite(metrics['gradient_norm'])
                if not finite:
                    raise FloatingPointError('Nonfinite probe loss/gradient norm')
                updates.append(dict(seconds=time.perf_counter()-update_start, **metrics,
                                    finite=finite, content_sha256=sample['content_sha256'],
                                    new_unique_graphs=unique_graphs()-graphs_before))
                if spec.get('compile_step') and updates[-1]['new_unique_graphs']:
                    raise RuntimeError('A graph compiled after the ready barrier; timed throughput is invalid')
            torch.cuda.synchronize(device); ended = time.perf_counter()
            finite = all(bool(parameter.detach().isfinite().all()) for parameter in model.parameters())
            if not finite:
                raise FloatingPointError('Nonfinite final model parameter')
            after = source_hashes()
            if after != source:
                raise RuntimeError('Source changed during probe; results cannot be combined')
            if spec.get('compile_step') and result['adapter_script_sha256'] != hashlib.sha256(adapter_path.read_bytes()).hexdigest():
                raise RuntimeError('CUDA adapter source changed during probe')
            valid = sum(row['valid_targets'] for row in updates)
            result.update(status='completed', mock=False, config=cfg, model=asdict(mc), updates=updates,
                          source_hashes=source, data_manifest_sha256=digest(corpus.manifest),
                          valid_targets=valid, warmup_targets=sum(row['metrics']['valid_targets'] for row in warmups),
                          warmup_updates_completed=len(warmups), acceleration=acceleration,
                          timed_new_unique_graphs=sum(row['new_unique_graphs'] for row in updates),
                          timed_start=started, timed_end=ended, timed_seconds=ended-started,
                          tokens_per_second=valid/(ended-started), go_perf_counter=go['go_perf_counter'], finite=finite,
                          runtime_state_bytes=state_bytes(state), cuda_device=torch.cuda.get_device_name(device),
                          cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                          cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                          cuda_reserved_bytes=torch.cuda.memory_reserved(device))
        write_json(folder/'worker.json', result)
        return 0
    except BaseException:
        result.update(error=traceback.format_exc())
        write_json(folder/'worker.json', result)
        return 1


def run_group(args, concurrency, folder, mock=False, inject_failure=False):
    folder.mkdir(parents=True, exist_ok=False)
    processes, logs = [], []
    launched = time.perf_counter(); go_time = None; failure = None
    try:
        for index in range(concurrency):
            worker_folder = folder/f'worker_{index}'; worker_folder.mkdir()
            spec = dict(index=index, folder=str(worker_folder), nodes=args.nodes,
                        seed=args.seed+index, config=str(args.config.resolve()), data_root=str(args.data_root.resolve()),
                        device=args.device, warmup_updates=args.warmup_updates, probe_updates=args.probe_updates,
                        timeout=args.timeout, go=str(folder/'go.json'), mock=mock,
                        inject_failure=inject_failure and index == 0,compile_step=args.compile_step)
            write_json(worker_folder/'spec.json', spec)
            log = (worker_folder/'process.log').open('w'); logs.append(log)
            env = os.environ | dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
            processes.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker-spec', str(worker_folder/'spec.json')],
                                              cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT))
        while not all((folder/f'worker_{index}'/'ready.json').exists() for index in range(concurrency)):
            if any(process.poll() is not None for process in processes):
                raise RuntimeError('Worker exited before every independent model completed warmup')
            if time.perf_counter()-launched > args.timeout:
                raise TimeoutError('Ready barrier timed out')
            time.sleep(.02)
        go_time = time.perf_counter()
        write_json(folder/'go.json', dict(go_perf_counter=go_time, ready_workers=concurrency))
        while any(process.poll() is None for process in processes):
            if any(process.poll() not in (None, 0) for process in processes):
                raise RuntimeError('Worker failed during synchronized probe')
            if time.perf_counter()-go_time > args.timeout:
                raise TimeoutError('Timed probe did not finish before timeout')
            time.sleep(.02)
        if any(process.returncode != 0 for process in processes):
            raise RuntimeError('A probe process returned nonzero')
    except BaseException:
        failure = traceback.format_exc()
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        ended = time.perf_counter()
        for log in logs:
            log.close()

    workers = []
    for index in range(concurrency):
        path = folder/f'worker_{index}'/'worker.json'
        if path.exists():
            row = read_json(path)
        else:
            row = dict(status='aborted', worker=index, reason='Parent stopped incomplete worker; inspect process.log')
            write_json(path, row)
        if index < len(processes):
            row['process_returncode'] = processes[index].returncode
        workers.append(row)
    completed = failure is None and all(row['status'] == 'completed' and row.get('finite') for row in workers)
    result = dict(concurrency=concurrency, status='completed' if completed else 'failed', mock=mock,
                  workers=workers, all_processes_exited=all(p.returncode is not None for p in processes),
                  total_wall_seconds=ended-launched, go_to_all_process_exit_seconds=ended-go_time if go_time else None,
                  note='Fresh processes; one independent full job per worker on the same GPU. Parent waits for every process to exit. Imports/warmup are included only in total_wall_seconds.')
    if failure:
        result['error'] = failure
    if completed:
        fingerprints = {json.dumps(row['source_hashes'], sort_keys=True) for row in workers}
        if len(fingerprints) != 1:
            result.update(status='failed', error='Workers used different source hashes')
        else:
            targets = sum(row['valid_targets'] for row in workers)
            compute_span = max(row['timed_end'] for row in workers)-min(row['timed_start'] for row in workers)
            result.update(valid_targets=targets, aggregate_tokens_per_second=targets/(ended-go_time),
                          simultaneous_compute_span_seconds=compute_span,
                          compute_span_tokens_per_second=targets/compute_span,
                          end_to_end_tokens_per_second=(targets+sum(row['warmup_targets'] for row in workers))/(ended-launched),
                          sum_worker_tokens_per_second=sum(row['tokens_per_second'] for row in workers))
    write_json(folder/'group.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--config', type=Path, default=PROJECT/'configs/v07/base.json')
    parser.add_argument('--data-root', type=Path, default=PROJECT/'data/v06/wikitext2')
    parser.add_argument('--nodes', type=int, default=128)
    parser.add_argument('--concurrency', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--probe-updates', type=int, default=3)
    parser.add_argument('--warmup-updates', type=int, default=1)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--compile-step', action='store_true', help='Attach the shared CUDA step adapter; finish compilation before the ready barrier')
    parser.add_argument('--timeout', type=float, default=900.)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--worker-spec', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_spec:
        return worker(args.worker_spec)
    if (min(args.concurrency+[args.nodes, args.probe_updates, args.warmup_updates]) < 1 or
            len(set(args.concurrency)) != len(args.concurrency) or args.timeout <= 0):
        parser.error('Counts/timeout must be positive and concurrency values unique')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root = args.root or PROJECT/'results/v07/cuda_concurrency_benchmark'/stamp
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        parser.error('Output root must be empty so previous probes and barrier files are preserved')
    groups = []
    for concurrency in args.concurrency:
        result = run_group(args, concurrency, root/f'concurrency_{concurrency}', mock=args.self_test)
        groups.append(result)
        print(json.dumps({key: result.get(key) for key in ('concurrency','status','valid_targets','aggregate_tokens_per_second','total_wall_seconds')}), flush=True)
    completed = [row for row in groups if row['status'] == 'completed']
    winner = max(completed, key=lambda row: row['aggregate_tokens_per_second']) if completed else None
    summary = dict(formal_training=False, cpu_mock=args.self_test, model_variant='full', nodes=args.nodes,
                   benchmark_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   compile_step=args.compile_step,
                   batch_size=2, sequence_tokens=128, bptt_tokens=128, probe_updates=args.probe_updates,
                   warmup_updates=args.warmup_updates, groups=groups,
                   selected_concurrency=winner['concurrency'] if winner and not args.self_test else None,
                   selection_metric='Sum of actual valid probe targets / parent wall time from go through all worker process exits',
                   note='Individual rates are diagnostics; their sum is not used for selection. This is a short isolated engineering probe, not a capability evaluation.')
    if args.self_test:
        assert len(completed) == len(args.concurrency)
        for row in groups:
            assert row['valid_targets'] == row['concurrency']*args.probe_updates*256
            assert row['all_processes_exited'] and row['aggregate_tokens_per_second'] > 0
            assert all(worker['timed_start'] >= worker['go_perf_counter'] for worker in row['workers'])
            assert all(worker['compile_step'] == args.compile_step for worker in row['workers'])
        failed = run_group(args, 2, root/'injected_failure', mock=True, inject_failure=True)
        assert failed['status'] == 'failed' and failed['all_processes_exited']
        assert any('Intentional CPU mock failure' in row.get('error','') for row in failed['workers'])
        summary['failure_cleanup_self_test'] = failed
        summary['self_test_passed'] = True
    write_json(root/'summary.json', summary)
    return 0 if completed else 1


if __name__ == '__main__':
    raise SystemExit(main())
