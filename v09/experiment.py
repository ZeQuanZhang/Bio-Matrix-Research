"""Local MPS comparison queue; persist every job and resume only frozen protocols."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch
from tqdm import tqdm

from v07.common import PROJECT, RunLock, digest, file_hash, write
from v07.data import ROOT
from .benchmark import frozen_corpus, sources


FORMAT = 'v09-local-comparison-1'
CONDITIONS = ('v08_seq', 'content_only', 'v09_dense', 'v09_sparse50', 'transformer')
DEFAULT_CONDITIONS = tuple(name for name in CONDITIONS if name != 'v08_seq')
NODES = (32, 128, 512)
SEEDS = (11, 23, 37)


def now():
    return datetime.now(timezone.utc).isoformat()


def plan(conditions=DEFAULT_CONDITIONS, nodes=NODES, seeds=SEEDS, tokens=12000000):
    jobs, cells, seen = [], [], set()
    for seed in seeds:
        for n in nodes:
            for condition in conditions:
                canonical = 32 if condition == 'content_only' else n
                identity = f'{condition}_n{canonical}_s{seed}'
                cells.append(dict(id=f'{condition}_n{n}_s{seed}', condition=condition,
                                  nodes=n, seed=seed, job_id=identity,
                                  reference=(condition == 'content_only' and n != canonical)))
                if identity not in seen:
                    jobs.append(dict(id=identity, condition=condition, nodes=canonical, display_nodes=n, seed=seed,
                                     tokens=tokens, status='not_executed', training_status='not_executed',
                                     evaluation_status='not_executed', completed_tokens=0, attempts=[]))
                    seen.add(identity)
    return dict(jobs=jobs, cells=cells)


def source_manifest():
    extra = (PROJECT/'run_biomatrix_v09_compare.py',
             PROJECT/'.run/Bio-Matrix v0.9 对比实验.run.xml',
             PROJECT/'.run/Bio-Matrix v0.9 训练.run.xml')
    return sources() | {str(path.relative_to(PROJECT)): file_hash(path) for path in extra if path.is_file()}


def data_manifest(data_root):
    corpus = frozen_corpus(data_root)
    return dict(root=str(Path(data_root).resolve()), manifest_sha256=digest(corpus.manifest),
                tokenizer_sha256=corpus.manifest['prepared_hashes']['tokenizer.json'],
                raw=corpus.manifest['raw'], prepared_hashes=corpus.manifest['prepared_hashes'])


def data_manifest_hash(data_root):
    value = read_json(Path(data_root)/'manifest.json')
    if not value:
        raise ValueError('Frozen data manifest is missing or unreadable')
    return digest(value)


def check_frozen(protocol):
    current = source_manifest()
    expected = protocol['source_hashes']
    if current != expected:
        changed = sorted(name for name in set(current) | set(expected) if current.get(name) != expected.get(name))
        raise RuntimeError('Comparison source changed during execution: '+', '.join(changed)+
                           '; results preserved; use a new RUN_DIRECTORY')
    if data_manifest_hash(protocol['config']['data_root']) != protocol['data']['manifest_sha256']:
        raise RuntimeError('Frozen data manifest changed during comparison; results preserved')


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def tail_jsonl(path):
    """Skip a partial append; never reread a growing training log every second."""
    try:
        with Path(path).open('rb') as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size-262144))
            lines = handle.read().decode('utf-8', errors='replace').splitlines()
        for line in reversed(lines):
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    return value
            except ValueError:
                continue
    except OSError:
        pass
    return {}


def progress(job_root):
    job_root = Path(job_root)
    status = read_json(job_root/'status.json')
    metric = tail_jsonl(job_root/'metrics.jsonl')
    validation = tail_jsonl(job_root/'validation.jsonl')
    live = read_json(job_root/'live.json')
    result = status | metric | live
    result['completed_tokens'] = max(int(row.get('completed_tokens', 0)) for row in (status, metric, live))
    result['last_safe_tokens'] = int(status.get('completed_tokens', 0))
    result['validation'] = live.get('validation', validation)
    result['evaluation'] = read_json(job_root/'evaluation/live.json')
    return result


def initialize(root, protocol, layout):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root/'manifest.json'
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if previous.get('protocol') != protocol:
            changed = [key for key in protocol if previous.get('protocol', {}).get(key) != protocol[key]]
            raise ValueError('Existing comparison protocol/source/data changed: '+', '.join(changed)+
                             '; use a new RUN_DIRECTORY. Retry/max-jobs belong in CLI arguments.')
        registry = read_json(root/'registry.json')
        if not registry or [job['id'] for job in registry.get('jobs', [])] != [job['id'] for job in layout['jobs']]:
            raise ValueError('Missing or inconsistent comparison registry; preserve this directory and choose a new one')
        return registry
    if any(path.name != '.queue.lock' for path in root.iterdir()):
        raise FileExistsError(f'Refusing to initialize a nonempty comparison directory: {root}')
    registry = dict(format=FORMAT, created_at=now(), updated_at=now(), status='not_executed', **layout)
    write(manifest_path, dict(format=FORMAT, created_at=now(), protocol=protocol,
                             cells=len(layout['cells']), independent_jobs=len(layout['jobs'])))
    for name in protocol['source_hashes']:
        target = root/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((PROJECT/name).read_bytes())
    persist(root, registry)
    return registry


def persist(root, registry):
    root = Path(root); registry['updated_at'] = now()
    lookup = {job['id']: job for job in registry['jobs']}
    rows = [cell | dict(status=lookup[cell['job_id']]['status'],
                        completed_tokens=lookup[cell['job_id']].get('completed_tokens', 0))
            for cell in registry['cells']]
    completed = sum(job['status'] == 'completed' for job in registry['jobs'])
    summary = dict(status=registry['status'], updated_at=registry['updated_at'],
                   completed_jobs=completed, independent_jobs=len(lookup),
                   completed_cells=sum(row['status'] == 'completed' for row in rows), cells=rows,
                   completed_tokens=sum(min(job['tokens'], job.get('completed_tokens', 0)) for job in lookup.values()),
                   target_tokens=sum(job['tokens'] for job in lookup.values()),
                   jobs=registry['jobs'])
    write(root/'registry.json', registry); write(root/'summary.json', summary)
    lines = ['# Bio-Matrix v0.9 本机对比实验', '',
             f'状态：{registry["status"]}；独立运行 {completed}/{len(lookup)}；'
             f'配对格 {summary["completed_cells"]}/{len(rows)}。', '',
             f'独立训练有效目标：{summary["completed_tokens"]:,}/{summary["target_tokens"]:,}。', '',
             'content_only 每种子只训练一次；N128/N512 的对应格引用 N32，不算独立重复。', '',
             '| seed | N | 模型 | 状态 | 有效目标 | 训练 | 评估 | 结果目录 |',
             '|---|---|---|---|---:|---|---|---|']
    for cell in rows:
        job = lookup[cell['job_id']]
        name = cell['condition']+('（引用 N32）' if cell['reference'] else '')
        lines.append(f'| {cell["seed"]} | {cell["nodes"]} | {name} | {job["status"]} | '
                     f'{job.get("completed_tokens", 0):,} | {job["training_status"]} | '
                     f'{job["evaluation_status"]} | [{job["id"]}]({job["id"]}/) |')
    lines += ['', '全部训练/评估日志、检查点、失败和中止记录保留。失败不自动重试；使用 --retry-failed 显式重试。',
              '训练与评估均为本机 MPS 顺序执行；本入口不连接远端。',
              'WikiText-2 结果属于已使用语料的探索性比较；此队列不读取 test。', '']
    (root/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def command(job, root, config, stage):
    python = str(PROJECT/'.venv/bin/python')
    folder = Path(root)/job['id']
    if stage == 'evaluation':
        return [python, '-m', 'v09.evaluate', '--checkpoint', str(folder/'last.pt'),
                '--output', str(folder/'evaluation'), '--device', 'mps',
                '--windows', str(config['evaluation_windows'])]
    base = [python, '-m', 'v09.train', '--device', 'mps', '--threads', '4', '--quiet']
    if (folder/'last.pt').is_file():
        # The child restores its immutable budget and schedule from this checkpoint.
        return base+['--resume', str(folder/'last.pt')]
    return base+['--context-limit', '4096', '--validation-windows', '128', '--eval-every', '500000',
                 '--condition', job['condition'], '--nodes', str(job['nodes']), '--tokens', str(job['tokens']),
                 '--seed', str(job['seed']), '--data-root', config['data_root'], '--output', str(folder)]


class StopRequest:
    def __init__(self):
        self.requested = False; self.previous = {}

    def __enter__(self):
        for name in (signal.SIGINT, signal.SIGTERM):
            self.previous[name] = signal.signal(name, self._request)
        return self

    def _request(self, *_):
        self.requested = True

    def __exit__(self, *_):
        for name, previous in self.previous.items():
            signal.signal(name, previous)


class Display:
    """Stable lines for totals, selected model families, phase and resources."""
    def __init__(self, registry, enabled=True):
        import shutil
        width = max(100, shutil.get_terminal_size((120, 30)).columns)
        options = dict(ncols=width, mininterval=1., disable=not enabled, leave=True)
        # Child metrics own the speed estimate; queued/resumed tokens have no new elapsed time.
        meter = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {postfix}'
        self.total = tqdm(total=sum(job['tokens'] for job in registry['jobs']), desc='总有效目标', position=0, bar_format=meter, **options)
        self.cells = tqdm(total=len(registry['cells']), desc='配对格', position=1, bar_format=meter, **options)
        conditions = tuple(dict.fromkeys(cell['condition'] for cell in registry['cells']))
        self.details = {name: tqdm(total=1, desc=name, position=i+2, bar_format=meter, **options) for i, name in enumerate(conditions)}
        self.stage = tqdm(total=0, desc='准备', position=len(conditions)+2, **options)
        self.resources = tqdm(total=0, desc='本机资源', position=len(conditions)+3, **options)

    def refresh(self, registry, job=None, snapshot=None, pid=None):
        lookup = {row['id']: row for row in registry['jobs']}
        if job is not None:
            self._last_job = job
        else:
            job = getattr(self, '_last_job', None)
        if snapshot is not None:
            self._last_snapshot = snapshot
        snapshot = snapshot if snapshot is not None else getattr(self, '_last_snapshot', {})
        self.total.n = sum(min(row['tokens'], row.get('completed_tokens', 0)) for row in lookup.values())
        completed_jobs = sum(row['status'] == 'completed' for row in lookup.values())
        self.total.set_postfix_str(f'独立运行 {completed_jobs}/{len(lookup)}', refresh=False)
        self.cells.n = sum(lookup[cell['job_id']]['status'] == 'completed' for cell in registry['cells'])
        if job is not None:
            seed, n = job['seed'], job.get('display_nodes', job['nodes'])
            for name, bar in self.details.items():
                cell = next((cell for cell in registry['cells'] if cell['seed'] == seed and
                             cell['nodes'] == n and cell['condition'] == name), None)
                if cell is None:
                    bar.n = 0; bar.total = 1
                    bar.set_postfix_str('本次未选', refresh=False)
                else:
                    row = lookup[cell['job_id']]; bar.total = row['tokens']; bar.n = row.get('completed_tokens', 0)
                    suffix = ' · 引用 N32' if cell['reference'] else ''
                    bar.set_description_str(f's{seed} N{n} {name}{suffix}', refresh=False)
                    text = row['status']
                    if row['id'] == job['id']:
                        loss = snapshot.get('loss'); speed = snapshot.get('tokens_per_second')
                        if loss is not None: text += f' loss={loss:.4f}'
                        if speed is not None: text += f' 目标/s={speed:.1f}'
                    bar.set_postfix_str(text, refresh=False)
            phase = snapshot.get('phase', job.get('stage', '准备'))
            evaluation = snapshot.get('evaluation') or {}
            validation = snapshot.get('validation') or {}
            detail = evaluation if job.get('stage') == 'evaluation' else validation
            if job.get('stage') == 'evaluation': phase = evaluation.get('phase', '同龄历史评估')
            done = detail.get('completed', detail.get('windows_completed'))
            count = detail.get('total', detail.get('windows_total'))
            if done is not None and count is not None:
                phase += f' {done}/{count}'
            nll = detail.get('nll')
            if nll is not None: phase += f' NLL={nll:.4f}'
            self.stage.set_description_str(f'{job["id"]} · {phase}', refresh=False)
            resource = evaluation if job.get('stage') == 'evaluation' else snapshot
            self.resources.set_description_str(resource_line(pid, resource), refresh=False)
        for bar in (self.total, self.cells, *self.details.values(), self.stage, self.resources):
            bar.refresh()

    def close(self):
        for bar in (self.resources, self.stage, *reversed(tuple(self.details.values())), self.cells, self.total):
            bar.close()


def resource_line(pid, snapshot):
    parts = ['本机 MPS']
    if pid:
        try:
            result = subprocess.run(['ps', '-p', str(pid), '-o', 'pcpu=,rss='],
                                    capture_output=True, text=True, timeout=2, check=False)
            cpu, rss = result.stdout.split()[:2]
            parts += [f'子进程 CPU {float(cpu):.0f}%', f'RSS {float(rss)/1048576:.2f} GiB']
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    memory = snapshot.get('resource') or snapshot.get('resources') or {}
    for name, label in (('mps_allocated_bytes', 'MPS已分配'), ('mps_driver_bytes', '驱动')):
        if memory.get(name) is not None:
            parts.append(f'{label} {memory[name]/2**30:.2f} GiB')
    return ' | '.join(parts)


def archive(path, destination):
    path, destination = Path(path), Path(destination)
    if not path.exists(): return None
    destination.mkdir(parents=True, exist_ok=True)
    target = destination/(path.name+'_'+datetime.now().strftime('%Y%m%dT%H%M%S')+f'_{time.time_ns()%1000000000:09d}')
    path.rename(target)
    return str(target)


def run_child(root, registry, job, config, stage, ui, stop, popen=None, sleep=time.sleep):
    """Keep output on disk and send one soft interrupt before waiting for save."""
    root = Path(root); folder = root/job['id']; popen = popen or subprocess.Popen
    filename = 'console.log' if stage == 'training' else 'console_eval.log'
    staging = root/'.console'; staging.mkdir(exist_ok=True)
    log_path = folder/filename if folder.is_dir() else staging/(job['id']+'.log')
    arguments = command(job, root, config, stage)
    job.update(status='running', stage=stage, started_at=now(), command=arguments)
    job[stage+'_status'] = 'running'; persist(root, registry)
    sent = False
    with log_path.open('ab') as output:
        try:
            child = popen(arguments, cwd=str(PROJECT), stdout=output, stderr=subprocess.STDOUT)
        except OSError as error:
            job.update(status='failed', failure_stage=stage, error=repr(error), finished_at=now())
            job[stage+'_status'] = 'failed'
            output.write((repr(error)+'\n').encode('utf-8'))
            if log_path.parent == staging:
                folder.mkdir(parents=True, exist_ok=True); log_path.rename(folder/filename)
            persist(root, registry)
            return 'failed'
        job['pid'] = child.pid; persist(root, registry)
        while True:
            if log_path.parent == staging and folder.is_dir():
                target = folder/filename
                if target.exists():
                    target = folder/f'console_attempt_{time.time_ns()}.log'
                log_path.rename(target); log_path = target
            snapshot = progress(folder)
            job['completed_tokens'] = min(job['tokens'], snapshot['completed_tokens'])
            job['last_safe_tokens'] = snapshot['last_safe_tokens']
            if snapshot.get('validation'): job['last_validation'] = snapshot['validation']
            ui.refresh(registry, job, snapshot, child.pid)
            persist(root, registry)
            code = child.poll()
            if code is not None:
                break
            if stop.requested and not sent:
                child.send_signal(signal.SIGINT); sent = True
                job['stop_requested_at'] = now(); persist(root, registry)
            sleep(1.)
    if log_path.parent == staging:
        folder.mkdir(parents=True, exist_ok=True)
        log_path.rename(folder/filename)
    status_root = folder if stage == 'training' else folder/'evaluation'
    final = read_json(status_root/'status.json') or read_json(status_root/'result.json') or read_json(status_root/'summary.json')
    state = final.get('status')
    if state == 'completed' and code == 0:
        outcome = 'completed'
    elif state == 'aborted' or stop.requested or code in (-signal.SIGINT, -signal.SIGTERM, -signal.SIGKILL):
        outcome = 'aborted'
    else:
        outcome = 'failed'
    job[stage+'_status'] = outcome
    overall = 'running' if stage == 'training' and outcome == 'completed' and config['evaluate_on_complete'] else outcome
    job.update(returncode=code, finished_at=now(), status=overall)
    if stage == 'training':
        job['completed_tokens'] = min(job['tokens'], int(final.get('completed_tokens', job.get('last_safe_tokens', 0))))
        if outcome == 'completed': job['completed_tokens'] = job['tokens']
    if outcome == 'failed': job['failure_stage'] = stage
    job.pop('pid', None); persist(root, registry)
    return outcome


def reconcile(root, job, evaluation_required=True):
    folder = Path(root)/job['id']
    training = read_json(folder/'status.json')
    evaluation = read_json(folder/'evaluation/status.json') or read_json(folder/'evaluation/result.json') or read_json(folder/'evaluation/summary.json')
    for stage, record in (('training', training), ('evaluation', evaluation)):
        if record.get('status') in ('completed', 'failed', 'aborted'):
            job[stage+'_status'] = record['status']
    if job['training_status'] == 'completed': job['completed_tokens'] = job['tokens']
    if job['training_status'] == 'failed' or evaluation_required and job['evaluation_status'] == 'failed':
        job['status'] = 'failed'
    elif job['training_status'] == 'completed' and (not evaluation_required or job['evaluation_status'] == 'completed'):
        job['status'] = 'completed'
    elif job['status'] in ('running', 'completed'):
        job['status'] = 'aborted'


def execute(args, popen=None, sleep=time.sleep):
    if (args.tokens < 1 or args.max_jobs is not None and args.max_jobs < 1 or args.evaluation_windows < 1
            or not args.nodes or not args.seeds or not args.conditions
            or len(set(args.nodes)) != len(args.nodes) or len(set(args.seeds)) != len(args.seeds)
            or len(set(args.conditions)) != len(args.conditions) or any(n not in NODES for n in args.nodes)
            or any(c not in CONDITIONS for c in args.conditions) or any(s < 0 for s in args.seeds)):
        raise ValueError('Invalid comparison selection, budget or job limit')
    root = Path(args.output).resolve()
    config = dict(tokens=args.tokens, nodes=list(args.nodes), seeds=list(args.seeds), conditions=list(args.conditions),
                  evaluate_on_complete=not args.skip_evaluation, evaluation_windows=args.evaluation_windows,
                  data_root=str(Path(args.data_root).resolve()), device='mps', threads=4,
                  context_limit=4096, validation_windows=128, eval_every=500000)
    protocol = dict(config=config, source_hashes=source_manifest(), data=data_manifest(args.data_root),
                    python=str(PROJECT/'.venv/bin/python'), python_version=sys.version,
                    torch_version=str(torch.__version__))
    layout = plan(args.conditions, args.nodes, args.seeds, args.tokens)
    root.mkdir(parents=True, exist_ok=True)
    with RunLock(root/'.queue.lock'):
        registry = initialize(root, protocol, layout)
        if args.plan_only:
            persist(root, registry)
            return root, registry
        if not torch.backends.mps.is_available():
            raise RuntimeError('本机 MPS 不可用；本队列不回退 CPU/CUDA，也不连接远端')
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
            raise RuntimeError('请关闭 PYTORCH_ENABLE_MPS_FALLBACK 后运行本机原生 MPS 对比')
        registry['status'] = 'running'; persist(root, registry)
        ui = None
        launched = 0
        try:
            ui = Display(registry, not args.quiet)
            with StopRequest() as stop:
                def guarded_child(job, stage):
                    check_frozen(protocol)
                    outcome = run_child(root, registry, job, config, stage, ui, stop, popen, sleep)
                    check_frozen(protocol)
                    return outcome

                for job in registry['jobs']:
                    reconcile(root, job, config['evaluate_on_complete'])
                    if job['status'] == 'completed': continue
                    if job['status'] == 'failed' and not args.retry_failed: continue
                    if stop.requested or args.max_jobs is not None and launched >= args.max_jobs: break
                    folder = root/job['id']
                    if (folder/'.training.lock').exists():
                        with RunLock(folder/'.training.lock', record_owner=False):
                            pass
                    launched += 1
                    if job['training_status'] != 'completed':
                        if folder.exists() and not (folder/'last.pt').is_file():
                            previous = archive(folder, root/'attempts')
                            job['attempts'].append(dict(stage='training', path=previous, archived_at=now()))
                            job['completed_tokens'] = 0
                        outcome = guarded_child(job, 'training')
                        if outcome != 'completed':
                            if stop.requested: break
                            continue
                    if config['evaluate_on_complete'] and job['evaluation_status'] != 'completed':
                        if stop.requested:
                            job['status'] = 'aborted'; break
                        if (folder/'evaluation').exists():
                            previous = archive(folder/'evaluation', folder/'evaluation_attempts')
                            job['attempts'].append(dict(stage='evaluation', path=previous, archived_at=now()))
                        outcome = guarded_child(job, 'evaluation')
                        if outcome != 'completed':
                            if stop.requested: break
                            continue
                    if not config['evaluate_on_complete']: job['evaluation_status'] = 'not_executed'
                    job['status'] = 'completed'; persist(root, registry)
                registry['status'] = ('completed' if all(job['status'] == 'completed' for job in registry['jobs'])
                                      else 'aborted' if stop.requested else 'incomplete')
                if args.max_jobs is not None: registry['last_launch_job_limit'] = args.max_jobs
        except BaseException as error:
            registry.update(status='failed', error=repr(error), failed_at=now())
            raise
        finally:
            persist(root, registry)
            if ui is not None:
                try:
                    ui.refresh(registry)
                finally:
                    ui.close()
    return root, registry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT/'results/v09/comparison_no_v08')
    parser.add_argument('--data-root', type=Path, default=ROOT)
    parser.add_argument('--tokens', type=int, default=12000000)
    parser.add_argument('--nodes', nargs='+', type=int, default=list(NODES))
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=list(DEFAULT_CONDITIONS))
    parser.add_argument('--evaluation-windows', type=int, default=64)
    parser.add_argument('--skip-evaluation', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--max-jobs', type=int)
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    root, registry = execute(parser.parse_args(argv))
    print(f'v0.9 对比队列 {registry["status"]}: {root}；'
          f'{len(registry["cells"])} 配对格 / {len(registry["jobs"])} 独立运行')


if __name__ == '__main__':
    main()
