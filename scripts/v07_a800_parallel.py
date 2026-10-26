"""Run the immutable v0.7 core grid concurrently on one A800 without changing training.

Example:
  python scripts/v07_a800_parallel.py --release /opt/bio/release --workspace /opt/bio/a800 --workers 3 --prepare-only
  python scripts/v07_a800_parallel.py --release /opt/bio/release --workspace /opt/bio/a800 --workers 3

An interrupted condition needs --resume-aborted; failed conditions need --retry-failed.
The default policy keeps each seed/scale group as a barrier. Backfill changes only
dispatch order and concurrency. This program never changes batch, precision,
training steps, data schedules, model settings, or the implementation inside v07.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time


PACKAGES = ('v07', 'v03', 'v04', 'v05', 'matrix_lab')
EXPECTED = dict(nodes=[32, 128, 512], seeds=[11, 23, 37],
                variants=['message', 'independent', 'full', 'transformer', 'stateless'],
                stage='core', batch_size=2, sequence_tokens=128, bptt_tokens=128, train_steps=24000)
ACCELERATION_SETTINGS = dict(backend='inductor', mode='default', fullgraph=True, dynamic=False)
ACCELERATION_CACHE_POLICY = dict(recompile_limit=32, accumulated_recompile_limit=256)
GPU_RESERVATIONS_MIB = {32: 1536, 128: 3584, 512: 11264}


def gpu_memory():
    """Read the selected GPU without importing torch or creating a CUDA context."""
    selected = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    output = subprocess.check_output(['nvidia-smi', '-i', selected, '--query-gpu=memory.total,memory.free',
                                      '--format=csv,noheader,nounits'], text=True)
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError('Backfill requires exactly one selected GPU')
    total, free = (int(value.strip()) for value in rows[0].split(','))
    return dict(total_mib=total, free_mib=free)


def reservation(case):
    return 1024 if case['variant'] in ('transformer', 'stateless') else GPU_RESERVATIONS_MIB[case['nodes']]
WRAPPER = '''"""Scheduler lease and durable process marker; executes the unmodified runner."""
import fcntl, json, os, pathlib, runpy, sys, time, traceback
launch = json.loads(pathlib.Path(sys.argv[1]).read_text())
def write(path, value):
    path = pathlib.Path(path); temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2)); temporary.replace(path)
with pathlib.Path(launch['lease']).open('a+') as lease:
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        start = pathlib.Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
    except OSError:
        start = None
    write(launch['pid_marker'], dict(pid=os.getpid(), process_start=start, attempt=launch['attempt']))
    code = 0
    try:
        from scripts.v07_cuda_acceleration import install, write_record
        acceleration = launch['acceleration']
        write_record(launch['acceleration_record'], acceleration | dict(status='pending' if acceleration['mode']=='compiled_step' else 'eager'))
        if acceleration['mode'] == 'compiled_step':
            install(device='cuda', record_path=launch['acceleration_record'], **acceleration['settings'])
        sys.argv = ['v07.run', *launch['argv']]
        runpy.run_module('v07.run', run_name='__main__')
    except SystemExit as error:
        code = error.code if isinstance(error.code, int) else (1 if error.code else 0)
    except BaseException:
        traceback.print_exc(); code = 1
    finally:
        write(launch['finish_marker'], dict(exit_code=code, finished_at=time.time(), attempt=launch['attempt']))
    raise SystemExit(code)
'''


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


class Lock:
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open('a+')
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.close()
            raise RuntimeError(f'Another scheduler/condition holds {self.path}; do not delete its lock file') from None
        return self

    def __exit__(self, *_):
        self.handle.close()


def source_files(release):
    files = []
    for package in PACKAGES:
        folder = release/package
        if not folder.is_dir():
            raise FileNotFoundError(f'Missing release package: {folder}')
        for path in folder.rglob('*'):
            if '__pycache__' in path.parts or path.suffix in ('.pyc', '.pyo'):
                continue
            if path.is_symlink():
                raise ValueError(f'Source must be a real immutable file/directory: {path}')
            if path.is_file():
                files.append(path)
    files.append(release/'configs/v07/base.json')
    files.append(release/'scripts/v07_cuda_acceleration.py')
    for name in ('run_biomatrix_v07.py', 'requirements.txt', 'requirements.lock.txt'):
        if (release/name).is_file():
            files.append(release/name)
    return {str(path.relative_to(release)): file_hash(path) for path in sorted(files)}


def data_identity(folder):
    manifest_path = folder/'manifest.json'; manifest = read(manifest_path)
    expected = dict(manifest['prepared_hashes'])
    expected.update({f'{split}-00000-of-00001.parquet': row['sha256'] for split, row in manifest.get('raw', {}).items()})
    actual = {}
    for name, sha in expected.items():
        path = (folder/name).resolve()
        if not path.is_relative_to(folder.resolve()) or file_hash(path) != sha:
            raise ValueError(f'Frozen data hash mismatch or unsafe filename: {name}')
        actual[name] = sha
    return dict(manifest_file_sha256=file_hash(manifest_path), manifest_sha256=digest(manifest), files=actual)


def grid(cfg):
    return [dict(id=f'core_n{nodes}_{variant}_s{seed}', stage='core', nodes=nodes, variant=variant, seed=seed)
            for seed in cfg['seeds'] for nodes in cfg['nodes'] for variant in cfg['variants']]


def child_config(cfg, case):
    return cfg | dict(nodes=[case['nodes']], seeds=[case['seed']], variants=[case['variant']], device='cuda')


def process_start(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except OSError:
        return None


def alive(pid, start=None):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        if Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] == 'Z':
            return False
    except OSError:
        pass
    return start is None or process_start(pid) == start


class Scheduler:
    def __init__(self, args):
        self.args = args; self.release = Path(args.release).resolve(); self.root = Path(args.workspace).resolve()
        if self.root.is_relative_to(self.release) or self.release.is_relative_to(self.root):
            raise ValueError('Workspace and immutable release must not contain each other')
        self.root.mkdir(parents=True, exist_ok=True)
        self.data = self.release/'data/v06/wikitext2'
        release_cfg = read(self.release/'configs/v07/base.json')
        if any(release_cfg.get(name) != value for name, value in EXPECTED.items()):
            raise ValueError('Release is not the fixed 45-condition B2/T128/BPTT128/24000-step core protocol')
        self.cfg = release_cfg | {'cpu_threads': args.cpu_threads}
        if read(self.data/'manifest.json')['vocab_size'] != self.cfg['vocab_size']:
            raise ValueError('Frozen tokenizer vocabulary differs from the release config')
        self.cases = grid(self.cfg); self.active = {}; self.stopping = 0; self.relayed = 0
        self.identity = dict(schema='biomatrix-v07-a800-parallel-1', release=str(self.release),
                             source_hashes=source_files(self.release), base_config=release_cfg, effective_config=self.cfg,
                             config_sha256=digest(self.cfg), data=str(self.data.resolve()), data_identity=data_identity(self.data),
                             child_config_sha256={case['id']: digest(child_config(self.cfg, case)) for case in self.cases},
                             python=str(Path(args.python).absolute()), scheduler_sha256=file_hash(__file__),
                             wrapper_sha256=hashlib.sha256(WRAPPER.encode()).hexdigest(),
                             cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', 'default'),
                             dtype='float32', scheduling_only=True,
                             acceleration=dict(compile_step=args.compile_step, settings=ACCELERATION_SETTINGS,
                                               cache_policy=ACCELERATION_CACHE_POLICY,
                                               adapter_sha256=file_hash(self.release/'scripts/v07_cuda_acceleration.py')),
                             barriers=[dict(seed=seed, nodes=nodes) for seed in self.cfg['seeds'] for nodes in self.cfg['nodes']],
                             dispatch_policy=args.dispatch_policy, gpu_reserve_mib=args.gpu_reserve_mib,
                             gpu_reservations_mib={str(key): value for key, value in GPU_RESERVATIONS_MIB.items()} | {'reference': 1024})
        path = self.root/'manifest.json'
        if path.exists():
            if read(path) != self.identity:
                if args.migrate_scheduling:
                    self.migrate_scheduling(read(path))
                else:
                    raise ValueError('Immutable release/config/data/runtime selection changed; use a new workspace or explicit --migrate-scheduling OLD_MANIFEST_SHA')
            elif args.migrate_scheduling:
                self.recover_scheduling_migration()
        else:
            write(path, self.identity)
        state_path = self.root/'status.json'
        if state_path.exists():
            self.state = read(state_path)
            if self.state['manifest_sha256'] != digest(self.identity) or [row['case'] for row in self.state['conditions']] != self.cases:
                raise ValueError('Scheduler status does not match its immutable manifest/grid')
        else:
            self.state = dict(manifest_sha256=digest(self.identity), status='not_executed',
                              conditions=[dict(case=case, status='not_executed', attempts=[]) for case in self.cases])
        for case in self.cases:
            path = self.config_path(case)
            value = child_config(self.cfg, case)
            if path.exists() and read(path) != value:
                raise ValueError(f'Child config changed: {path}')
            write(path, value)
        for slot in range(args.workers):
            self.prepare_worker(slot)
        self.persist()

    def recover_scheduling_migration(self):
        matches = []
        for path in (self.root/'scheduling_migrations').glob('*/receipt.json'):
            receipt = read(path)
            if (receipt['old_manifest_sha256'] == self.args.migrate_scheduling
                    and receipt['new_manifest_sha256'] == digest(self.identity)):
                matches.append((path, receipt))
        if len(matches) != 1:
            raise ValueError('Scheduling migration is already current without one matching recovery receipt')
        path, receipt = matches[0]
        with ExitStack() as locks:
            if not getattr(self.args, '_scheduler_lock_held', False):
                locks.enter_context(Lock(self.root/'.scheduler.lock'))
            current = read(self.root/'status.json')
            if current['manifest_sha256'] == receipt['old_manifest_sha256']:
                if current != read(path.parent/'status.before.json'):
                    raise ValueError('Interrupted migration status differs from archived pre-migration state')
                for row in current['conditions']:
                    locks.enter_context(Lock(self.run_root(row['case'])/'.condition.lock'))
                write(self.root/'status.json', read(path.parent/'status.after.json'))
            elif current['manifest_sha256'] != receipt['new_manifest_sha256']:
                raise ValueError('Migration recovery status has an unknown manifest digest')
            write(path, receipt | dict(committed=True, recovery_verified_at=time.time()))

    def migrate_scheduling(self, old):
        old_sha = digest(old)
        if self.args.migrate_scheduling != old_sha:
            raise ValueError('Scheduling migration old manifest digest does not match --migrate-scheduling')
        mutable = {'scheduler_sha256', 'dispatch_policy', 'gpu_reserve_mib', 'gpu_reservations_mib'}
        if ({key: value for key, value in old.items() if key not in mutable}
                != {key: value for key, value in self.identity.items() if key not in mutable}):
            raise ValueError('Scheduling migration changed immutable training/source/data/adapter/wrapper identity')
        previous_state = read(self.root/'status.json')
        if previous_state.get('manifest_sha256') != old_sha or [row['case'] for row in previous_state['conditions']] != self.cases:
            raise ValueError('Scheduling migration status does not match the old immutable manifest/grid')
        with ExitStack() as locks:
            if not getattr(self.args, '_scheduler_lock_held', False):
                locks.enter_context(Lock(self.root/'.scheduler.lock'))
            for row in previous_state['conditions']:
                if read(self.config_path(row['case'])) != child_config(self.cfg, row['case']):
                    raise ValueError('Scheduling migration child training config changed')
                for attempt in row['attempts']:
                    marker_path = Path(attempt.get('pid_marker', '/does-not-exist'))
                    marker = read(marker_path) if marker_path.is_file() else attempt
                    if alive(marker.get('pid'), marker.get('process_start')):
                        raise RuntimeError('Scheduling migration requires all old child processes to have stopped')
                locks.enter_context(Lock(self.run_root(row['case'])/'.condition.lock'))
            for folder in sorted((self.root/'workers').glob('worker-*')):
                self.verify_worker(int(folder.name.split('-')[-1]))
                for version in ('v06', 'v07'):
                    locks.enter_context(Lock(folder/'results'/version/'.queue.lock'))
            self.verify_release()
            self.state = previous_state
            for row in previous_state['conditions']:
                if row['status'] == 'completed' and row.get('artifacts') != self.validate_result(row['case']):
                    raise ValueError('Scheduling migration completed artifact changed')
            archive = self.root/'scheduling_migrations'/f'{time.time_ns()}-{old_sha[:12]}'
            archive.mkdir(parents=True, exist_ok=False)
            shutil.copy2(self.root/'manifest.json', archive/'manifest.before.json')
            shutil.copy2(self.root/'status.json', archive/'status.before.json')
            updated = previous_state | dict(manifest_sha256=digest(self.identity))
            receipt = dict(schema='biomatrix-v07-scheduling-migration-1', migrated_at=time.time(),
                           old_manifest_sha256=old_sha, new_manifest_sha256=digest(self.identity),
                           old_scheduler_sha256=old['scheduler_sha256'], new_scheduler_sha256=self.identity['scheduler_sha256'],
                           old_dispatch_policy=old.get('dispatch_policy', 'barrier'), new_dispatch_policy=self.args.dispatch_policy,
                           immutable_identity_verified=True, all_parent_child_leases_verified_free=True,
                           raw_paths_preserved=True, conditions_before={row['case']['id']: row['status'] for row in previous_state['conditions']})
            write(archive/'manifest.after.json', self.identity)
            write(archive/'status.after.json', updated)
            write(archive/'receipt.json', receipt)
            write(self.root/'manifest.json', self.identity)
            write(self.root/'status.json', updated)
            write(archive/'receipt.json', receipt | dict(committed=True))

    def config_path(self, case):
        return self.root/'configs'/f'{case["id"]}.json'

    def run_root(self, case):
        return self.root/'raw'/case['id']

    def worker(self, slot):
        return self.root/'workers'/f'worker-{slot:02d}'

    def prepare_worker(self, slot):
        destination = self.worker(slot)
        if not destination.exists():
            temporary = Path(tempfile.mkdtemp(prefix=f'.worker-{slot:02d}-', dir=self.root))
            try:
                for name in self.identity['source_hashes']:
                    target = temporary/name; target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(self.release/name, target)
                data = temporary/'data/v06/wikitext2'; data.parent.mkdir(parents=True, exist_ok=True)
                data.symlink_to(self.data.resolve(), target_is_directory=True)
                (temporary/'worker_entry.py').write_text(WRAPPER)
                destination.parent.mkdir(parents=True, exist_ok=True); temporary.replace(destination)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        self.verify_worker(slot)

    def verify_worker(self, slot):
        folder = self.worker(slot)
        if folder.resolve() != folder:
            raise ValueError(f'Worker PROJECT must be a real independent directory: {folder}')
        for name, expected in self.identity['source_hashes'].items():
            path = folder/name
            if path.resolve() != path or file_hash(path) != expected:
                raise ValueError(f'Worker source changed or symlinked: {path}')
        if file_hash(folder/'worker_entry.py') != self.identity['wrapper_sha256']:
            raise ValueError('Worker launcher changed')
        if not (folder/'data/v06/wikitext2').is_symlink() or (folder/'data/v06/wikitext2').resolve() != self.data.resolve():
            raise ValueError('Worker does not share the frozen dataset symlink')

    def persist(self):
        self.state.update(updated_at=time.time(), workers=self.args.workers)
        write(self.root/'status.json', self.state)

    def verify_release(self):
        if source_files(self.release) != self.identity['source_hashes'] or data_identity(self.data) != self.identity['data_identity']:
            raise ValueError('Immutable release or shared data changed while the scheduler was running')

    def validate_result(self, case):
        folder = self.run_root(case)/case['id']
        result, protocol = read(folder/'result.json'), read(folder/'protocol.json')
        expected_cfg = child_config(self.cfg, case)
        expected_cfg |= {name: value for name, value in case.items() if name in expected_cfg}
        source = {name: sha for name, sha in self.identity['source_hashes'].items()
                  if (name.startswith('v07/') and name.endswith('.py') and not Path(name).name.startswith('test'))
                  or name in ('v04/model.py', 'v05/model.py', 'v05/run.py')}
        if (result.get('status') != 'completed' or result.get('case') != case or protocol.get('config') != expected_cfg
                or protocol.get('source_hashes') != source or protocol.get('data_manifest_sha256') != self.identity['data_identity']['manifest_sha256']
                or protocol.get('dtype') != 'float32' or protocol.get('device') != 'cuda'
                or result.get('protocol_sha256') != digest(protocol)):
            raise ValueError(f'Completed result protocol mismatch: {case["id"]}')
        runtime = {name: protocol.get(name) for name in ('torch', 'device', 'dtype')}
        if 'runtime_identity' in self.state and self.state['runtime_identity'] != runtime:
            raise ValueError('Completed conditions used different torch/device/dtype runtime versions')
        self.state['runtime_identity'] = runtime
        acceleration = read(folder/'acceleration.json')
        expected_acceleration = self.acceleration(case)
        if (any(acceleration.get(name) != value for name, value in expected_acceleration.items() if name != 'device')
                or not acceleration.get('device', '').startswith('cuda')
                or acceleration.get('status') != ('active' if expected_acceleration['mode']=='compiled_step' else 'eager')):
            raise ValueError(f'Completed acceleration identity mismatch: {case["id"]}')
        return dict(result_sha256=file_hash(folder/'result.json'), protocol_sha256=digest(protocol),
                    acceleration_sha256=file_hash(folder/'acceleration.json'))

    def acceleration(self, case):
        active = self.args.compile_step and case['variant'] in ('message', 'independent', 'full')
        return dict(schema='biomatrix-v07-acceleration-1', compile_step=self.args.compile_step,
                    mode='compiled_step' if active else 'eager', variant=case['variant'],
                    compiler='torch.compile' if active else None, settings=ACCELERATION_SETTINGS if active else None,
                    cache_policy=ACCELERATION_CACHE_POLICY if active else None,
                    device='cuda', dtype='float32',
                    precision_policy='strict_fp32_tf32_disabled' if active else 'core_float32',
                    adapter_sha256=self.identity['acceleration']['adapter_sha256'])

    def reconcile(self):
        for row in self.state['conditions']:
            if row['status'] == 'completed':
                artifacts = self.validate_result(row['case'])
                if row.get('artifacts') != artifacts:
                    raise ValueError(f'Completed artifact changed: {row["case"]["id"]}')
                continue
            if row['status'] in ('running', 'starting'):
                attempt = row['attempts'][-1]
                marker = read(attempt['pid_marker']) if Path(attempt['pid_marker']).exists() else attempt
                if alive(marker.get('pid'), marker.get('process_start')):
                    raise RuntimeError(f'Orphan child {marker["pid"]} still runs {row["case"]["id"]}; wait or stop it before resuming')
                with Lock(self.run_root(row['case'])/'.condition.lock'):
                    finish = read(attempt['finish_marker']) if Path(attempt['finish_marker']).exists() else None
                if finish and finish['exit_code'] == 0 and (self.run_root(row['case'])/row['case']['id']/'result.json').exists():
                    row['artifacts'] = self.validate_result(row['case']); row['status'] = 'completed'
                else:
                    row['status'] = 'aborted' if finish is None or attempt.get('stop_requested') else 'failed'
                    row['reason'] = 'Previous scheduler stopped; last atomic model checkpoint is retained'
                attempt.update(status=row['status'], recovered_after_parent_exit=True,
                               finished_at=finish['finished_at'] if finish else time.time())
                if finish:
                    attempt['exit_code'] = finish['exit_code']
            if row['status'] == 'failed' and self.args.retry_failed:
                row['status'] = 'not_executed'; row['reason'] = 'Explicit --retry-failed requested'
            if row['status'] == 'aborted' and self.args.resume_aborted:
                row['status'] = 'not_executed'; row['reason'] = 'Explicit --resume-aborted requested'
        self.persist()

    def launch(self, row, slot):
        self.verify_release(); self.verify_worker(slot)
        case = row['case']; run_root = self.run_root(case); run_root.mkdir(parents=True, exist_ok=True)
        if digest(read(self.config_path(case))) != self.identity['child_config_sha256'][case['id']]:
            raise ValueError(f'Child config changed before dispatch: {case["id"]}')
        with Lock(run_root/'.condition.lock'):
            pass
        number = len(row['attempts'])+1
        folder = self.root/'attempts'/case['id']/f'{number:03d}'; folder.mkdir(parents=True, exist_ok=False)
        argv = ['--config', str(self.config_path(case)), '--root', str(run_root),
                '--data-root', str(self.worker(slot)/'data/v06/wikitext2'), '--device', 'cuda', '--mode', 'paired']
        launch = dict(attempt=number, lease=str(run_root/'.condition.lock'), pid_marker=str(folder/'pid.json'),
                      finish_marker=str(folder/'finished.json'), argv=argv,
                      acceleration=self.acceleration(case), acceleration_record=str(run_root/case['id']/'acceleration.json'))
        write(folder/'launch.json', launch)
        attempt = launch | dict(worker=slot, status='starting', started_at=time.time(), stdout=str(folder/'stdout.log'))
        row['attempts'].append(attempt); row['status'] = 'starting'; self.persist()
        output = (folder/'stdout.log').open('xb')
        env = os.environ.copy(); env.update(PYTHONPATH=str(self.worker(slot)), PYTHONUNBUFFERED='1',
                                           TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS=str(self.cfg['cpu_threads']))
        if self.args.compile_step:
            env.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')
        try:
            process = subprocess.Popen([self.args.python, str(self.worker(slot)/'worker_entry.py'), str(folder/'launch.json')],
                                       cwd=self.worker(slot), env=env, stdout=output, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        except Exception as error:
            output.close(); row.update(status='failed', reason=repr(error)); attempt.update(status='failed', error=repr(error)); self.persist()
            return False
        attempt.update(pid=process.pid, process_start=process_start(process.pid), status='running')
        row['status'] = 'running'; self.active[slot] = (process, output, row); self.persist()
        print(f'Start {case["id"]} worker={slot} pid={process.pid}', flush=True)
        return True

    def reap(self):
        for slot, (process, output, row) in list(self.active.items()):
            code = process.poll()
            if code is None:
                continue
            output.close(); attempt = row['attempts'][-1]
            try:
                self.verify_worker(slot); self.verify_release()
                if code == 0 and (self.run_root(row['case'])/row['case']['id']/'result.json').exists():
                    row['artifacts'] = self.validate_result(row['case']); row['status'] = 'completed'
                else:
                    row['status'] = 'aborted' if self.stopping else 'failed'
                    row['reason'] = f'Child exit={code}; completed result absent or child unsuccessful; see stdout'
            except Exception as error:
                row.update(status='failed', reason=repr(error))
            attempt.update(status=row['status'], exit_code=code, finished_at=time.time())
            del self.active[slot]; self.persist()
            print(f'{row["status"]}: {row["case"]["id"]} exit={code}', flush=True)

    def relay_stop(self):
        if self.stopping and self.relayed != self.stopping:
            forwarded = signal.SIGTERM if not self.relayed else signal.SIGINT
            for process, _, row in self.active.values():
                row['attempts'][-1]['stop_requested'] = True
                try:
                    os.killpg(process.pid, forwarded)
                except ProcessLookupError:
                    pass
            self.relayed = self.stopping; self.persist()

    def dispatch_candidates(self):
        if self.args.dispatch_policy == 'backfill':
            return [row for row in self.state['conditions'] if row['status'] == 'not_executed']
        group = next(([row for row in self.state['conditions'] if row['case']['seed'] == seed and row['case']['nodes'] == nodes]
                      for seed in self.cfg['seeds'] for nodes in self.cfg['nodes']
                      if any(row['status'] != 'completed' and row['case']['seed'] == seed and row['case']['nodes'] == nodes
                             for row in self.state['conditions'])), [])
        if any(row['status'] in ('failed', 'aborted') for row in group):
            return []
        return [row for row in group if row['status'] == 'not_executed']

    def eligible_candidate(self, candidates):
        if self.args.dispatch_policy != 'backfill':
            return candidates[0] if candidates else None
        memory = gpu_memory()
        reserved = sum(reservation(row['case']) for _, _, row in self.active.values())
        self.state['gpu_budget'] = memory | dict(reserved_mib=reserved, reserve_mib=self.args.gpu_reserve_mib,
                                                sampled_at=time.time())
        for row in candidates:
            needed = reservation(row['case'])
            if (reserved+needed <= memory['total_mib']-self.args.gpu_reserve_mib
                    and needed <= memory['free_mib']-self.args.gpu_reserve_mib):
                return row
        return None

    def terminal_status(self):
        rows = self.state['conditions']
        if all(row['status'] == 'completed' for row in rows):
            return 'completed'
        if any(row['status'] == 'failed' for row in rows):
            return 'failed'
        return 'aborted'

    def run(self):
        self.reconcile()
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        def stop(sig, _):
            self.stopping = sig if not self.stopping else self.stopping+256
        for sig in previous:
            signal.signal(sig, stop)
        self.state['status'] = 'running'; self.persist()
        try:
            while True:
                self.relay_stop(); self.reap()
                if self.stopping:
                    if not self.active:
                        self.state['status'] = 'aborted'; self.persist()
                        return 128+(self.stopping % 256)
                else:
                    pending = self.dispatch_candidates()
                    if not self.active and not pending:
                        self.state['status'] = self.terminal_status(); self.persist()
                        if self.state['status'] != 'completed':
                            print('Queue retains failed/aborted conditions; use --retry-failed / --resume-aborted explicitly', flush=True)
                        return 0 if self.state['status'] == 'completed' else 1
                    for slot in range(self.args.workers):
                        if slot in self.active or self.stopping:
                            continue
                        candidate = self.eligible_candidate(pending)
                        if candidate is None:
                            break
                        pending.remove(candidate)
                        self.launch(candidate, slot)
                    self.persist()
                time.sleep(self.args.poll_seconds)
        except BaseException:
            self.stopping = signal.SIGTERM
            self.relay_stop()
            while self.active:
                self.reap(); time.sleep(self.args.poll_seconds)
            self.state['status'] = 'failed'; self.persist()
            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--release', required=True, type=Path); parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--dispatch-policy', choices=('barrier', 'backfill'), default='barrier')
    parser.add_argument('--gpu-reserve-mib', type=int, default=12288, help='GPU headroom outside conservative active reservations')
    parser.add_argument('--migrate-scheduling', metavar='OLD_MANIFEST_SHA', help='Stopped-workspace migration; immutable training identity must match')
    parser.add_argument('--workers', type=int, default=3); parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--cpu-threads', type=int, default=1, help='Hardware thread count per child; model/data/step budgets unchanged')
    parser.add_argument('--compile-step', action='store_true', help='Explicit strict-FP32 torch.compile(Network.step) adapter; never changes v07 core/config')
    parser.add_argument('--mode', choices=('queue', 'benchmark'), default='queue')
    parser.add_argument('--prepare-only', action='store_true'); parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--resume-aborted', action='store_true'); parser.add_argument('--poll-seconds', type=float, default=2.)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 16 or args.poll_seconds <= 0 or args.cpu_threads < 1 or args.gpu_reserve_mib < 0:
        parser.error('--workers must be 1..16, --cpu-threads positive, --poll-seconds positive, --gpu-reserve-mib nonnegative')
    args.python = shutil.which(args.python) or args.python
    return args


def main(argv=None):
    args = arguments(argv)
    if args.mode == 'benchmark':
        raise SystemExit('Benchmark is a separate engineering protocol; this scheduler only implements --mode queue')
    release, workspace = args.release.resolve(), args.workspace.resolve()
    if workspace.is_relative_to(release) or release.is_relative_to(workspace):
        raise ValueError('Workspace and immutable release must not contain each other')
    global_lock = Path(tempfile.gettempdir())/f'biomatrix-v07-a800-parallel-{os.getuid()}.lock'
    with ExitStack() as stack:
        stack.enter_context(Lock(global_lock)); stack.enter_context(Lock(args.workspace/'.scheduler.lock'))
        args._scheduler_lock_held = True
        scheduler = Scheduler(args)
        if args.prepare_only:
            print(f'Prepared 45 conditions and {args.workers} independent worker projects; no training started', flush=True)
            return 0
        return scheduler.run()


if __name__ == '__main__':
    raise SystemExit(main())
