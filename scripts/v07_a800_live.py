"""Adopt a frozen scheduler's existing workers without restarting training.

The base scheduler remains the immutable experiment identity. This adapter has
its own durable receipt and never changes the manifest or training attempts.
Stop only the verified old supervisor before adoption; SIGTERM on that old
supervisor would stop its workers. --preflight-only is entirely read-only.
"""
import argparse
from contextlib import ExitStack
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time


def load_base(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location('biomatrix_frozen_supervisor', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lease_held(path):
    """Test an existing lease; never create or replace its inode."""
    import fcntl
    with Path(path).open('r') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def process_description(pid):
    proc = Path('/proc')/str(pid)
    return dict(argv=(proc/'cmdline').read_bytes().decode().strip('\0').split('\0'),
                cwd=(proc/'cwd').resolve(), pgid=os.getpgid(pid))


def inspect_workspace(base, args, manifest_sha):
    """Reuse base identity construction while rejecting every attempted write.

    The resulting view can explicitly prepare a missing worker directory later;
    its persist() remains disabled. Inspection itself does not prepare workers.
    """
    root = Path(args.workspace).resolve()
    if not root.is_dir() or not (root/'manifest.json').is_file() or not (root/'status.json').is_file():
        raise ValueError('Adoption requires an existing workspace, manifest and status')
    manifest = base.read(root/'manifest.json')
    if base.digest(manifest) != manifest_sha:
        raise ValueError('Adoption old manifest digest does not match')
    if manifest['scheduler_sha256'] != base.file_hash(base.__file__):
        raise ValueError('Base scheduler source differs from frozen manifest')
    if args.migrate_scheduling or args.resume_aborted or args.retry_failed:
        raise ValueError('Live adoption cannot migrate training identity or retry existing conditions')

    class ReadOnlyView(base.Scheduler):
        def persist(self):
            pass

        def prepare_worker(self, slot):
            if not getattr(self, '_inspecting', True):
                return super().prepare_worker(slot)
            if self.worker(slot).exists():
                self.verify_worker(slot)

    original_write = base.write

    def verify_existing(path, value):
        if not Path(path).is_file() or base.read(path) != value:
            raise ValueError(f'Preflight would change or create an immutable file: {path}')

    try:
        base.write = verify_existing
        view = ReadOnlyView(args)
    finally:
        base.write = original_write
    view._inspecting = False
    view.verify_release()
    occupied, plans = set(), []
    for row in view.state['conditions']:
        if row['status'] == 'completed':
            if row.get('artifacts') != view.validate_result(row['case']):
                raise ValueError('Completed artifact changed during adoption')
            continue
        if row['status'] not in ('running', 'starting'):
            continue
        attempts = row.get('attempts', [])
        if not attempts:
            raise ValueError('Active condition has no durable attempt')
        attempt = attempts[-1]
        number, slot = attempt['attempt'], attempt['worker']
        if slot in occupied or not 0 <= slot < args.workers:
            raise ValueError('Duplicate or out-of-range active worker slot')
        occupied.add(slot)
        view.verify_worker(slot)
        folder = root/'attempts'/row['case']['id']/f'{number:03d}'
        lease = view.run_root(row['case'])/'.condition.lock'
        argv = ['--config', str(view.config_path(row['case'])), '--root', str(view.run_root(row['case'])),
                '--data-root', str(view.worker(slot)/'data/v06/wikitext2'), '--device', 'cuda', '--mode', 'paired']
        expected = dict(attempt=number, lease=str(lease), pid_marker=str(folder/'pid.json'),
                        finish_marker=str(folder/'finished.json'), argv=argv,
                        acceleration=view.acceleration(row['case']),
                        acceleration_record=str(view.run_root(row['case'])/row['case']['id']/'acceleration.json'))
        if base.read(folder/'launch.json') != expected or any(attempt.get(key) != value for key, value in expected.items()):
            raise ValueError('Active attempt launch identity changed')
        marker = base.read(expected['pid_marker'])
        pid, start = marker.get('pid'), marker.get('process_start')
        if not isinstance(pid, int) or not start or marker.get('attempt') != number:
            raise ValueError('Active attempt has no valid PID/start/attempt marker')
        if attempt.get('pid') not in (None, pid) or attempt.get('process_start') not in (None, start):
            raise ValueError('PID marker and attempt identity differ')
        alive = base.alive(pid, start)
        if base.alive(pid) and base.process_start(pid) != start:
            raise ValueError('Active PID was reused by another process')
        held = lease_held(lease)
        if alive:
            description = process_description(pid)
            expected_cmd = [args.python, str(view.worker(slot)/'worker_entry.py'), str(folder/'launch.json')]
            if (description['argv'] != expected_cmd or description['cwd'] != view.worker(slot)
                    or description['pgid'] != pid or not held):
                raise ValueError('Active process command/cwd/group/condition lease differs')
            # A process may finish during inspection. The proxy will reconcile it
            # without a new attempt; a different process identity is never adopted.
            if base.alive(pid) and base.process_start(pid) != start:
                raise ValueError('Active PID identity changed during inspection')
        elif held:
            raise ValueError('Exited worker still has a held condition lease')
        finish_path = Path(expected['finish_marker'])
        if finish_path.exists():
            finish = base.read(finish_path)
            if finish.get('attempt') != number or not isinstance(finish.get('exit_code'), int):
                raise ValueError('Finish marker belongs to another attempt or is invalid')
        plans.append(dict(case_id=row['case']['id'], slot=slot, pid=pid, process_start=start,
                          attempt=number, finish_marker=expected['finish_marker'], lease=str(lease), alive=alive))
    view.adoption_plans = plans
    return view


class AdoptedProcess:
    """A non-child PID cannot be waitpid()ed; its lease and marker define exit."""
    def __init__(self, base, plan):
        self.base = base
        self.pid, self.start = plan['pid'], plan['process_start']
        self.finish, self.lease = Path(plan['finish_marker']), Path(plan['lease'])
        self.attempt = plan['attempt']

    def poll(self):
        if self.base.alive(self.pid, self.start):
            return None
        if lease_held(self.lease):
            return None
        if not self.finish.exists():
            return 125
        marker = self.base.read(self.finish)
        if marker.get('attempt') != self.attempt or not isinstance(marker.get('exit_code'), int):
            return 125
        return marker['exit_code']


def scheduler_class(base, adapter_identity, receipt_path, manifest_sha):
    class LiveScheduler(base.Scheduler):
        def persist(self):
            self.state['supervisor_adapter'] = adapter_identity
            super().persist()

        def reconcile(self):
            # Validation finishes for every condition before any active map or
            # attempt is changed. Failure here leaves all training PIDs alone.
            view = inspect_workspace(base, self.args, manifest_sha)
            self.state = view.state
            rows = {row['case']['id']: row for row in self.state['conditions']}
            active = {plan['slot']: (AdoptedProcess(base, plan), io.StringIO(), rows[plan['case_id']])
                      for plan in view.adoption_plans}
            self.active = active
            self.persist()
            receipt = base.read(receipt_path)
            base.write(receipt_path, receipt | dict(committed=True, adopted_at=time.time(),
                       adopted=view.adoption_plans, status_after=copy.deepcopy(self.state)))

        def relay_stop(self):
            if not self.stopping or self.relayed == self.stopping:
                return
            forwarded = signal.SIGTERM if not self.relayed else signal.SIGINT
            for process, _, row in self.active.values():
                attempt = row['attempts'][-1]
                start = process.start if isinstance(process, AdoptedProcess) else attempt['process_start']
                if not base.alive(process.pid, start):
                    continue
                try:
                    if os.getpgid(process.pid) != process.pid or base.process_start(process.pid) != start:
                        raise RuntimeError('Refusing to signal an unverified worker process group')
                    attempt['stop_requested'] = True
                    os.killpg(process.pid, forwarded)
                except ProcessLookupError:
                    pass
            self.relayed = self.stopping
            self.persist()

        def run(self):
            self.reconcile()
            previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

            def request_stop(sig, _):
                self.stopping = sig if not self.stopping else self.stopping+256

            for sig in previous:
                signal.signal(sig, request_stop)
            self.state['status'] = 'running'
            self.persist()
            try:
                while True:
                    self.relay_stop()
                    self.reap()
                    if self.stopping:
                        if not self.active:
                            self.state['status'] = 'aborted'
                            self.persist()
                            return 128+(self.stopping % 256)
                    else:
                        pending = self.dispatch_candidates()
                        if not self.active and not pending:
                            self.state['status'] = self.terminal_status()
                            self.persist()
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
                # A failed supervisor must be replaceable while its training
                # continues. Only an explicit operator signal is relayed.
                self.state['status'] = 'supervisor_failed'
                self.persist()
                raise
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)

    return LiveScheduler


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--base-scheduler', type=Path, default=Path(os.environ.get(
        'BIOMATRIX_BASE_SCHEDULER', str(Path(__file__).with_name('v07_a800_parallel.py')))))
    parser.add_argument('--adopt-running')
    parser.add_argument('--preflight-only', action='store_true')
    options, remaining = parser.parse_known_args(argv)
    base = load_base(options.base_scheduler)
    args = base.arguments(remaining)
    if not options.adopt_running:
        if options.preflight_only:
            raise ValueError('--preflight-only requires --adopt-running OLD_MANIFEST_SHA')
        return base.main(remaining)
    if args.mode != 'queue' or args.prepare_only:
        raise ValueError('Adoption requires queue mode; use --preflight-only for inspection')
    if options.preflight_only:
        view = inspect_workspace(base, args, options.adopt_running)
        print(json.dumps(dict(status='preflight_passed', workers=args.workers,
                              manifest_sha256=options.adopt_running, active=view.adoption_plans)), flush=True)
        return 0
    global_lock = Path(tempfile.gettempdir())/f'biomatrix-v07-a800-parallel-{os.getuid()}.lock'
    with ExitStack() as stack:
        stack.enter_context(base.Lock(global_lock))
        stack.enter_context(base.Lock(args.workspace/'.scheduler.lock'))
        args._scheduler_lock_held = True
        view = inspect_workspace(base, args, options.adopt_running)
        adapter = dict(schema='biomatrix-v07-live-supervisor-1', adapter_path=str(Path(__file__).resolve()),
                       adapter_sha256=base.file_hash(__file__), base_path=str(Path(base.__file__).resolve()),
                       base_sha256=base.file_hash(base.__file__), workers=args.workers)
        receipt = view.root/'live_adoptions'/f'{time.time_ns()}-{adapter["adapter_sha256"][:12]}'/'receipt.json'
        base.write(receipt, dict(schema='biomatrix-v07-live-adoption-1', committed=False,
                   created_at=time.time(), adapter=adapter, manifest_sha256=options.adopt_running,
                   manifest_file_sha256=base.file_hash(view.root/'manifest.json'),
                   status_before=copy.deepcopy(view.state), inspected=view.adoption_plans))
        scheduler = scheduler_class(base, adapter, receipt, options.adopt_running)(args)
        return scheduler.run()


if __name__ == '__main__':
    raise SystemExit(main())
