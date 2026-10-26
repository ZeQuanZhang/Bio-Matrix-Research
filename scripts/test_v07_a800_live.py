"""Live-supervisor CPU tests with the existing fake runner; zero ML updates."""
from contextlib import redirect_stdout
import copy
import importlib.util
import io
import os
from pathlib import Path
import signal
import sys
import time
import unittest
from unittest.mock import patch


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


live = load('live_adapter_test_subject', 'v07_a800_live.py')
fixture = load('live_fixture', 'test_v07_a800_parallel.py')
base = fixture.module


class LiveTests(unittest.TestCase):
    args = fixture.SchedulerTests.args

    def setUp(self):
        fixture.SchedulerTests.setUp(self)
        self.root, self.release, self.workspace = self.root.resolve(), self.release.resolve(), self.workspace.resolve()
        self.processes = {}
        self.outputs = []
        self.descriptions = {}
        self.patches = [patch.object(base, 'process_start', side_effect=lambda pid: f'test-start-{pid}'),
                        patch.object(base, 'alive', side_effect=self.alive),
                        patch.object(live, 'process_description', side_effect=lambda pid: self.descriptions[pid])]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for process in self.processes.values():
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)
        for output in self.outputs:
            output.close()
        for item in reversed(self.patches):
            item.stop()
        fixture.SchedulerTests.tearDown(self)

    def alive(self, pid, start=None):
        return (pid in self.processes and self.processes[pid].poll() is None
                and (start is None or start == f'test-start-{pid}'))

    def prepare(self, active=0, duration='3'):
        os.environ['FAKE_DURATION'] = duration
        scheduler = base.Scheduler(self.args('--workers', '12', '--dispatch-policy', 'backfill'))
        with redirect_stdout(io.StringIO()):
            for slot, row in enumerate(scheduler.state['conditions'][:active]):
                scheduler.launch(row, slot)
                self.register(scheduler, slot)
        self.old = scheduler
        self.sha = base.digest(scheduler.identity)
        return scheduler

    def register(self, scheduler, slot):
        process, output, row = scheduler.active[slot]
        self.processes[process.pid] = process
        self.outputs.append(output)
        attempt = row['attempts'][-1]
        marker_path = Path(attempt['pid_marker'])
        deadline = time.monotonic()+5
        while not marker_path.exists() or not live.lease_held(attempt['lease']):
            if process.poll() is not None or time.monotonic() > deadline:
                self.fail('Fake worker failed to acquire its lease')
            time.sleep(.005)
        # macOS has no /proc: substitute the same synthetic start identity in
        # the fixture marker, while the real stdlib child keeps its real lease.
        marker = base.read(marker_path)
        marker['process_start'] = f'test-start-{process.pid}'
        base.write(marker_path, marker)
        self.descriptions[process.pid] = dict(
            argv=[scheduler.args.python, str(scheduler.worker(slot)/'worker_entry.py'),
                  str(marker_path.parent/'launch.json')], cwd=scheduler.worker(slot), pgid=process.pid)

    def adoption_args(self, workers=14):
        return self.args('--workers', str(workers), '--dispatch-policy', 'backfill')

    def new_supervisor(self, workers=14):
        args = self.adoption_args(workers)
        receipt = self.workspace/'live_adoptions'/str(time.time_ns())/'receipt.json'
        base.write(receipt, dict(committed=False, status_before=copy.deepcopy(base.read(self.workspace/'status.json'))))
        cls = live.scheduler_class(base, dict(adapter_sha256='test-adapter', workers=workers), receipt, self.sha)
        return cls(args), receipt

    def test_preflight_is_read_only_and_explicit_prepare_creates_only_new_workers(self):
        self.prepare()
        paths = list(self.workspace.rglob('*.py'))+list((self.workspace/'configs').glob('*.json'))
        paths += [self.workspace/'manifest.json', self.workspace/'status.json']
        before = {path: (base.file_hash(path), path.stat().st_mtime_ns) for path in paths}
        view = live.inspect_workspace(base, self.adoption_args(), self.sha)
        self.assertEqual(before, {path: (base.file_hash(path), path.stat().st_mtime_ns) for path in paths})
        self.assertFalse((self.workspace/'workers/worker-12').exists())
        view.prepare_worker(12)
        view.prepare_worker(13)
        self.assertTrue((self.workspace/'workers/worker-13/worker_entry.py').is_file())
        self.assertEqual(before, {path: (base.file_hash(path), path.stat().st_mtime_ns) for path in paths})

    def test_live_twelve_adopt_and_expand_without_new_attempts_or_pid_changes(self):
        self.prepare(active=12, duration='2')
        original = {row['case']['id']: copy.deepcopy(row['attempts']) for row in self.old.state['conditions'][:12]}
        manifest = (self.workspace/'manifest.json').read_bytes()
        new, receipt = self.new_supervisor()
        new.reconcile()
        self.assertEqual(len(new.active), 12)
        self.assertTrue(base.read(receipt)['committed'])
        for row in new.state['conditions'][:12]:
            self.assertEqual(row['attempts'], original[row['case']['id']])
        self.assertEqual(manifest, (self.workspace/'manifest.json').read_bytes())
        with redirect_stdout(io.StringIO()):
            for slot, row in zip((12, 13), new.state['conditions'][12:14]):
                new.launch(row, slot)
                self.register(new, slot)
        self.assertEqual(len(new.active), 14)
        second, _ = self.new_supervisor()
        second.reconcile()
        self.assertEqual(len(second.active), 14)
        self.assertTrue(all(len(row['attempts']) == 1 for row in second.state['conditions'][:14]))
        deadline = time.monotonic()+6
        with redirect_stdout(io.StringIO()):
            while second.active and time.monotonic() < deadline:
                second.reap()
                time.sleep(.01)
        self.assertFalse(second.active)
        self.assertTrue(all(row['status'] == 'completed' for row in second.state['conditions'][:14]))
        self.assertEqual(manifest, (self.workspace/'manifest.json').read_bytes())

    def test_adoption_validation_failure_does_not_signal_or_change_attempt(self):
        self.prepare(active=1)
        path = self.old.worker(0)/'worker_entry.py'
        path.write_text(path.read_text()+'\n# tampered\n')
        before = (self.workspace/'status.json').read_bytes()
        with patch.object(os, 'killpg') as kill:
            with self.assertRaisesRegex(ValueError, 'launcher changed'):
                live.inspect_workspace(base, self.adoption_args(), self.sha)
        kill.assert_not_called()
        self.assertEqual(before, (self.workspace/'status.json').read_bytes())
        self.assertTrue(all(process.poll() is None for process in self.processes.values()))

    def test_launch_tampering_and_reused_pid_are_rejected(self):
        self.prepare(active=1)
        row = self.old.state['conditions'][0]
        path = Path(row['attempts'][-1]['pid_marker']).with_name('launch.json')
        original = base.read(path)
        base.write(path, original | dict(attempt=99))
        with self.assertRaisesRegex(ValueError, 'launch identity'):
            live.inspect_workspace(base, self.adoption_args(), self.sha)
        base.write(path, original)
        with patch.object(base, 'process_start', return_value='reused-start'):
            with self.assertRaisesRegex(ValueError, 'reused'):
                live.inspect_workspace(base, self.adoption_args(), self.sha)

    def test_live_process_requires_held_condition_lease_and_unique_slot(self):
        self.prepare(active=2)
        with patch.object(live, 'lease_held', return_value=False):
            with self.assertRaisesRegex(ValueError, 'lease differs'):
                live.inspect_workspace(base, self.adoption_args(), self.sha)
        state = base.read(self.workspace/'status.json')
        state['conditions'][1]['attempts'][-1]['worker'] = 0
        base.write(self.workspace/'status.json', state)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            live.inspect_workspace(base, self.adoption_args(), self.sha)

    def test_supervisor_exception_keeps_workers_live_and_allows_readoption(self):
        self.prepare(active=1)
        new, _ = self.new_supervisor()
        with patch.object(new, 'eligible_candidate', side_effect=RuntimeError('test supervisor failure')), \
             patch.object(os, 'killpg') as kill:
            with self.assertRaisesRegex(RuntimeError, 'test supervisor failure'):
                new.run()
        kill.assert_not_called()
        self.assertEqual(base.read(self.workspace/'status.json')['status'], 'supervisor_failed')
        again, _ = self.new_supervisor()
        again.reconcile()
        self.assertEqual(len(again.active), 1)
        self.assertEqual(len(again.state['conditions'][0]['attempts']), 1)
        self.assertTrue(all(process.poll() is None for process in self.processes.values()))

    def test_missing_finish_is_failed_without_new_attempt(self):
        self.prepare(active=1, duration='.04')
        process = next(iter(self.processes.values()))
        process.wait(timeout=5)
        attempt = self.old.state['conditions'][0]['attempts'][-1]
        Path(attempt['finish_marker']).unlink()
        new, _ = self.new_supervisor()
        new.reconcile()
        with redirect_stdout(io.StringIO()):
            new.reap()
        self.assertEqual(new.state['conditions'][0]['status'], 'failed')
        self.assertEqual(len(new.state['conditions'][0]['attempts']), 1)

    def test_finish_marker_waits_for_lease_and_same_attempt(self):
        lease = self.root/'lease'
        lease.touch()
        finish = self.root/'finish.json'
        plan = dict(pid=99999999, process_start='fake', finish_marker=str(finish), lease=str(lease), attempt=2)
        proxy = live.AdoptedProcess(base, plan)
        with base.Lock(lease):
            self.assertIsNone(proxy.poll())
        base.write(finish, dict(exit_code=0, attempt=1))
        self.assertEqual(proxy.poll(), 125)
        base.write(finish, dict(exit_code=0, attempt=2))
        self.assertEqual(proxy.poll(), 0)

    def test_relay_refuses_reused_pid_and_nonisolated_process_group(self):
        self.prepare(active=1)
        new, _ = self.new_supervisor()
        new.reconcile()
        new.stopping = signal.SIGTERM
        with patch.object(base, 'process_start', return_value='reused'), patch.object(os, 'killpg') as kill:
            with self.assertRaisesRegex(RuntimeError, 'unverified'):
                new.relay_stop()
        kill.assert_not_called()
        with patch.object(os, 'getpgid', return_value=1), patch.object(os, 'killpg') as kill:
            with self.assertRaisesRegex(RuntimeError, 'unverified'):
                new.relay_stop()
        kill.assert_not_called()

    def test_completed_result_identity_and_base_hash_are_verified(self):
        self.prepare(active=1, duration='.04')
        next(iter(self.processes.values())).wait(timeout=5)
        with redirect_stdout(io.StringIO()):
            self.old.reap()
        live.inspect_workspace(base, self.adoption_args(), self.sha)
        path = self.old.run_root(self.old.cases[0])/self.old.cases[0]['id']/'result.json'
        path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            live.inspect_workspace(base, self.adoption_args(), self.sha)
        manifest = base.read(self.workspace/'manifest.json')
        manifest['scheduler_sha256'] = 'modified-base'
        base.write(self.workspace/'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'Base scheduler source'):
            live.inspect_workspace(base, self.adoption_args(), base.digest(manifest))

    def test_exclusive_lock_prevents_second_supervisor_before_any_mutation(self):
        self.prepare()
        before = (self.workspace/'status.json').read_bytes()
        command = ['--base-scheduler', str(fixture.SCRIPT), '--adopt-running', self.sha,
                   '--release', str(self.release), '--workspace', str(self.workspace),
                   '--workers', '14', '--dispatch-policy', 'backfill']
        with base.Lock(self.workspace/'.scheduler.lock'):
            with self.assertRaisesRegex(RuntimeError, 'Another scheduler'):
                live.main(command)
        self.assertEqual(before, (self.workspace/'status.json').read_bytes())

    def test_retry_and_immutable_training_changes_are_rejected(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, 'cannot migrate'):
            live.inspect_workspace(base, self.adoption_args().__class__(**(vars(self.adoption_args()) | dict(resume_aborted=True))), self.sha)
        with self.assertRaisesRegex(ValueError, 'runtime selection changed'):
            live.inspect_workspace(base, self.args('--workers', '14', '--dispatch-policy', 'backfill', '--cpu-threads', '2'), self.sha)


if __name__ == '__main__':
    unittest.main(verbosity=2)
