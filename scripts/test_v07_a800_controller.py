"""Local controller tests: no network, GPU, training or real process signals."""
from contextlib import contextmanager, redirect_stdout
import base64
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET


PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('a800_controller', PROJECT/'run_biomatrix_v07_a800.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.root = Path(self.temporary.name).resolve()
        self.release = self.root/'release'; self.release.mkdir()
        self.workspace = self.root/'workspace'; self.workspace.mkdir()
        self.scheduler = self.root/'scheduler.py'; self.scheduler.write_text('# fixture only\n')
        self.known_hosts = self.root/'known_hosts'; self.known_hosts.write_text('fixed-host-key-fixture\n')
        self.config_path = self.root/'connection.local.json'
        self.cfg = dict(host='fixture.invalid', port=2200, username='fixture', password='synthetic-secret-not-real',
                        known_hosts=str(self.known_hosts), control_path=str(self.root/'control'),
                        remote_python=sys.executable, release=str(self.release), workspace=str(self.workspace),
                        scheduler=str(self.scheduler), workers=5, cpu_threads=1, compile_step=True,
                        dispatch_policy='barrier', gpu_reserve_mib=12288,
                        environment={'CUDA_MPS_PIPE_DIRECTORY': '/tmp/fake-mps', 'OMP_NUM_THREADS': '1'})
        self.config_path.write_text(json.dumps(self.cfg)); self.config_path.chmod(0o600)
        self.procs = {}; self.proc_reads = {}; self.changed_start_after = None

    def tearDown(self):
        self.temporary.cleanup()

    def scheduler_argv(self, workspace=None):
        return [sys.executable, str(self.scheduler), '--release', str(self.release),
                '--workspace', str(workspace or self.workspace), '--workers', '5']

    def remote(self, operation, **flags):
        case = self
        class ProcPath:
            def __init__(self, value):
                self.value = value
            def iterdir(self):
                return [SimpleNamespace(name=str(pid)) for pid in case.procs]
            def read_text(self):
                pid = int(self.value.split('/')[2]); row = case.procs.get(pid)
                if row is None:
                    raise FileNotFoundError
                case.proc_reads[pid] = case.proc_reads.get(pid, 0)+1
                start = row['start']
                if case.changed_start_after and case.proc_reads[pid] > case.changed_start_after:
                    start += '-changed'
                return f'{pid} (fixture scheduler) '+row.get('state', 'S')+' '+' '.join(['0']*18)+' '+start
            def read_bytes(self):
                pid = int(self.value.split('/')[2]); row = case.procs.get(pid)
                if row is None:
                    raise FileNotFoundError
                return ('\0'.join(row['argv'])+'\0').encode()
        def path(*parts):
            result = Path(*parts)
            return ProcPath(str(result)) if str(result) == '/proc' or str(result).startswith('/proc/') else result
        def launch(argv, **kwargs):
            self.procs[1234] = dict(start='999', argv=argv)
            return SimpleNamespace(pid=1234)
        payload = {name: self.cfg[name] for name in ('remote_python', 'release', 'workspace', 'scheduler',
                                                   'workers', 'cpu_threads', 'environment', 'compile_step',
                                                   'dispatch_policy', 'gpu_reserve_mib')}
        payload.update(operation=operation, **flags)
        output = io.StringIO()
        with patch.dict(sys.modules, {'pathlib': SimpleNamespace(Path=path)}), \
             patch.object(sys, 'argv', ['remote', base64.b64encode(json.dumps(payload).encode()).decode()]), \
             patch('subprocess.Popen', side_effect=launch) as popen, \
             patch('subprocess.run', return_value=getattr(self, 'gpu_result', subprocess.CompletedProcess([], 1, '', '')),
                   side_effect=getattr(self, 'gpu_error', None)) as query, \
             patch('os.kill') as kill, redirect_stdout(output):
            self.gpu_query = query
            exec(compile(module.REMOTE, '<remote-controller-fixture>', 'exec'), {})
        return json.loads(output.getvalue()), popen, kill

    def test_askpass_protocol_outputs_only_synthetic_password_and_static_errors(self):
        helper = PROJECT/'scripts/v07_ssh_askpass.py'
        result = subprocess.run([sys.executable, str(helper), 'ignored prompt'],
                                env=os.environ | {'BIOMATRIX_SSH_CONFIG': str(self.config_path)}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0); self.assertEqual(result.stdout, self.cfg['password']+'\n')
        self.assertEqual(result.stderr, '')
        failure = subprocess.run([sys.executable, str(helper)], env=os.environ | {'BIOMATRIX_SSH_CONFIG': str(self.root/'absent')},
                                 capture_output=True, text=True)
        self.assertNotEqual(failure.returncode, 0); self.assertEqual(failure.stdout, '')
        self.assertNotIn(self.cfg['password'], failure.stderr)

    def test_native_ssh_force_askpass_pinning_and_secret_exclusion(self):
        with module.connection(self.config_path) as ssh:
            wrapper = Path(ssh.env['SSH_ASKPASS'])
            self.assertEqual(wrapper.stat().st_mode & 0o777, 0o700)
            self.assertIn(sys.executable, wrapper.read_text()); self.assertNotIn(self.cfg['password'], wrapper.read_text())
            self.assertEqual(ssh.env['SSH_ASKPASS_REQUIRE'], 'force')
            args = ssh.command('start', resume_aborted=False, retry_failed=False)
            self.assertNotIn(self.cfg['password'], repr(args))
            self.assertIn('StrictHostKeyChecking=yes', args); self.assertIn('UserKnownHostsFile='+str(self.known_hosts), args)
            remote = module.shlex.split(args[-1]); payload = json.loads(base64.b64decode(remote[-1]))
            self.assertNotIn('password', payload); self.assertNotIn('host', payload)
            self.assertEqual(payload['environment'], self.cfg['environment']); self.assertTrue(payload['compile_step'])
            self.assertEqual(payload['dispatch_policy'], 'barrier')
            self.assertEqual(payload['gpu_reserve_mib'], 12288)
            completed = subprocess.CompletedProcess(args, 0, '{"status":"running"}', '')
            with patch.object(module.subprocess, 'run', return_value=completed) as run:
                self.assertEqual(ssh.request('status')['status'], 'running')
            self.assertEqual(run.call_args.kwargs['stdin'], subprocess.DEVNULL)
        self.assertFalse(wrapper.exists())

    def test_ssh_errors_are_redacted(self):
        with module.connection(self.config_path) as ssh:
            completed = subprocess.CompletedProcess([], 1, '', 'error '+self.cfg['password'])
            with patch.object(module.subprocess, 'run', return_value=completed):
                with self.assertRaisesRegex(RuntimeError, r'\[redacted\]') as result:
                    ssh.request('status')
            self.assertNotIn(self.cfg['password'], str(result.exception))

    def test_saved_config_requires_private_permissions_and_fixed_public_key(self):
        self.config_path.chmod(0o644)
        with self.assertRaises(PermissionError):
            module.load_connection(self.config_path)
        self.config_path.chmod(0o600); self.known_hosts.unlink()
        with self.assertRaises(FileNotFoundError):
            module.load_connection(self.config_path)

    def test_more_parallel_workers_are_supported_up_to_32(self):
        self.cfg['workers'] = 16
        self.config_path.write_text(json.dumps(self.cfg))
        self.assertEqual(module.load_connection(self.config_path)['workers'], 16)
        self.cfg['workers'] = 33
        self.config_path.write_text(json.dumps(self.cfg))
        with self.assertRaisesRegex(ValueError, '1..32'):
            module.load_connection(self.config_path)

    def test_scheduling_configuration_defaults_validation_and_public_transmission(self):
        for name in ('dispatch_policy', 'gpu_reserve_mib'):
            self.cfg.pop(name)
        self.config_path.write_text(json.dumps(self.cfg))
        loaded = module.load_connection(self.config_path)
        self.assertEqual((loaded['dispatch_policy'], loaded['gpu_reserve_mib']), ('barrier', 12288))
        self.cfg.update(dispatch_policy='backfill', gpu_reserve_mib=8192)
        self.config_path.write_text(json.dumps(self.cfg))
        with module.connection(self.config_path) as ssh:
            payload = json.loads(base64.b64decode(module.shlex.split(ssh.command('start')[-1])[-1]))
        self.assertEqual((payload['dispatch_policy'], payload['gpu_reserve_mib']), ('backfill', 8192))
        self.cfg['dispatch_policy'] = 'unknown'; self.config_path.write_text(json.dumps(self.cfg))
        with self.assertRaisesRegex(ValueError, 'dispatch_policy'):
            module.load_connection(self.config_path)
        self.cfg['dispatch_policy'] = 'backfill'
        for invalid in (-1, True, '8192'):
            self.cfg['gpu_reserve_mib'] = invalid; self.config_path.write_text(json.dumps(self.cfg))
            with self.assertRaisesRegex(ValueError, 'gpu_reserve_mib'):
                module.load_connection(self.config_path)

    def test_start_detaches_once_with_exact_scheduler_flags_and_environment(self):
        state, popen, kill = self.remote('start', resume_aborted=True, retry_failed=True)
        self.assertEqual(state['action'], 'started'); self.assertTrue(state['scheduler_alive'])
        argv = popen.call_args.args[0]
        self.assertEqual(argv[:3], ['nohup', sys.executable, str(self.scheduler)])
        for flag in ('--resume-aborted', '--retry-failed', '--compile-step'):
            self.assertIn(flag, argv)
        self.assertNotIn('--train-steps', argv); self.assertNotIn('--batch-size', argv)
        self.assertEqual(argv[argv.index('--dispatch-policy')+1], 'barrier')
        self.assertEqual(argv[argv.index('--gpu-reserve-mib')+1], '12288')
        self.assertEqual(popen.call_args.kwargs['env']['CUDA_MPS_PIPE_DIRECTORY'], '/tmp/fake-mps')
        self.assertEqual(popen.call_args.kwargs['stdin'], subprocess.DEVNULL)
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        marker = json.loads((self.workspace/'scheduler.pid.json').read_text())
        self.assertEqual((marker['pid'], marker['process_start']), (1234, '999')); kill.assert_not_called()
        second, again, kill = self.remote('start')
        self.assertEqual(second['action'], 'attached'); again.assert_not_called(); kill.assert_not_called()

    def test_pause_signals_only_verified_parent_and_rejects_pid_reuse(self):
        self.procs[50] = dict(start='555', argv=self.scheduler_argv())
        state, popen, kill = self.remote('pause')
        self.assertEqual(state['action'], 'pause_requested'); kill.assert_called_once_with(50, signal.SIGTERM)
        popen.assert_not_called()
        self.proc_reads.clear(); self.changed_start_after = 1
        with patch('os.kill') as outer:
            with self.assertRaisesRegex(RuntimeError, 'identity changed'):
                self.remote('pause')
            outer.assert_not_called()
        self.changed_start_after = None; self.proc_reads.clear()
        self.procs[50]['argv'] = self.scheduler_argv(self.root/'other-workspace')
        state, popen, kill = self.remote('pause'); self.assertEqual(state['action'], 'already_stopped')
        kill.assert_not_called()

    def test_attach_refuses_acceleration_or_thread_identity_changes(self):
        self.procs[50] = dict(start='555', argv=self.scheduler_argv()+['--cpu-threads', '1'])
        with self.assertRaisesRegex(RuntimeError, 'mode differs'):
            self.remote('start')
        self.procs[50]['argv'] += ['--compile-step']
        (self.workspace/'manifest.json').write_text(json.dumps({'acceleration': {'compile_step': False}}))
        with self.assertRaisesRegex(RuntimeError, 'mode differs'):
            self.remote('start')
        (self.workspace/'manifest.json').write_text(json.dumps({'acceleration': {'compile_step': True}, 'effective_config': {'cpu_threads': 2}}))
        with self.assertRaisesRegex(RuntimeError, 'mode differs'):
            self.remote('start')

    def test_backfill_launch_flags_and_attach_require_matching_runtime_and_manifest(self):
        self.cfg.update(dispatch_policy='backfill', gpu_reserve_mib=8192)
        state, popen, _ = self.remote('start')
        self.assertEqual(state['action'], 'started')
        argv = popen.call_args.args[0]
        self.assertEqual(argv[argv.index('--dispatch-policy')+1], 'backfill')
        self.assertEqual(argv[argv.index('--gpu-reserve-mib')+1], '8192')
        state, popen, _ = self.remote('start')
        self.assertEqual(state['action'], 'attached'); popen.assert_not_called()
        self.cfg['dispatch_policy'] = 'barrier'
        with self.assertRaisesRegex(RuntimeError, 'mode differs'):
            self.remote('start')
        self.cfg['dispatch_policy'] = 'backfill'; self.cfg['gpu_reserve_mib'] = 4096
        with self.assertRaisesRegex(RuntimeError, 'mode differs'):
            self.remote('start')
        self.cfg['gpu_reserve_mib'] = 8192
        (self.workspace/'manifest.json').write_text(json.dumps({'dispatch_policy': 'barrier', 'gpu_reserve_mib': 8192}))
        with self.assertRaisesRegex(RuntimeError, 'mode differs'):
            self.remote('start')
        (self.workspace/'manifest.json').write_text(json.dumps({'dispatch_policy': 'backfill', 'gpu_reserve_mib': 4096}))
        with self.assertRaisesRegex(RuntimeError, 'mode differs'):
            self.remote('start')
        state, popen, kill = self.remote('status')
        self.assertTrue(state['scheduler_alive']); popen.assert_not_called(); kill.assert_not_called()

    def test_status_reads_double_case_path_and_detects_orphan_child(self):
        case = dict(id='core_n32_full_s11', seed=11, nodes=32, variant='full')
        self.procs[77] = dict(start='777', argv=['python', 'worker_entry.py'])
        status = dict(status='running', conditions=[dict(case=case, status='running', attempts=[dict(pid=77, process_start='777', worker=2)])])
        (self.workspace/'status.json').write_text(json.dumps(status))
        folder = self.workspace/'raw'/case['id']/case['id']; folder.mkdir(parents=True)
        (folder/'metrics.jsonl').write_text('{"step":4,"loss":3.2,"tokens_per_second":100}\n{"partial":')
        (folder.parent/'events.jsonl').write_text('{"stage":"validation"}\n')
        (folder/'events.jsonl').write_text('{"stage":"wrong-inner-directory"}\n')
        state, popen, kill = self.remote('status')
        self.assertEqual(state['conditions'][0]['step'], 4)
        self.assertEqual(state['conditions'][0]['stage'], 'validation')
        self.assertEqual(state['orphan_children'], [dict(id=case['id'], pid=77)])
        popen.assert_not_called(); kill.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, 'Old child'):
            self.remote('start')
        (folder/'acceleration.json').write_text('{"status":"compiling"}')
        compiled, _, _ = self.remote('status')
        self.assertEqual(compiled['conditions'][0]['stage'], '首次编译')
        self.assertEqual(compiled['conditions'][0]['acceleration_status'], 'compiling')
        (folder/'acceleration.json').write_text('{"status":"active"}')
        compiled, _, _ = self.remote('status')
        self.assertEqual(compiled['conditions'][0]['stage'], 'validation')
        (folder/'acceleration.json').write_text('{"status":"failed"}')
        compiled, _, _ = self.remote('status')
        self.assertEqual(compiled['conditions'][0]['stage'], '编译失败')

    def test_zombies_do_not_count_as_live_schedulers(self):
        self.procs[50] = dict(start='555', argv=self.scheduler_argv(), state='Z')
        state, popen, kill = self.remote('status')
        self.assertFalse(state['scheduler_alive']); popen.assert_not_called(); kill.assert_not_called()

    def test_follow_ctrl_c_never_requests_remote_pause(self):
        ssh = Mock(cfg={'workers': 5})
        ssh.request.return_value = dict(status='running', scheduler_alive=True, conditions=[])
        with patch.object(module, 'Progress'), patch.object(module.time, 'sleep', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                module.follow(ssh)
        self.assertEqual(ssh.request.call_args_list[0].args, ('status',))
        self.assertFalse(any(call.args[0] == 'pause' for call in ssh.request.call_args_list))

    def test_human_startup_is_flushed_before_connection_or_first_status_returns(self):
        class Output(io.StringIO):
            def __init__(self):
                super().__init__(); self.flushed = []
            def flush(self):
                self.flushed.append(self.getvalue()); super().flush()
        output = Output(); ssh = Mock(cfg=self.cfg)
        snapshot = dict(status='running', scheduler_alive=True, conditions=[])
        def assert_startup_visible():
            self.assertIn('[A800]', output.getvalue())
            self.assertTrue(any('[A800]' in text for text in output.flushed))
            self.assertNotIn(self.cfg['password'], output.getvalue())
        @contextmanager
        def fake_connection(path):
            assert_startup_visible()
            yield ssh
        def request(*args, **kwargs):
            assert_startup_visible()
            return snapshot
        ssh.request.side_effect = request
        with redirect_stdout(output), patch.object(module, 'connection', fake_connection), \
             patch.object(module, 'Progress'), patch.object(module.shutil, 'which', return_value='/usr/bin/ssh'):
            self.assertEqual(module.main(['--mode', 'follow', '--once']), 0)
        ssh.request.assert_called_once_with('status')

    def test_progress_exists_before_first_status_response(self):
        ssh = Mock(cfg=self.cfg)
        snapshot = dict(status='running', scheduler_alive=True, conditions=[])
        with patch.object(module, 'Progress') as progress:
            def request(*args, **kwargs):
                progress.assert_called_once()
                return snapshot
            ssh.request.side_effect = request
            self.assertEqual(module.follow(ssh, once=True), snapshot)
        progress.return_value.close.assert_called_once()

    def test_follow_recovers_visible_initial_connection_errors(self):
        for error in (RuntimeError('fixture SSH failure'), subprocess.TimeoutExpired(['ssh'], 60)):
            with self.subTest(error=type(error).__name__):
                ssh = Mock(cfg=self.cfg)
                ssh.redact.side_effect = lambda text: text.replace(self.cfg['password'], '[redacted]')
                snapshot = dict(status='completed', scheduler_alive=False, conditions=[])
                ssh.request.side_effect = [error, snapshot]
                with patch.object(module, 'Progress') as progress, patch.object(module.time, 'sleep') as sleep:
                    self.assertEqual(module.follow(ssh), snapshot)
                self.assertEqual([call.args for call in ssh.request.call_args_list], [('status',), ('status',)])
                sleep.assert_called_once()
                progress.return_value.connection.assert_called()
                self.assertRegex(progress.return_value.connection.call_args.args[0], '连接|重连|中断')
                progress.return_value.update.assert_called_once_with(snapshot)
                progress.return_value.close.assert_called_once()

    def test_once_and_json_report_initial_failure_without_retry(self):
        for flags in ({'once': True}, {'json_output': True}):
            with self.subTest(flags=flags):
                ssh = Mock(cfg=self.cfg); ssh.request.side_effect = RuntimeError('fixture SSH failure')
                with patch.object(module, 'Progress'), patch.object(module.time, 'sleep') as sleep:
                    with self.assertRaisesRegex(RuntimeError, 'fixture SSH failure'):
                        module.follow(ssh, **flags)
                ssh.request.assert_called_once_with('status'); sleep.assert_not_called()

    def test_json_mode_has_no_startup_or_progress_noise(self):
        ssh = Mock(cfg=self.cfg)
        snapshot = dict(status='running', scheduler_alive=True, conditions=[])
        ssh.request.return_value = snapshot
        @contextmanager
        def fake_connection(path):
            yield ssh
        output = io.StringIO()
        with redirect_stdout(output), patch.object(module, 'connection', fake_connection), \
             patch.object(module, 'Progress') as progress, patch.object(module.shutil, 'which', return_value='/usr/bin/ssh'):
            self.assertEqual(module.main(['--mode', 'follow', '--json']), 0)
        self.assertEqual(json.loads(output.getvalue()), snapshot)
        progress.assert_not_called()
        ssh.request.assert_called_once_with('status')

    def test_pycharm_and_regular_console_use_the_same_progress_controller(self):
        for hosted, isatty in (('1', True), ('1', False), ('0', True)):
            with self.subTest(hosted=hosted, isatty=isatty):
                ssh = Mock(cfg=self.cfg)
                snapshot = dict(status='running', scheduler_alive=True, conditions=[])
                ssh.request.return_value = snapshot
                output = Mock(); output.isatty.return_value = isatty
                with patch.dict(os.environ, {'PYCHARM_HOSTED': hosted}), patch.object(module.sys, 'stdout', output), \
                     patch.object(module, 'Progress') as progress:
                    self.assertEqual(module.follow(ssh, once=True), snapshot)
                progress.assert_called_once_with(self.cfg['workers'])
                progress.return_value.connection.assert_called_once()
                progress.return_value.update.assert_called_once_with(snapshot)
                progress.return_value.close.assert_called_once()
        self.assertFalse(hasattr(module, 'PyCharmProgress'))

    def test_fixed_bars_match_local_console_and_only_log_condition_transitions(self):
        row = dict(id='full-n128-s11', status='running', worker=0, step=100, stage='训练',
                   loss=2.5, tokens_per_second=100., case=dict(variant='full', nodes=128, seed=11))
        snapshot = dict(status='running', scheduler_alive=True, workers=12, conditions=[row])
        created = {}
        def new_bar(**options):
            bar = Mock(n=options.get('initial', 0), total=options['total'])
            created[options['position']] = bar
            return bar
        with patch('tqdm.tqdm', side_effect=new_bar) as tqdm, \
             patch.object(module.os, 'get_terminal_size', side_effect=OSError):
            progress = module.Progress(12)
            self.assertEqual(sorted(created), list(range(14)))
            for call in tqdm.call_args_list:
                options = call.kwargs
                self.assertEqual(options['leave'], False)
                self.assertEqual(options['dynamic_ncols'], False)
                self.assertEqual(options['mininterval'], 1.)
                self.assertEqual(options['maxinterval'], 5.)
                self.assertEqual(options['disable'], False)
                self.assertEqual(options['gui'], False)
                self.assertEqual(options['delay'], 0.)
                self.assertGreaterEqual(options['nrows'], 15)
                self.assertEqual(options['ncols'], 120)
                self.assertIs(options['file'], module.sys.stdout)
            for bar in created.values():
                self.assertTrue(callable(bar.dynamic_ncols))
            progress.update(snapshot)
            self.assertEqual(tqdm.write.call_count, 1)
            for step, stage in ((200, '训练'), (200, '验证'), (300, '训练')):
                row.update(step=step, stage=stage)
                progress.update(snapshot)
            self.assertEqual(tqdm.write.call_count, 1)
            row.update(status='completed')
            progress.update(snapshot); progress.update(snapshot)
            self.assertEqual(tqdm.write.call_count, 2)
            self.assertIn('完成 full-n128-s11', tqdm.write.call_args.args[0])
            self.assertEqual(progress.bar.n, 24000)
            self.assertEqual(progress.bar.total, 24000)
            self.assertEqual(progress.refresh_count, 6)
            self.assertEqual(tqdm.call_count, 14)
            progress.close()
            for bar in created.values():
                bar.close.assert_called_once()

    def test_real_detail_bars_repeated_updates_return_to_the_same_rows(self):
        output = io.StringIO()
        row = dict(id='full-n128-s11', status='running', worker=0, step=100, stage='训练',
                   loss=2.5, tokens_per_second=100., case=dict(variant='full', nodes=128, seed=11))
        snapshot = dict(status='running', scheduler_alive=True, workers=12, conditions=[row])
        with patch.object(module.sys, 'stdout', output):
            progress = module.Progress(12)
            progress.connection('正在连接 SSH 并读取训练进度…')
            self.assertIn('正在连接 SSH', output.getvalue())
            progress.update(snapshot)
            output.seek(0); output.truncate(0)
            for step, stage in ((200, '训练'), (200, '验证'), (300, '训练')):
                row.update(step=step, stage=stage)
                progress.update(snapshot)
            refreshed = output.getvalue()
            self.assertNotIn('more hidden', refreshed)
            self.assertNotIn('\x1b', refreshed.replace('\x1b[A', ''))
            cursor_row = maximum_row = 0
            for movement in re.findall(r'\x1b\[A|\n', refreshed):
                cursor_row += 1 if movement == '\n' else -1
                self.assertGreaterEqual(cursor_row, 0)
                maximum_row = max(maximum_row, cursor_row)
            self.assertEqual(cursor_row, 0)
            self.assertLessEqual(maximum_row, 13)
            self.assertIn('300/24000', refreshed)
            progress.close()

    def test_first_import_tqdm_environment_cannot_hide_or_move_the_bar(self):
        code = '''import importlib.util, io, json, sys
from contextlib import redirect_stdout
from pathlib import Path
spec = importlib.util.spec_from_file_location('fixture_controller', sys.argv[1])
controller = importlib.util.module_from_spec(spec); spec.loader.exec_module(controller)
controller.PROJECT = Path(sys.argv[2])
row = dict(id='full', status='running', worker=0, step=321, stage='训练',
           case=dict(variant='full', nodes=128, seed=11))
snapshot = dict(status='running', scheduler_alive=True, conditions=[row])
with redirect_stdout(io.StringIO()) as output:
    progress = controller.Progress(12)
    state = dict(disable=progress.bar.disable, gui=progress.bar.gui,
                 delay=progress.bar.delay, position=progress.bar.pos,
                 ncols=progress.bar.ncols, nrows=progress.bar.nrows)
    progress.connection('正在连接 SSH')
    progress.update(snapshot)
    initial = output.getvalue()
    output.seek(0); output.truncate(0)
    for step in (322, 323, 324):
        row['step'] = step; progress.update(snapshot)
    updates = output.getvalue()
    output.seek(0); output.truncate(0)
    progress.close()
    closed = output.getvalue()
print(json.dumps(dict(state=state, initial=initial, updates=updates, closed=closed)))
'''
        environment = dict(os.environ, PYCHARM_HOSTED='1', TQDM_DISABLE='1', TQDM_POSITION='7',
                           TQDM_GUI='1', TQDM_DELAY='3600', TQDM_NROWS='1', TQDM_NCOLS='0')
        result = subprocess.run([sys.executable, '-c', code, str(PROJECT/'run_biomatrix_v07_a800.py'), str(self.root)],
                                env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        rendered = json.loads(result.stdout)
        state = rendered['state']
        self.assertEqual((state['disable'], state['gui'], state['delay'], state['position']), (False, False, 0., 0))
        self.assertEqual(state['ncols'], 120)
        self.assertGreaterEqual(state['nrows'], 15)
        self.assertIn('正在连接 SSH', rendered['initial'])
        self.assertIn('|', rendered['initial'])
        self.assertIn('321/24000', rendered['initial'])
        self.assertIn('324/24000', rendered['updates'])
        for text in (rendered['updates'], rendered['closed']):
            self.assertNotIn('more hidden', text)
            self.assertNotIn('\x1b', text.replace('\x1b[A', ''))

    def test_default_entry_continuously_polls_three_times_until_local_interrupt(self):
        self.assertEqual(module.MODE, 'follow')
        ssh = Mock(cfg={'workers': 16})
        snapshots = [dict(status='running', scheduler_alive=True, workers=16, conditions=[], sampled_at=iteration)
                     for iteration in (1, 2, 3)]
        ssh.request.side_effect = snapshots
        @contextmanager
        def fake_connection(path):
            yield ssh
        with patch.object(module, 'connection', fake_connection), patch.object(module, 'Progress') as progress, \
             patch.object(module.time, 'sleep', side_effect=[None, None, KeyboardInterrupt]), \
             patch.object(module.shutil, 'which', return_value='/usr/bin/ssh'):
            with self.assertRaises(KeyboardInterrupt):
                module.main([])
        self.assertEqual(ssh.request.call_count, 3)
        self.assertEqual([call.args for call in ssh.request.call_args_list], [('status',)]*3)
        self.assertEqual([call.args[0]['sampled_at'] for call in progress.return_value.update.call_args_list], [1, 2, 3])
        progress.return_value.close.assert_called_once()

    def test_real_zero_size_pty_and_resize_keep_every_worker_visible_without_hanging(self):
        code = '''import fcntl, importlib.util, io, json, os, pty, select, struct, sys, termios, threading, time
from contextlib import redirect_stdout
spec = importlib.util.spec_from_file_location('fixture_controller', sys.argv[1])
controller = importlib.util.module_from_spec(spec); spec.loader.exec_module(controller)
master, slave = pty.openpty()
def resize(rows, columns):
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', rows, columns, 0, 0))
resize(int(sys.argv[2]), int(sys.argv[3]))
def drain():
    try:
        while True:
            ready, _, _ = select.select([master], [], [], .1)
            if ready and not os.read(master, 65536): return
    except (OSError, ValueError): return
thread = threading.Thread(target=drain, daemon=True); thread.start()
class Capture:
    def __init__(self):
        self.stream = os.fdopen(os.dup(slave), 'w', encoding='utf-8', buffering=1)
        self.parts = []
    def write(self, text):
        self.parts.append(text); return self.stream.write(text)
    def flush(self): self.stream.flush()
    def fileno(self): return self.stream.fileno()
    def isatty(self): return self.stream.isatty()
capture = Capture()
rows = [dict(id=f'full-n128-s{worker}', status='running', worker=worker, step=100+worker,
             stage='训练', loss=2.5+worker/100, tokens_per_second=100+worker,
             case=dict(variant='full', nodes=128, seed=worker)) for worker in range(12)]
snapshot = dict(status='running', scheduler_alive=True, workers=12, conditions=rows,
                gpu_telemetry=[dict(name='A800', utilization_percent=95., memory_used_mib=40960., memory_total_mib=81920.)])
phases = []
try:
    with redirect_stdout(capture):
        started = time.perf_counter(); progress = controller.Progress(12)
        progress.connection('正在连接 SSH 并读取训练进度…'); progress.update(snapshot)
        startup_seconds = time.perf_counter()-started
        from tqdm import tqdm
        bars = sorted(list(tqdm._instances), key=lambda bar: abs(bar.pos))
        for height, width in ((22,162), (0,0), (14,0), (22,162)):
            resize(height, width)
            first = len(capture.parts); started = time.perf_counter()
            for row in rows: row['step'] += 1
            progress.update(snapshot)
            phases.append(dict(seconds=time.perf_counter()-started,
                               size=[height,width], rendered=''.join(capture.parts[first:]),
                               bars=[dict(position=abs(bar.pos), ncols=bar.ncols, nrows=bar.nrows,
                                          disable=bar.disable, gui=bar.gui, n=bar.n,
                                          desc=bar.desc, postfix=bar.postfix) for bar in bars]))
        progress.close()
finally:
    capture.stream.close(); os.close(slave); os.close(master); thread.join(timeout=1)
print(json.dumps(dict(startup_seconds=startup_seconds, phases=phases)))
'''
        for height, width in ((0, 0), (14, 0), (14, 1), (14, 2), (14, 20)):
            with self.subTest(initial_height=height, initial_width=width):
                result = subprocess.run([sys.executable, '-c', code, str(PROJECT/'run_biomatrix_v07_a800.py'), str(height), str(width)],
                                        env=dict(os.environ, PYCHARM_HOSTED='0'), capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout)
                self.assertLess(report['startup_seconds'], 2.)
                for phase in report['phases']:
                    self.assertLess(phase['seconds'], 2.)
                    self.assertEqual([bar['position'] for bar in phase['bars']], list(range(14)))
                    for bar in phase['bars']:
                        self.assertFalse(bar['disable']); self.assertFalse(bar['gui'])
                        self.assertEqual(bar['ncols'], 161 if phase['size'][1] == 162 else 120)
                        self.assertGreaterEqual(bar['nrows'], 15)
                    rendered = phase['rendered']
                    self.assertNotIn('more hidden', rendered)
                    self.assertNotIn('\x1b[2J', rendered)
                    self.assertNotIn('\x1b[H', rendered)
                    self.assertNotIn('\x1b', rendered.replace('\x1b[A', ''))
                    cursor_row = maximum_row = 0
                    for movement in re.findall(r'\x1b\[A|\n', rendered):
                        cursor_row += 1 if movement == '\n' else -1
                        self.assertGreaterEqual(cursor_row, 0)
                        maximum_row = max(maximum_row, cursor_row)
                    self.assertEqual(cursor_row, 0)
                    self.assertLessEqual(maximum_row, 13)
                for worker, bar in enumerate(report['phases'][-1]['bars'][2:]):
                    self.assertEqual(bar['n'], 104+worker)
                    self.assertIn('full', bar['desc'])
                    self.assertIn('128', bar['desc'])
                    self.assertIn(str(worker), bar['desc'])
                    self.assertIn(f'loss={2.5+worker/100:.4f}', bar['postfix'])
                    self.assertIn(f'{100+worker:.1f} target/s', bar['postfix'])
                    self.assertIn('训练', bar['postfix'])

    def test_explicit_status_fetches_exactly_one_snapshot(self):
        ssh = Mock(cfg={'workers': 5})
        ssh.request.return_value = dict(status='running', scheduler_alive=True, conditions=[])
        @contextmanager
        def fake_connection(path):
            yield ssh
        with patch.object(module, 'connection', fake_connection), patch.object(module, 'Progress'), \
             patch.object(module.time, 'sleep') as sleep, patch.object(module.shutil, 'which', return_value='/usr/bin/ssh'):
            self.assertEqual(module.main(['--mode', 'status']), 0)
        ssh.request.assert_called_once_with('status'); sleep.assert_not_called()

    def test_default_one_second_cadence_subtracts_request_and_render_time(self):
        self.assertEqual(module.POLL_SECONDS, 1.)
        for elapsed, expected_sleep in ((.2, .8), (1.2, 0.)):
            with self.subTest(request_and_render_seconds=elapsed):
                ssh = Mock(cfg=self.cfg)
                snapshots = [dict(status='running', scheduler_alive=True, conditions=[]),
                             dict(status='completed', scheduler_alive=False, conditions=[])]
                ssh.request.side_effect = snapshots
                with patch.object(module, 'Progress') as progress, \
                     patch.object(module.time, 'monotonic', side_effect=[10., 10.+elapsed, 12.]), \
                     patch.object(module.time, 'sleep') as sleep:
                    self.assertEqual(module.follow(ssh), snapshots[-1])
                sleep.assert_called_once()
                self.assertAlmostEqual(sleep.call_args.args[0], expected_sleep)
                self.assertEqual(ssh.request.call_count, 2)
                self.assertEqual(progress.return_value.update.call_count, 2)
                progress.return_value.close.assert_called_once()

    def test_remote_gpu_telemetry_is_read_only_and_tolerates_unavailable_fields(self):
        self.gpu_result = subprocess.CompletedProcess([], 0, 'NVIDIA A800 80GB PCIe, 91, 16384, 81920, 280.4, 300, 55\n', '')
        snapshot, popen, kill = self.remote('status')
        query = self.gpu_query
        query.assert_called_once()
        self.assertIn('--format=csv,noheader,nounits', query.call_args.args[0])
        self.assertEqual(query.call_args.kwargs['timeout'], 3)
        gpu = snapshot['gpu_telemetry'][0]
        self.assertEqual(gpu['utilization_percent'], 91.)
        self.assertEqual(gpu['memory_used_mib'], 16384.)
        self.assertGreater(snapshot['sampled_at'], 0)
        popen.assert_not_called(); kill.assert_not_called()
        self.gpu_error = FileNotFoundError
        unavailable, _, _ = self.remote('status')
        self.assertEqual(unavailable['gpu_telemetry'], [])

    def test_live_progress_updates_total_resources_and_each_worker(self):
        row = dict(status='running', worker=15, step=123, loss=2.5, tokens_per_second=321., stage='训练',
                   case=dict(variant='full', nodes=128, seed=11))
        snapshot = dict(status='running', scheduler_alive=True, workers=16, conditions=[row],
                        gpu_telemetry=[dict(name='A800', utilization_percent=95., memory_used_mib=40960.,
                                            memory_total_mib=81920., power_watts=260., temperature_c=60.)])
        created = {}
        def new_bar(**options):
            bar = Mock(n=options.get('initial', 0), total=options['total'])
            created[options['position']] = bar
            return bar
        with patch('tqdm.tqdm', side_effect=new_bar) as bars:
            progress = module.Progress(5)
            progress.update(snapshot); progress.update(snapshot)
            self.assertEqual(bars.call_count, 18)
            self.assertEqual(sorted(created), list(range(18)))
            self.assertEqual(progress.workers, 16)
            self.assertEqual(progress.refresh_count, 2)
            self.assertEqual(progress.bar.n, 123)
            self.assertEqual(progress.bar.total, 24000)
            self.assertIn('A800 总训练', progress.bar.set_description_str.call_args.args[0])
            resources = ' '.join(str(call.args[0]) for bar in (created[0], created[1])
                                 for method in (bar.set_description_str, bar.set_postfix_str)
                                 for call in method.call_args_list)
            for fragment in ('完成0/1', '活跃1/16', '321 target/s', 'GPU95%', '40.0/80G'):
                self.assertIn(fragment, resources)
            self.assertRegex(resources, r'\d{2}:\d{2}:\d{2}')
            self.assertEqual(progress.bar.refresh.call_count, 2)
            detail = created[17]
            self.assertEqual(detail.n, 123)
            self.assertEqual(detail.total, 24000)
            self.assertEqual(detail.refresh.call_count, 2)
            detail_text = ' '.join(str(call.args[0]) for method in (detail.set_description_str, detail.set_postfix_str)
                                   for call in method.call_args_list)
            for fragment in ('full', '128', '11', '训练', '2.500', '321'):
                self.assertIn(fragment, detail_text)
            progress.close()
            for bar in created.values():
                bar.close.assert_called_once()

    def test_modes_require_explicit_resume_and_retry(self):
        for mode, resume, retry in [('start', False, False), ('resume', True, False), ('retry_failed', False, True)]:
            ssh = Mock(cfg={'workers': 5})
            ssh.request.return_value = dict(status='running', scheduler_alive=True, conditions=[])
            @contextmanager
            def fake_connection(path):
                yield ssh
            with patch.object(module, 'connection', fake_connection), patch.object(module, 'Progress'), patch.object(module.shutil, 'which', return_value='/usr/bin/ssh'):
                self.assertEqual(module.main(['--mode', mode, '--once']), 0)
            ssh.request.assert_called_once_with('start', resume_aborted=resume, retry_failed=retry)

    def test_fetch_preserves_prior_archives_and_uses_no_remote_signal(self):
        destination = self.root/'archive.tar.gz'; destination.write_bytes(b'old')
        with module.connection(self.config_path) as ssh:
            with self.assertRaises(FileExistsError):
                ssh.fetch(destination)
            self.assertEqual(destination.read_bytes(), b'old')
            target = self.root/'new.tar.gz'; child = Mock(stdout=io.BytesIO(b'archive fixture'))
            child.wait.return_value = 0
            with patch.object(module.subprocess, 'Popen', return_value=child) as popen, patch('tqdm.tqdm'):
                self.assertEqual(ssh.fetch(target), target)
            self.assertEqual(target.read_bytes(), b'archive fixture')
            self.assertEqual(popen.call_args.kwargs['stdin'], subprocess.DEVNULL)
            child.terminate.assert_not_called()

    def test_fetch_failure_retains_partial_archive(self):
        destination = self.root/'failed.tar.gz'
        def failed_transfer(*args, **kwargs):
            kwargs['stderr'].write(b'fixture permission failure')
            child = Mock(stdout=io.BytesIO(b'partial archive'))
            child.wait.return_value = 1; child.poll.return_value = 1
            return child
        with module.connection(self.config_path) as ssh:
            with patch.object(module.subprocess, 'Popen', side_effect=failed_transfer), patch('tqdm.tqdm'):
                with self.assertRaisesRegex(RuntimeError, 'fixture permission failure'):
                    ssh.fetch(destination)
        self.assertFalse(destination.exists())
        self.assertEqual(destination.with_suffix('.gz.part').read_bytes(), b'partial archive')

    def test_early_scheduler_exit_is_launch_failure_but_empty_status_is_readable(self):
        ssh = Mock(cfg={'workers': 5})
        ssh.request.return_value = dict(status='not_executed', scheduler_alive=False, conditions=[])
        @contextmanager
        def fake_connection(path):
            yield ssh
        with patch.object(module, 'connection', fake_connection), patch.object(module, 'Progress'), patch.object(module.shutil, 'which', return_value='/usr/bin/ssh'):
            self.assertEqual(module.main(['--mode', 'start', '--once']), 1)
            self.assertEqual(module.main(['--mode', 'status', '--once']), 0)

    def test_real_remote_tar_pins_open_inode_and_skips_temporary_missing_and_symlink_entries(self):
        folder = self.workspace/'raw'/'case'/'case'; folder.mkdir(parents=True)
        migrations = self.workspace/'scheduling_migrations'; migrations.mkdir()
        (migrations/'manifest.before.json').write_text('{"dispatch_policy":"barrier"}')
        (migrations/'receipt.json').write_text('{"status":"migrated"}')
        adoptions = self.workspace/'live_adoptions'; adoptions.mkdir()
        (adoptions/'receipt.json').write_text('{"workers":14,"training_processes_preserved":true}')
        (self.workspace/'backfill_before.json').write_text('{"workers":5}')
        checkpoint = folder/'last.pt'; old = b'old-inode-checkpoint\0'*4096; new = b'new-inode\0'*17
        checkpoint.write_bytes(old); (folder/'last.pt.tmp').write_bytes(b'temporary')
        temporary = folder/'pending.tmp'; temporary.mkdir(); (temporary/'never.pt').write_bytes(b'temporary directory')
        disappearing = folder/'vanishing.pt'; disappearing.write_bytes(b'gone during walk')
        outside = self.root/'outside'; outside.mkdir(); (outside/'not-exported').write_bytes(b'external fixture')
        (folder/'external-file').symlink_to(outside/'not-exported')
        (folder/'external-directory').symlink_to(outside, target_is_directory=True)
        replacement = folder/'replacement.tmp'; swapped = []
        original_header = tarfile.TarFile.gettarinfo; original_open = os.open
        def header_after_replacement(archive, name=None, arcname=None, fileobj=None):
            if arcname == 'raw/case/case/last.pt':
                self.assertIsNotNone(fileobj)
                replacement.write_bytes(new); replacement.replace(checkpoint); swapped.append(True)
            return original_header(archive, name=name, arcname=arcname, fileobj=fileobj)
        def open_after_disappearance(path, flags, *args, **kwargs):
            if path == 'vanishing.pt':
                os.unlink(path, dir_fd=kwargs['dir_fd'])
            return original_open(path, flags, *args, **kwargs)
        payload = dict(operation='fetch', workspace=str(self.workspace), release=str(self.release), scheduler=str(self.scheduler))
        output = io.BytesIO()
        with patch.object(sys, 'argv', ['remote', base64.b64encode(json.dumps(payload).encode()).decode()]), \
             patch.object(tarfile.TarFile, 'gettarinfo', header_after_replacement), \
             patch('os.open', side_effect=open_after_disappearance), redirect_stdout(SimpleNamespace(buffer=output)):
            with self.assertRaises(SystemExit) as ended:
                exec(compile(module.REMOTE, '<real-local-remote-fetch>', 'exec'), {})
        self.assertEqual(ended.exception.code, 0); self.assertEqual(swapped, [True])
        self.assertEqual(checkpoint.read_bytes(), new)
        output.seek(0)
        with tarfile.open(fileobj=output, mode='r:gz') as archive:
            names = archive.getnames()
            self.assertEqual(archive.extractfile('raw/case/case/last.pt').read(), old)
            self.assertEqual(archive.getmember('raw/case/case/last.pt').size, len(old))
            self.assertEqual(json.load(archive.extractfile('scheduling_migrations/manifest.before.json')), {'dispatch_policy': 'barrier'})
            self.assertEqual(json.load(archive.extractfile('scheduling_migrations/receipt.json')), {'status': 'migrated'})
            self.assertEqual(json.load(archive.extractfile('live_adoptions/receipt.json')), {'workers': 14, 'training_processes_preserved': True})
            self.assertEqual(json.load(archive.extractfile('backfill_before.json')), {'workers': 5})
            self.assertFalse(any(name.endswith('.tmp') or 'pending.tmp/' in name or 'external-' in name or name.endswith('vanishing.pt')
                                 for name in names))
        def refuse_checkpoint(path, flags, *args, **kwargs):
            if path == 'last.pt':
                raise PermissionError('fixture non-ENOENT failure')
            return original_open(path, flags, *args, **kwargs)
        with patch.object(sys, 'argv', ['remote', base64.b64encode(json.dumps(payload).encode()).decode()]), \
             patch('os.open', side_effect=refuse_checkpoint), redirect_stdout(SimpleNamespace(buffer=io.BytesIO())):
            with self.assertRaisesRegex(PermissionError, 'non-ENOENT'):
                exec(compile(module.REMOTE, '<real-local-remote-fetch-error>', 'exec'), {})

    def test_pycharm_uses_local_python_and_emulated_terminal_without_secrets(self):
        path = PROJECT/'.run/Bio-Matrix v0.7 A800.run.xml'; root = ET.fromstring(path.read_text())
        options = {row.attrib['name']: row.attrib['value'] for row in root.findall('.//option')}
        self.assertEqual(options['SCRIPT_NAME'], '$PROJECT_DIR$/run_biomatrix_v07_a800.py')
        self.assertEqual(options['EMULATE_TERMINAL'], 'true')
        self.assertEqual(options['PARAMETERS'], '--mode follow')
        self.assertNotIn(self.cfg['password'], path.read_text())


if __name__ == '__main__':
    unittest.main()
