"""Comparison orchestration tests: mocked local children, no GPU or training."""
import argparse
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from v07.common import write
from . import experiment as queue


class FakeDisplay:
    def __init__(self, *args):
        self.updates = []

    def refresh(self, *args):
        self.updates.append(args)

    def close(self):
        pass


class Children:
    """Produce the same disk contract as train/evaluate without executing them."""
    def __init__(self, tokens=256, fail_stage=None):
        self.tokens = tokens; self.fail_stage = fail_stage; self.calls = []

    def __call__(self, arguments, **options):
        self.calls.append(arguments)
        stage = 'training' if arguments[2] == 'v09.train' else 'evaluation'
        if '--resume' in arguments:
            folder = Path(arguments[arguments.index('--resume')+1]).parent
        else:
            folder = Path(arguments[arguments.index('--output')+1])
        if stage == 'training' and '--resume' not in arguments:
            if folder.exists() and any(folder.iterdir()):
                raise AssertionError('parent polluted the fresh train output before launch')
        folder.mkdir(parents=True, exist_ok=True)
        failed = stage == self.fail_stage
        if stage == 'training':
            (folder/'last.pt').write_bytes(b'checkpoint preserved')
            write(folder/'status.json', dict(status='failed' if failed else 'completed',
                                           completed_tokens=64 if failed else self.tokens))
        else:
            write(folder/'result.json', dict(status='failed' if failed else 'completed', windows=1))
        options['stdout'].write((stage+' child output\n').encode())
        return SimpleNamespace(pid=123456, poll=lambda: 3 if failed else 0,
                               send_signal=lambda sig: None)


class ExperimentChecks(unittest.TestCase):
    def arguments(self, output, **options):
        values = dict(output=Path(output), data_root=Path(output)/'data', tokens=256,
                      nodes=[32], seeds=[11], conditions=['content_only', 'v09_dense'],
                      evaluation_windows=1, skip_evaluation=False, plan_only=False,
                      max_jobs=None, retry_failed=False, quiet=True)
        values.update(options)
        return argparse.Namespace(**values)

    def execute(self, args, children):
        with patch.object(queue, 'source_manifest', return_value={}), \
                patch.object(queue, 'data_manifest', return_value={'manifest_sha256': 'frozen'}), \
                patch.object(queue, 'data_manifest_hash', return_value='frozen'), \
                patch.object(queue.torch.backends.mps, 'is_available', return_value=True), \
                patch.object(queue, 'Display', FakeDisplay), \
                patch.dict(queue.os.environ, {'PYTORCH_ENABLE_MPS_FALLBACK': '0'}):
            return queue.execute(args, popen=children, sleep=lambda _: None)

    def test_default_plan_deduplicates_only_content_and_preserves_pair_order(self):
        layout = queue.plan()
        self.assertEqual(len(layout['cells']), 36)
        self.assertEqual(len(layout['jobs']), 30)
        self.assertEqual(sum(row['condition'] == 'content_only' for row in layout['jobs']), 3)
        self.assertEqual(sum(row['reference'] for row in layout['cells']), 6)
        self.assertEqual([row['condition'] for row in layout['jobs'][:4]], list(queue.DEFAULT_CONDITIONS))
        self.assertEqual({row['nodes'] for row in layout['jobs'][:4]}, {32})
        self.assertEqual({row['seed'] for row in layout['jobs'][:10]}, {11})
        self.assertEqual(sum(row['tokens'] for row in layout['jobs']), 360000000)
        self.assertFalse(any(row['condition'] == 'v08_seq' for row in layout['jobs']))

    def test_plan_only_never_launches_and_freezes_source_and_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.arguments(temporary, plan_only=True)
            children = Children()
            root, registry = self.execute(args, children)
            self.assertFalse(children.calls)
            self.assertTrue((root/'manifest.json').is_file())
            self.assertTrue((root/'REPORT.md').is_file())
            self.assertTrue(all(row['status'] == 'not_executed' for row in registry['jobs']))
            args.tokens = 512
            with self.assertRaisesRegex(ValueError, 'protocol/source/data changed'):
                self.execute(args, children)

    def test_existing_manifest_rejects_source_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            protocol = dict(source_hashes={}, data={'sha256': 'a'}, config={'tokens': 256})
            layout = queue.plan(['content_only'], [32], [11], 256)
            queue.initialize(temporary, protocol, layout)
            with self.assertRaisesRegex(ValueError, 'source_hashes'):
                queue.initialize(temporary, protocol | {'source_hashes': {'changed.py': 'other'}}, layout)

    def test_five_detail_bars_show_canonical_reference_without_double_counting_tokens(self):
        class Bar:
            def __init__(self, total=0, desc='', **options):
                self.total = total; self.n = 0; self.desc = desc; self.postfix = ''
                self.bar_format = options.get('bar_format', '')
            def set_postfix_str(self, value, **options): self.postfix = value
            def set_description_str(self, value, **options): self.desc = value
            def refresh(self): pass
            def close(self): pass

        registry = queue.plan(queue.CONDITIONS)
        content = next(job for job in registry['jobs'] if job['condition'] == 'content_only')
        content.update(status='completed', training_status='completed', evaluation_status='completed', completed_tokens=12000000)
        current = next(job for job in registry['jobs'] if job['condition'] == 'v09_dense' and job['nodes'] == 128)
        current['stage'] = 'evaluation'
        with patch.object(queue, 'tqdm', Bar), patch.object(queue, 'resource_line', return_value='local'):
            ui = queue.Display(registry)
            ui.refresh(registry, current, {'evaluation': {'phase': 'related', 'completed': 3, 'total': 64}})
        self.assertEqual(ui.total.n, 12000000)
        self.assertEqual(ui.cells.n, 3)
        self.assertEqual(len(ui.details), 5)
        self.assertEqual(ui.details['content_only'].n, 12000000)
        self.assertIn('引用 N32', ui.details['content_only'].desc)
        self.assertIn('related 3/64', ui.stage.desc)
        for bar in (ui.total, ui.cells, *ui.details.values()):
            self.assertNotIn('rate', bar.bar_format)
            self.assertNotIn('remaining', bar.bar_format)
        current.update(status='completed', completed_tokens=current['tokens'])
        with patch.object(queue, 'resource_line', return_value='local'):
            ui.refresh(registry)
        self.assertTrue(ui.details['v09_dense'].postfix.startswith('completed'))
        custom = queue.plan(['content_only'], [128], [11], 256)
        self.assertEqual(custom['jobs'][0]['nodes'], 32)
        self.assertEqual(custom['jobs'][0]['display_nodes'], 128)

    def test_default_display_omits_v08_and_keeps_positions_contiguous(self):
        bars = []

        class Bar:
            def __init__(self, **options):
                self.options = options; bars.append(self)

        with patch.object(queue, 'tqdm', Bar):
            ui = queue.Display(queue.plan())
        self.assertEqual(list(ui.details), list(queue.DEFAULT_CONDITIONS))
        self.assertEqual([bar.options['position'] for bar in bars], list(range(8)))
        self.assertEqual(bars[0].options['total'], 360000000)
        self.assertEqual(bars[1].options['total'], 36)

    def test_local_launch_saves_all_logs_and_completed_jobs_are_not_retrained(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.arguments(temporary)
            children = Children()
            root, registry = self.execute(args, children)
            self.assertEqual(registry['status'], 'completed')
            self.assertEqual(len(children.calls), 4)
            for arguments in children.calls:
                self.assertEqual(arguments[0], str(queue.PROJECT/'.venv/bin/python'))
                self.assertEqual(arguments[arguments.index('--device')+1], 'mps')
                self.assertFalse(any('ssh' in item or 'A800' in item for item in arguments))
            for job in registry['jobs']:
                folder = root/job['id']
                self.assertIn('training child output', (folder/'console.log').read_text())
                self.assertIn('evaluation child output', (folder/'console_eval.log').read_text())
                self.assertTrue((folder/'last.pt').is_file())
                self.assertEqual(job['evaluation_status'], 'completed')
            again = Children()
            self.execute(args, again)
            self.assertFalse(again.calls)

    def test_four_family_queue_launches_all_30_jobs_without_v08(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.arguments(temporary, conditions=list(queue.DEFAULT_CONDITIONS),
                                  nodes=list(queue.NODES), seeds=list(queue.SEEDS))
            children = Children()
            _, registry = self.execute(args, children)
            training = [call for call in children.calls if call[2] == 'v09.train']
            self.assertEqual(registry['status'], 'completed')
            self.assertEqual(len(registry['cells']), 36)
            self.assertEqual(len(training), 30)
            self.assertEqual(len(children.calls), 60)
            self.assertTrue(all('v08_seq' not in call for call in training))
            self.assertEqual(sum(call[call.index('--condition')+1] == 'content_only'
                                 for call in training), 3)

    def test_training_failure_is_preserved_and_retry_uses_existing_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.arguments(temporary, conditions=['v09_dense'])
            root, registry = self.execute(args, Children(fail_stage='training'))
            self.assertEqual(registry['jobs'][0]['status'], 'failed')
            skipped = Children()
            self.execute(args, skipped)
            self.assertFalse(skipped.calls)
            args.retry_failed = True
            children = Children()
            _, restored = self.execute(args, children)
            self.assertEqual(restored['status'], 'completed')
            self.assertIn('--resume', children.calls[0])
            self.assertNotIn('--tokens', children.calls[0])
            self.assertNotIn('--condition', children.calls[0])
            self.assertGreater((root/restored['jobs'][0]['id']/'console.log').stat().st_size, 0)

    def test_evaluation_failure_never_discards_completed_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.arguments(temporary, conditions=['v09_dense'])
            root, registry = self.execute(args, Children(fail_stage='evaluation'))
            job = registry['jobs'][0]
            self.assertEqual(job['training_status'], 'completed')
            self.assertEqual(job['evaluation_status'], 'failed')
            self.assertEqual(job['status'], 'failed')
            args.retry_failed = True
            children = Children()
            _, registry = self.execute(args, children)
            self.assertEqual(len(children.calls), 1)
            self.assertEqual(children.calls[0][2], 'v09.evaluate')
            self.assertEqual(registry['status'], 'completed')
            self.assertEqual(len(registry['jobs'][0]['attempts']), 1)
            self.assertTrue(Path(registry['jobs'][0]['attempts'][0]['path']).is_dir())
            self.assertTrue((root/job['id']/'last.pt').is_file())

    def test_soft_stop_waits_for_child_checkpoint_and_records_aborted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = queue.plan(['v09_dense'], [32], [11], 256)
            registry['status'] = 'running'
            job = registry['jobs'][0]
            stop = SimpleNamespace(requested=False)
            signals = []

            def launch(arguments, **options):
                folder = root/job['id']; folder.mkdir()
                calls = []

                def poll():
                    if not calls:
                        calls.append(1); stop.requested = True
                        return None
                    return 0

                def interrupt(sig):
                    signals.append(sig)
                    (folder/'last.pt').write_bytes(b'last safe update')
                    write(folder/'status.json', dict(status='aborted', completed_tokens=64))

                return SimpleNamespace(pid=123456, poll=poll, send_signal=interrupt)

            config = {'evaluation_windows': 1, 'data_root': 'unused', 'evaluate_on_complete': True}
            outcome = queue.run_child(root, registry, job, config, 'training', FakeDisplay(), stop,
                                      popen=launch, sleep=lambda _: None)
            self.assertEqual(outcome, 'aborted')
            self.assertEqual(signals, [signal.SIGINT])
            self.assertEqual(job['completed_tokens'], 64)
            self.assertTrue((root/job['id']/'last.pt').is_file())

    def test_aborted_training_resumes_and_max_jobs_limits_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.arguments(temporary, max_jobs=1)
            children = Children()
            root, registry = self.execute(args, children)
            self.assertEqual(registry['jobs'][0]['status'], 'completed')
            self.assertEqual(registry['jobs'][1]['status'], 'not_executed')
            self.assertEqual(len(children.calls), 2)
            job = registry['jobs'][1]; folder = root/job['id']; folder.mkdir()
            (folder/'last.pt').write_bytes(b'safe')
            write(folder/'status.json', dict(status='aborted', completed_tokens=64))
            job.update(status='aborted', training_status='aborted', completed_tokens=64)
            queue.persist(root, registry)
            resumed = Children()
            _, registry = self.execute(args, resumed)
            self.assertIn('--resume', resumed.calls[0])
            self.assertEqual(registry['status'], 'completed')

    def test_partial_metrics_and_live_validation_feed_real_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root/'status.json', {'status': 'running', 'completed_tokens': 64})
            (root/'metrics.jsonl').write_text('{"completed_tokens":128,"loss":2.5}\n{"completed_tokens":')
            write(root/'live.json', {'phase': 'validation', 'completed_tokens': 128,
                                    'validation': {'completed': 2, 'total': 128, 'nll': 3.}})
            snapshot = queue.progress(root)
            self.assertEqual(snapshot['completed_tokens'], 128)
            self.assertEqual(snapshot['last_safe_tokens'], 64)
            self.assertEqual(snapshot['loss'], 2.5)
            self.assertEqual(snapshot['validation']['completed'], 2)

    def test_spawn_error_is_a_saved_failure_without_destroying_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.arguments(temporary, conditions=['v09_dense'])

            def missing_python(*args, **kwargs):
                raise FileNotFoundError('local interpreter missing')

            root, registry = self.execute(args, missing_python)
            job = registry['jobs'][0]
            self.assertEqual(job['status'], 'failed')
            self.assertEqual(job['failure_stage'], 'training')
            self.assertIn('local interpreter missing', (root/job['id']/'console.log').read_text())

    def test_parent_death_between_training_and_evaluation_only_starts_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.arguments(temporary, conditions=['v09_dense'], plan_only=True)
            root, registry = self.execute(args, Children())
            job = registry['jobs'][0]; folder = root/job['id']; folder.mkdir()
            (folder/'last.pt').write_bytes(b'completed training')
            write(folder/'status.json', dict(status='completed', completed_tokens=256))
            job.update(status='completed', training_status='completed')
            queue.persist(root, registry)
            args.plan_only = False
            children = Children()
            _, registry = self.execute(args, children)
            self.assertEqual([call[2] for call in children.calls], ['v09.evaluate'])
            self.assertEqual(registry['status'], 'completed')

    def test_source_edit_after_first_job_stops_before_any_other_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.arguments(temporary)
            changed = []
            children = Children()

            def launch(arguments, **options):
                child = children(arguments, **options)
                if arguments[2] == 'v09.evaluate': changed.append(True)
                return child

            with patch.object(queue, 'source_manifest', side_effect=lambda: {'edited.py': 'new'} if changed else {}), \
                    patch.object(queue, 'data_manifest', return_value={'manifest_sha256': 'frozen'}), \
                    patch.object(queue, 'data_manifest_hash', return_value='frozen'), \
                    patch.object(queue.torch.backends.mps, 'is_available', return_value=True), \
                    patch.object(queue, 'Display', FakeDisplay), \
                    patch.dict(queue.os.environ, {'PYTORCH_ENABLE_MPS_FALLBACK': '0'}):
                with self.assertRaisesRegex(RuntimeError, 'source changed during execution'):
                    queue.execute(args, popen=launch, sleep=lambda _: None)
            self.assertEqual(len(children.calls), 2)
            saved = queue.read_json(Path(temporary)/'registry.json')
            self.assertEqual(saved['status'], 'failed')
            self.assertIn('edited.py', saved['error'])
            self.assertEqual(saved['jobs'][1]['status'], 'not_executed')
            self.assertTrue((Path(temporary)/saved['jobs'][0]['id']/'last.pt').is_file())

    def test_unexpected_child_exception_persists_failed_queue_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.arguments(temporary)
            with patch.object(queue, 'run_child', side_effect=RuntimeError('injected monitor failure')):
                with self.assertRaisesRegex(RuntimeError, 'injected monitor failure'):
                    self.execute(args, Children())
            saved = queue.read_json(Path(temporary)/'registry.json')
            self.assertEqual(saved['status'], 'failed')
            self.assertIn('injected monitor failure', saved['error'])

    def test_data_manifest_edit_stops_before_first_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.arguments(temporary)
            children = Children()
            with patch.object(queue, 'source_manifest', return_value={}), \
                    patch.object(queue, 'data_manifest', return_value={'manifest_sha256': 'old'}), \
                    patch.object(queue, 'data_manifest_hash', return_value='changed'), \
                    patch.object(queue.torch.backends.mps, 'is_available', return_value=True), \
                    patch.object(queue, 'Display', FakeDisplay), \
                    patch.dict(queue.os.environ, {'PYTORCH_ENABLE_MPS_FALLBACK': '0'}):
                with self.assertRaisesRegex(RuntimeError, 'data manifest changed'):
                    queue.execute(args, popen=children, sleep=lambda _: None)
            self.assertFalse(children.calls)
            self.assertEqual(queue.read_json(Path(temporary)/'registry.json')['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
