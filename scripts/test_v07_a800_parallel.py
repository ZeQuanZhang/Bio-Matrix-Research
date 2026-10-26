"""CPU scheduler tests. Fake runner imports no torch and performs zero ML updates."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).with_name('v07_a800_parallel.py')
spec = importlib.util.spec_from_file_location('a800_scheduler', SCRIPT)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

FAKE = '''"""Fake CPU-only runner for scheduler control-flow tests; no torch, no training."""
import argparse, fcntl, json, os, pathlib, signal, time
from contextlib import ExitStack
from .common import PROJECT
def digest(v):
    import hashlib
    return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def sha(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()
p=argparse.ArgumentParser()
for name in ('config','root','data-root','device','mode'):p.add_argument('--'+name,required=True)
a=p.parse_args(); cfg=json.loads(pathlib.Path(a.config).read_text())
seed,nodes,variant=cfg['seeds'][0],cfg['nodes'][0],cfg['variants'][0]
case=dict(id=f'core_n{nodes}_{variant}_s{seed}',stage='core',nodes=nodes,variant=variant,seed=seed)
folder=pathlib.Path(a.root)/case['id'];folder.mkdir(parents=True,exist_ok=True)
def event(kind):
    if not os.getenv('FAKE_EVENTS'):return
    fd=os.open(os.environ['FAKE_EVENTS'],os.O_APPEND|os.O_CREAT|os.O_WRONLY,0o644)
    os.write(fd,(json.dumps(dict(kind=kind,case=case,time=time.time()))+'\\n').encode());os.close(fd)
stop=False
def request(*args):
    global stop
    stop=True
signal.signal(signal.SIGTERM,request);signal.signal(signal.SIGINT,request)
with ExitStack() as stack:
    for name in ('v06','v07'):
        path=PROJECT/'results'/name/'.queue.lock';path.parent.mkdir(parents=True,exist_ok=True)
        handle=stack.enter_context(path.open('a+'));fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    event('start');started=time.monotonic()
    while time.monotonic()-started<float(os.getenv('FAKE_DURATION','.005')) and not stop:time.sleep(.005)
    (folder/'last.pt').write_text('fake checkpoint; zero ML updates')
    if stop:event('aborted');raise SystemExit(0)
    if os.getenv('FAKE_FAIL')==case['id']:event('failed');raise SystemExit(17)
    paths=list((PROJECT/'v07').glob('*.py'))+[PROJECT/'v04/model.py',PROJECT/'v05/model.py',PROJECT/'v05/run.py']
    sources={str(path.relative_to(PROJECT)):sha(path) for path in paths if not path.name.startswith('test')}
    effective=cfg|{key:value for key,value in case.items() if key in cfg}
    manifest=json.loads((pathlib.Path(a.data_root)/'manifest.json').read_text())
    protocol=dict(config=effective,source_hashes=sources,data_manifest_sha256=digest(manifest),device='cuda',dtype='float32',torch='fake-no-torch')
    (folder/'protocol.json').write_text(json.dumps(protocol))
    (folder/'result.json').write_text(json.dumps(dict(status='completed',case=case,protocol_sha256=digest(protocol),fake=True,ml_updates=0)))
    event('completed')
'''


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.root = Path(self.temporary.name)
        self.release = self.root/'release'; self.workspace = self.root/'workspace'
        for package in module.PACKAGES:
            folder = self.release/package; folder.mkdir(parents=True)
            (folder/'__init__.py').write_text('')
        for name in ('v04/model.py', 'v05/model.py', 'v05/run.py'):
            (self.release/name).write_text('# fake source fixture\n')
        (self.release/'v07/common.py').write_text('from pathlib import Path\nPROJECT=Path(__file__).resolve().parents[1]\n')
        (self.release/'v07/run.py').write_text(FAKE)
        (self.release/'scripts').mkdir()
        shutil.copy2(SCRIPT.with_name('v07_cuda_acceleration.py'), self.release/'scripts/v07_cuda_acceleration.py')
        config = json.loads((SCRIPT.parents[1]/'configs/v07/base.json').read_text())
        folder = self.release/'configs/v07'; folder.mkdir(parents=True)
        module.write(folder/'base.json', config)
        data = self.release/'data/v06/wikitext2'; data.mkdir(parents=True)
        (data/'tiny').write_text('frozen fake data; zero training')
        module.write(data/'manifest.json', dict(vocab_size=config['vocab_size'], raw={}, prepared_hashes={'tiny': module.file_hash(data/'tiny')}))
        self.events = self.root/'events.jsonl'
        self.environment = patch.dict(os.environ, dict(FAKE_EVENTS=str(self.events), FAKE_DURATION='.005', FAKE_FAIL=''))
        self.environment.start()

    def tearDown(self):
        self.environment.stop(); self.temporary.cleanup()

    def args(self, *extra):
        return module.arguments(['--release', str(self.release), '--workspace', str(self.workspace),
                                 '--workers', '3', '--poll-seconds', '.005', *extra])

    def run_scheduler(self, *extra):
        with redirect_stdout(io.StringIO()):
            return module.Scheduler(self.args(*extra)).run()

    def timeline(self):
        return [json.loads(line) for line in self.events.read_text().splitlines()] if self.events.exists() else []

    def test_prepare_real_worker_sources_and_frozen_protocol(self):
        scheduler = module.Scheduler(self.args())
        self.assertEqual(len(scheduler.state['conditions']), 45)
        self.assertTrue(all(row['status']=='not_executed' for row in scheduler.state['conditions']))
        for slot in range(3):
            worker = scheduler.worker(slot)
            self.assertFalse((worker/'v07').is_symlink())
            self.assertEqual((worker/'v07/common.py').resolve(), worker/'v07/common.py')
            self.assertTrue((worker/'data/v06/wikitext2').is_symlink())
            self.assertNotEqual((worker/'v07/common.py').resolve(), self.release/'v07/common.py')
        cfg = module.read(scheduler.config_path(scheduler.cases[0]))
        self.assertEqual((cfg['batch_size'],cfg['sequence_tokens'],cfg['bptt_tokens'],cfg['train_steps']), (2,128,128,24000))
        self.assertEqual(cfg['cpu_threads'], 1)
        self.assertEqual(scheduler.identity['base_config']['cpu_threads'], 4)
        self.assertFalse(self.events.exists())
        self.assertFalse(scheduler.identity['acceleration']['compile_step'])
        self.assertTrue((scheduler.worker(0)/'scripts/v07_cuda_acceleration.py').is_file())
        self.assertFalse((scheduler.worker(0)/'scripts/v07_cuda_acceleration.py').is_symlink())
        module.Scheduler(self.args('--workers','2'))

    def test_explicit_compilation_identity_and_mock_worker_install(self):
        # A stdlib fake verifies wrapper installation, without CUDA/Torch/ML.
        adapter = self.release/'scripts/v07_cuda_acceleration.py'
        adapter.write_text('''import json\nfrom pathlib import Path\ndef write_record(path,value):\n p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))\ndef install(**kwargs):\n p=Path(kwargs['record_path']);v=json.loads(p.read_text());v.update(status='active',device='cuda:0');write_record(p,v)\n''')
        scheduler = module.Scheduler(self.args('--compile-step'))
        self.assertTrue(scheduler.identity['acceleration']['compile_step'])
        self.assertEqual(scheduler.identity['acceleration']['cache_policy'], dict(recompile_limit=32,accumulated_recompile_limit=256))
        self.assertEqual(scheduler.cfg, module.read(self.release/'configs/v07/base.json') | dict(cpu_threads=1))
        with self.assertRaisesRegex(ValueError, 'runtime selection changed'):
            module.Scheduler(self.args())
        with redirect_stdout(io.StringIO()):
            self.assertEqual(scheduler.run(), 0)
        for row in scheduler.state['conditions']:
            value = module.read(scheduler.run_root(row['case'])/row['case']['id']/'acceleration.json')
            self.assertTrue(value['compile_step'])
            self.assertEqual(value['mode'], 'eager' if row['case']['variant'] in ('transformer','stateless') else 'compiled_step')
            self.assertEqual(value['cache_policy'], None if value['mode']=='eager' else scheduler.identity['acceleration']['cache_policy'])
            self.assertEqual(value['adapter_sha256'], module.file_hash(adapter))

    def test_barriers_all_45_conditions_and_completed_resume(self):
        self.assertEqual(self.run_scheduler(),0)
        events = self.timeline(); starts = [event for event in events if event['kind']=='start']
        self.assertEqual(len(starts),45)
        previous_end = 0
        for seed in (11,23,37):
            for nodes in (32,128,512):
                group=[event for event in events if event['case']['seed']==seed and event['case']['nodes']==nodes]
                self.assertEqual(sum(event['kind']=='completed' for event in group),5)
                self.assertGreaterEqual(min(event['time'] for event in group if event['kind']=='start'), previous_end)
                previous_end=max(event['time'] for event in group if event['kind']=='completed')
        self.assertEqual(self.run_scheduler(),0)
        self.assertEqual(self.timeline(),events)
        state=module.read(self.workspace/'status.json')
        self.assertTrue(all(len(row['attempts'])==1 for row in state['conditions']))
        # Completed artifact edits must not be silently accepted on resume.
        case=state['conditions'][0]['case']; path=self.workspace/'raw'/case['id']/case['id']/'result.json'
        value=module.read(path);value['changed_metric']=True;module.write(path,value)
        with self.assertRaisesRegex(ValueError,'artifact changed'):
            self.run_scheduler()

    def test_failure_requires_explicit_retry_and_cannot_cross_barrier(self):
        os.environ['FAKE_FAIL']='core_n32_full_s11'
        self.assertEqual(self.run_scheduler(),1)
        events=self.timeline()
        self.assertFalse(any(event['case']['nodes']!=32 or event['case']['seed']!=11 for event in events))
        self.assertEqual(self.run_scheduler(),1)
        self.assertEqual(self.timeline(),events)
        os.environ['FAKE_FAIL']=''
        self.assertEqual(self.run_scheduler('--retry-failed'),0)
        rows=module.read(self.workspace/'status.json')['conditions']
        self.assertEqual(next(row for row in rows if row['case']['id']=='core_n32_full_s11')['attempts'][-1]['status'],'completed')
        self.assertEqual(len(next(row for row in rows if row['case']['id']=='core_n32_full_s11')['attempts']),2)

    def test_stop_relays_to_children_and_resumes_only_when_requested(self):
        os.environ['FAKE_DURATION']='2'
        scheduler=module.Scheduler(self.args()); timer=threading.Timer(.15,lambda:setattr(scheduler,'stopping',signal.SIGTERM))
        timer.start()
        try:
            with redirect_stdout(io.StringIO()):code=scheduler.run()
        finally:
            timer.join()
        self.assertEqual(code,143)
        rows=module.read(self.workspace/'status.json')['conditions']
        self.assertTrue(any(row['status']=='aborted' for row in rows))
        self.assertTrue(list((self.workspace/'raw').glob('*/*/last.pt')))
        before=len(self.timeline());self.assertEqual(self.run_scheduler(),1)
        self.assertEqual(len(self.timeline()),before)
        os.environ['FAKE_DURATION']='.005'
        self.assertEqual(self.run_scheduler('--resume-aborted'),0)

    def test_source_mutation_and_duplicate_global_lock_are_rejected(self):
        scheduler=module.Scheduler(self.args())
        path=scheduler.worker(0)/'v07/common.py';path.write_text(path.read_text()+'# changed\n')
        with self.assertRaisesRegex(ValueError,'Worker source changed'):
            module.Scheduler(self.args())
        lock=self.root/'global.lock'
        with module.Lock(lock):
            with self.assertRaises(RuntimeError):
                with module.Lock(lock):pass

    def test_backfill_uses_idle_slots_across_scale_barriers(self):
        os.environ['FAKE_DURATION'] = '.12'
        with patch.object(module, 'gpu_memory', return_value=dict(total_mib=81920, free_mib=80000)):
            self.assertEqual(self.run_scheduler('--dispatch-policy','backfill','--workers','8'), 0)
        events = self.timeline()
        first_large = min(event['time'] for event in events if event['kind']=='start' and event['case']['nodes']==128)
        last_small = max(event['time'] for event in events if event['kind']=='completed' and event['case']['nodes']==32 and event['case']['seed']==11)
        self.assertLess(first_large,last_small)
        self.assertEqual(sum(event['kind']=='start' for event in events),45)

    def test_backfill_reserves_future_peak_and_checks_actual_free_memory(self):
        scheduler = module.Scheduler(self.args('--dispatch-policy','backfill'))
        rows = scheduler.state['conditions']
        large = next(row for row in rows if row['case']['nodes']==512 and row['case']['variant']=='full')
        small = next(row for row in rows if row['case']['nodes']==32 and row['case']['variant']=='stateless')
        # A just-started job may have allocated nothing yet; its full peak still counts.
        scheduler.active = {slot:(None,None,large) for slot in range(6)}
        with patch.object(module,'gpu_memory',return_value=dict(total_mib=81920,free_mib=80000)):
            self.assertIs(scheduler.eligible_candidate([large,small]),small)
        with patch.object(module,'gpu_memory',return_value=dict(total_mib=81920,free_mib=12288+1023)):
            self.assertIsNone(scheduler.eligible_candidate([small]))
        self.assertEqual(scheduler.state['gpu_budget']['reserved_mib'], 6*11264)

    def test_backfill_keeps_failed_condition_but_finishes_independent_cases(self):
        os.environ['FAKE_FAIL'] = 'core_n32_full_s11'
        with patch.object(module,'gpu_memory',return_value=dict(total_mib=81920,free_mib=80000)):
            self.assertEqual(self.run_scheduler('--dispatch-policy','backfill','--workers','8'),1)
        rows = module.read(self.workspace/'status.json')['conditions']
        self.assertEqual(sum(row['status']=='completed' for row in rows),44)
        failed = next(row for row in rows if row['status']=='failed')
        self.assertEqual(failed['case']['id'],'core_n32_full_s11')
        self.assertEqual(len(failed['attempts']),1)

    def make_legacy_manifest(self, scheduler):
        old = dict(scheduler.identity)
        old['scheduler_sha256'] = 'old-frozen-scheduler-hash'
        for key in ('dispatch_policy','gpu_reserve_mib','gpu_reservations_mib'):
            del old[key]
        module.write(self.workspace/'manifest.json',old)
        state = module.read(self.workspace/'status.json');state['manifest_sha256']=module.digest(old)
        module.write(self.workspace/'status.json',state)
        return old

    def test_explicit_stopped_migration_archives_identity_and_preserves_results(self):
        scheduler = module.Scheduler(self.args())
        for slot,row in enumerate(scheduler.state['conditions'][:2]):
            scheduler.launch(row,slot)
        while scheduler.active:
            scheduler.reap();time.sleep(.005)
        third = scheduler.state['conditions'][2]
        third['status']='aborted'
        checkpoint=scheduler.run_root(third['case'])/third['case']['id']/'last.pt'
        checkpoint.parent.mkdir(parents=True,exist_ok=True);checkpoint.write_text('existing checkpoint')
        scheduler.persist();old = self.make_legacy_manifest(scheduler)
        old_status=module.read(self.workspace/'status.json')
        with self.assertRaisesRegex(ValueError,'old manifest digest'):
            module.Scheduler(self.args('--dispatch-policy','backfill','--migrate-scheduling','wrong'))
        with module.Lock(self.workspace/'.scheduler.lock'):
            with self.assertRaises(RuntimeError):
                module.Scheduler(self.args('--dispatch-policy','backfill','--migrate-scheduling',module.digest(old)))
        migrated = module.Scheduler(self.args('--dispatch-policy','backfill','--migrate-scheduling',module.digest(old)))
        self.assertEqual(checkpoint.read_text(),'existing checkpoint')
        self.assertEqual([row['status'] for row in migrated.state['conditions'][:3]],['completed','completed','aborted'])
        archive=next((self.workspace/'scheduling_migrations').iterdir())
        self.assertEqual(module.read(archive/'manifest.before.json'),old)
        self.assertEqual(module.read(archive/'status.before.json'),old_status)
        self.assertTrue(module.read(archive/'receipt.json')['committed'])
        with patch.object(module,'gpu_memory',return_value=dict(total_mib=81920,free_mib=80000)):
            resumed = module.Scheduler(self.args('--dispatch-policy','backfill','--resume-aborted','--workers','8'))
            with redirect_stdout(io.StringIO()):self.assertEqual(resumed.run(),0)
        starts=[event['case']['id'] for event in self.timeline() if event['kind']=='start']
        for row in resumed.state['conditions'][:2]:
            self.assertEqual(starts.count(row['case']['id']),1)
        self.assertEqual(resumed.state['conditions'][2]['status'],'completed')

    def test_migration_rejects_training_or_adapter_tampering(self):
        scheduler=module.Scheduler(self.args());old=self.make_legacy_manifest(scheduler)
        path=self.release/'scripts/v07_cuda_acceleration.py';path.write_text(path.read_text()+'\n# changed adapter\n')
        with self.assertRaisesRegex(ValueError,'immutable training/source/data/adapter/wrapper'):
            module.Scheduler(self.args('--dispatch-policy','backfill','--migrate-scheduling',module.digest(old)))
        self.assertEqual(module.read(self.workspace/'manifest.json'),old)
        self.assertFalse((self.workspace/'scheduling_migrations').exists())

    def test_migration_rejects_live_or_locked_children(self):
        scheduler=module.Scheduler(self.args());old=self.make_legacy_manifest(scheduler)
        case=scheduler.cases[0]
        with module.Lock(scheduler.run_root(case)/'.condition.lock'):
            with self.assertRaises(RuntimeError):
                module.Scheduler(self.args('--dispatch-policy','backfill','--migrate-scheduling',module.digest(old)))
        state=module.read(self.workspace/'status.json')
        state['conditions'][0]['attempts']=[dict(pid=os.getpid(),process_start=module.process_start(os.getpid()))]
        module.write(self.workspace/'status.json',state)
        with self.assertRaisesRegex(RuntimeError,'all old child processes'):
            module.Scheduler(self.args('--dispatch-policy','backfill','--migrate-scheduling',module.digest(old)))
        self.assertEqual(module.read(self.workspace/'manifest.json'),old)

    def test_interrupted_manifest_status_commit_requires_explicit_verified_recovery(self):
        scheduler=module.Scheduler(self.args());old=self.make_legacy_manifest(scheduler)
        args=self.args('--dispatch-policy','backfill','--migrate-scheduling',module.digest(old))
        migrated=module.Scheduler(args)
        archive=next((self.workspace/'scheduling_migrations').iterdir())
        # Simulate interruption between the two atomic document writes.
        module.write(self.workspace/'status.json',module.read(archive/'status.before.json'))
        with self.assertRaisesRegex(ValueError,'does not match'):
            module.Scheduler(self.args('--dispatch-policy','backfill'))
        recovered=module.Scheduler(args)
        self.assertEqual(recovered.state['manifest_sha256'],module.digest(migrated.identity))
        self.assertTrue(module.read(archive/'receipt.json')['committed'])


if __name__=='__main__':
    unittest.main()
