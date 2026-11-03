"""Finish predeclared diagnostics after the frozen training/evaluation process completes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT/'results/repaired/local333k_corrected_20261103'
RUNTIME = PROJECT/'results/tiny_arch/runtime/bin/python'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    while True:
        state = json.loads((ROOT/'registry.json').read_text())
        if state['status'] == 'completed':
            break
        if state['status'] in ('failed', 'aborted'):
            raise RuntimeError(f'Parent experiment {state["status"]}; diagnostics not executed')
        time.sleep(5.)
    sys.path.insert(0, str(PROJECT))
    from v07.common import RunLock, write
    with RunLock(ROOT/'.analysis.lock'):
        protocol = json.loads((ROOT/'protocol.json').read_text())
        assert all(sha(PROJECT/p) == value for p, value in protocol['sources'].items())
        assert len(state['conditions']) == 91 and all(v['status'] == 'completed' for v in state['conditions'].values())
        tests = json.loads((ROOT/'heldout/registry.json').read_text())
        assert len(tests) == 21 and all(v['status'] == 'completed' for v in tests.values())
        checkpoints = sorted(ROOT.glob('final_*/last.pt')); assert len(checkpoints) == 42
        before = {str(p.relative_to(ROOT)): sha(p) for p in checkpoints}
        write(ROOT/'analysis_checkpoint_hashes_before.json', before)
        jobs = [('qa_priors', PROJECT/'scripts/repaired_qa_prior_controls.py'),
                ('bigram', PROJECT/'scripts/repaired_bigram_control.py'),
                ('lm_history', PROJECT/'scripts/repaired_lm_history_controls.py'),
                ('plots', ROOT/'analysis/plot_results.py')]
        status_file = ROOT/'analysis_status.json'
        status = json.loads(status_file.read_text()) if status_file.exists() else dict(status='running', conditions={n: dict(status='not_executed') for n, _ in jobs})
        for name, script in jobs:
            if status['conditions'][name]['status'] == 'completed':
                continue
            status['status'] = 'running'; status['conditions'][name] = dict(status='running', source_sha256=sha(script))
            write(status_file, status); print('后处理开始', name, flush=True)
            with (ROOT/f'analysis_{name}.private.log').open('w') as output:
                result = subprocess.run([str(RUNTIME), str(script)], cwd=PROJECT, env=os.environ.copy(), stdout=output, stderr=subprocess.STDOUT)
            status['conditions'][name]['status'] = 'completed' if result.returncode == 0 else 'failed'
            status['conditions'][name]['returncode'] = result.returncode
            write(status_file, status)
            if result.returncode:
                status['status'] = 'failed'; write(status_file, status)
                raise RuntimeError(f'{name} failed; see its retained private log')
        after = {str(p.relative_to(ROOT)): sha(p) for p in checkpoints}
        assert before == after and all(sha(PROJECT/p) == value for p, value in protocol['sources'].items())
        write(ROOT/'analysis_checkpoint_integrity.json', dict(status='completed', checkpoints=42, frozen_sources=len(protocol['sources']),
            weights_unchanged=True, source_files_unchanged=True, note='All supplementary measurements are read-only'))
        status['status'] = 'completed'; write(status_file, status)
        print('全部预先登记诊断和图已完成；42份最终检查点保持不变。', flush=True)


if __name__ == '__main__':
    main()
