"""Wait for the serial training queue, then run diagnostics without GPU contention."""
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import unittest
from .run import ROOT, cases, write


def main():
    cfg = json.loads(Path('configs/v03/base.json').read_text())
    identifiers = [c['id'] for stage in ('pilot', 'main', 'capacity', 'iterations', 'topology', 'public') for c in cases(stage, cfg)]
    write(ROOT/'postprocess_state.json', {'status': 'waiting_for_training', 'training_conditions': len(identifiers)})
    while True:
        statuses = json.loads((ROOT/'conditions.json').read_text())
        failed = [k for k in identifiers if statuses[k]['status'] in ('failed', 'aborted')]
        if failed:
            write(ROOT/'postprocess_state.json', {'status': 'blocked_by_training_condition', 'conditions': failed})
            raise RuntimeError(f'Training stopped: {failed}')
        if all(statuses[k]['status'] == 'completed' for k in identifiers):
            break
        time.sleep(45)
    time.sleep(1)
    try:
        for module in ('v03.baselines', 'v03.diagnose'):
            write(ROOT/'postprocess_state.json', {'status': 'running', 'module': module})
            subprocess.run([sys.executable, '-u', '-m', module], check=True)
        from . import test_v03, test_controls
        from matrix_lab import test_core, test_extensions
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in (test_v03, test_controls, test_core, test_extensions))
        stream = io.StringIO(); outcome = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        (ROOT/'P0/final_tests.log').write_text(stream.getvalue())
        write(ROOT/'P0/final_tests.json', {'tests': outcome.testsRun, 'skipped': len(outcome.skipped), 'passed': outcome.wasSuccessful()})
        if not outcome.wasSuccessful():
            raise RuntimeError('Final tests failed')
        for module in ('v03.verify', 'v03.report'):
            write(ROOT/'postprocess_state.json', {'status': 'running', 'module': module})
            subprocess.run([sys.executable, '-u', '-m', module], check=True)
    except Exception as error:
        write(ROOT/'postprocess_state.json', {'status': 'failed', 'error': repr(error)}); raise
    write(ROOT/'postprocess_state.json', {'status': 'completed'})


if __name__ == '__main__':
    main()
