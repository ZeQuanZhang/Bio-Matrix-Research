"""Exercise the real comparison launcher in fresh PyCharm-sized PTYs, without training."""
import fcntl
import json
import os
from pathlib import Path
import pty
import select
import struct
import subprocess
import sys
import termios
import time
import unittest


PROJECT = Path(__file__).resolve().parents[1]
PROBE = """
import json, runpy
launcher = runpy.run_path('run_biomatrix_v09_compare.py', run_name='display_probe')
from v09.experiment import Display, plan
registry = plan(launcher['CONDITIONS'])
ui = Display(registry)
job = registry['jobs'][0]
job.update(status='running', stage='training', completed_tokens=128)
ui.refresh(registry, job, {'loss': 7., 'tokens_per_second': 400., 'phase': 'training'})
bars = (ui.total, ui.cells, *ui.details.values(), ui.stage, ui.resources)
print('DISPLAY_PROBE=' + json.dumps({'rows': [bar.nrows for bar in bars],
                                   'columns': [bar.ncols for bar in bars]}))
ui.close()
"""


class ComparisonDisplayChecks(unittest.TestCase):
    def test_zero_and_normal_terminal_show_all_selected_bars(self):
        for rows, columns in ((0, 0), (13, 162)):
            with self.subTest(rows=rows, columns=columns):
                master, slave = pty.openpty()
                fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', rows, columns, 0, 0))
                environment = os.environ.copy()
                environment.pop('TQDM_NROWS', None)
                child = None
                output = bytearray()
                try:
                    child = subprocess.Popen([sys.executable, '-c', PROBE], cwd=PROJECT,
                                             stdout=slave, stderr=slave, env=environment)
                    os.close(slave); slave = None
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        if select.select([master], [], [], .1)[0]:
                            try:
                                chunk = os.read(master, 65536)
                            except OSError:
                                break
                            if not chunk:
                                break
                            output.extend(chunk)
                        elif child.poll() is not None:
                            break
                    self.assertIsNotNone(child.poll(), 'progress display hung')
                    text = output.decode('utf-8', errors='replace')
                    self.assertEqual(child.returncode, 0, text)
                    marker = next(line for line in text.splitlines() if 'DISPLAY_PROBE=' in line)
                    state = json.loads(marker.split('DISPLAY_PROBE=', 1)[1])
                    self.assertEqual(state['rows'], [24] * 8)
                    self.assertTrue(all(width >= 100 for width in state['columns']))
                    for label in ('总有效目标', '配对格', 'content_only', 'v09_dense',
                                  'v09_sparse50', 'transformer', 'training', '本机 MPS'):
                        self.assertIn(label, text)
                    self.assertNotIn('more hidden', text)
                    self.assertNotIn('v08_seq', text)
                    self.assertIn('\x1b[A', text)
                finally:
                    if child is not None and child.poll() is None:
                        child.kill(); child.wait()
                    if slave is not None:
                        os.close(slave)
                    os.close(master)


if __name__ == '__main__':
    unittest.main()
