"""Focused regressions for data leakage checks, terminal rendering and exact resume."""
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.v2_console import width
from v07.common import write
from . import data, run
from .console import IDEStream, LABELS, Progress


class LocalChecks(unittest.TestCase):
    def test_zero_size_terminal_and_ide_non_tty_keep_all_bars(self):
        p = Progress(False)
        with patch('os.get_terminal_size', return_value=os.terminal_size((0, 0))):
            self.assertEqual(p.size(), (120, 24))
        for columns, rows in ((120, 24), (80, 8), (80, 13), (40, 13)):
            lines = p.frame(columns, rows, 'CPU 10% | RSS 1 GiB')
            self.assertLessEqual(len(lines), rows-1)
            self.assertTrue(all(width(line) <= columns-2 for line in lines))
            for label in LABELS.values():
                self.assertIn(label, '\n'.join(lines))
        with patch.dict(os.environ, {'PYCHARM_HOSTED': '1'}):
            self.assertTrue(IDEStream(io.StringIO()).isatty())
        sink = io.StringIO(); p.stream = sink; p.enabled = True; p.terminal = True
        p.resources = lambda: 'CPU 10%'; p.draw(); p.draw()
        self.assertEqual(sink.getvalue().count('\x1b[H'), 2)
        self.assertEqual(sink.getvalue().count('BioD'), 2)
        p.enabled = False; p.close()

    def test_normalized_overlap_filter(self):
        phrase = 'one two three four five six seven eight nine ten eleven twelve thirteen'
        locked = set(data.grams(data.words(phrase)))
        self.assertTrue(locked.intersection(data.grams(data.words('prefix '+phrase.upper()+' suffix'))))
        self.assertFalse(locked.intersection(data.grams(data.words('unrelated science and architecture'))))
        self.assertEqual(list(data.strings({'q': ['abc', {'answer': 'def'}], 'id': 1})), ['abc', 'def'])

    def test_interruption_resume_matches_uninterrupted_training_and_pairing(self):
        torch.set_num_threads(1)
        c = run.settings(run.CONFIG)
        c.update(model=dict(vocab_size=40, width=16, layers=1, heads=2, ff=24, nodes=4,
                            phase_bands=2, segment_tokens=4, context_limit=8),
                 sequence=8, pretrain_tokens=32, milestones=[16, 32], warmup_tokens=8,
                 microbatch=1, accumulation=2, validation_sequences=2, eval_every=2,
                 checkpoint_every=1, activation_checkpoint=False)
        stop = SimpleNamespace(requested=False); device = torch.device('cpu')
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); corpus = base/'data'; corpus.mkdir()
            files = []
            for split in ('train', 'validation'):
                values = np.arange(129, dtype='<u2') % 40
                (corpus/f'{split}.bin').write_bytes(values.tobytes())
                files.append(dict(path=f'{split}.bin', split=split))
            manifest = dict(files=files)
            fresh, resumed = base/'fresh', base/'resumed'
            fresh.mkdir(); resumed.mkdir()
            with patch.object(run.data, 'prepare', return_value=manifest), patch.object(run.evaluate, 'run'):
                run.train(fresh, corpus, c, Progress(False), stop, device, smoke=True)
                with self.assertRaises(KeyboardInterrupt):
                    run.train(resumed, corpus, c, Progress(False), stop, device, smoke=True, max_updates=1)
                self.assertEqual(json.loads((resumed/'registry.json').read_text())['status'], 'aborted')
                run.train(resumed, corpus, c, Progress(False), stop, device, smoke=True)
                for name in run.CONDITIONS:
                    a = torch.load(fresh/name/'last.pt', weights_only=True)
                    b = torch.load(resumed/name/'last.pt', weights_only=True)
                    self.assertEqual(a['counters']['batch_chain'], b['counters']['batch_chain'])
                    self.assertEqual(a['counters']['tokens'], 32)
                    for key in a['model']:
                        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
                self.assertEqual(json.loads((resumed/'paired_32.json').read_text())['status'], 'completed')
                with self.assertRaisesRegex(ValueError, '恢复被拒绝'):
                    run.train(resumed, corpus, c | dict(learning_rate=.002), Progress(False), stop, device, smoke=True)


if __name__ == '__main__':
    unittest.main()
