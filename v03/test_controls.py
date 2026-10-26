"""Control matching and sample isolation, separate from frozen training source files."""
import unittest
import json
from pathlib import Path
import numpy as np
import torch
from .data import example, collate
from .model import Config, Network
from .diagnose import counterfactual_pairs


class Controls(unittest.TestCase):
    def test_repeat_plan_is_complete_and_paired(self):
        from collections import Counter, defaultdict
        from .repeat import plan
        rows = plan(json.loads(Path('configs/v03/base.json').read_text()))
        self.assertEqual(len(rows), 90); self.assertEqual(len({c['id'] for c in rows}), 90)
        self.assertEqual(Counter(c['stage'] for c in rows), {'capacity': 18, 'iterations': 36, 'topology': 24, 'public': 12})
        groups = defaultdict(set)
        for c in rows:
            self.assertIn(c['seed'], (23, 37)); self.assertEqual(c['steps'], 2048)
            self.assertTrue(c['id'].endswith(f"_s{c['seed']}"))
            groups[tuple(c[k] for k in ('stage', 'nodes', 'state_dim', 'iterations', 'topology', 'seed'))].add(c['model'])
        self.assertTrue(all(models == {'C1', 'C2', 'C3'} for models in groups.values()))

    def test_counterfactuals_preserve_question_and_encoded_labels(self):
        from .data import bank, solve
        for a, b in counterfactual_pairs(bank('test', 2)):
            self.assertEqual(a['question'], b['question'])
            self.assertNotEqual(a['target'], b['target'])
            self.assertEqual(b['target'], solve(b['world'], b['spec']))
            rows = {r[0]: r for r in b['records']}
            for i, oid in enumerate(b['world']['ids']):
                self.assertEqual(rows[oid+1][1], 513+b['world']['attrs'][i][0])

    def test_c3_without_coupling_matches_c2(self):
        torch.set_num_threads(2); torch.manual_seed(73)
        a = Network(Config(model='C2', iterations=4, width=32, bottleneck=8))
        torch.manual_seed(73)
        b = Network(Config(model='C3', iterations=4, width=32, bottleneck=8))
        for name, parameter in a.named_parameters():
            torch.testing.assert_close(parameter, dict(b.named_parameters())[name], atol=0, rtol=0)
        batch = collate([example(np.random.default_rng(i), 8, 2) for i in range(2)])
        inputs = {k: v for k, v in batch.items() if k != 'target'}
        torch.testing.assert_close(a(**inputs)[0], b(**inputs, intervention='no_coupling')[0], atol=1e-6, rtol=1e-5)

    def test_no_cross_sample_state_or_label_input(self):
        torch.set_num_threads(2); torch.manual_seed(13)
        model = Network(Config(iterations=3, width=32, bottleneck=8))
        batch = collate([example(np.random.default_rng(i), 8, 2) for i in range(2)])
        inputs = {k: v for k, v in batch.items() if k != 'target'}
        both = model(**inputs)[0]
        alone = model(**{k: v[:1] for k, v in inputs.items()})[0]
        torch.testing.assert_close(both[:1], alone, atol=1e-6, rtol=1e-5)
        with self.assertRaises(TypeError):
            model(**batch)


if __name__ == '__main__':
    unittest.main()
