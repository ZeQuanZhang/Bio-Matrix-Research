"""CPU protocol tests; these do not launch the benchmark queue or use a GPU."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from v07.model import Config as TransformerConfig, Transformer
from .benchmark import (CONDITIONS, build, compare_pair, execute, frozen_corpus,
                        initialize_model, plan, report, validate)


class BenchmarkProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def args(self, **changes):
        return SimpleNamespace(nodes=[32, 128, 512], conditions=list(CONDITIONS), sequence_tokens=64,
                               batch=2, threads=4, group_size=16, repeats=3, warmup=6,
                               **changes)

    def test_queue_pairs_all_six_conditions_at_every_scale(self):
        records = plan([32, 128, 512])
        self.assertEqual(len(records), 18)
        self.assertEqual(len({row['id'] for row in records}), 18)
        self.assertTrue(all(row['status'] == 'not_executed' for row in records))
        self.assertEqual([row['condition'] for row in records[:6]], list(CONDITIONS))
        self.assertEqual([row['nodes'] for row in records], [32]*6+[128]*6+[512]*6)

    def test_build_declares_equivalence_and_new_dynamics_separately(self):
        old, old_cfg = build(dict(condition='v08_seq', nodes=32), 64)
        model, cfg = build(dict(condition='v09_E', nodes=32), 64)
        self.assertFalse(cfg.content_base)
        self.assertEqual(cfg.address_size, 0)
        self.assertEqual(cfg.active_fraction, 1.)
        self.assertTrue(cfg.reuse_controls and cfg.batch_feedback)
        self.assertEqual(cfg.credit_horizon, old_cfg.credit_horizon)
        initialize_model(old, 'v08_seq', 11); initialize_model(model, 'v09_E', 11)
        for name, value in old.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0, msg=name)
        for condition, fraction in (('v09_dense', 1.), ('v09_sparse75', .75), ('v09_sparse50', .5)):
            _, cfg = build(dict(condition=condition, nodes=32), 64)
            self.assertTrue(cfg.content_base)
            self.assertEqual(cfg.address_size, 8)
            self.assertEqual(cfg.active_fraction, fraction)

    def test_no_manifest_cannot_trigger_download_or_tokenizer_fit(self):
        with TemporaryDirectory() as folder, patch('v09.benchmark.prepare') as prepare:
            with self.assertRaises(FileNotFoundError):
                frozen_corpus(folder)
            prepare.assert_not_called()

    def test_existing_results_are_not_overwritten_before_any_execution(self):
        with TemporaryDirectory() as folder:
            target = Path(folder)/'results'; target.mkdir()
            (target/'keep.txt').write_text('unchanged')
            args = self.args(output=target)
            with self.assertRaises(FileExistsError):
                execute(args)
            self.assertEqual((target/'keep.txt').read_text(), 'unchanged')

    def test_validation_rejects_ambiguous_or_invalid_counts(self):
        args = self.args()
        validate(args)
        for key, value in (('nodes', [32, 32]), ('nodes', [64]), ('sequence_tokens', 17),
                           ('repeats', 2), ('warmup', 0), ('group_size', 0)):
            with self.subTest(key=key, value=value):
                changed = SimpleNamespace(**(vars(args) | {key: value}))
                with self.assertRaises(ValueError):
                    validate(changed)

    def test_native_comparison_checks_states_and_parameter_gradients_on_cpu(self):
        cfg = TransformerConfig(vocab_size=32, transformer_width=16, transformer_ff=32,
                                transformer_layers=1, context_limit=32)
        models = [Transformer(cfg), Transformer(cfg)]
        models[1].load_state_dict(models[0].state_dict())
        values = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
        # Transformer state contains nested KV and is not this checker's Bio contract.
        class TinyBio(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(32, 4)
            def forward(self, tokens):
                hidden = self.embedding(tokens)
                logits = torch.nn.functional.linear(hidden, self.embedding.weight)
                return logits, dict(x=hidden.sum(1), count=torch.ones(len(tokens), dtype=torch.long)), {}
        left, right = TinyBio(), TinyBio()
        right.load_state_dict(left.state_dict())
        result = compare_pair(left, right, values, torch.device('cpu'))
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(result['gradients']['embedding.weight']['passed'])
        self.assertEqual(set(result['states']), {'x', 'count'})

    def test_report_retains_failures_and_actual_active_fraction(self):
        records = plan([32])
        records[0].update(status='failed', error='kept')
        records[1].update(status='completed', result=dict(median_seconds=.1, tokens_per_second=1280.,
                         parameters=123, state_bytes=456, actual_active_fraction=[.8125, .75, .75]))
        with TemporaryDirectory() as folder:
            report(Path(folder), records)
            text = (Path(folder)/'REPORT.md').read_text()
            self.assertIn('| v08_seq_n32 | failed |', text)
            self.assertIn('0.771', text)
            self.assertIn('not_executed', text)
            self.assertIn('完成 1/6', text)


if __name__ == '__main__':
    unittest.main()
