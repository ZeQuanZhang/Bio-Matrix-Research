"""Small CPU checks for intervention attribution and strict-JSON metrics."""
import json
import unittest

import torch

from .diagnostics import cut_feedback_check, gradient_summary, input_interventions
from .model import Config, Network, initialize


class DiagnosticChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_input_attribution_preserves_runtime_flags_and_is_json_serializable(self):
        model = Network(Config(vocab_size=32, nodes=16, matrix_size=4, width=16,
                         bands=2, modes=2, block_size=4, group_size=2, active_fraction=.5)).double()
        initialize(model, 11)
        history = torch.tensor([[1, 2, 3, 5, 8, 13, 21, 2]])
        probe = torch.tensor([[7, 9, 11, 13]]); labels = torch.tensor([[9, 11, 13, 15]])
        with torch.no_grad():
            _, state, _ = model(history)
        record = input_interventions(model, probe, labels, state, history, 11)
        self.assertTrue(model.training)
        self.assertTrue(model.runtime_bio_read_enabled)
        self.assertEqual(record['token_replacement']['prefix_logit_max_abs'], 0.)
        self.assertEqual(record['input_difference_by_shell']['resolution'], 'final_internal_round_only')
        self.assertGreater(record['token_replacement']['logit_max_abs'], .001)
        base = record['content_base_only']
        self.assertLess(base['warm_vs_cold']['logit_max_abs'], 1e-10)
        self.assertLess(base['clear_slow']['logit_max_abs'], 1e-10)
        self.assertLess(base['ordered_vs_shuffled_history']['logit_max_abs'], 1e-10)
        self.assertGreater(base['token_replacement']['logit_max_abs'], .001)
        json.dumps(record, allow_nan=False)

    def test_cut_warm_pending_feedback_has_inner_positive_control(self):
        record = cut_feedback_check(torch.device('cpu'), 11)
        self.assertEqual(record['status'], 'completed')
        self.assertEqual(record['outer_slow_difference'], 0.)
        self.assertEqual(record['logits_max_difference'], 0.)
        self.assertGreater(record['reachable_inner_positive_control'], 0.)
        json.dumps(record, allow_nan=False)

    def test_failed_gradient_elements_are_explicit_and_strict_json(self):
        model = torch.nn.Linear(2, 2)
        model.weight.grad = torch.full_like(model.weight, float('inf'))
        record = gradient_summary(model)
        self.assertFalse(record['all_elements_finite'])
        self.assertIsNone(record['global_norm64'])
        json.dumps(record, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
