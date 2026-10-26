"""CPU checks of the independently trainable, stateless content baseline."""
import unittest

import torch
from torch.nn import functional as F

from v07.common import detach
from .baselines import Config as ContentConfig, ContentOnly, initialize as initialize_content
from .benchmark import build, initialize_model
from .model import Config, Network, initialize


class ContentChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_canonical_baseline_has_only_active_parameters_and_empty_state(self):
        model, cfg = build(dict(condition='content_only'), 4096, context_limit=4096)
        initialize_model(model, 'content_only', 11)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 303680)
        self.assertFalse(hasattr(cfg, 'nodes'))
        self.assertEqual(cfg.context_limit, 4096)
        values = torch.tensor([[3, 7, 9]])
        logits, state, diagnostics = model(values, {'unused': torch.ones(2)})
        self.assertEqual(state, {}); self.assertEqual(diagnostics, {})
        logits.square().mean().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_full_base_only_formula_values_outputs_and_all_predictive_gradients(self):
        # Public-size d8/w64 and all three node scales; warm Bio state changes
        # cannot alter this map. No optimizer or training run is performed.
        tokens = torch.tensor([[3, 7, 9, 4, 5, 2, 10, 8, 4, 1, 11, 6, 12, 15, 18, 2],
                               [7, 3, 4, 9, 2, 5, 11, 6, 10, 8, 1, 12, 18, 2, 15, 4]])
        target = tokens.roll(-1, 1)
        pure = ContentOnly(ContentConfig(vocab_size=32)).double()
        initialize_content(pure, 11)
        pure_logits, _, _ = pure(tokens)
        F.cross_entropy(pure_logits.flatten(0, 1), target.flatten()).backward()
        pure_parameters = dict(pure.named_parameters())
        for nodes in (32, 128, 512):
            with self.subTest(nodes=nodes):
                full = Network(Config(vocab_size=32, nodes=nodes)).double()
                initialize(full, 11); full.runtime_bio_read_enabled = False
                with torch.no_grad():
                    _, warm, _ = full(tokens.flip(1))
                    warm = detach(warm)
                    for key in ('x', 'm', 'slow', 'trace'):
                        if key in warm:
                            warm[key] = warm[key]*.4+.03
                actual, _, _ = full(tokens, warm)
                torch.testing.assert_close(actual, pure_logits, atol=2e-10, rtol=2e-10)
                F.cross_entropy(actual.flatten(0, 1), target.flatten()).backward()
                for name, parameter in full.named_parameters():
                    if name in pure_parameters:
                        expected = pure_parameters[name]
                        value = parameter[:, :64] if name == 'read_projection.weight' else parameter
                        gradient = parameter.grad[:, :64] if name == 'read_projection.weight' else parameter.grad
                        torch.testing.assert_close(value, expected, atol=0, rtol=0, msg=name)
                        torch.testing.assert_close(gradient, expected.grad, atol=5e-10, rtol=5e-10, msg=name)
                        if name == 'read_projection.weight':
                            torch.testing.assert_close(parameter.grad[:, 64:], torch.zeros_like(parameter.grad[:, 64:]), atol=0, rtol=0)
                    elif parameter.grad is not None:
                        torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter.grad), atol=5e-10, rtol=0, msg=name)

    def test_transformer_context_is_explicit_and_benchmark_default_stays_256(self):
        _, default = build(dict(condition='transformer', nodes=32), 32)
        _, training = build(dict(condition='transformer', nodes=32), 32, context_limit=4096)
        self.assertEqual(default.context_limit, 256)
        self.assertEqual(training.context_limit, 4096)


if __name__ == '__main__':
    unittest.main()
