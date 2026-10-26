import unittest
import numpy as np
import torch

from .data import generate
from .extensions import ExtendedMemory


class ExtensionTests(unittest.TestCase):
    def test_noisy_observation_targets_and_signal_no_future_input(self):
        from .extension_run import noisy_batch
        from .signal_run import examples, SignalModel
        events, labels, queries = noisy_batch(np.random.default_rng(3), batch=8, probability=0.)
        for row, target in zip(events, labels):
            memory = {}
            for op, key, val in row.tolist():
                if op == 0:
                    memory[key] = val
                if op == 2:
                    self.assertEqual(target[0].item(), memory[key])
                    self.assertEqual(val, 4)
        x, y = examples(np.random.default_rng(4), 4, 16, [.08, .17])
        self.assertTrue(bool((x[:, 16:] == 0).all()))
        model = SignalModel('spectral_learned')
        prediction = model(x)
        self.assertEqual(prediction.shape, y.shape)
        prediction.square().mean().backward()
        for p in model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(bool(p.grad.isfinite().all()))
        changed = x.clone(); changed[:, :16, 0] *= -1
        torch.testing.assert_close(model(x, reset=True), model(changed, reset=True))

    def test_filter_jacobian_matches_autograd(self):
        model = ExtendedMemory('bayes', width=8, state=8).double()
        state = torch.randn(1, 8, dtype=torch.float64)*.2
        encoded = torch.randn(1, 16, dtype=torch.float64)
        expected = torch.autograd.functional.jacobian(lambda s: model.core.step(encoded, s[None])[0], state[0])
        torch.testing.assert_close(model.jacobian(state, encoded)[0], expected)

    def test_positive_variance_causal_reset_and_gradients(self):
        events, _, queries = generate(np.random.default_rng(8), batch=4, pairs=3, delay=6, queries=2)
        for mode in ('bayes', 'spectral_zero', 'spectral_fixed', 'spectral_learned', 'full'):
            model = ExtendedMemory(mode)
            logits, state = model(events, queries)
            self.assertTrue(bool(logits.isfinite().all() and state.isfinite().all()))
            self.assertEqual(state.shape, (4, model.state_floats))
            if model.bayesian:
                self.assertTrue(bool((state[:, 32:64] > 0).all()))
            early, _ = model(events[:, :queries[0]+1], (queries[0],))
            torch.testing.assert_close(early, logits[:, :1])
            reset, _ = model(events[:, :queries[0]+1], (queries[0],), intervention='reset')
            direct, _ = model(events[:, queries[0]:queries[0]+1], (0,))
            torch.testing.assert_close(reset, direct)
            solo, _ = model(events[1:2], queries)
            torch.testing.assert_close(logits[1:2], solo)
            logits.square().mean().backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(bool(parameter.grad.isfinite().all()), name)

    def test_rotation_and_forgetting_are_distinct(self):
        model = ExtendedMemory('spectral_fixed')
        state = torch.zeros(2, 32)
        state[:, 0] = 1.
        x = torch.zeros(2, 32)
        for _ in range(100):
            state = model.oscillator_step(state, x)
        self.assertTrue(bool(state.isfinite().all()))
        self.assertLessEqual(float(state.view(2, -1, 2).norm(dim=-1).max().detach()), 2**.5 + 1e-5)


if __name__ == '__main__':
    unittest.main()
