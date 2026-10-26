"""Numerical, gradient, causality and shape contracts for the v0.8 scan.

Run: .venv/bin/python -m unittest v08.test_recurrence -v
"""
import unittest

import torch
from torch.nn import functional as F

from .recurrence import parallel_delta, sequential_delta, triangular_delta


KERNELS = (sequential_delta, parallel_delta, triangular_delta)


class DeltaRecurrenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def inputs(self, steps, dtype=torch.float64, leading=(2, 3, 2), d=4, seed=19):
        generator = torch.Generator().manual_seed(seed)
        initial = torch.randn(*leading, d, d, generator=generator, dtype=dtype)*.2
        key = F.normalize(torch.randn(*leading, steps, d, generator=generator, dtype=dtype), dim=-1)
        value = torch.randn(*leading, steps, d, generator=generator, dtype=dtype)*.3
        alpha = .7+.29*torch.rand(*leading, steps, generator=generator, dtype=dtype)
        beta = .1+.8*torch.rand(*leading, steps, generator=generator, dtype=dtype)
        return initial, key, value, alpha, beta

    def test_outputs_and_nonzero_initial_for_single_and_non_power_of_two_lengths(self):
        for steps in (1, 2, 3, 7, 16, 31):
            with self.subTest(steps=steps):
                inputs = self.inputs(steps)
                expected = sequential_delta(*inputs)
                for kernel in KERNELS[1:]:
                    actual = kernel(*inputs)
                    self.assertEqual(actual[0].shape, (2, 3, 2, steps, 4, 4))
                    for left, right in zip(actual, expected):
                        torch.testing.assert_close(left, right, atol=2e-14, rtol=2e-13)
                    torch.testing.assert_close(actual[1], actual[0][..., -1, :, :])

    def test_single_step_matches_literal_dense_equation(self):
        initial, key, value, alpha, beta = self.inputs(1)
        k, v, a, b = key[..., 0, :], value[..., 0, :], alpha[..., 0], beta[..., 0]
        expected = (a[..., None, None]*(initial @ (torch.eye(4, dtype=initial.dtype)
                    - b[..., None, None]*k.unsqueeze(-1)*k.unsqueeze(-2)))
                    + b[..., None, None]*v.unsqueeze(-1)*k.unsqueeze(-2))
        for kernel in KERNELS:
            actual, final = kernel(initial, key, value, alpha, beta)
            torch.testing.assert_close(final, expected, atol=1e-15, rtol=1e-14)
            torch.testing.assert_close(actual[..., 0, :, :], expected, atol=1e-15, rtol=1e-14)

    def test_float32_longer_scan(self):
        inputs = self.inputs(129, dtype=torch.float32, leading=(2, 2, 3), d=8)
        for kernel in KERNELS[1:]:
            for actual, expected in zip(kernel(*inputs), sequential_delta(*inputs)):
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_all_input_gradients_match_rank_one_reference(self):
        for steps in (1, 7, 17):
            with self.subTest(steps=steps):
                raw = self.inputs(steps, leading=(1, 2, 2), d=3)
                weight = torch.randn(1, 2, 2, steps, 3, 3, dtype=torch.float64,
                                     generator=torch.Generator().manual_seed(71))
                outputs, gradients = [], []
                for kernel in KERNELS:
                    inputs = tuple(tensor.clone().requires_grad_() for tensor in raw)
                    states, final = kernel(*inputs)
                    loss = (states.sin()*weight).sum()+.37*final.square().sum()
                    gradients.append(torch.autograd.grad(loss, inputs))
                    outputs.append(loss)
                for output, gradient in zip(outputs[1:], gradients[1:]):
                    torch.testing.assert_close(outputs[0], output, atol=2e-13, rtol=2e-12)
                    for actual, expected in zip(gradient, gradients[0]):
                        self.assertTrue(bool(actual.isfinite().all()))
                        torch.testing.assert_close(actual, expected, atol=5e-13, rtol=5e-11)

    def test_parallel_autograd_matches_finite_differences(self):
        inputs = tuple(tensor.requires_grad_() for tensor in self.inputs(3, leading=(), d=2))
        for kernel in KERNELS[1:]:
            self.assertTrue(torch.autograd.gradcheck(kernel, inputs, eps=1e-6,
                                                    atol=2e-6, rtol=2e-5))

    def test_broadcast_batch_node_and_band_axes_with_gradients(self):
        initial, key, value, alpha, beta = self.inputs(5, leading=(2, 3, 2), d=3)
        raw = (initial[:, :1], key[:1, :, :1], value[:, :, :1], alpha[0, 0, 0], beta[:, :1])
        gradients = []
        for kernel in KERNELS:
            inputs = tuple(tensor.clone().requires_grad_() for tensor in raw)
            states, final = kernel(*inputs)
            self.assertEqual(states.shape, (2, 3, 2, 5, 3, 3))
            full = tuple(tensor.expand_as(target) for tensor, target in zip(inputs, (initial, key, value, alpha, beta)))
            reference = sequential_delta(*full)
            torch.testing.assert_close(states, reference[0], atol=2e-14, rtol=2e-13)
            torch.testing.assert_close(final, reference[1], atol=2e-14, rtol=2e-13)
            gradients.append(torch.autograd.grad(states.square().sum()+final.square().sum(), inputs))
        for gradient in gradients[1:]:
            for left, right in zip(gradient, gradients[0]):
                torch.testing.assert_close(left, right, atol=5e-13, rtol=5e-11)

    def test_future_perturbation_cannot_change_prefix_states_or_gradients(self):
        raw = self.inputs(11, leading=(1, 2, 1), d=3)
        cutoff = 5
        for kernel in KERNELS[1:]:
            reference, _ = kernel(*raw)
            for index in range(1, 5):
                modified = [tensor.clone() for tensor in raw]
                if index in (1, 2):
                    modified[index][..., cutoff:, :] += .53
                else:
                    modified[index][..., cutoff:] += .19
                states, _ = kernel(*modified)
                torch.testing.assert_close(states[..., :cutoff, :, :], reference[..., :cutoff, :, :],
                                           atol=0, rtol=0)
                self.assertGreater(float((states[..., cutoff:, :, :]-reference[..., cutoff:, :, :]).abs().max()), 1e-5)
            inputs = tuple(tensor.clone().requires_grad_() for tensor in raw)
            states, _ = kernel(*inputs)
            gradients = torch.autograd.grad(states[..., :cutoff, :, :].square().sum(), inputs)
            for index, gradient in enumerate(gradients[1:], start=1):
                future = gradient[..., cutoff:, :] if index in (1, 2) else gradient[..., cutoff:]
                torch.testing.assert_close(future, torch.zeros_like(future), atol=0, rtol=0)

    def test_chunked_execution_preserves_all_states_and_final_memory(self):
        initial, key, value, alpha, beta = self.inputs(13)
        for kernel in KERNELS[1:]:
            whole, final = kernel(initial, key, value, alpha, beta)
            memory, parts = initial, []
            for start, end in ((0, 4), (4, 9), (9, 13)):
                states, memory = kernel(memory, key[..., start:end, :], value[..., start:end, :],
                                        alpha[..., start:end], beta[..., start:end])
                parts.append(states)
            torch.testing.assert_close(torch.cat(parts, dim=-3), whole, atol=2e-14, rtol=2e-13)
            torch.testing.assert_close(memory, final, atol=2e-14, rtol=2e-13)

    def test_homogeneous_transition_is_nonexpansive_with_valid_local_bounds(self):
        initial, key, value, alpha, beta = self.inputs(19)
        value.zero_()
        beta = beta*2
        for kernel in KERNELS[1:]:
            states, _ = kernel(initial, key, value, alpha, beta)
            previous = torch.cat((initial.unsqueeze(-3), states[..., :-1, :, :]), dim=-3)
            self.assertTrue(bool((torch.linalg.matrix_norm(states)
                                  <= torch.linalg.matrix_norm(previous)+1e-13).all()))

    def test_zero_retention_coefficients_preserve_outputs_and_all_gradients(self):
        raw = self.inputs(9, leading=(1, 2, 2), d=3)
        for zero_indices in ((0,), (3, 4, 7), tuple(range(9))):
            alpha = raw[3].clone()
            alpha[..., list(zero_indices)] = 0
            inputs_raw = (*raw[:3], alpha, raw[4])
            expected, gradients = None, None
            for kernel in KERNELS:
                inputs = tuple(tensor.clone().requires_grad_() for tensor in inputs_raw)
                states, final = kernel(*inputs)
                grad = torch.autograd.grad(states.square().sum()+final.sin().sum(), inputs)
                if expected is None:
                    expected, gradients = states, grad
                else:
                    torch.testing.assert_close(states, expected, atol=2e-14, rtol=2e-13)
                    for actual, reference in zip(grad, gradients):
                        self.assertTrue(bool(actual.isfinite().all()))
                        torch.testing.assert_close(actual, reference, atol=5e-13, rtol=5e-11)

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS is unavailable')
    def test_native_mps_float32_outputs_and_all_input_gradients(self):
        raw = self.inputs(17, dtype=torch.float32, leading=(1, 2, 2), d=4)
        reference, reference_grad = None, None
        for kernel in KERNELS:
            inputs = tuple(tensor.to('mps').requires_grad_() for tensor in raw)
            states, final = kernel(*inputs)
            gradient = torch.autograd.grad(states.square().sum()+final.square().sum(), inputs)
            if reference is None:
                reference, reference_grad = states.cpu(), [tensor.cpu() for tensor in gradient]
            else:
                torch.testing.assert_close(states.cpu(), reference, atol=2e-6, rtol=2e-5)
                for actual, expected in zip(gradient, reference_grad):
                    torch.testing.assert_close(actual.cpu(), expected, atol=3e-6, rtol=3e-5)

    def test_empty_sequence_retains_initial_memory(self):
        inputs = self.inputs(0)
        for kernel in KERNELS:
            states, final = kernel(*inputs)
            self.assertEqual(states.shape, (2, 3, 2, 0, 4, 4))
            torch.testing.assert_close(final, inputs[0], atol=0, rtol=0)

    def test_invalid_shape_dtype_and_broadcast_are_rejected(self):
        raw = self.inputs(3)
        variants = ((0, raw[0][..., :2, :]), (2, raw[2][..., :2, :]),
                    (3, raw[3][..., :2]), (4, raw[4].float()),
                    (1, raw[1][:, :2]), (2, raw[2].long()))
        for kernel in KERNELS:
            for index, tensor in variants:
                with self.subTest(kernel=kernel.__name__, index=index), self.assertRaises((ValueError, TypeError)):
                    inputs = list(raw)
                    inputs[index] = tensor
                    kernel(*inputs)


if __name__ == '__main__':
    unittest.main()
