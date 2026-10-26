import unittest
import pickle
import tempfile
from pathlib import Path

import torch

from bio_lm import AffineScan, BioState, Config, LanguageModel, load_checkpoint


class UnsupportedCheckpointObject:
    pass


class BioTests(unittest.TestCase):
    def test_checkpoint_loading_restricts_pickle_globals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"version": torch.__version__, "weight": torch.ones(2)}, path)
            torch.testing.assert_close(load_checkpoint(path)["weight"], torch.ones(2))
            torch.save({"unsupported": UnsupportedCheckpointObject()}, path)
            with self.assertRaises(pickle.UnpicklingError):
                load_checkpoint(path)

    @unittest.skipUnless(torch.backends.mps.is_available(), "Metal is unavailable")
    def test_metal_scan_matches_cpu(self):
        torch.manual_seed(11)
        tensors = [torch.rand(2, 33, 5), torch.randn(2, 33, 5), torch.randn(2, 5)]
        reference = [x.clone().requires_grad_() for x in tensors]
        metal = [x.to("mps").requires_grad_() for x in tensors]
        expected = AffineScan.apply(*reference)
        actual = AffineScan.apply(*metal)
        torch.testing.assert_close(actual.cpu(), expected, atol=2e-6, rtol=2e-5)
        expected.square().mean().backward(); actual.square().mean().backward()
        for x, y in zip(reference, metal):
            torch.testing.assert_close(x.grad, y.grad.cpu(), atol=2e-6, rtol=2e-5)
        cpu_cell = BioState(7)
        metal_cell = BioState(7).to("mps")
        metal_cell.load_state_dict(cpu_cell.state_dict())
        x = torch.randn(2, 67, 7)
        wanted, wanted_state = cpu_cell(x)
        got, got_state = metal_cell(x.to("mps"))
        torch.testing.assert_close(wanted, got.cpu(), atol=5e-6, rtol=5e-5)
        torch.testing.assert_close(wanted_state, got_state.cpu(), atol=5e-6, rtol=5e-5)
        wanted.square().mean().backward(); got.square().mean().backward()
        for a, b in zip(cpu_cell.parameters(), metal_cell.parameters()):
            torch.testing.assert_close(a.grad, b.grad.cpu(), atol=5e-6, rtol=5e-4)

    def test_scan_matches_serial_and_gradients(self):
        torch.manual_seed(4)
        a = torch.rand(2, 11, 3, dtype=torch.float64, requires_grad=True)
        b = torch.randn_like(a, requires_grad=True)
        initial = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        expected, current = [], initial
        for t in range(a.shape[1]):
            current = a[:, t] * current + b[:, t]
            expected.append(current)
        expected = torch.stack(expected, 1)
        actual = AffineScan.apply(a, b, initial)
        torch.testing.assert_close(actual, expected)
        weight = torch.randn_like(actual)
        wanted = torch.autograd.grad((expected * weight).sum(), (a, b, initial), retain_graph=True)
        got = torch.autograd.grad((actual * weight).sum(), (a, b, initial))
        for x, y in zip(got, wanted):
            torch.testing.assert_close(x, y)
        self.assertTrue(torch.autograd.gradcheck(AffineScan.apply, (a, b, initial)))

    def test_state_bounds_causality_and_chunk_equivalence(self):
        torch.manual_seed(2)
        cell = BioState(7).double()
        x = torch.randn(2, 37, 7, dtype=torch.float64) * 3
        y, state = cell(x)
        early, _ = cell(x[:, :19])
        torch.testing.assert_close(y[:, :19], early)
        first, carried = cell(x[:, :19])
        second, final = cell(x[:, 19:], carried)
        torch.testing.assert_close(y, torch.cat((first, second), 1))
        torch.testing.assert_close(state, final)
        positive = state[:, [0, 1, 2, 3, 5]]
        self.assertTrue(bool(((positive >= 0) & (positive <= 1)).all()))
        self.assertTrue(bool((state.abs() <= 1).all()))
        self.assertGreater((second - cell(x[:, 19:])[0]).abs().max().item(), 1e-6)
        independent = cell(x[1:])[0]
        torch.testing.assert_close(y[1:], independent)
        no_history = cell(x, mode="stateless")[0]
        changed = x.clone(); changed[:, :19] *= -1
        torch.testing.assert_close(no_history[:, 19:], cell(changed, mode="stateless")[0][:, 19:])

    def test_model_causality_and_exact_parameter_match(self):
        cfg = dict(vocab_size=128, width=32, layers=2, heads=4, context=32)
        baseline = LanguageModel(Config(**cfg, ff_width=136))
        bio = LanguageModel(Config(**cfg, ff_width=128, variant="bio"))
        self.assertEqual(baseline.parameter_count(), bio.parameter_count())
        tokens = torch.randint(0, 128, (2, 23))
        for model in (baseline, bio):
            torch.testing.assert_close(model(tokens)[0][:, :12], model(tokens[:, :12])[0], atol=2e-6, rtol=2e-5)
        loss = bio(tokens)[0].square().mean(); loss.backward()
        for name, parameter in bio.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(parameter.grad.isfinite().all()), name)


if __name__ == "__main__":
    unittest.main()
