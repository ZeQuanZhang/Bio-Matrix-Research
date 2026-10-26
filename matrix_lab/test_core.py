import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from .data import generate, training_batch
from .models import Config, MemoryModel, VARIANTS


class CoreTests(unittest.TestCase):
    def test_gru_matches_torch_reference(self):
        model = MemoryModel(Config(cell='gru'))
        reference = torch.nn.GRUCell(32, 32)
        with torch.no_grad():
            reference.weight_ih.copy_(model.input_state.weight)
            reference.bias_ih.copy_(model.input_state.bias)
            reference.weight_hh.copy_(model.state_state.weight)
            reference.bias_hh.copy_(model.state_state.bias)
        x, state = torch.randn(5, 32), torch.randn(5, 32)
        torch.testing.assert_close(model.step(model.input_state(x), state), reference(x, state))

    def test_data_targets_overwrite_and_reproducibility(self):
        a = generate(np.random.default_rng(11), batch=32, pairs=4, delay=8, overwrite=True, queries=3)
        b = generate(np.random.default_rng(11), batch=32, pairs=4, delay=8, overwrite=True, queries=3)
        torch.testing.assert_close(a[0], b[0]); torch.testing.assert_close(a[1], b[1])
        for row, targets in zip(a[0], a[1]):
            memory, answer = {}, []
            for op, key, val in row.tolist():
                if op == 0:
                    memory[key] = val
                if op == 2:
                    self.assertEqual(val, 4)
                    answer.append(memory[key])
            self.assertEqual(answer, targets.tolist())
        self.assertGreater(len(set(a[1][:, 0].tolist())), 1)

    def test_factorized_matrix_matches_explicit_and_gradients(self):
        model = MemoryModel(Config()).double()
        x, s = torch.randn(3, 32, dtype=torch.float64), torch.randn(3, 32, dtype=torch.float64)
        gate = model.gate(s).sigmoid()
        factored = model.u(gate * model.v(x))
        matrices = model.u.weight[None] @ torch.diag_embed(gate) @ model.v.weight[None]
        explicit = torch.einsum('bij,bj->bi', matrices, x)
        torch.testing.assert_close(factored, explicit)
        wanted = torch.autograd.grad(explicit.square().sum(), tuple(model.u.parameters()) + tuple(model.v.parameters()), retain_graph=True)
        got = torch.autograd.grad(factored.square().sum(), tuple(model.u.parameters()) + tuple(model.v.parameters()))
        for a, b in zip(wanted, got):
            torch.testing.assert_close(a, b)

    def test_causal_isolated_bounded_and_reset(self):
        events, _, positions = generate(np.random.default_rng(4), batch=4, pairs=3, queries=2)
        for cell, readout in VARIANTS.values():
            model = MemoryModel(Config(cell=cell, readout=readout)).double()
            logits, state = model(events, positions)
            self.assertTrue(bool((state.abs() <= 1).all()))
            solo, _ = model(events[1:2], positions)
            torch.testing.assert_close(logits[1:2], solo)
            early, _ = model(events[:, :positions[0] + 1], (positions[0],))
            torch.testing.assert_close(logits[:, :1], early)
            _, prefix = model(events[:, :positions[0]])
            suffix, _ = model(events[:, positions[0]:], tuple(p - positions[0] for p in positions), prefix)
            torch.testing.assert_close(logits, suffix)
            reset, _ = model(events[:, :positions[0] + 1], (positions[0],), intervention="reset")
            direct, _ = model(events[:, positions[0]:positions[0] + 1], (0,))
            torch.testing.assert_close(reset, direct)

    def test_gradients_initialization_and_checkpoint_resume(self):
        events, targets, positions = training_batch(np.random.default_rng(8), 400, 16)
        for cell, readout in VARIANTS.values():
            torch.manual_seed(18); model = MemoryModel(Config(cell=cell, readout=readout))
            torch.manual_seed(18); other = MemoryModel(Config(cell=cell, readout=readout))
            for a, b in zip(model.parameters(), other.parameters()):
                torch.testing.assert_close(a, b)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            loss = torch.nn.functional.cross_entropy(model(events, positions)[0].flatten(0, 1), targets.flatten())
            self.assertTrue(bool(loss.isfinite())); loss.backward()
            for name, p in model.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(bool(p.grad.isfinite().all()), name)
            optimizer.step()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'checkpoint.pt'
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, path)
                saved = torch.load(path, weights_only=True)
                other.load_state_dict(saved['model'])
                resumed = torch.optim.AdamW(other.parameters(), lr=.001)
                resumed.load_state_dict(saved['optimizer'])
                torch.testing.assert_close(model(events, positions)[0], other(events, positions)[0])
                for net, opt in ((model, optimizer), (other, resumed)):
                    opt.zero_grad()
                    net(events, positions)[0].square().mean().backward(); opt.step()
                for a, b in zip(model.parameters(), other.parameters()):
                    torch.testing.assert_close(a, b)

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable')
    def test_mps_forward_backward_matches_cpu(self):
        events, _, positions = generate(np.random.default_rng(2), batch=4, pairs=2, delay=3)
        for cell in ('gru', 'linear', 'gated', 'reaction'):
            cpu = MemoryModel(Config(cell=cell))
            mps = copy.deepcopy(cpu).to('mps')
            expected = cpu(events, positions)[0]
            actual = mps(events.to('mps'), positions)[0]
            torch.testing.assert_close(expected, actual.cpu(), atol=3e-6, rtol=3e-5)
            expected.square().mean().backward(); actual.square().mean().backward()
            for a, b in zip(cpu.parameters(), mps.parameters()):
                torch.testing.assert_close(a.grad, b.grad.cpu(), atol=3e-6, rtol=5e-4)


if __name__ == '__main__':
    unittest.main()
