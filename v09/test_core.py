"""CPU references for full-content delivery and equivalent algebra reuse."""
from dataclasses import replace
import unittest

import torch
from torch.nn import functional as F

from v08.model import Config as PreviousConfig, Network as PreviousNetwork
from v08.model import initialize as previous_initialize
from .core import Config, Network, initialize


class CoreChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def options(self):
        return dict(vocab_size=32, nodes=16, matrix_size=4, width=16,
                    bands=2, modes=2, block_size=4, internal_steps=4)

    def model(self, **options):
        model = Network(Config(**(self.options() | options))).double()
        initialize(model, 901); model.gate_blend = 0.
        return model

    def compare_states(self, reference, actual):
        self.assertEqual(reference.keys(), actual.keys())
        for name in reference:
            torch.testing.assert_close(reference[name], actual[name], atol=3e-11, rtol=2e-9, msg=name)

    def test_execution_reuse_matches_v08_all_states_outputs_and_gradients(self):
        old = PreviousNetwork(PreviousConfig(**self.options(), implementation='sequential')).double()
        previous_initialize(old, 901); old.gate_blend = 0.
        models = [old]+[self.model(content_base=False, address_size=0, reuse_controls=reuse) for reuse in (False, True)]
        for model in models[1:]:
            model.load_state_dict(old.state_dict())
        tokens = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]])
        results = []
        for model in models:
            output, state, _ = model(tokens)
            F.cross_entropy(output[:, :-1].flatten(0, 1), tokens[:, 1:].flatten()).backward()
            results.append((output, state))
        for output, state in results[1:]:
            torch.testing.assert_close(results[0][0], output, atol=2e-11, rtol=2e-9)
            self.compare_states(results[0][1], state)
        reference = dict(old.named_parameters())
        for model in models[1:]:
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(reference[name].grad, parameter.grad, atol=3e-10, rtol=2e-8, msg=name)

    def test_base_reaches_actual_read_ports_without_attenuation_or_new_edges(self):
        model = self.model().eval()
        tokens = torch.tensor([[1, 3, 5, 7]])
        state = model.initial_state(1)
        with torch.no_grad():
            coefficients = model.block_coefficients(tokens, state)
            fresh = model.content_projection(model.embedding(tokens)).reshape(4, 4, 4)
        read = model.read_ports.bool()
        mass = coefficients['content_base_mass']
        torch.testing.assert_close(mass[:, read], torch.ones_like(mass[:, read]), atol=0, rtol=0)
        for node in read.nonzero().flatten():
            torch.testing.assert_close(coefficients['content_base_transport'][:, node], fresh, atol=0, rtol=0)
        parent, valid = model.base_parent()
        for target in valid.nonzero().flatten():
            legal = model.neighbors[target][model.edge_mask[target].bool()]
            self.assertIn(int(parent[target]), legal.tolist())
            self.assertEqual(int(model.shell[target]-model.shell[parent[target]]), 1)
        short = self.model(internal_steps=1).eval()
        with torch.no_grad():
            partial = short.block_coefficients(tokens, short.initial_state(1))
        self.assertEqual(float((partial['content_base_mass']*short.read_ports).sum()), 0.)

    def test_cut_graph_and_closed_edges_prevent_fresh_output_bypass(self):
        tokens = torch.tensor([[1, 3, 5, 7]])
        changed = torch.tensor([[2, 4, 6, 8]])
        for cut in ('cut', 'closed'):
            with self.subTest(cut=cut):
                model = self.model(topology='cut' if cut == 'cut' else 'connected').eval()
                if cut == 'closed':
                    model.edge_mask.zero_()
                with torch.no_grad():
                    a, _, diag = model(tokens)
                    b, _, _ = model(changed)
                torch.testing.assert_close(a, b, atol=0, rtol=0)
                self.assertEqual(float(diag['content_base_surface_mass']), 0.)

    def test_current_content_is_distinguishable_and_full_matrix_with_small_addresses(self):
        model = self.model(address_size=2).eval()
        tokens = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15]])
        changed = tokens.clone(); changed[:, 2] = 4
        output, state, _ = model(tokens)
        other, _, _ = model(changed)
        torch.testing.assert_close(output[:, :2], other[:, :2], atol=0, rtol=0)
        self.assertGreater(float((output[:, 2]-other[:, 2]).detach().norm()), 1e-3)
        F.cross_entropy(output[:, :-1].flatten(0, 1), tokens[:, 1:].flatten()).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(parameter.grad.isfinite().all()), name)
        self.assertGreater(float(model.content_projection.weight.grad.norm()), 0.)
        self.assertGreater(float(model.address_projection.weight.grad.norm()), 0.)
        self.assertEqual(state['x'].shape, (1, 16, 2, 4, 4))
        self.assertEqual(model.address_projection.out_features, 2)

    def test_scaled_direct_slow_read_local_direction_matches_autograd(self):
        model = self.model().eval(); b, n, d = 2, 16, 4
        state = model.initial_state(b)
        with torch.no_grad():
            coefficients = model.block_coefficients(torch.tensor([[1], [2]]), state)
        slow = (torch.randn(b, n, d, d, dtype=torch.float64)*.01).requires_grad_()
        state = state | {'slow': slow}
        fast = torch.randn(b, n, model.cfg.bands, 1, d, d, dtype=torch.float64)*.01
        logits, _, reads = model.read_block(fast, coefficients, state, b, 1)
        labels = torch.tensor([3, 5])
        loss = F.cross_entropy(logits[:, 0], labels, reduction='sum')
        reference, = torch.autograd.grad(loss, slow)
        cached = model.cache_feedback(state, reads['hidden'], logits[:, 0], reads['query'],
                   reads['weight'], reads['scale'], reads['gain'])
        local = model.local_direction(cached, model.embedding(labels))
        torch.testing.assert_close(local, -reference, atol=1e-11, rtol=1e-9)
        self.assertFalse(local.requires_grad)

    def test_compact_graph_accessors_execute_fewer_actual_nodes(self):
        class Compact(Network):
            def __init__(self, cfg, keep):
                super().__init__(cfg)
                self.register_buffer('keep', keep)
                mapping = torch.full((cfg.nodes,), -1, dtype=torch.long)
                mapping[keep] = torch.arange(len(keep))
                neighbor = mapping[self.neighbors[keep]]
                self.view = {name: getattr(self, name)[keep] for name in
                             ('positions', 'shell', 'outward', 'write_ports', 'read_ports')}
                self.view['edge_mask'] = self.edge_mask[keep]*(neighbor >= 0)
                self.view['neighbors'] = neighbor.clamp_min(0)

            def graph_buffer(self, name):
                return self.view[name]

            def node_anchor(self):
                return self.node_shape[self.keep]

            def phase_initializer(self):
                return self.initial_phase[self.keep]

        model = Compact(Config(**self.options(), address_size=2), torch.arange(10)).double()
        # Nonregistered view tensors are regenerated after the test's dtype conversion.
        model.view = {name: value.to(dtype=torch.float64) if value.is_floating_point() else value
                      for name, value in model.view.items()}
        initialize(model, 901)
        tokens = torch.tensor([[1, 3, 5, 7]])
        output, state, diag = model(tokens)
        output.square().mean().backward()
        self.assertEqual(state['m'].shape[1], 10)
        self.assertEqual(state['x'].shape[1], 10)
        self.assertEqual(state['credit_query'].shape[2], 10)
        self.assertEqual(float(diag['active_nodes']), 10.)
        self.assertTrue(bool(output.isfinite().all()))
        self.assertTrue(bool(model.node_shape.grad[10:].eq(0).all()))

    def test_base_only_read_is_exact_linear_content_baseline_but_keeps_states(self):
        model = self.model().eval(); model.runtime_bio_read_enabled = False
        tokens = torch.tensor([[1, 3, 5, 7]])
        state = model.initial_state(1)
        changed = state | {'slow': torch.randn_like(state['slow'])*.1,
                           'm': torch.randn_like(state['m'])*.1,
                           'x': torch.randn_like(state['x'])*.1}
        with torch.no_grad():
            actual, end, _ = model(tokens, state)
            other, _, _ = model(tokens, changed)
            content = model.content_projection(model.embedding(tokens))
            pooled = torch.cat((content, torch.zeros_like(content)), -1)
            reference = F.linear(model.head(model.read_projection(pooled)), model.embedding.weight)
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(other, reference, atol=1e-12, rtol=1e-10)
        self.assertGreater(float(end['x'].abs().sum()), 0.)
        self.assertGreater(int(end['evidence_count'].sum()), 0)
        self.assertEqual(float(model.local_direction(end, model.embedding(torch.tensor([9]))).abs().sum()), 0.)
        cut = self.model(topology='cut').eval(); cut.runtime_bio_read_enabled = False
        with torch.no_grad():
            output, _, diag = cut(tokens)
            zero = torch.zeros(1, 4, 2*model.cfg.matrix_size**2, dtype=torch.float64)
            empty = F.linear(cut.head(cut.read_projection(zero)), cut.embedding.weight)
        torch.testing.assert_close(output, empty, atol=1e-12, rtol=1e-10)
        self.assertEqual(float(diag['content_base_surface_mass']), 0.)


if __name__ == '__main__':
    unittest.main()
