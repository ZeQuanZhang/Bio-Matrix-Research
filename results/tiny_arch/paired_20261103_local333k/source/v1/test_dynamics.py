"""Mechanism and causal-interface checks; no training or ability claims."""
import copy
import unittest

import torch
from torch.testing import assert_close

from .dynamics import BioField


class FieldTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(203)
        self.model = BioField(16, 2, 4, nodes=8).double()
        self.hidden = torch.randn(2, 4, 16, dtype=torch.float64)
        self.controls = {name: torch.randn(2, 4, 3, 2, 4, dtype=torch.float64)
                         for name in ('query', 'key', 'value')}
        self.controls['band_weights'] = torch.randn(2, 4, 3, 2, dtype=torch.float64).softmax(2)
        self.query = self.controls['query'].mean(2)

    def state(self):
        return self.model.initial_state(2, dtype=torch.float64)

    def committed(self):
        return self.model.write(self.hidden, self.controls, self.state())

    def test_read_only_causal_and_fixed_state(self):
        state = self.committed()
        old = {name: value.clone() for name, value in state.items()}
        before, _ = self.model.read(self.hidden, self.query, state)
        changed = self.hidden.clone(); changed[:, -1] += 100
        after, _ = self.model.read(changed, self.query, state)
        assert_close(before[:, :-1], after[:, :-1], atol=0, rtol=0)
        for name in state:
            assert_close(state[name], old[name], atol=0, rtol=0)
        for _ in range(10):
            state = self.model.write(self.hidden, self.controls, state)
        self.assertEqual({k: v.shape for k, v in old.items()}, {k: v.shape for k, v in state.items()})
        self.assertEqual(state['credit_valid'].shape, (2, 8))

    def test_summary_split_and_lane_commit(self):
        empty = self.model.empty_summary(2, dtype=torch.float64)
        full = self.model.accumulate(empty, self.controls, self.hidden)
        half = {k: v[:, :2] for k, v in self.controls.items()}
        rest = {k: v[:, 2:] for k, v in self.controls.items()}
        split = self.model.accumulate(self.model.accumulate(empty, half, self.hidden[:, :2]), rest, self.hidden[:, 2:])
        for name in full:
            assert_close(full[name], split[name], atol=1e-15, rtol=1e-15)
        state = self.state()
        result = self.model.commit(full, state, torch.tensor([True, False]))
        for name in state:
            assert_close(state[name][1], result[name][1], atol=0, rtol=0)
        self.assertGreater((result['m'][0]-state['m'][0]).abs().max().item(), 1e-5)

    def test_all_learning_paths_have_finite_gradients(self):
        state = self.committed()
        state = self.model.write(self.hidden*.7, self.controls, state)
        out, _ = self.model.read(self.hidden, self.query, state)
        (out.square().mean()+.01*state['slow'].square().sum()).backward()
        names = ('node_shape', 'input_projection.weight', 'query_projection.weight',
                 'input_diagonal', 'input_rank', 'output_diagonal', 'output_rank',
                 'group_keys', 'raw_tau', 'raw_trace_tau', 'raw_frequency',
                 'raw_amplitude', 'raw_coupling', 'fourier', 'raw_reliability',
                 'raw_variance', 'raw_local_rate')
        parameters = dict(self.model.named_parameters())
        for name in names:
            gradient = parameters[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(bool(torch.isfinite(gradient).all()), name)
            self.assertGreater(gradient.abs().max().item(), 0, name)

    def test_probability_phase_trace_and_dynamic_affect_computation(self):
        state = self.committed()
        base, _ = self.model.read(self.hidden, self.query, state)
        variants = []
        p = state['p'].clone(); p[:, 0] *= 20
        variants.append(state | {'p': p})
        trace = state['trace'].clone(); trace[:, 0, :, :, 0] += 2
        variants.append(state | {'trace': trace})
        phase = state['phase'].clone(); phase[:, 0] = phase[:, 0].flip(-1)
        variants.append(state | {'phase': phase})
        for altered in variants:
            value, _ = self.model.read(self.hidden, self.query, altered)
            self.assertGreater((value-base).abs().max().item(), 1e-8)
        self.model.dynamic_enabled = False
        plain, _ = self.model.read(self.hidden, self.query, state)
        self.assertGreater((plain-base).abs().max().item(), 1e-8)
        eig = torch.linalg.eigvalsh(state['p'])
        self.assertGreater(eig.min().item(), 0)
        assert_close(state['phase'].square().sum(-1), torch.ones_like(state['phase'][..., 0]), atol=1e-6, rtol=1e-6)

    def test_phase_disabled_disables_coupling_and_fourier(self):
        self.model.phase_enabled = False
        one = self.committed()
        with torch.no_grad():
            self.model.raw_frequency.add_(9)
            self.model.fourier.add_(20)
            self.model.raw_coupling.add_(10)
            self.model.raw_amplitude.add_(10)
        two = self.committed()
        for name in one:
            assert_close(one[name], two[name], atol=0, rtol=0)

    def test_fourier_coupling_and_group_membership_enter_messages(self):
        state = self.committed()
        base = self.model.write(self.hidden, self.controls, state)
        for name in ('fourier', 'raw_coupling', 'group_keys'):
            altered = copy.deepcopy(self.model)
            with torch.no_grad():
                parameter = dict(altered.named_parameters())[name]
                if name == 'group_keys':
                    parameter[0].add_(2)
                else:
                    parameter.add_(2)
            result = altered.write(self.hidden, self.controls, state)
            out, _ = altered.read(self.hidden, self.query, result)
            reference, _ = self.model.read(self.hidden, self.query, base)
            self.assertGreater((out-reference).abs().max().item(), 1e-9, name)

    def test_zero_packet_does_not_jump_outward(self):
        state = self.state()
        zero = torch.zeros(2, 8, 2, 4, dtype=torch.float64)
        mass = torch.zeros(2, 8, dtype=torch.float64); mass[:, 0] = 1
        message, _, edges = self.model._flow(zero, mass, state, state['phase'])
        assert_close(message, zero, atol=0, rtol=0)
        self.assertEqual(edges[:, self.model.outward].abs().max().item(), 0)
        with self.assertRaises(ValueError):
            BioField(16, 2, 4, nodes=8, active_fraction=0)

    def test_sparse_view_executes_compact_legal_subgraph(self):
        self.model.active_fraction = .5
        state = self.state()
        route = self.model.route(state)
        view = self.model.induced_view(route)
        self.assertLess(view.nodes, self.model.nodes)
        self.assertEqual(view.node_shape.shape[0], len(route))
        self.assertTrue(bool(torch.isin(self.model.required_path, route).all()))
        for target in range(view.nodes):
            for slot in range(view.neighbors.shape[1]):
                if view.edge_mask[target, slot]:
                    source = route[view.neighbors[target, slot]]
                    legal = ((self.model.neighbors[route[target]] == source)
                             & self.model.edge_mask[route[target]].bool()).any()
                    self.assertTrue(bool(legal))
        observed = []
        original = BioField._mapped
        from unittest.mock import patch
        def track(field, *args, **kwargs):
            observed.append(field.nodes)
            return original(field, *args, **kwargs)
        with patch.object(BioField, '_mapped', track):
            changed = self.model.write(self.hidden, self.controls, state)
            _, diag = self.model.read(self.hidden, self.query, changed)
        self.assertTrue(observed)
        self.assertLess(max(observed), self.model.nodes)
        self.assertLess(diag['active_fraction'].item(), 1)

    def test_sparse_inactive_states_frozen_and_credit_ids_shift_once(self):
        self.model.active_fraction = .5
        self.model.automatic_feedback = False
        state = self.state()
        state['credit_key'] = torch.arange(state['credit_key'].numel(), dtype=torch.float64).reshape_as(state['credit_key'])
        state['credit_weight'].fill_(1)
        state['credit_valid'].fill_(True)
        route = self.model.route(state)
        inactive = torch.tensor([i for i in range(8) if i not in route.tolist()])
        result = self.model.write(self.hidden, self.controls, state)
        for name in ('m', 'p', 'fast', 'slow', 'trace', 'support', 'innovation', 'phase'):
            assert_close(result[name][:, inactive], state[name][:, inactive], atol=0, rtol=0)
        for name in ('credit_key', 'credit_value', 'credit_prediction', 'credit_weight'):
            assert_close(result[name][:, 0, inactive], torch.zeros_like(result[name][:, 0, inactive]), atol=0, rtol=0)
            assert_close(result[name][:, 1:, inactive], state[name][:, :-1, inactive], atol=0, rtol=0)
        self.assertEqual({k: v.shape for k, v in state.items()}, {k: v.shape for k, v in result.items()})

    def test_sparse_future_credit_is_not_applied_to_reassigned_node(self):
        self.model.active_fraction = .5
        state = self.committed()
        route = self.model.route(state)
        inactive = torch.tensor([i for i in range(8) if i not in route.tolist()])
        state['credit_weight'][:, :, inactive] = 100
        before = state['slow'][:, inactive].clone()
        result = self.model.write(self.hidden, self.controls, state)
        assert_close(result['slow'][:, inactive], before, atol=0, rtol=0)
        # The selected-view credit frame follows global IDs rather than view order.
        old = self.model._gather(state, 0, route)
        summary = self.model.accumulate(self.model.empty_summary(2, dtype=torch.float64), self.controls, self.hidden)
        local = self.model.induced_view(route).commit({k: v[:1] for k, v in summary.items()}, old)
        assert_close(result['slow'][:1, route], local['slow'], atol=0, rtol=0)

    def test_sparse_full_selection_is_dense_and_preserves_parameter_gradients(self):
        state = self.committed()
        dense, _ = self.model.read(self.hidden, self.query, state)
        all_nodes = torch.arange(8)
        compact_all = self.model.induced_view(all_nodes)
        reference, _ = compact_all.read(self.hidden, self.query, state)
        assert_close(dense, reference, atol=0, rtol=0)
        self.model.active_fraction = .5
        result = self.model.write(self.hidden, self.controls, state)
        value, _ = self.model.read(self.hidden, self.query, result)
        value.square().sum().backward()
        for name in ('node_shape', 'node_address', 'raw_frequency', 'raw_amplitude', 'input_rank'):
            grad = dict(self.model.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(bool(torch.isfinite(grad).all()), name)
            self.assertGreater(grad.abs().max().item(), 0, name)

    def test_sparse_network_arbitrary_calls_and_future_causality(self):
        from .model import Config, Network
        from .test_model import assert_tree_close
        config = Config(vocab_size=31, width=16, layers=1, heads=2, ff=32,
                        context_limit=8, segment_tokens=4, memory_heads=2,
                        memory_dim=4, bands=2, nodes=8, active_fraction=.5)
        model = Network(config).double()
        tokens = torch.randint(31, (2, 13))
        output, state, _ = model(tokens)
        outputs, split_state, at = [], None, 0
        for length in (1, 2, 5, 1, 4):
            value, split_state, _ = model(tokens[:, at:at+length], split_state)
            at += length; outputs.append(value)
        assert_close(torch.cat(outputs, 1), output, atol=2e-9, rtol=2e-9)
        assert_tree_close(self, split_state, state)
        future = tokens.clone(); future[:, 7:] = (future[:, 7:]+1) % 31
        perturbed, _, _ = model(future)
        assert_close(perturbed[:, :7], output[:, :7], atol=0, rtol=0)

    def test_local_feedback_consumed_once_and_preserves_orthogonal_subspace(self):
        state = self.state()
        key = torch.zeros(2, 8, 2, 4, dtype=torch.float64); key[..., 0] = 1
        state['credit_key'][:, 0] = key
        state['credit_valid'][:, 0] = True
        state['credit_weight'][:, 0] = 1
        state['slow'][..., 1] = .2
        old = state['slow'].clone()
        observed = torch.ones_like(key)*.3
        changed = self.model.apply_feedback(state, observed)
        self.assertGreater((changed['slow'][..., 0]-old[..., 0]).abs().max().item(), 1e-6)
        assert_close(changed['slow'][..., 1:], old[..., 1:], atol=0, rtol=0)
        repeated = self.model.apply_feedback(changed, observed)
        assert_close(changed['slow'], repeated['slow'], atol=0, rtol=0)
        with self.assertRaises(ValueError):
            self.model.apply_feedback(state, observed, lag=9)

    def test_delayed_credit_is_finite_and_explicit(self):
        self.model.automatic_feedback = False
        state = self.committed()
        state = self.model.write(self.hidden, self.controls, state)
        self.assertTrue(bool(state['credit_valid'][:, :2].all()))
        observed = torch.ones_like(state['credit_prediction'][:, 1])*.2
        changed = self.model.apply_feedback(state, observed, lag=2)
        self.assertTrue(bool(changed['credit_valid'][:, 0].all()))
        self.assertFalse(bool(changed['credit_valid'][:, 1].any()))
        self.assertGreater(changed['slow'].norm().item(), 0)

    def test_consolidation_transaction_persists_and_can_be_corrected(self):
        model = self.model
        state = {name: value[:1].clone() for name, value in self.state().items()}
        state['support'].fill_(4); state['innovation'].zero_()
        state['slow'].zero_(); state['slow'][..., 0, 0] = 1
        old = model.node_shape.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters())
        optimizer.state[model.node_shape] = {'step': torch.tensor(1.)}
        proposal = model.consolidate_proposal(state, rate=.1)
        self.assertTrue(bool(proposal['eligible'].all()))
        diag = model.apply_consolidation(proposal, optimizer)
        self.assertGreater(diag['update_norm'], 0)
        self.assertNotIn(model.node_shape, optimizer.state)
        fresh = model.initial_state(1, dtype=torch.float64)
        self.assertGreater((fresh['m'][0]-old).abs().max().item(), 1e-4)
        assert_close(fresh['slow'], torch.zeros_like(fresh['slow']), atol=0, rtol=0)
        forward = model.node_shape.detach().clone()
        corrected = fresh | {'support': torch.full_like(fresh['support'], 4),
                             'innovation': torch.zeros_like(fresh['innovation']),
                             'slow': -torch.ones_like(fresh['slow'])}
        model.consolidate(corrected, rate=.1)
        self.assertLess(model.node_shape[..., 0, 0].mean().item(), forward[..., 0, 0].mean().item())
        with self.assertRaises(ValueError):
            model.apply_consolidation(proposal)
        with self.assertRaises(ValueError):
            model.consolidate_proposal(self.state())

    def test_unverified_experience_cannot_consolidate(self):
        state = self.state()
        low = self.model.consolidate_proposal(state, allow_multilane=True)
        assert_close(low['delta'], torch.zeros_like(low['delta']), atol=0, rtol=0)
        state['support'].fill_(10); state['innovation'].fill_(1)
        conflict = self.model.consolidate_proposal(state, allow_multilane=True)
        assert_close(conflict['delta'], torch.zeros_like(conflict['delta']), atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
