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
                        memory_dim=8, bands=2, nodes=8, active_fraction=.5)
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

    def test_ordered_association_distinguishes_equal_mean_sequences(self):
        field = BioField(16, 2, 4, nodes=8, segment_tokens=8, phase_enabled=False,
                         dynamic_enabled=False, field_write_mode='ordered').double()
        with torch.no_grad():
            field.node_shape.zero_()
        controls = {name: torch.zeros(1, 2, 3, 2, 4, dtype=torch.float64)
                    for name in ('key', 'query', 'value')}
        controls['key'][..., 0] = 1; controls['query'][..., 0] = 1
        controls['value'][:, 0, ..., 1] = 1; controls['value'][:, 1, ..., 1] = -1
        reverse = {k: v.flip(1) for k, v in controls.items()}
        hidden = torch.zeros(1, 2, 16, dtype=torch.float64)
        first = field.write(hidden, controls, field.initial_state(1))
        second = field.write(hidden, reverse, field.initial_state(1))
        self.assertGreater((first['association']-second['association']).abs().max().item(), 1e-4)
        query = controls['query'].mean(2)
        read, _ = field.read(hidden, query, first)
        reversed_read, _ = field.read(hidden, query, second)
        self.assertGreater((read-reversed_read).abs().max().item(), 1e-5)
        summary = copy.deepcopy(field); summary.field_write_mode = 'summary'
        a = summary.write(hidden, controls, summary.initial_state(1))
        b = summary.write(hidden, reverse, summary.initial_state(1))
        assert_close(a['association'], b['association'], atol=1e-12, rtol=1e-12)
        self.assertEqual({k: v.shape for k, v in first.items()}, {k: v.shape for k, v in a.items()})
        self.assertEqual(first['association_updates'].item(), 2)
        self.assertEqual(a['association_updates'].item(), 1)
        for mode in ('ordered', 'summary'):
            field.field_write_mode = mode
            whole = field.accumulate(field.empty_summary(1), controls, hidden)
            split = field.accumulate(field.accumulate(field.empty_summary(1),
                    {k: v[:, :1] for k, v in controls.items()}, hidden[:, :1]),
                    {k: v[:, 1:] for k, v in controls.items()}, hidden[:, 1:])
            result = field.commit(whole, field.initial_state(1))
            split_result = field.commit(split, field.initial_state(1))
            for name in result:
                assert_close(result[name], split_result[name], atol=1e-12, rtol=1e-12)

    def test_fixed_control_buffers_overflow_and_budget_match(self):
        self.model.segment_tokens = 4
        empty = self.model.empty_summary(2, dtype=torch.float64)
        filled = self.model.accumulate(empty, self.controls, self.hidden)
        self.assertEqual(filled['tokens_key'].shape, (2, 4, 3, 2, 4))
        with self.assertRaises(ValueError):
            self.model.accumulate(filled, {k: v[:, :1] for k, v in self.controls.items()}, self.hidden[:, :1])
        self.model.field_write_mode = 'summary'
        other = self.model.empty_summary(2, dtype=torch.float64)
        self.assertEqual(sum(v.numel() for v in empty.values()), sum(v.numel() for v in other.values()))

    def test_scatter_buffers_match_old_loop_values_gradients_and_masked_splits(self):
        def old_loop(field, summary, controls, hidden, valid):
            key, value, query = (controls[name] for name in ('key', 'value', 'query'))
            b, t, _, h, d = key.shape
            weight = controls['band_weights']
            weight = weight/weight.sum(2, keepdim=True).clamp_min(1e-6)
            values = {name: (tensor*weight[..., None]).sum(2)
                      for name, tensor in (('key', key), ('value', value), ('query', query))}
            values['hidden'] = field.input_projection(hidden).reshape(b, t, h, d).tanh()
            result = dict(summary)
            for at in range(t):
                slot = torch.nn.functional.one_hot(result['count'].long().clamp_max(field.segment_tokens-1),
                                                   field.segment_tokens).to(key.dtype)*valid[:, at, None]
                for name, tensor in (('key', key), ('value', value), ('query', query)):
                    result['tokens_'+name] = result['tokens_'+name]+slot[:, :, None, None, None]*tensor[:, at, None]
                result['tokens_weight'] = result['tokens_weight']+slot[:, :, None, None]*weight[:, at, None]
                result['tokens_hidden'] = result['tokens_hidden']+slot[:, :, None, None]*values['hidden'][:, at, None]
                for name in values:
                    result[name] = result[name]+values[name][:, at]*valid[:, at, None, None]
                result['count'] = result['count']+valid[:, at].to(key.dtype)
            return result

        field = self.model
        initial = field.empty_summary(2, dtype=torch.float64)
        initial['count'] = torch.tensor([2., 1.], dtype=torch.float64)
        for name in initial:
            if name == 'count':
                continue
            initial[name] = torch.randn_like(initial[name])
            if name.startswith('tokens_'):
                initial[name][0, 2:] = 0; initial[name][1, 1:] = 0
        valid = torch.tensor([[True, False, True, False], [False, True, True, True]])
        coefficients = {name: torch.randn_like(value) for name, value in initial.items() if name != 'count'}

        def run(model, operation):
            controls = {name: value.detach().clone().requires_grad_() for name, value in self.controls.items()}
            hidden = self.hidden.detach().clone().requires_grad_()
            summary = {name: value.detach().clone().requires_grad_(name != 'count')
                       for name, value in initial.items()}
            result = operation(model, summary, controls, hidden, valid)
            loss = sum((result[name]*coefficient).sum() for name, coefficient in coefficients.items())
            inputs = [*controls.values(), hidden, *(value for name, value in summary.items() if name != 'count'),
                      model.input_projection.weight]
            gradients = torch.autograd.grad(loss, inputs)
            for gradient in gradients[:5]:
                assert_close(gradient[~valid], torch.zeros_like(gradient[~valid]), atol=0, rtol=0)
            return result, gradients

        actual, actual_gradient = run(field, lambda m, s, c, h, v: m.accumulate(s, c, h, v))
        expected, reference_gradient = run(copy.deepcopy(field), old_loop)
        for name in actual:
            assert_close(actual[name], expected[name], atol=0, rtol=0)
        for actual_value, reference_value in zip(actual_gradient, reference_gradient):
            assert_close(actual_value, reference_value, atol=2e-14, rtol=2e-14)
        split, split_gradient = run(copy.deepcopy(field), lambda m, s, c, h, v: m.accumulate(
            m.accumulate(s, {k: x[:, :1] for k, x in c.items()}, h[:, :1], v[:, :1]),
            {k: x[:, 1:] for k, x in c.items()}, h[:, 1:], v[:, 1:]))
        for name in actual:
            assert_close(actual[name], split[name], atol=1e-15, rtol=1e-15)
        for actual_value, split_value in zip(actual_gradient, split_gradient):
            assert_close(actual_value, split_value, atol=2e-14, rtol=2e-14)

    def test_pulse_fresh_coverage_and_insufficient_depth_regression(self):
        field = BioField(16, 2, 4, nodes=32, internal_steps=3).double()
        pulse = field.calibrate_pulse(.2)
        self.assertTrue(all(row['fresh_fraction'] == 1 for row in pulse['shells']))
        zero = field.calibrate_pulse(0)
        self.assertTrue(all(row['fresh_fraction'] == 0 for row in zero['shells'][1:]))
        field.internal_steps = 2
        old = field.calibrate_pulse(.2)
        self.assertEqual(old['shells'][-1]['fresh_fraction'], 0)
        with self.assertRaises(ValueError):
            BioField(16, 2, 4, nodes=32, internal_steps=2)

    def test_all_shells_receive_observations_and_time_constants_are_token_based(self):
        field = BioField(16, 2, 4, nodes=32, internal_steps=3, segment_tokens=4).double()
        hidden = self.hidden[:1]
        controls = {k: v[:1] for k, v in self.controls.items()}
        state = field.write(hidden, controls, field.initial_state(1))
        for level in range(field.shell_count):
            chosen = field.shell == level
            self.assertTrue(bool((state['fresh_reached'][:, chosen] > 0).all()))
            self.assertTrue(bool((state['support'][:, chosen] > 0).all()))
            self.assertGreater(state['association'][:, chosen].norm().item(), 0)
        # With zero observations, the isolated old-association term decays by elapsed tokens.
        empty_controls = {k: torch.zeros_like(v) for k, v in controls.items()}
        empty_controls.pop('band_weights', None)
        old = field.initial_state(1)
        old['association'].fill_(.1)
        new = field.write(torch.zeros_like(hidden), empty_controls, old)
        expected = old['association']*torch.exp(-4/field.raw_tau[-1].exp())
        assert_close(new['association'], expected, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
