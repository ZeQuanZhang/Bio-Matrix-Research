"""CPU checks of real node selection, global credit, and fixed-block streaming."""
import unittest
from unittest.mock import patch

import torch

from .core import initialize
from .model import Config, Network


class RoutingStreamChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, **options):
        settings = dict(vocab_size=32, nodes=16, matrix_size=4, width=16,
                        bands=2, modes=2, block_size=4, implementation='sequential',
                        group_size=2, active_fraction=.5, credit_horizon=8)
        settings.update(options)
        model = Network(Config(**settings)).double()
        initialize(model, 909)
        model.gate_blend = 0.
        return model.eval()

    def clone_state(self, state):
        return {name: value.detach().clone() for name, value in state.items()}

    def test_full_selection_skips_compaction_and_keeps_router_gradient(self):
        model = self.model(group_size=16, route_mode='summary')
        tokens = torch.tensor([[1, 2, 3, 4]])
        with patch.object(model, '_gather', side_effect=AssertionError('full selection must not gather')):
            output, state, diag = model(tokens)
        self.assertEqual(float(diag['active_fraction']), 1.)
        self.assertEqual(state['x'].shape[1], model.cfg.nodes)
        output.square().mean().backward()
        for parameter in (model.route_key.weight, model.route_query.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(parameter.grad.isfinite().all()))
            self.assertGreater(float(parameter.grad.norm()), 0.)

    def assert_state_close(self, expected, actual, atol=2e-11, rtol=2e-9):
        self.assertEqual(expected.keys(), actual.keys())
        for name in expected:
            torch.testing.assert_close(expected[name], actual[name], atol=atol, rtol=rtol, msg=name)

    def active_ids(self, model, lane=0, block=-1):
        return model.last_routes[block][lane]['active_ids']

    def test_selected_nodes_are_the_only_nodes_executed_and_idle_state_freezes(self):
        model = self.model(route_mode='cyclic')
        model.runtime_automatic_feedback = False
        state = model.initial_state(1)
        generator = torch.Generator().manual_seed(42)
        for name in ('m', 'x', 'slow', 'trace', 'support', 'innovation', 'response'):
            state[name] = torch.randn(state[name].shape, generator=generator, dtype=torch.float64)*.02
        before = self.clone_state(state)
        executed = []
        original = model.block_coefficients

        def capture(tokens, compact):
            executed.append(compact['m'].shape[1])
            return original(tokens, compact)

        with torch.no_grad(), patch.object(model, 'block_coefficients', side_effect=capture):
            _, end, _ = model(torch.tensor([[1, 2, 3, 4]]), state)
        ids = self.active_ids(model)
        self.assertLess(len(ids), model.cfg.nodes)
        self.assertEqual(executed, [len(ids)])
        active = torch.zeros(model.cfg.nodes, dtype=torch.bool)
        active[ids] = True
        for name in ('m', 'p', 'x', 'slow', 'z', 'trace', 'support', 'innovation', 'response', 'evidence_count'):
            torch.testing.assert_close(end[name][:, ~active], before[name][:, ~active], atol=0, rtol=0, msg=name)
        self.assertGreater(float((end['x'][:, active]-before['x'][:, active]).abs().max()), 0.)

    def test_old_global_credit_rolls_on_original_nodes_when_selection_changes(self):
        model = self.model(route_mode='cyclic')
        model.runtime_automatic_feedback = False
        state = model.initial_state(1)
        n, d = model.cfg.nodes, model.cfg.matrix_size
        state['read_query'] = torch.arange(n*d, dtype=torch.float64).reshape(1, n, d)+.25
        state['read_weight'] = torch.arange(n, dtype=torch.float64).reshape(1, n)+.5
        state['feedback_gain'] = torch.arange(n, dtype=torch.float64).reshape(1, n)+.75
        state['credit_query'] = torch.arange((model.cfg.credit_horizon-1)*n*d,
                                           dtype=torch.float64).reshape(1, -1, n, d)+1000.
        state['credit_weight'].fill_(17.)
        state['credit_gain'].fill_(19.)
        state['feedback_valid'].fill_(True)
        state['credit_valid'].fill_(True)
        before = self.clone_state(state)
        with torch.no_grad():
            _, end, _ = model(torch.tensor([[1, 2, 3, 4]]), state)
        for current, past in (('read_query', 'credit_query'), ('read_weight', 'credit_weight'),
                              ('feedback_gain', 'credit_gain')):
            torch.testing.assert_close(end[past][:, 3], before[current], atol=0, rtol=0, msg=current)
            torch.testing.assert_close(end[past][:, 4], before[past][:, 0], atol=0, rtol=0, msg=past)
        self.assertTrue(bool(end['credit_valid'].all()))
        with torch.no_grad():
            _, second, _ = model(torch.tensor([[5, 6, 7, 8]]), end)
        # Four more observations move the first block's final read to lag five.
        for current, past in (('read_query', 'credit_query'), ('read_weight', 'credit_weight'),
                              ('feedback_gain', 'credit_gain')):
            torch.testing.assert_close(second[past][:, 3], end[current], atol=0, rtol=0, msg=past)

    def test_pending_feedback_is_consumed_globally_before_new_compact_execution(self):
        model = self.model(route_mode='cyclic')
        model.runtime_automatic_feedback = False
        first = torch.tensor([[1, 2, 3, 4]])
        following = torch.tensor([[5, 6, 7, 8]])
        with torch.no_grad():
            _, state, _ = model(first)
            previous_ids = self.active_ids(model).clone()
            for _ in range(len(model.groups)+1):
                _, candidate, _ = model(following, self.clone_state(state))
                next_ids = self.active_ids(model).clone()
                abandoned = [int(node) for node in previous_ids if not bool((next_ids == node).any())]
                if abandoned:
                    break
                state, previous_ids = candidate, next_ids
        self.assertTrue(abandoned, 'cyclic routing must exercise a change of selected nodes')
        node = abandoned[0]
        pending = self.clone_state(state)
        pending['slow'].zero_()
        pending['read_query'].zero_(); pending['read_query'][0, node, 0] = 1.
        pending['read_weight'].zero_(); pending['read_weight'][0, node] = 1.
        pending['feedback_gain'].zero_(); pending['feedback_gain'][0, node] = 1.
        pending['read_scale'].fill_(1.)
        pending['prediction'].zero_()
        pending['feedback_h'].fill_(.02)
        pending['feedback_valid'].fill_(True)
        model.runtime_automatic_feedback = True
        with torch.no_grad():
            expected, _ = model._apply_feedback(self.clone_state(pending), following[:, 0])
            _, actual, _ = model(following, pending)
        self.assertFalse(bool((self.active_ids(model) == node).any()))
        self.assertGreater(float(expected['slow'][0, node].abs().sum()), 0.)
        torch.testing.assert_close(actual['slow'][0, node], expected['slow'][0, node], atol=0, rtol=0)
        torch.testing.assert_close(expected['group_summary'], model._summaries(expected), atol=0, rtol=0)
        torch.testing.assert_close(actual['group_summary'], model._summaries(actual), atol=0, rtol=0)
        self.assertFalse(bool(actual['credit_valid'][0, 3]))

    def test_each_batch_lane_routes_and_predicts_independently(self):
        model = self.model(route_mode='summary')
        model.runtime_automatic_feedback = False
        tokens = torch.tensor([[1, 2, 3, 4], [9, 8, 7, 6]])
        state = model.initial_state(2)
        state['x'][1, model.cfg.nodes//2:] = .3
        state['slow'][1, :model.cfg.nodes//2] = .1
        # Force two distinct routes, exercising separate compact batches.
        with torch.no_grad():
            model.route_key.weight.zero_(); model.route_query.weight.zero_()
            model.route_key.weight[:2, :2] = torch.eye(2)
            model.route_query.weight[:2, :2] = torch.eye(2)
        state['group_summary'].zero_(); state['group_summary'][..., 1] = .01
        optional = [group for group in model.components[-1] if group not in model.mandatory_groups]
        self.assertGreaterEqual(len(optional), 3)
        for lane, group in enumerate(optional[-2:]):
            state['group_summary'][lane, group].zero_()
            state['group_summary'][lane, group, 0] = 2.
        with torch.no_grad():
            batched, end, _ = model(tokens, self.clone_state(state))
            batched_ids = self.active_ids(model, lane=0).clone()
            other_ids = self.active_ids(model, lane=1).clone()
            alone, single, _ = model(tokens[:1], {name: value[:1].clone() for name, value in state.items()})
            single_ids = self.active_ids(model).clone()
        self.assertFalse(torch.equal(batched_ids, other_ids))
        torch.testing.assert_close(batched_ids, single_ids, atol=0, rtol=0)
        torch.testing.assert_close(batched[:1], alone, atol=2e-11, rtol=2e-9)
        self.assert_state_close({name: value[:1] for name, value in end.items()}, single)

    def test_cut_prevents_input_and_history_from_selecting_or_influencing_other_component(self):
        model = self.model(topology='cut', route_mode='summary')
        model.runtime_automatic_feedback = False
        tokens = torch.tensor([[1, 2, 3, 4]])
        changed_tokens = torch.tensor([[8, 7, 6, 5]])
        state = model.initial_state(1)
        inner = model.shell < model.cfg.shells//2
        outer = ~inner
        changed = self.clone_state(state)
        changed['x'][:, inner] = .4
        changed['slow'][:, inner] = -.2
        changed['m'][:, inner] = .1
        changed['group_summary'] = model._summaries(changed)
        with torch.no_grad():
            reference, end, _ = model(tokens, self.clone_state(state))
            ids = self.active_ids(model).clone()
            other, altered_end, _ = model(changed_tokens, changed)
            changed_ids = self.active_ids(model).clone()
        torch.testing.assert_close(ids[outer[ids]], changed_ids[outer[changed_ids]], atol=0, rtol=0)
        torch.testing.assert_close(reference, other, atol=0, rtol=0)
        for name in ('m', 'p', 'x', 'slow', 'z', 'trace', 'support', 'innovation', 'response'):
            torch.testing.assert_close(end[name][:, outer], altered_end[name][:, outer], atol=0, rtol=0, msg=name)

    def test_future_tokens_do_not_change_past_outputs_or_block_start_route(self):
        model = self.model(route_mode='summary')
        tokens = torch.tensor([[1, 2, 3, 4]])
        changed = tokens.clone(); changed[:, 2:] = torch.tensor([[14, 15]])
        with torch.no_grad():
            output, _, _ = model(tokens)
            ids = self.active_ids(model).clone()
            altered, _, _ = model(changed)
            changed_ids = self.active_ids(model).clone()
        torch.testing.assert_close(ids, changed_ids, atol=0, rtol=0)
        torch.testing.assert_close(output[:, :2], altered[:, :2], atol=0, rtol=0)
        self.assertGreater(float((output[:, 2:]-altered[:, 2:]).abs().max()), 1e-8)

    def test_cut_blocks_global_teacher_feedback_into_warm_unreachable_memories(self):
        model = self.model(topology='cut', route_mode='summary')
        state = model.initial_state(1)
        unreachable = ~model.feedback_reachable
        self.assertTrue(bool(unreachable.any()))
        generator = torch.Generator().manual_seed(2718)
        for name in ('slow', 'x', 'm', 'trace', 'response'):
            state[name][:, unreachable] = torch.randn(state[name][:, unreachable].shape,
                                      generator=generator, dtype=torch.float64)*.03
        state['support'][:, unreachable] = 4.
        state['read_query'][:, unreachable, 0] = 1.
        state['read_weight'][:, unreachable] = 1.
        state['feedback_gain'][:, unreachable] = 1.
        state['feedback_valid'].fill_(True)
        state['feedback_h'] = torch.randn(state['feedback_h'].shape, generator=generator, dtype=torch.float64)*.02
        state['group_summary'] = model._summaries(state)
        before = self.clone_state(state)
        with torch.no_grad():
            direction = model.local_direction(state, model.embedding(torch.tensor([1])))
            self.assertGreater(float(direction[:, unreachable].abs().sum()), 0.)
            consumed, _ = model._apply_feedback(self.clone_state(state), torch.tensor([1]))
            a, end_a, _ = model(torch.tensor([[1, 2, 3, 4]]), self.clone_state(state))
            b, end_b, _ = model(torch.tensor([[8, 7, 6, 5]]), self.clone_state(state))
        self.assertFalse(bool(consumed['feedback_valid'].any()))
        torch.testing.assert_close(consumed['slow'][:, unreachable], before['slow'][:, unreachable], atol=0, rtol=0)
        torch.testing.assert_close(end_a['slow'][:, unreachable], before['slow'][:, unreachable], atol=0, rtol=0)
        torch.testing.assert_close(end_b['slow'][:, unreachable], before['slow'][:, unreachable], atol=0, rtol=0)
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_memory_interventions_refresh_cached_group_summaries(self):
        model = self.model()
        state = model.initial_state(1)
        state['slow'].fill_(.2); state['x'].fill_(.1)
        state['group_summary'] = model._summaries(state)
        state['feedback_valid'].fill_(True); state['credit_valid'].fill_(True)
        cleared = model.clear_slow(state)
        torch.testing.assert_close(cleared['group_summary'], model._summaries(cleared), atol=0, rtol=0)
        self.assertGreater(float((state['group_summary']-cleared['group_summary']).abs().sum()), 0.)
        self.assertFalse(bool(cleared['feedback_valid'].any()))
        self.assertFalse(bool(cleared['credit_valid'].any()))
        reset = model.reset_fast(state)
        torch.testing.assert_close(reset['group_summary'], model._summaries(reset), atol=0, rtol=0)
        torch.testing.assert_close(reset['slow'], state['slow'], atol=0, rtol=0)
        torch.testing.assert_close(reset['x'], torch.zeros_like(reset['x']), atol=0, rtol=0)

    def test_training_curriculum_delays_actual_sparse_execution(self):
        model = self.model()
        for completed, gate, fraction in ((0, 0., 1.), (5, .5, 1.), (10, 1., 1.),
                                           (20, 1., .75), (30, 1., .5), (100, 1., .5)):
            with self.subTest(completed=completed):
                schedule = model.training_progress(completed, 100)
                self.assertAlmostEqual(schedule['gate_blend'], gate)
                self.assertAlmostEqual(schedule['active_fraction'], fraction)
                self.assertAlmostEqual(model.gate_blend, gate)
                self.assertAlmostEqual(model.active_fraction, fraction)
        for completed, total in ((-1, 100), (101, 100), (0, 0)):
            with self.assertRaises(ValueError):
                model.training_progress(completed, total)

    def test_streaming_preserves_full_block_schedule_outputs_and_every_state_field(self):
        model = self.model(route_mode='cyclic')
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [9, 8, 7, 6, 5, 4, 3, 2]])
        with torch.no_grad():
            expected, end, _ = model(tokens)
            stream = model.initial_stream_state(2)
            outputs = []
            for position in range(tokens.shape[1]):
                before = self.clone_state(stream['committed'])
                output, stream, _ = model.step(tokens[:, position], stream)
                outputs.append(output)
                if (position+1) % model.cfg.block_size:
                    self.assert_state_close(before, stream['committed'], atol=0, rtol=0)
                    self.assertEqual(stream['tokens'].shape[1], (position+1) % model.cfg.block_size)
            self.assertEqual(stream['tokens'].shape[1], 0)
        torch.testing.assert_close(expected, torch.stack(outputs, 1), atol=2e-11, rtol=2e-9)
        self.assert_state_close(end, stream['committed'])

    def test_stream_rejects_parameter_changes_while_prefix_is_pending(self):
        model = self.model()
        with torch.no_grad():
            stream = model.initial_stream_state(1)
            _, stream, _ = model.step(torch.tensor([1]), stream)
            model.raw_content_gain.add_(.01)
            with self.assertRaises((ValueError, RuntimeError)):
                model.step(torch.tensor([2]), stream)

    def test_stream_rejects_budget_runtime_and_configuration_changes(self):
        model = self.model()
        with torch.no_grad():
            stream = model.initial_stream_state(1)
            _, stream, _ = model.step(torch.tensor([1]), stream)
            for owner, name, replacement in ((model, 'active_fraction', 1.),
                    (model, 'gate_blend', .25), (model, 'runtime_local_enabled', False),
                    (model, 'runtime_slow_enabled', False), (model, 'runtime_automatic_feedback', False),
                    (model, 'runtime_bio_read_enabled', False), (model.cfg, 'block_size', 8),
                    (model.cfg, 'route_mode', 'cyclic')):
                with self.subTest(field=name):
                    previous = getattr(owner, name)
                    setattr(owner, name, replacement)
                    with self.assertRaisesRegex(RuntimeError, 'protocol'):
                        model.step(torch.tensor([2]), stream)
                    with self.assertRaisesRegex(RuntimeError, 'protocol'):
                        model.flush_stream(stream)
                    setattr(owner, name, previous)

    def test_explicit_stream_flush_matches_one_partial_block_and_is_idempotent(self):
        model = self.model(route_mode='summary')
        tokens = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            expected, end, _ = model(tokens)
            stream = model.initial_stream_state(1)
            outputs = []
            for token in tokens.unbind(1):
                output, stream, _ = model.step(token, stream)
                outputs.append(output)
            flushed = model.flush_stream(stream)
            repeated = model.flush_stream(flushed)
        torch.testing.assert_close(expected, torch.stack(outputs, 1), atol=2e-11, rtol=2e-9)
        self.assert_state_close(end, flushed['committed'])
        self.assert_state_close(flushed['committed'], repeated['committed'], atol=0, rtol=0)
        self.assertEqual(flushed['tokens'].shape[1], 0)


if __name__ == '__main__':
    unittest.main()
