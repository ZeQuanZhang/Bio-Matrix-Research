"""Small CPU references for the v0.7 mechanism corrections."""
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from .common import detach, state_bytes
from .model import (Config, Network, Stateless, Transformer, initialize,
                    invalidate_feedback, joseph, bounded, unit_rotate)
from .graph import solid_ball


class ModelCorrections(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, **options):
        cfg = Config(vocab_size=48, matrix_size=4, width=16, modes=2, **options)
        model = Network(cfg)
        initialize(model, 71)
        return model

    def test_head_vjp_matches_autograd(self):
        model = self.model().double()
        torch.manual_seed(3)
        hidden = torch.randn(2, 16, dtype=torch.float64, requires_grad=True)
        direction = torch.randn_like(hidden)
        reference, = torch.autograd.grad((model.head(hidden)*direction).sum(), hidden)
        torch.testing.assert_close(model.head_vjp(hidden, direction), reference, atol=1e-11, rtol=1e-10)

    def test_direct_associative_feedback_matches_autograd_and_finite_difference(self):
        model = self.model().double().eval(); b, n, d = 2, 32, 4
        torch.manual_seed(19)
        slow = (torch.randn(b, n, d, d, dtype=torch.float64)*.01).requires_grad_()
        query = F.normalize(torch.randn(b, n, d, dtype=torch.float64), dim=-1)
        response = torch.rand(b, n, d, d, dtype=torch.float64)
        weight = model.read_distribution(response)*.4
        base = torch.randn(b, 16, dtype=torch.float64)*.1
        label = torch.tensor([7, 21])

        def prediction(matrix):
            read = model.slow_read(matrix, query, weight)
            pooled = torch.cat((torch.zeros(b, d*d, dtype=torch.float64), read.flatten(1)), -1)
            hidden = base+model.read_projection(pooled)
            return F.linear(model.head(hidden), model.embedding.weight), hidden

        logits, hidden = prediction(slow)
        loss = F.cross_entropy(logits, label, reduction='sum')
        gradient, = torch.autograd.grad(loss, slow)
        state = model.initial_state(b) | dict(slow=slow.detach(), feedback_h=hidden.detach(),
                prediction=(logits.softmax(-1)@model.embedding.weight).detach(), read_query=query,
                read_weight=weight, response=torch.ones_like(slow), support=torch.ones(b, n, d, dtype=slow.dtype),
                feedback_gain=torch.ones(b, n, dtype=slow.dtype),
                feedback_valid=torch.ones(b, dtype=torch.bool))
        local = model.local_direction(state, model.embedding(label))
        torch.testing.assert_close(local, -gradient, atol=1e-11, rtol=1e-10)
        self.assertGreater(float(local[:, model.shell != model.cfg.shells-1].detach().abs().max()), 0.)
        direction = F.normalize(torch.randn_like(slow).flatten(1), dim=-1).reshape_as(slow)
        eps = 1e-5
        plus = F.cross_entropy(prediction(slow+eps*direction)[0], label, reduction='sum')
        minus = F.cross_entropy(prediction(slow-eps*direction)[0], label, reduction='sum')
        finite = (plus-minus)/(2*eps)
        torch.testing.assert_close(finite, (gradient*direction).sum(), atol=1e-9, rtol=1e-5)
        update = model.local_update(state, model.embedding(label))-slow.detach()
        self.assertLess(float((gradient*update).detach().sum()), 0.)
        disabled = invalidate_feedback(state)
        torch.testing.assert_close(model.local_update(disabled, model.embedding(label)), state['slow'])
        torch.testing.assert_close(disabled['slow'], state['slow'])

    def test_zero_semantic_pulse_closes_and_large_pulse_opens_outward_gate(self):
        model = self.model().eval(); state = model.initial_state(1); d = model.cfg.matrix_size
        features = model.features(state); controls = model.controls(features)
        content = torch.randn_like(state['m'])
        drive = torch.zeros(1, model.cfg.nodes, 2*d+1)
        # Both routing fields remain nonzero; neither may force a semantic jump.
        drive[..., d:2*d] = F.normalize(torch.ones_like(drive[..., d:2*d]), dim=-1)
        drive[..., -1] = 1
        _, low, _ = model.weights(content, state['x'], drive, controls, features, state)
        drive[..., :d] = 1
        _, high, _ = model.weights(content, state['x'], drive, controls, features, state)
        outward = model.outward&model.edge_mask.bool()
        self.assertEqual(float(low[:, outward].sum()), 0.)
        self.assertTrue(bool((high[:, outward] == 1).all()))
        # Backward at exactly zero semantic amplitude must also stay finite.
        drive.zero_(); drive.requires_grad_()
        model.train()
        weights, gates, _ = model.weights(content, state['x'], drive, controls, features, state)
        (weights.sum()+gates.sum()).backward()
        self.assertTrue(bool(drive.grad.isfinite().all()))

    def test_time_bands_follow_their_declared_time_constants(self):
        model = self.model(variant='message').eval()
        with torch.no_grad():
            for parameter in model.message.parameters():
                parameter.zero_()
            model.raw_tau.copy_(torch.tensor([math.log(4), math.log(16), math.log(64)]))
        state = model.initial_state(1)
        state['x'] = torch.ones_like(state['x'])*.01
        initial = state['x'].clone()
        drive = torch.zeros(1, 32, 9); packet = torch.zeros(1, 9)
        with torch.no_grad():
            for _ in range(model.cfg.internal_steps):
                state, drive, *_ = model.tick(state, drive, packet)
        expected = initial*torch.exp(-1/model.raw_tau.exp())[None, None, :, None, None]
        torch.testing.assert_close(state['x'], expected, atol=1e-8, rtol=1e-6)

    def test_carrier_rotation_uses_one_token_time_independent_of_internal_steps(self):
        reference = None
        for steps in (2, 4, 8):
            model = self.model(internal_steps=steps).eval()
            with torch.no_grad():
                for parameter in model.controls.parameters():
                    parameter.zero_()
            state = model.initial_state(1)
            state['z'] = torch.zeros_like(state['z']); state['z'][..., 0] = 1
            state['x'] = torch.arange(state['x'].numel()).reshape_as(state['x'])*1e-5
            drive = torch.zeros(1, 32, 9); packet = torch.zeros(1, 9)
            # Isolate carrier integration: no transport/forcing, phase held fixed.
            weight = torch.zeros(1, 32, model.cfg.neighbors)
            sine = torch.zeros(*weight.shape, model.cfg.modes)
            with torch.no_grad(), patch.object(model, 'weights', return_value=(weight, weight, sine)), patch('v07.model.rotate', side_effect=lambda z, speed: z):
                for _ in range(steps):
                    state, drive, *_ = model.tick(state, drive, packet)
            if reference is None:
                reference = state['x']
            else:
                torch.testing.assert_close(state['x'], reference, atol=1e-8, rtol=1e-5)

    def test_stream_equivalence_fixed_state_and_lane_reset(self):
        model = self.model(variant='no_threshold').eval()
        token = torch.tensor([[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13]])
        with torch.no_grad():
            whole, end, _ = model(token)
            first, state, _ = model(token[:, :3])
            last, split_end, _ = model(token[:, 3:], state)
            reset, _, _ = model(token[:, 3:], state, torch.tensor([True, False]))
            fresh, _, _ = model(token[:1, 3:])
        torch.testing.assert_close(whole, torch.cat((first, last), 1))
        torch.testing.assert_close(reset[:1], fresh)
        torch.testing.assert_close(reset[1:], last[1:])
        self.assertEqual(state_bytes(state), state_bytes(end))
        for key in end:
            torch.testing.assert_close(end[key], split_end[key])
        self.assertFalse(end['feedback_h'].requires_grad)
        self.assertFalse(end['read_query'].requires_grad)
        self.assertLessEqual(float(end['slow'].norm(dim=(-1, -2)).max()), 4.00001)

    def test_transformer_reset_masks_cached_history_per_lane(self):
        cfg = Config(vocab_size=48, variant='transformer', transformer_width=16,
                     transformer_ff=32, context_limit=8)
        model = Transformer(cfg).eval(); initialize(model, 71)
        token = torch.tensor([[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13]])
        with torch.no_grad():
            whole, _, _ = model(token)
            first, state, _ = model(token[:, :3])
            last, _, _ = model(token[:, 3:], state)
            reset, after, _ = model(token[:, 3:], state, torch.tensor([True, False]))
            fresh, _, _ = model(token[:1, 3:])
            _, bounded, _ = model(token[:, :3], after)
        torch.testing.assert_close(whole, torch.cat((first, last), 1), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(reset[:1], fresh, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(reset[1:], last[1:], atol=1e-6, rtol=1e-5)
        self.assertFalse(bool(after['valid'][0, :3].any()))
        self.assertTrue(bool(after['valid'][1].all()))
        torch.testing.assert_close(after['position'], torch.tensor([3, 6]))
        self.assertEqual(bounded['valid'].shape[-1], 8)
        self.assertEqual(bounded['kv'][0][0].shape[-2], 8)

    def test_transformer_sliding_context_is_chunk_size_invariant_after_reset(self):
        cfg = Config(vocab_size=48, variant='transformer', transformer_width=16,
                     transformer_ff=32, context_limit=8)
        model = Transformer(cfg).eval(); initialize(model, 71)
        tokens = torch.arange(36).reshape(2, 18)%48
        with torch.no_grad():
            prefix, state, _ = model(tokens[:, :7])
            reset = torch.tensor([True, False])
            chunks, chunk_state = [], state
            for start in (7, 11, 15):
                output, chunk_state, _ = model(tokens[:, start:min(start+4, 18)], chunk_state,
                                                reset if start == 7 else None)
                chunks.append(output)
            singles, single_state = [], state
            for start in range(7, 18):
                output, single_state, _ = model(tokens[:, start:start+1], single_state,
                                                 reset if start == 7 else None)
                singles.append(output)
        torch.testing.assert_close(torch.cat(chunks, 1), torch.cat(singles, 1), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(chunk_state['valid'], single_state['valid'])
        for chunk, single in zip(chunk_state['kv'], single_state['kv']):
            for left, right in zip(chunk, single):
                torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)

    def test_default_has_no_current_embedding_bypass(self):
        model = self.model().eval()
        token = torch.tensor([[2, 3]])
        with torch.no_grad():
            model.read_projection.weight.zero_()
            ordinary, _, _ = model(token)
            model.cfg = replace(model.cfg, variant='bypass')
            bypass, _, _ = model(token)
        torch.testing.assert_close(ordinary[:, :1], ordinary[:, 1:])
        torch.testing.assert_close(bypass, model.head_only(token))
        self.assertGreater(float((ordinary-bypass).detach().abs().max()), 1e-5)
        stateless = Stateless(model.cfg); initialize(stateless, 71)
        self.assertEqual(stateless.embedding.embedding_dim, model.embedding.embedding_dim)

    def test_tiny_cpu_training_keeps_finite_gradients_and_persistent_slow(self):
        model = self.model(); model.gate_blend = 0.
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        tokens = torch.tensor([[2, 5, 9, 3, 2, 5, 9, 3]])
        state = None
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            logits, state, diag = model(tokens[:, :-1], state)
            loss = F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())
            loss.backward()
            self.assertTrue(bool(loss.isfinite()))
            for name, parameter in model.named_parameters():
                self.assertTrue(parameter.grad is None or bool(parameter.grad.isfinite().all()), name)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); state = invalidate_feedback(detach(state))
        self.assertGreater(float(state['slow'].abs().sum()), 0.)
        self.assertFalse(bool(state['feedback_valid'].any()))
        self.assertTrue(bool(diag['local_update_rms'].isfinite()))

    def test_controls_have_finite_gradients_and_slow_intervention_is_complete(self):
        tokens = torch.tensor([[2, 5, 9, 3]])
        for variant in ('message', 'independent', 'full', 'no_local', 'no_slow', 'no_probability',
                        'no_dynamic', 'single_timescale', 'no_threshold', 'bypass'):
            with self.subTest(variant=variant):
                model = self.model(variant=variant); model.gate_blend = 0.
                logits, _, _ = model(tokens)
                F.cross_entropy(logits[:, :-1].flatten(0, 1), tokens[:, 1:].flatten()).backward()
                for name, parameter in model.named_parameters():
                    self.assertTrue(parameter.grad is None or bool(parameter.grad.isfinite().all()), name)
        model = self.model().eval(); state = model.initial_state(1)
        other = dict(state); other['slow'] = torch.randn_like(state['slow'])*.01
        model.runtime_slow_enabled = False
        with torch.no_grad():
            a, _, _ = model(tokens, state); b, _, _ = model(tokens, other)
        torch.testing.assert_close(a, b)

    def test_distributed_slow_read_updates_reachable_inner_nodes_and_respects_cut(self):
        tokens = torch.tensor([[1, 4, 5, 8, 9, 10, 4, 5]])
        changed = torch.tensor([[1, 24, 25, 8, 9, 10, 4, 5]])
        for topology in ('connected', 'cut'):
            with self.subTest(topology=topology):
                model = self.model(variant='no_threshold', topology=topology).eval()
                with torch.no_grad():
                    logits, state, _ = model(tokens)
                    other, _, _ = model(changed)
                inner = model.shell < model.cfg.shells//2
                if topology == 'connected':
                    self.assertGreater(float(state['read_weight'][:, inner].sum()), 0.)
                    self.assertGreater(float(state['slow'][:, inner].abs().sum()), 0.)
                    self.assertGreater(float((logits[:, -1]-other[:, -1]).abs().max()), 1e-7)
                else:
                    self.assertEqual(float(state['read_weight'][:, inner].sum()), 0.)
                    self.assertEqual(float(state['slow'][:, inner].abs().sum()), 0.)
                    torch.testing.assert_close(logits, other, atol=1e-7, rtol=0.)

    def test_checkpoint_events_preserves_parameter_gradients(self):
        normal = self.model(); checked = self.model(checkpoint_events=True)
        checked.load_state_dict(normal.state_dict())
        tokens = torch.tensor([[2, 5, 9]])
        for model in (normal, checked):
            model.gate_blend = 0.; logits, _, _ = model(tokens)
            F.cross_entropy(logits[:, :-1].flatten(0, 1), tokens[:, 1:].flatten()).backward()
        for (name, left), (_, right) in zip(normal.named_parameters(), checked.named_parameters()):
            if left.grad is not None:
                torch.testing.assert_close(left.grad, right.grad, atol=1e-6, rtol=1e-5, msg=name)

    def test_input_mapping_enters_the_identical_fast_and_joseph_observation(self):
        model = self.model(variant='no_threshold').eval()
        observed = []
        original = model.observe
        def record(*args):
            result = original(*args); observed.append(result)
            return result
        with torch.no_grad(), patch.object(model, 'observe', side_effect=record), patch('v07.model.joseph', wraps=joseph) as update:
            model(torch.tensor([[7]]))
        self.assertEqual(len(observed), model.cfg.internal_steps)
        self.assertEqual(update.call_count, 1)
        torch.testing.assert_close(update.call_args.args[2], observed[-1][1], atol=0, rtol=0)
        torch.testing.assert_close(update.call_args.args[3], observed[-1][0], atol=0, rtol=0)
        value = torch.tensor([[[.7, .2, .1, -.1]]]).expand(1, 32, -1)
        key = torch.tensor([[[1., 0., 0., 0.]]]).expand_as(value)
        controls = torch.zeros(1, 32, 21)
        controls[..., 5] = 1; controls[..., 9] = 1
        controls[:, 1, 5] = 0; controls[:, 1, 6] = 1
        mapped_value, mapped_key = model.observe(value, key, controls)
        self.assertGreater(float((mapped_key[:, 0]-mapped_key[:, 1]).detach().abs().max()), .01)
        self.assertGreater(float((mapped_value[:, 0]-mapped_value[:, 1]).detach().abs().max()), .01)
        zeros = torch.zeros(1, 32, 4, 4)
        covariance = torch.eye(4).expand_as(zeros)
        posterior, _ = joseph(zeros, covariance, mapped_key, mapped_value, torch.ones(1, 32)*.1,
                               torch.ones(1, 32)*.5, torch.zeros(1, 32))
        self.assertGreater(float((posterior[:, 0]-posterior[:, 1]).detach().abs().max()), .01)

    def test_history_probability_and_phase_change_real_paths_and_observations(self):
        model = self.model().eval(); state = model.initial_state(1)
        altered = dict(state); altered['m'] = bounded(state['m']+.2)
        value = torch.full((1, 32, 4), .2); key = F.normalize(torch.ones_like(value), dim=-1)
        with torch.no_grad():
            old_observation = model.observe(value, key, model.controls(model.features(state)))
            new_observation = model.observe(value, key, model.controls(model.features(altered)))
            self.assertGreater(float((old_observation[0]-new_observation[0]).abs().max()), 1e-7)
            content = torch.arange(32*16).reshape(1, 32, 4, 4).float()*.0001
            drive = torch.ones(1, 32, 9)*.03
            base_features = model.features(state)
            base_weight, _, _ = model.weights(content, state['x'], drive, model.controls(base_features), base_features, state)
            uncertain = dict(state); uncertain['p'] = state['p']*8
            uncertain_features = model.features(uncertain)
            uncertain_weight, _, _ = model.weights(content, uncertain['x'], drive, model.controls(uncertain_features), uncertain_features, uncertain)
            phase = dict(state); phase['z'] = unit_rotate(state['z'], torch.arange(32)[None, :, None]*.2)
            phase_features = model.features(phase)
            phase_weight, _, _ = model.weights(content, phase['x'], drive, model.controls(phase_features), phase_features, phase)
            self.assertGreater(float((base_weight-uncertain_weight).abs().max()), 1e-8)
            self.assertGreater(float((base_weight-phase_weight).abs().max()), 1e-7)
            a, _, _ = model(torch.tensor([[7]]), state)
            b, _, _ = model(torch.tensor([[7]]), uncertain)
            self.assertGreater(float((a-b).abs().max()), 1e-7)

    def test_stationary_covariance_has_psd_bound_and_correct_unobserved_limit(self):
        model = self.model(covariance_tau=64).double()
        state = model.initial_state(1)
        key = torch.zeros(1, 32, 4, dtype=torch.float64); key[..., 0] = 1
        value = torch.ones_like(key)*.2
        variance = torch.ones(1, 32, dtype=torch.float64)*.1
        reliability = torch.ones_like(variance)*.5
        iterations = 1024
        with torch.no_grad():
            for _ in range(iterations):
                mean, covariance = model.stationary_prior(state)
                m, p = joseph(mean, covariance, key, value, variance, reliability,
                               torch.ones_like(variance)*model.cfg.covariance_process)
                state = state | {'m': bounded(m), 'p': p}
        rho = math.exp(-1/model.cfg.covariance_tau)
        expected = 1+(model.cfg.covariance_process+1e-6)*(1-rho**iterations)/(1-rho)
        torch.testing.assert_close(state['p'][..., 1, 1], torch.full_like(variance, expected), atol=1e-11, rtol=1e-11)
        eigenvalues = torch.linalg.eigvalsh(state['p'])
        self.assertGreater(float(eigenvalues.min()), 0.)
        self.assertLessEqual(float(eigenvalues.max()), model.covariance_upper_bound()+1e-10)
        self.assertTrue(bool((model.features(state)[..., 3] < 1).all()))

    def test_phase_and_all_runtime_numeric_bounds_are_fixed(self):
        phase = torch.tensor([[1., 0.], [0., 1.]])
        for _ in range(10000):
            phase = unit_rotate(phase, torch.tensor([.031, .027]))
        torch.testing.assert_close(phase.norm(dim=-1), torch.ones(2), atol=2e-7, rtol=0.)
        model = self.model(variant='no_threshold').eval()
        with torch.no_grad():
            _, short, _ = model(torch.tensor([[1, 5]]))
            _, state, _ = model((torch.arange(48)%47+1)[None])
        self.assertEqual(state_bytes(short), state_bytes(state))
        for name in ('m', 'x', 'slow'):
            self.assertLessEqual(float(state[name].norm(dim=(-1, -2)).max()), 4.00001)
        self.assertLessEqual(float(state['trace'].norm(dim=-1).max()), 1.00001)
        self.assertLessEqual(float(state['response'].max()), 2.00001)
        self.assertLessEqual(float(state['support'].max()), 100.)
        self.assertLessEqual(float(state['innovation'].max()), 25.)
        self.assertLessEqual(float(torch.linalg.eigvalsh(state['p']).max()), model.covariance_upper_bound()+1e-3)
        torch.testing.assert_close(state['z'].norm(dim=-1), torch.ones_like(state['z'][..., 0]), atol=2e-7, rtol=0.)
        state['evidence_count'] = torch.full_like(state['evidence_count'], 2**31-2)
        with torch.no_grad():
            _, state, _ = model(torch.tensor([[1, 2, 3]]), state)
        self.assertEqual(int(state['evidence_count'].max()), 2**31-1)

    def test_local_write_participation_normalization_matches_population_response(self):
        changes = {'none': [], 'participation': []}
        for mode in changes:
            for nodes in (32, 128, 512):
                model = self.model(nodes=nodes, local_write_normalization=mode).double()
                state = model.initial_state(1)
                distribution = torch.ones(1, nodes, dtype=torch.float64)/nodes
                scale, effective = model.write_scale(distribution)
                torch.testing.assert_close(effective, torch.tensor([float(nodes)], dtype=torch.float64))
                state['read_scale'] = scale
                state['feedback_gain'] = torch.ones(1, nodes, dtype=torch.float64)
                state['read_query'][..., 0] = 1
                # The exact uniform direct-read negative gradient is proportional to a_n.
                direction = torch.zeros_like(state['slow']); direction[..., 0, 0] = 1/nodes
                with torch.no_grad(), patch.object(model, 'local_direction', return_value=direction):
                    updated = model.local_update(state, model.embedding(torch.tensor([7])))
                query = torch.zeros(1, nodes, 4, dtype=torch.float64); query[..., 0] = 1
                changes[mode].append(float(model.slow_read(updated, query, distribution).norm()))
        self.assertAlmostEqual(changes['none'][0]/changes['none'][1], 4., places=10)
        self.assertAlmostEqual(changes['none'][1]/changes['none'][2], 4., places=10)
        torch.testing.assert_close(torch.tensor(changes['participation']), torch.tensor([changes['participation'][0]]*3))

    def test_effective_participation_uses_actual_read_weights_without_erasing_confidence(self):
        model = self.model(variant='no_threshold').eval()
        with torch.no_grad():
            _, state, _ = model(torch.tensor([[7, 11]]))
            full_scale, full_effective = model.write_scale(state['read_weight'])
            low = state | {'read_weight': state['read_weight']*.1}
            low_scale, low_effective = model.write_scale(low['read_weight'])
            torch.testing.assert_close(full_effective, low_effective)
            torch.testing.assert_close(full_scale, low_scale)
            token = model.embedding(torch.tensor([17]))
            torch.testing.assert_close(model.local_direction(low, token), .1*model.local_direction(state, token))
            one = torch.zeros_like(state['read_weight']); one[:, 3] = .001
            _, effective = model.write_scale(one)
            torch.testing.assert_close(effective, torch.ones(1))
            _, none = model.write_scale(torch.zeros_like(one))
            torch.testing.assert_close(none, torch.zeros(1))

    def test_delayed_feedback_uses_old_query_once_and_respects_horizon(self):
        model = self.model(variant='no_threshold').eval(); model.runtime_automatic_feedback = False
        with torch.no_grad():
            _, state, _ = model(torch.tensor([[2, 5, 9, 3, 7, 11, 4, 8]]))
        # Cache slots store eligibility metadata, never copies of fast/slow node states.
        self.assertEqual(state['credit_query'].shape[1], 7)
        old = model.feedback_frame(state, 8)
        old['credit_query'] = state['credit_query'].clone()
        old['credit_query'][:, 6] = 0; old['credit_query'][:, 6, :, 0] = 1
        old['read_query'] = torch.zeros_like(state['read_query']); old['read_query'][..., 1] = 1
        old['credit_gain'] = torch.ones_like(state['credit_gain'])
        with torch.no_grad():
            updated = model.apply_feedback(old, torch.tensor([17]), lag=8)
            again = model.apply_feedback(updated, torch.tensor([17]), lag=8)
        delta = updated['slow']-old['slow']
        self.assertGreater(float(delta[..., 0].abs().sum()), 0.)
        self.assertEqual(float(delta[..., 1:].abs().sum()), 0.)
        torch.testing.assert_close(again['slow'], updated['slow'], atol=0, rtol=0)
        self.assertFalse(bool(updated['credit_valid'][:, 6].any()))
        with self.assertRaisesRegex(ValueError, 'horizon'):
            model.apply_feedback(state, torch.tensor([17]), lag=9)
        with torch.no_grad():
            _, expired, _ = model(torch.tensor([[2]]), updated)
        # Every optimizer version invalidation clears all eligibility slots.
        invalid = invalidate_feedback(expired)
        self.assertFalse(bool(invalid['feedback_valid'].any()))
        self.assertFalse(bool(invalid['credit_valid'].any()))

    def test_rule_feedback_preserves_orthogonal_experience_and_resets_credit(self):
        model = self.model(variant='no_threshold').eval(); model.runtime_automatic_feedback = False
        with torch.no_grad():
            _, state, _ = model(torch.tensor([[2, 5, 9], [7, 4, 11]]))
        state['read_query'] = torch.zeros_like(state['read_query']); state['read_query'][..., 0] = 1
        state['feedback_gain'] = torch.ones_like(state['feedback_gain'])
        state['slow'][..., 1] = .03
        with torch.no_grad():
            revised = model.apply_feedback(state, torch.tensor([17, 18]))
        self.assertGreater(float((revised['slow'][..., 0]-state['slow'][..., 0]).abs().sum()), 0.)
        torch.testing.assert_close(revised['slow'][..., 1:], state['slow'][..., 1:], atol=0, rtol=0)
        fast_reset = model.reset_fast(revised)
        torch.testing.assert_close(fast_reset['slow'], revised['slow'])
        torch.testing.assert_close(fast_reset['support'], revised['support'])
        self.assertFalse(bool(fast_reset['feedback_valid'].any()))
        self.assertFalse(bool(fast_reset['credit_valid'].any()))
        cleared = model.clear_slow(revised)
        self.assertEqual(float(cleared['slow'].abs().sum()), 0.)
        torch.testing.assert_close(cleared['x'], revised['x'])
        with torch.no_grad():
            _, partially_reset, _ = model(torch.tensor([[3], [4]]), revised, torch.tensor([True, False]))
            _, fresh, _ = model(torch.tensor([[3]]))
        for name in ('slow', 'credit_valid', 'credit_query'):
            torch.testing.assert_close(partially_reset[name][:1], fresh[name])

    def test_capacity_projection_preserves_orthogonal_memory_on_immediate_and_delayed_feedback(self):
        model = self.model(variant='no_threshold').double().eval(); model.runtime_automatic_feedback = False
        with torch.no_grad():
            _, state, _ = model(torch.tensor([[2, 5, 9]]))
        state['slow'].zero_(); state['slow'][..., 0, 1] = 4.
        state['read_query'].zero_(); state['read_query'][..., 0] = 1
        state['feedback_gain'].fill_(1)
        state['credit_query'][:, 0] = state['read_query']
        state['credit_prediction'][:, 0] = state['prediction']
        state['credit_h'][:, 0] = state['feedback_h']
        state['credit_weight'][:, 0] = state['read_weight']
        state['credit_gain'][:, 0] = state['feedback_gain']
        state['credit_scale'][:, 0] = state['read_scale']
        state['credit_valid'][:, 0] = True
        token = torch.tensor([17])
        with torch.no_grad():
            direct = model.local_update(state, model.embedding(token))
            immediate, diagnostics = model._apply_feedback(state, token, 1)
            delayed = model.apply_feedback(state, token, 2)
        self.assertGreater(float(diagnostics['capacity_clip_fraction']), 0.)
        self.assertEqual(float(diagnostics['local_effective_step_ratio']), 0.)
        for updated in (direct, immediate['slow'], delayed['slow']):
            torch.testing.assert_close(updated[..., 1:], state['slow'][..., 1:], atol=0, rtol=0)
            self.assertLessEqual(float(updated.norm(dim=(-1, -2)).max()), 4.+1e-12)
        torch.testing.assert_close(direct, immediate['slow'], atol=0, rtol=0)
        torch.testing.assert_close(direct, delayed['slow'], atol=0, rtol=0)
        # With a little remaining capacity, only the queried component can change.
        for retained in (4., 3.9999):
            almost_full = state['slow'].detach().clone(); almost_full[..., 0, 1] = retained
            almost_full.requires_grad_()
            proposal = torch.zeros_like(almost_full); proposal[..., 1, 0] = .05
            updated, clipped = model.project_slow_update(almost_full, proposal, state['read_query'])
            updated.square().sum().backward()
            self.assertTrue(bool(almost_full.grad.isfinite().all()))
            self.assertTrue(bool(clipped.all()))
            torch.testing.assert_close(updated[..., 1:], almost_full[..., 1:], atol=0, rtol=0)
            self.assertLessEqual(float(updated.detach().norm(dim=(-1, -2)).max()), 4.+1e-12)

    def test_diagnostic_mask_changes_only_valid_token_statistics(self):
        model = self.model(variant='no_threshold').eval()
        tokens = torch.tensor([[1, 4, 5], [7, 8, 9]])
        mask = torch.tensor([[True, False, False], [True, True, False]])
        with torch.no_grad():
            all_logits, all_state, _ = model(tokens)
            masked_logits, masked_state, summary = model(tokens, diagnostic_mask=mask)
            state = None; rows = []
            for t in range(tokens.shape[1]):
                logits, state, row = model.step(tokens[:, t], model.initial_state(2) if state is None else state)
                rows.append(row)
        torch.testing.assert_close(masked_logits, all_logits, atol=0, rtol=0)
        for name in all_state:
            torch.testing.assert_close(masked_state[name], all_state[name], atol=0, rtol=0)
        reached = torch.stack([row['reached'] for row in rows], 1)
        minimum = torch.stack([row['covariance_min_diagonal'] for row in rows], 1)
        torch.testing.assert_close(summary['reached'], reached[mask].mean())
        torch.testing.assert_close(summary['covariance_min_diagonal'], minimum[mask].min())

    def test_solid_ball_candidate_budget_and_radial_paths(self):
        for nodes in (32, 128, 512):
            graph = solid_ball(nodes)
            self.assertEqual(tuple(graph['neighbors'].shape), (nodes, 8))
            torch.testing.assert_close(graph['positions'].norm(dim=-1), graph['shell']/3, atol=1e-6, rtol=0.)
            reached = graph['write_ports'].bool()
            for _ in range(3):
                reached = reached | (reached[graph['neighbors']] & graph['edge_mask'].bool()).any(-1)
            self.assertTrue(bool(reached.all()))


if __name__ == '__main__':
    unittest.main()
