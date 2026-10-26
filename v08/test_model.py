"""CPU FP64 checks for the new block schedule, not v0.7 equivalence claims."""
import unittest

import torch
from torch.nn import functional as F

from v07.common import detach
from v07.model import invalidate_feedback
from .model import Config, Network, initialize


class BlockModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, implementation='sequential', **options):
        cfg = Config(vocab_size=32, nodes=16, matrix_size=4, width=16,
                     bands=2, modes=2, block_size=4, implementation=implementation, **options)
        model = Network(cfg).double()
        initialize(model, 801)
        model.gate_blend = 0.
        return model

    def compare_states(self, left, right):
        self.assertEqual(left.keys(), right.keys())
        for name in left:
            torch.testing.assert_close(left[name], right[name], atol=2e-11, rtol=2e-9, msg=name)

    def test_three_implementations_match_logits_states_and_all_parameter_gradients(self):
        tokens = torch.tensor([[1, 2, 3, 5, 8, 13, 21, 2], [3, 7, 9, 11, 13, 15, 17, 19]])
        models = [self.model(implementation) for implementation in ('sequential', 'parallel', 'triangular')]
        for model in models[1:]:
            model.load_state_dict(models[0].state_dict())
        results = []
        for model in models:
            logits, state, diag = model(tokens)
            # The second block exposes differentiable writes from the first block.
            F.cross_entropy(logits[:, :-1].flatten(0, 1), tokens[:, 1:].flatten()).backward()
            results.append((logits, state))
            self.assertTrue(all(bool(value.isfinite().all()) for value in state.values()))
            self.assertTrue(all(bool(value.isfinite()) for value in diag.values()))
            self.assertEqual(state['x'].shape, (2, 16, 2, 4, 4))
            self.assertEqual(state['slow'].shape, (2, 16, 4, 4))
            self.assertGreater(float(state['slow'].detach().abs().sum()), 0.)
            self.assertGreater(float(model.raw_slow_rate.grad.abs()), 0.)
            self.assertGreater(float(model.content_projection.weight.grad.norm()), 0.)
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(bool(parameter.grad.isfinite().all()), name)
        for other in results[1:]:
            torch.testing.assert_close(results[0][0], other[0], atol=2e-11, rtol=2e-9)
            self.compare_states(results[0][1], other[1])
        reference = dict(models[0].named_parameters())
        for model in models[1:]:
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(reference[name].grad, parameter.grad,
                                           atol=2e-10, rtol=2e-8, msg=name)

    def test_future_tokens_cannot_change_previous_outputs_inside_or_across_blocks(self):
        model = self.model('triangular').eval()
        tokens = torch.tensor([[1, 2, 3, 5, 8, 13, 21, 2]])
        with torch.no_grad():
            original, _, _ = model(tokens)
            for position in (1, 3, 4, 6):
                changed = tokens.clone(); changed[:, position:] = (changed[:, position:]+7)%32
                output, _, _ = model(changed)
                torch.testing.assert_close(original[:, :position], output[:, :position], atol=0, rtol=0)
                self.assertGreater(float((original[:, position:]-output[:, position:]).abs().max()), 1e-8)

    def test_complete_blocks_are_call_invariant_with_lane_reset(self):
        model = self.model('triangular').eval()
        tokens = torch.tensor([[1, 2, 3, 5, 8, 13, 21, 2], [4, 6, 8, 10, 12, 14, 16, 18]])
        with torch.no_grad():
            whole, end, _ = model(tokens)
            first, state, _ = model(tokens[:, :4])
            second, split, _ = model(tokens[:, 4:], state)
            reset, reset_state, _ = model(tokens[:, 4:], state, torch.tensor([True, False]))
            fresh, fresh_state, _ = model(tokens[:1, 4:])
        torch.testing.assert_close(whole, torch.cat((first, second), 1), atol=0, rtol=0)
        self.compare_states(end, split)
        torch.testing.assert_close(reset[:1], fresh, atol=1e-13, rtol=1e-11)
        torch.testing.assert_close(reset[1:], second[1:], atol=0, rtol=0)
        for name in reset_state:
            torch.testing.assert_close(reset_state[name][:1], fresh_state[name], atol=1e-13, rtol=1e-11)
        self.assertTrue(bool(end['feedback_valid'].all()))
        self.assertFalse(end['feedback_h'].requires_grad)
        invalid = invalidate_feedback(end)
        self.assertFalse(bool(invalid['feedback_valid'].any()))
        self.assertFalse(bool(invalid['credit_valid'].any()))
        torch.testing.assert_close(invalid['slow'], end['slow'], atol=0, rtol=0)

    def test_closed_graph_is_finite_and_current_input_has_no_head_bypass(self):
        model = self.model('triangular')
        model.edge_mask.zero_()
        tokens = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15]])
        logits, state, diag = model(tokens)
        F.cross_entropy(logits[:, :-1].flatten(0, 1), tokens[:, 1:].flatten()).backward()
        self.assertTrue(bool(logits.isfinite().all()))
        self.assertEqual(float(diag['attention_mass'].detach()), 0.)
        self.assertTrue(all(parameter.grad is None or bool(parameter.grad.isfinite().all())
                            for parameter in model.parameters()))
        self.assertTrue(all(bool(tensor.isfinite().all()) for tensor in state.values()))
        torch.testing.assert_close(logits[:, :4], logits[:, :1].expand(-1, 4, -1), atol=0, rtol=0)
        # Removing the sole matrix read removes all token-specific output variation.
        model = self.model('triangular').eval()
        with torch.no_grad():
            model.read_projection.weight.zero_()
            output, _, _ = model(tokens)
        torch.testing.assert_close(output, output[:, :1].expand_as(output), atol=0, rtol=0)

    def test_local_feedback_stops_direction_gradient_but_keeps_eta_trainable(self):
        model = self.model().eval()
        with torch.no_grad():
            _, state, _ = model(torch.tensor([[1, 2, 3, 4]]))
        proposal = model.local_proposal(state, model.embedding(torch.tensor([7])))
        proposal.square().sum().backward()
        self.assertGreater(float(model.raw_slow_rate.grad.abs()), 0.)
        self.assertIsNone(model.embedding.weight.grad)
        self.assertIsNone(model.ff[0].weight.grad)
        self.assertIsNone(model.read_projection.weight.grad)
        revised = model.apply_feedback(state, torch.tensor([7]))
        repeated = model.apply_feedback(revised, torch.tensor([7]))
        torch.testing.assert_close(revised['slow'], repeated['slow'], atol=0, rtol=0)
        self.assertLessEqual(float(revised['slow'].detach().norm(dim=(-1, -2)).max()), 4.+1e-12)

    def test_probabilistic_phase_and_state_bounds_and_training_steps(self):
        model = self.model('triangular')
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        tokens = torch.tensor([[1, 3, 5, 7, 1, 3, 5, 7, 1]])
        state = None
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            logits, state, _ = model(tokens[:, :-1], state)
            loss = F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); state = invalidate_feedback(detach(state))
        for name in ('m', 'x', 'slow'):
            self.assertLessEqual(float(state[name].norm(dim=(-1, -2)).max()), 4.+1e-11)
        self.assertGreater(float(torch.linalg.eigvalsh(state['p']).min()), 0.)
        self.assertLessEqual(float(torch.linalg.eigvalsh(state['p']).max()), model.covariance_upper_bound()+1e-10)
        torch.testing.assert_close(state['z'].norm(dim=-1), torch.ones_like(state['z'][..., 0]), atol=1e-12, rtol=0)
        self.assertGreater(float(state['support'].sum()), 0.)
        self.assertGreater(int(state['evidence_count'].max()), 0)
        self.assertGreater(float(state['trace'].abs().sum()), 0.)

    def test_slow_intervention_and_directional_capacity_projection(self):
        model = self.model('triangular').eval(); state = model.initial_state(1)
        altered = state | {'slow': torch.randn_like(state['slow'])*.1}
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            a, _, _ = model(tokens, state)
            b, _, _ = model(tokens, altered)
        self.assertGreater(float((a-b).abs().max()), 1e-6)
        model.runtime_slow_enabled = False
        with torch.no_grad():
            a, _, _ = model(tokens, state)
            b, _, _ = model(tokens, altered)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        slow = torch.zeros_like(state['slow']); slow[..., 0, 1] = 4.
        update = torch.zeros_like(slow); update[..., 1, 0] = .05
        query = torch.zeros_like(state['support']); query[..., 0] = 1
        projected, clipped = model.project_slow_update(slow, update, query)
        self.assertTrue(bool(clipped.all()))
        torch.testing.assert_close(projected[..., 1:], slow[..., 1:], atol=0, rtol=0)
        self.assertLessEqual(float(projected.norm(dim=(-1, -2)).max()), 4.+1e-12)

    def test_config_and_input_validation(self):
        for options in ({'block_size': 0}, {'block_size': True}, {'implementation': 'magic'}, {'variant': 'message'}):
            with self.assertRaises(ValueError):
                Config(**options)
        model = self.model()
        for tokens in (torch.zeros(1, 0, dtype=torch.long), torch.zeros(1, 4), torch.zeros(4, dtype=torch.long)):
            with self.assertRaises(ValueError):
                model(tokens)
        with self.assertRaises(ValueError):
            model(torch.ones(1, 4, dtype=torch.long), diagnostic_mask=torch.zeros(1, 4, dtype=torch.bool))
        with self.assertRaisesRegex(NotImplementedError, 'explicit block'):
            model.step(torch.ones(1, dtype=torch.long), model.initial_state(1))

    def test_fp32_hard_gate_scan_matches_reference_and_r1_entrance_is_orthogonal(self):
        models = [self.model(mode).float() for mode in ('sequential', 'triangular')]
        models[1].load_state_dict(models[0].state_dict())
        tokens = torch.tensor([[1, 2, 3, 5, 8, 13, 21, 2]])
        results = []
        for model in models:
            model.gate_blend = 1.
            output, state, _ = model(tokens)
            F.cross_entropy(output[:, :-1].flatten(0, 1), tokens[:, 1:].flatten()).backward()
            results.append((output, state))
        torch.testing.assert_close(results[0][0], results[1][0], atol=2e-6, rtol=2e-5)
        for name in results[0][1]:
            torch.testing.assert_close(results[0][1][name], results[1][1][name], atol=2e-6, rtol=2e-5, msg=name)
        for (name, left), (_, right) in zip(models[0].named_parameters(), models[1].named_parameters()):
            self.assertTrue(bool(left.grad.isfinite().all()), name)
            torch.testing.assert_close(left.grad, right.grad, atol=3e-6, rtol=5e-4, msg=name)
        entrance = models[0].content_projection.weight
        torch.testing.assert_close(entrance@entrance.T, torch.eye(16), atol=3e-7, rtol=1e-6)
        self.assertAlmostEqual(float((.3*models[0].raw_content_gain.sigmoid()).detach()), .1, places=7)

    def test_arrival_gate_blocks_zero_and_subthreshold_key_gradients_only(self):
        model = self.model()
        d = model.cfg.matrix_size
        mass = torch.tensor([0., .999e-4, 1.001e-4], dtype=torch.float64)
        value = mass[:, None]*torch.tensor([.3, -.2, .1, .4], dtype=torch.float64)
        key = mass[:, None]*F.normalize(torch.tensor([1., 2., 3., 4.], dtype=torch.float64), dim=0)
        drive = torch.cat((value, key, mass[:, None]), -1)[None].requires_grad_()
        control = model.controls(torch.zeros(1, 3, 12, dtype=torch.float64)).detach()
        observed_value, observed_key, arrived = model.final_observation(drive, control)
        torch.testing.assert_close(arrived, torch.tensor([[False, False, True]]))
        torch.testing.assert_close(observed_value[:, :2], torch.zeros_like(observed_value[:, :2]), atol=0, rtol=0)
        torch.testing.assert_close(observed_key[:, :2], torch.zeros_like(observed_key[:, :2]), atol=0, rtol=0)
        direction = torch.tensor([1., -.4, .7, -.2], dtype=torch.float64)
        ((observed_key*direction).sum()+observed_value.sum()).backward()
        self.assertTrue(bool(drive.grad.isfinite().all()))
        torch.testing.assert_close(drive.grad[:, :2], torch.zeros_like(drive.grad[:, :2]), atol=0, rtol=0)
        self.assertGreater(float(drive.grad[:, 2, :2*d].norm()), 0.)
        # The arrived observation retains the original normalized direction and derivative.
        direct = drive.detach()[:, 2:].clone().requires_grad_()
        direct_value, direct_key = model.observe((direct[..., :d]/.05).clamp(-2, 2),
                           F.normalize(direct[..., d:2*d], dim=-1, eps=1e-5), control[:, 2:])
        ((direct_key*direction).sum()+direct_value.sum()).backward()
        torch.testing.assert_close(drive.grad[:, 2:], direct.grad, atol=1e-10, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
