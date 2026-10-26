"""CPU comparisons against the unchanged v0.8 commit, including credit edges."""
from dataclasses import dataclass
import math
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from v07.common import detach
from v07.model import bounded, invalidate_feedback
from v08.model import Config as OldConfig, Network as OldNetwork, initialize
from .feedback import FeedbackMixin


@dataclass
class Config(OldConfig):
    batch_feedback: bool = True


class Network(FeedbackMixin, OldNetwork):
    pass


class CompactNetwork(Network):
    """A small stand-in for the caller's selected node-shape accessor."""
    def stationary_prior(self, state):
        rho = math.exp(-1/self.cfg.covariance_tau)
        anchor = bounded(self.node_shape)[self.selected_nodes][None]
        mean = math.sqrt(rho)*state['m']+(1-math.sqrt(rho))*anchor
        eye = torch.eye(self.cfg.matrix_size, dtype=state['p'].dtype, device=state['p'].device)
        return mean, rho*state['p']+(1-rho)*eye


class FeedbackChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def models(self, dtype=torch.float64, credit=5, block=4, compact=False):
        cls = CompactNetwork if compact else Network
        cfg = dict(vocab_size=32, nodes=16, matrix_size=4, width=16, bands=2,
                   modes=2, internal_steps=2, block_size=block,
                   implementation='sequential', credit_horizon=credit)
        models = [cls(Config(**cfg, batch_feedback=value)).to(dtype) for value in (False, True)]
        initialize(models[0], 9091); models[1].load_state_dict(models[0].state_dict())
        for model in models:
            model.gate_blend = 1.
            if compact:
                model.selected_nodes = torch.tensor([0, 3, 7, 12, 15])
        return models

    def compare_mapping(self, left, right, dtype=torch.float64):
        self.assertEqual(left.keys(), right.keys())
        tol = dict(atol=2e-11, rtol=2e-9) if dtype == torch.float64 else dict(atol=3e-6, rtol=4e-5)
        for name in left:
            self.assertEqual(left[name].requires_grad, right[name].requires_grad, name)
            torch.testing.assert_close(left[name], right[name], **tol, msg=name)

    def compare_gradients(self, models, dtype=torch.float64):
        tol = dict(atol=2e-10, rtol=2e-8) if dtype == torch.float64 else dict(atol=4e-6, rtol=5e-4)
        for (name, left), (other, right) in zip(models[0].named_parameters(), models[1].named_parameters()):
            self.assertEqual(name, other)
            self.assertEqual(left.grad is None, right.grad is None, name)
            if left.grad is not None:
                self.assertTrue(bool(left.grad.isfinite().all()), name)
                self.assertTrue(bool(right.grad.isfinite().all()), name)
                torch.testing.assert_close(left.grad, right.grad, **tol, msg=name)

    def test_full_model_fp64_outputs_states_diagnostics_and_all_parameter_gradients(self):
        tokens = torch.tensor([[1, 2, 3, 5, 8, 13, 21, 2, 7, 9, 11, 4],
                               [3, 7, 9, 11, 13, 15, 17, 19, 2, 8, 14, 20]])
        models = self.models()
        results = []
        for model in models:
            output, state, diag = model(tokens)
            loss = F.cross_entropy(output[:, :-1].flatten(0, 1), tokens[:, 1:].flatten())
            loss.backward(); results.append((output, state, diag))
            self.assertGreater(float(model.raw_slow_rate.grad.abs()), 0.)
            self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
        torch.testing.assert_close(results[0][0], results[1][0], atol=2e-11, rtol=2e-9)
        self.compare_mapping(results[0][1], results[1][1])
        self.compare_mapping(results[0][2], results[1][2])
        self.compare_gradients(models)

    def fixture(self, model, steps, dtype, compact=False):
        generator = torch.Generator().manual_seed(808)
        b, d, w = 2, model.cfg.matrix_size, model.cfg.width
        state = model.initial_state(b)
        if compact:
            ids = model.selected_nodes
            for name in ('m', 'p', 'x', 'slow', 'trace', 'response', 'support', 'innovation', 'z',
                         'read_query', 'read_weight', 'feedback_gain', 'evidence_count'):
                state[name] = state[name].index_select(1, ids)
            for name in ('credit_query', 'credit_weight', 'credit_gain'):
                state[name] = state[name].index_select(2, ids)
        n = state['m'].shape[1]
        rand = lambda *shape: torch.randn(*shape, generator=generator, dtype=dtype)
        state['slow'] = torch.zeros_like(state['slow'])
        state['slow'][..., 0, 1] = 4.
        state['support'] = torch.full_like(state['support'], 99.999)
        state['evidence_count'].fill_(2**31-2)
        state['feedback_valid'] = torch.tensor([False, True])
        state['credit_valid'] = torch.arange(b*state['credit_valid'].shape[1]).reshape_as(state['credit_valid'])%2 == 0
        state['opaque_state'] = rand(b, 3)
        tokens = torch.randint(0, model.cfg.vocab_size, (b, steps), generator=generator)
        key = F.normalize(rand(b*steps, n, d), dim=-1)
        arrived = torch.arange(b*steps*n).reshape(b*steps, n)%3 != 0
        key = key*arrived[..., None]
        coefficients = dict(local_key=key, local_value=rand(b*steps, n, d).tanh(),
                            arrived=arrived, control=rand(b*steps, n, 5),
                            response=rand(b*steps, n, d, d).sigmoid(),
                            phase=F.normalize(rand(b*steps, n, model.cfg.modes, 2), dim=-1),
                            attention=rand(b*steps, n, model.cfg.neighbors).sigmoid(),
                            opaque_coefficient={'keep': 'unchanged'})
        query = torch.zeros(b*steps, n, d, dtype=dtype); query[..., 0] = 1.
        reads = dict(hidden=rand(b*steps, w), query=query,
                     weight=torch.full((b*steps, n), 1/n, dtype=dtype),
                     scale=torch.ones(b*steps, dtype=dtype),
                     gain=torch.ones(b*steps, n, dtype=dtype),
                     effective=torch.full((b*steps,), n, dtype=dtype),
                     innovation=rand(b*steps, n).square())
        logits = F.linear(model.head(reads['hidden']), model.embedding.weight).reshape(b, steps, -1)
        physical = bounded(rand(b, steps, n, model.cfg.bands, d, d))
        return tokens, state, coefficients, reads, logits, physical

    def test_credit_capacity_arrival_and_unknown_fields_at_fp32_boundaries(self):
        for credit, steps in ((1, 1), (1, 5), (5, 1), (5, 2), (4, 7)):
            for compact in (False, True):
                with self.subTest(credit=credit, steps=steps, compact=compact):
                    models = self.models(torch.float32, credit=credit, compact=compact)
                    results = []
                    for model in models:
                        args = self.fixture(model, steps, torch.float32, compact)
                        extra = args[2]['opaque_coefficient']
                        state, diag = model.commit_block(*args)
                        self.assertIs(args[2]['opaque_coefficient'], extra)
                        torch.testing.assert_close(state['opaque_state'], args[1]['opaque_state'], atol=0, rtol=0)
                        self.assertTrue(bool(state['feedback_valid'].all()))
                        self.assertFalse(state['read_query'].requires_grad)
                        self.assertLessEqual(float(state['slow'].detach().norm(dim=(-1, -2)).max()), 4.+3e-6)
                        self.assertLessEqual(int(state['evidence_count'].max()), 2**31-1)
                        self.assertTrue(bool(torch.linalg.eigvalsh(state['p']).min() > 0))
                        # The saturated orthogonal memory forces real directional clipping.
                        if steps > 1:
                            self.assertGreater(float(diag['capacity_clip_fraction']), 0.)
                        (state['m'].square().mean()+state['slow'].square().mean()
                         +state['trace'].square().mean()+args[4].square().mean()).backward()
                        results.append((state, diag))
                    self.compare_mapping(results[0][0], results[1][0], torch.float32)
                    self.compare_mapping(results[0][1], results[1][1], torch.float32)
                    self.compare_gradients(models, torch.float32)

    def test_feedback_switches_preserve_consumption_and_finite_credit_drop(self):
        tokens = torch.tensor([[1, 2, 3, 5, 8, 13, 21, 2]])
        for flag in ('runtime_automatic_feedback', 'runtime_local_enabled', 'runtime_slow_enabled'):
            with self.subTest(flag=flag):
                models = self.models(credit=3)
                results = []
                for model in models:
                    setattr(model, flag, False)
                    with torch.no_grad():
                        output, state, diag = model(tokens)
                    results.append((output, state, diag))
                    self.assertTrue(bool(state['credit_valid'].all()))
                    torch.testing.assert_close(state['slow'], torch.zeros_like(state['slow']), atol=0, rtol=0)
                torch.testing.assert_close(results[0][0], results[1][0], atol=2e-11, rtol=2e-9)
                self.compare_mapping(results[0][1], results[1][1])
                self.compare_mapping(results[0][2], results[1][2])

    def test_f1_direction_detached_eta_and_old_slow_remain_differentiable(self):
        model = self.models()[1]
        args = list(self.fixture(model, 4, torch.float64))
        args[1]['slow'] = (torch.randn_like(args[1]['slow'])*.02).requires_grad_()
        old = args[1]['slow']
        state, _ = model.commit_block(*args)
        state['slow'].square().sum().backward()
        self.assertGreater(float(model.raw_slow_rate.grad.abs()), 0.)
        self.assertGreater(float(old.grad.abs().sum()), 0.)
        for name in ('embedding.weight', 'ff.0.weight', 'read_projection.weight'):
            self.assertIsNone(dict(model.named_parameters())[name].grad, name)

    def test_retained_lagged_credit_is_consumed_once_after_a_bulk_cache_commit(self):
        tokens = torch.tensor([[1, 2, 3, 5, 8, 13, 21, 2]])
        states = []
        for model in self.models(credit=5):
            model.runtime_automatic_feedback = False
            with torch.no_grad():
                _, state, _ = model(tokens)
                for lag in (5, 2, 1):
                    state = model.apply_feedback(state, torch.tensor([7]), lag=lag)
                    repeated = model.apply_feedback(state, torch.tensor([7]), lag=lag)
                    torch.testing.assert_close(state['slow'], repeated['slow'], atol=0, rtol=0)
                    self.assertFalse(bool(model.feedback_frame(state, lag)['feedback_valid'].any()))
            states.append(state)
        self.compare_mapping(states[0], states[1])

    def test_reset_invalidation_and_pending_are_unchanged_across_optimizer_updates(self):
        models = self.models()
        tokens = torch.tensor([[1, 2, 3, 5, 8, 13, 21, 2], [3, 7, 9, 11, 13, 15, 17, 19]])
        results = []
        for model in models:
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            state = None
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                output, state, _ = model(tokens, state, reset_mask=torch.tensor([step == 1, False]))
                F.cross_entropy(output[:, :-1].flatten(0, 1), tokens[:, 1:].flatten()).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step(); state = invalidate_feedback(detach(state))
                self.assertFalse(bool(state['feedback_valid'].any()))
                self.assertFalse(bool(state['credit_valid'].any()))
            results.append((output, state))
        torch.testing.assert_close(results[0][0], results[1][0], atol=3e-10, rtol=3e-8)
        self.compare_mapping(results[0][1], results[1][1])

    def test_head_vjp_and_credit_cache_work_is_batched_without_hiding_slow_steps(self):
        tokens = torch.tensor([[1, 2, 3, 5, 8, 13, 21, 2]])
        counts = []
        for model in self.models():
            with patch.object(model, 'head_vjp', wraps=model.head_vjp) as vjp, \
                    patch.object(model, 'cache_feedback', wraps=model.cache_feedback) as cache, \
                    patch.object(model, 'project_slow_update', wraps=model.project_slow_update) as project:
                model(tokens)
                counts.append((vjp.call_count, cache.call_count, project.call_count))
        self.assertEqual(counts[0], (8, 8, 8))
        self.assertEqual(counts[1], (4, 0, 8))


if __name__ == '__main__':
    unittest.main()
