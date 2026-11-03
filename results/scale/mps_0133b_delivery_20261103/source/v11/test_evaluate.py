"""Check the intervention contract, including masked QA scoring and weighted aggregation."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from .evaluate import evaluate_history


class Recorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(12).float()/20)
        self.cfg = SimpleNamespace(vocab_size=12, context_limit=2)
        self.calls = []

    def initial_state(self, batch):
        return dict(position=torch.zeros(batch, dtype=torch.long), memory=[torch.zeros(batch, 1)],
                    pending=[torch.zeros(batch, 1)], valid=None, kv=[None])

    def forward(self, tokens, state=None, reset_mask=None, memory_read=True, memory_write=True):
        state = self.initial_state(len(tokens)) if state is None else state
        self.calls.append((tokens.clone(), state['position'].clone(), memory_read,
                           state['memory'][0].clone(),
                           None if state['kv'][0] is None else state['kv'][0].clone()))
        new = state | dict(position=state['position']+tokens.shape[1],
                           memory=[state['memory'][0]+tokens.float().sum(1, keepdim=True)],
                           kv=[tokens.float().sum(1, keepdim=True)])
        return self.weight[None, None].expand(*tokens.shape, -1), new, {}


class EvaluationChecks(unittest.TestCase):
    def test_identical_targets_recent_context_and_no_parameter_updates(self):
        model = Recorder()
        windows = np.asarray([[1, 2, 3, 4, 5, 6, 7, 8, 9], [9, 8, 7, 6, 5, 4, 3, 2, 1]])
        original = model.weight.detach().clone()
        result = evaluate_history(model, windows, torch.device('cpu'), 6, batch=2)
        self.assertTrue(model.training)
        torch.testing.assert_close(model.weight, original)
        self.assertEqual(len(model.calls), 12)
        for index in (0, 4, 6):
            prefix = model.calls[index]
            torch.testing.assert_close(prefix[0][:, -2:], torch.tensor(windows[:, 4:6]))
        for index in (1, 2, 3, 5, 7):
            scored = model.calls[index]
            torch.testing.assert_close(scored[0], torch.tensor(windows[:, 6:-1]))
            torch.testing.assert_close(scored[1], torch.full((2,), 6))
            torch.testing.assert_close(scored[4], model.calls[1][4])
        torch.testing.assert_close(model.calls[2][3], torch.zeros(2, 1))
        self.assertFalse(model.calls[3][2])
        self.assertFalse(model.calls[8][2])
        self.assertFalse(model.calls[9][2])
        torch.testing.assert_close(model.calls[10][0][:, -2:], torch.tensor(windows[:, 4:6]))
        torch.testing.assert_close(model.calls[11][0], torch.tensor(windows[:, 6:-1]))
        self.assertEqual(result['conditions']['warm']['targets'], 4)
        self.assertEqual(result['memory_clear_cost'], 0.)
        self.assertEqual(result['remote_tokens'], 4)
        self.assertEqual(result['total_read_off_cost'], 0.)
        self.assertIn('Whole-path conditions may change KV', result['intervention_contract'])

    def test_invalid_and_inapplicable_conditions(self):
        model = Recorder()
        with self.assertRaises(ValueError):
            evaluate_history(model, [[1., 2., 3.]], torch.device('cpu'), 1)
        with self.assertRaises(ValueError):
            evaluate_history(model, [[1, 2, 12]], torch.device('cpu'), 1)
        result = evaluate_history(model, np.asarray([[1, 2, 3, 4]]), torch.device('cpu'), 1)
        self.assertEqual(result['conditions']['unrelated_history']['status'], 'not_applicable')
        self.assertEqual(result['conditions']['remote_history_reencoded']['status'], 'not_applicable')

    def test_variable_masks_use_token_weighted_nll_and_first_answer_metrics(self):
        model = Recorder()
        windows = np.asarray([[1, 2, 3, 11, 4, 5, 0], [1, 2, 3, 1, 11, 9, 0]])
        mask = np.zeros((2, 6), dtype=bool)
        mask[0, 2] = True
        mask[1, 2:5] = True
        first = np.zeros_like(mask)
        first[:, 2] = True
        result = evaluate_history(model, windows, torch.device('cpu'), 2,
                                  target_mask=mask, first_answer_mask=first)
        targets = torch.tensor(windows[:, 3:])
        logits = model.weight[None, None].expand(2, 4, -1)
        losses = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction='none').reshape(2, 4)
        selected = torch.tensor(mask[:, 2:])
        expected = float(losses[selected].mean().detach())
        warm = result['conditions']['warm']
        self.assertAlmostEqual(warm['nll'], expected, places=6)
        self.assertEqual(warm['targets'], 4)
        self.assertEqual(warm['per_window_targets'], [1, 3])
        self.assertNotAlmostEqual(warm['window_mean_nll'], warm['nll'], places=4)
        self.assertAlmostEqual(warm['first_answer_nll'], float(losses[:, 0].mean().detach()), places=6)
        self.assertEqual(warm['first_answer_accuracy'], .5)
        self.assertEqual(warm['first_answer_accuracy_per_window'], [1., 0.])
        self.assertEqual(warm['first_answer_targets_per_window'], [1, 1])
        for condition in result['conditions'].values():
            if condition['status'] == 'completed':
                self.assertEqual(condition['per_window_targets'], [1, 3])
        # Recorder.forward accepts no target mask; scoring metadata never enters it.
        torch.testing.assert_close(model.calls[1][0], torch.tensor(windows[:, 2:-1]))

    def test_zero_score_windows_are_reported_without_nan(self):
        windows = np.asarray([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
        mask = np.zeros((2, 4), dtype=bool)
        mask[1, 2:] = True
        result = evaluate_history(Recorder(), windows, torch.device('cpu'), 2, target_mask=mask)
        row = result['conditions']['warm']
        self.assertEqual(row['per_window_nll'][0], None)
        self.assertEqual(row['per_window_targets'], [0, 2])
        self.assertEqual(row['scored_windows'], 1)
        self.assertEqual(row['first_answer_nll'], None)
        self.assertTrue(np.isfinite(row['nll']))

    def test_bad_masks_are_rejected_before_forward(self):
        windows = np.asarray([[1, 2, 3, 4, 5]])
        bad = (np.ones((1, 5), dtype=bool), np.ones((1, 4), dtype=np.float32),
               np.zeros((1, 4), dtype=bool))
        for mask in bad:
            model = Recorder()
            with self.assertRaises(ValueError):
                evaluate_history(model, windows, torch.device('cpu'), 2, target_mask=mask)
            self.assertEqual(model.calls, [])
        target = np.zeros((1, 4), dtype=bool)
        target[:, 2] = True
        for first in (np.array([[True, False, False, False]]),
                      np.array([[False, False, False, True]])):
            with self.assertRaises(ValueError):
                evaluate_history(Recorder(), windows, torch.device('cpu'), 2,
                                 target_mask=target, first_answer_mask=first)

    def test_right_padding_does_not_change_causal_first_answer_score(self):
        from .model import Config, Network
        torch.manual_seed(8)
        model = Network(Config(vocab_size=12, width=16, layers=1, heads=2, ff=32,
                               context_limit=4, segment_tokens=2, memory_enabled=False))
        first = np.asarray([[1, 2, 3, 4, 0, 0, 0], [1, 2, 3, 5, 0, 0, 0]])
        second = first.copy()
        second[:, 4:] = [[6, 7, 8], [9, 10, 11]]
        mask = np.zeros((2, 6), dtype=bool)
        mask[:, 2] = True
        a = evaluate_history(model, first, torch.device('cpu'), 2,
                             target_mask=mask, first_answer_mask=mask)
        b = evaluate_history(model, second, torch.device('cpu'), 2,
                             target_mask=mask, first_answer_mask=mask)
        self.assertEqual(a['first_answer_nll'], b['first_answer_nll'])
        self.assertEqual(a['first_answer_accuracy'], b['first_answer_accuracy'])

    def test_reencoding_old_remote_tokens_cannot_change_frozen_local_transformer(self):
        from .model import Config, Network
        torch.manual_seed(27)
        model = Network(Config(vocab_size=12, width=16, layers=2, heads=2, ff=32,
                               context_limit=4, segment_tokens=2, memory_enabled=False)).double()
        windows = np.asarray([[1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                              [9, 8, 7, 6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
        before = {name: value.clone() for name, value in model.state_dict().items()}
        result = evaluate_history(model, windows, torch.device('cpu'), 12)
        self.assertEqual(result['baseline_receptive_tokens'], 8)
        self.assertEqual(result['remote_tokens'], 4)
        warm = result['conditions']['warm']
        remote = result['conditions']['remote_history_reencoded']
        self.assertEqual(warm['per_window_nll'], remote['per_window_nll'])
        self.assertEqual(result['reencoded_history_cost'], 0.)
        for name, value in before.items():
            torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)

    def test_all_path_read_off_matches_one_shot_full_model_read_off(self):
        from .model import Config, Network
        torch.manual_seed(31)
        model = Network(Config(vocab_size=12, width=16, layers=1, heads=2, ff=32,
                               context_limit=4, segment_tokens=2, nodes=8, bands=2)).double()
        windows = np.asarray([[1, 2, 3, 4, 5, 6, 7, 8, 9], [9, 8, 7, 6, 5, 4, 3, 2, 1]])
        result = evaluate_history(model, windows, torch.device('cpu'), 6)
        with torch.no_grad():
            inputs = torch.tensor(windows[:, :-1])
            targets = torch.tensor(windows[:, 1:])
            logits, _, _ = model(inputs, memory_read=False)
            expected = F.cross_entropy(logits[:, 6:].float().flatten(0, 1), targets[:, 6:].flatten(),
                                       reduction='none').reshape(2, -1).mean(1)
        actual = result['conditions']['memory_read_off_all']['per_window_nll']
        np.testing.assert_allclose(actual, expected.cpu().numpy(), atol=2e-7, rtol=2e-7)


if __name__ == '__main__':
    unittest.main()
