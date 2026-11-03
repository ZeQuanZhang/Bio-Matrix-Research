"""Check the intervention contract, independent of architecture quality."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch

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
        self.assertEqual(len(model.calls), 8)
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
        self.assertEqual(result['conditions']['warm']['targets'], 4)
        self.assertEqual(result['memory_clear_cost'], 0.)

    def test_invalid_and_inapplicable_conditions(self):
        model = Recorder()
        with self.assertRaises(ValueError):
            evaluate_history(model, [[1., 2., 3.]], torch.device('cpu'), 1)
        with self.assertRaises(ValueError):
            evaluate_history(model, [[1, 2, 12]], torch.device('cpu'), 1)
        result = evaluate_history(model, np.asarray([[1, 2, 3, 4]]), torch.device('cpu'), 1)
        self.assertEqual(result['conditions']['unrelated_history']['status'], 'not_applicable')


if __name__ == '__main__':
    unittest.main()
