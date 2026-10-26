"""Evaluation protocol checks; tiny fixtures verify plumbing, not model ability."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .common import Console
from .evaluate import calibration_error, evaluate, feed, history_subsets, runtime_intervention, windows
from .model import Config, Network, Stateless, Transformer, initialize, invalidate_feedback


class TinyCorpus:
    def sample(self, split, rng, batch, length):
        docs = rng.integers(0, 3, size=batch).tolist()
        starts = rng.integers(0, 20, size=batch).tolist()
        values = np.stack([(np.arange(length+1)+start+doc*3) % 29+3 for doc, start in zip(docs, starts)])
        return values, docs, starts


class RecordingModel(torch.nn.Module):
    def __init__(self, variant='full', local=True):
        super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(.01))
        self.cfg = SimpleNamespace(variant=variant); self.calls = []; self.fail_local = False
        if local:
            self.runtime_local_enabled = True; self.runtime_slow_enabled = True

    def forward(self, tokens, state=None):
        local = getattr(self, 'runtime_local_enabled', None)
        if self.fail_local and local is False:
            raise RuntimeError('deliberate local-probe failure')
        count = 0 if state is None else state
        outputs = []
        for token in tokens[0]:
            self.calls.append((int(token), count, local))
            count += 1
            outputs.append(torch.arange(32, device=tokens.device)*self.weight*(count+int(token)))
        return torch.stack(outputs)[None], count, {'age': torch.tensor(float(count)), 'local_update_rms': float(bool(local))}

    def head_only(self, tokens):
        return torch.zeros(*tokens.shape, 32, device=tokens.device)+self.weight


class BagHistoryModel(torch.nn.Module):
    """An order-blind counting fixture must receive no credit for ordered history."""
    def __init__(self):
        super().__init__(); self.cfg = SimpleNamespace(variant='stateless')

    def forward(self, tokens, state=None):
        count = torch.zeros(1, 32) if state is None else state
        outputs = []
        for t in range(tokens.shape[1]):
            count = count.scatter_add(1, tokens[:, t, None], torch.ones(1, 1))
            outputs.append(count*.1)
        return torch.stack(outputs, 1), count, {}


class EvaluationProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.cfg = dict(evaluation_seed=709, prefix_tokens=4, target_tokens=5,
                        bptt_tokens=3, validation_windows=2, test_windows=2)
        self.device = torch.device('cpu')

    def test_windows_are_paired_and_reproducible(self):
        a = windows(TinyCorpus(), 'test', self.cfg); b = windows(TinyCorpus(), 'test', self.cfg)
        self.assertEqual(a['sha256'], b['sha256'])
        self.assertTrue(all(x != y for x, y in zip(a['docs'], a['other'])))

    def test_calibration_bins_count_boundaries_and_certainty_once(self):
        self.assertEqual(calibration_error([[.5, 0.]]), .5)
        self.assertEqual(calibration_error([[.5, 1.]]), .5)
        self.assertEqual(calibration_error([[1., 0.]]), 1.)
        self.assertEqual(calibration_error([[1., 1.]]), 0.)
        # With all predictions wrong, each sample contributes its confidence divided by sample count.
        confidence = np.arange(11)/10
        self.assertAlmostEqual(calibration_error(np.stack((confidence, np.zeros_like(confidence)), -1)),
                               float(confidence.mean()), places=14)
        self.assertAlmostEqual(calibration_error([[.5, 0.], [.5, 1.], [1., 1.]]), 0., places=14)

    def test_every_condition_scores_identical_visible_inputs_and_targets(self):
        model = RecordingModel().train(); before = model.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate(model, TinyCorpus(), 'test', self.cfg, self.device, Console(enabled=False), directory, True)
            rows = json.loads((Path(directory)/'test_predictions.json').read_text())
        self.assertEqual(len(result['conditions']), 12)
        for i in range(self.cfg['test_windows']):
            window = [row for row in rows if row['window'] == i]
            for row in window:
                self.assertEqual(row['input_tokens'], window[0]['input_tokens'])
                self.assertEqual(row['target_tokens'], window[0]['target_tokens'])
                self.assertEqual(row['scored_tokens_sha256'], result['scored_tokens_sha256'])
                self.assertEqual(row['subset_masks'], window[0]['subset_masks'])
                self.assertEqual(len(row['per_token_diagnostics']), self.cfg['target_tokens'])
        age = result['conditions']['carry']['diagnostics']['age']
        self.assertEqual(age['samples'], 10); self.assertEqual(age['mean'], 7.)
        self.assertTrue(model.training); self.assertTrue(model.runtime_local_enabled)
        self.assertTrue(model.runtime_slow_enabled); torch.testing.assert_close(model.weight, before)

    def test_aba_controls_match_age_recent_history_and_remote_frequencies(self):
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate(RecordingModel(), TinyCorpus(), 'test', self.cfg, self.device,
                              Console(enabled=False), directory, True)
            rows = json.loads((Path(directory)/'test_predictions.json').read_text())
        for window in range(self.cfg['test_windows']):
            records = {row['condition']: row for row in rows if row['window'] == window}
            a = records['carry']['history_tokens']; b = records['unrelated']['history_tokens']
            self.assertEqual(records['A_B_A']['history_tokens'], a+b)
            self.assertEqual(records['B_B_A']['history_tokens'], b+b)
            self.assertEqual(records['B_A_A']['history_tokens'], b+a)
            shuffled = records['shuffled_A_B_A']['history_tokens']
            self.assertEqual(sorted(shuffled[:len(a)]), sorted(a)); self.assertEqual(shuffled[len(a):], b)
            for condition in ('A_B_A', 'B_B_A', 'B_A_A', 'shuffled_A_B_A'):
                self.assertEqual(records[condition]['initial_history_length'], 2*self.cfg['prefix_tokens'])
                self.assertEqual(records[condition]['per_token_diagnostics'][0]['age'], 2*self.cfg['prefix_tokens']+1)
        self.assertIn('mixes extra state age', result['contrast_interpretation']['return_cost'])

    def test_order_blind_history_cannot_pass_frequency_matched_order_control(self):
        result = evaluate(BagHistoryModel(), TinyCorpus(), 'test', self.cfg, self.device,
                          Console(enabled=False), interventions=True)
        self.assertEqual(result['contrasts']['ordered_remote_history_gain'], 0.)
        self.assertEqual(result['conditions']['A_B_A']['per_position_nll'],
                         result['conditions']['shuffled_A_B_A']['per_position_nll'])

    def test_prefix_state_probes_call_model_api_without_changing_inputs(self):
        model = RecordingModel(); calls = []
        model.reset_fast = lambda state: calls.append(('fast', state)) or 0
        model.clear_slow = lambda state: calls.append(('slow', state)) or state
        result = evaluate(model, TinyCorpus(), 'test', self.cfg, self.device,
                          Console(enabled=False), interventions=True)
        self.assertEqual(calls, [('fast', 4), ('slow', 4), ('fast', 4), ('slow', 4)])
        self.assertEqual(result['conditions']['reset_fast_after_prefix']['diagnostics']['age']['mean'], 3.)
        self.assertEqual(result['conditions']['clear_slow_after_prefix']['diagnostics']['age']['mean'], 7.)
        self.assertEqual(result['conditions']['reset_fast_after_prefix']['tokens'], 10)

    def test_scoring_subset_annotations_cannot_change_model_predictions(self):
        first = TinyCorpus(); first.tokens = {'train': np.array([5]*1000+[7]*2+[8]*2+[11]*200)}
        second = TinyCorpus(); second.tokens = {'train': np.arange(32).repeat(30)}
        a = evaluate(RecordingModel(), first, 'test', self.cfg, self.device, Console(enabled=False), interventions=True)
        b = evaluate(RecordingModel(), second, 'test', self.cfg, self.device, Console(enabled=False), interventions=True)
        self.assertNotEqual(a['subset_protocol']['train_frequency_sha256'], b['subset_protocol']['train_frequency_sha256'])
        self.assertEqual(a['conditions'], b['conditions'])

    def test_credit_only_probe_invalidates_metadata_after_prefix_without_content_loss(self):
        model = Network(Config(vocab_size=64, width=16, matrix_size=4, modes=2, internal_steps=2)).eval()
        initialize(model, 71); calls = []
        def checked_invalidation(state):
            calls.append(state['evidence_count'].clone())
            result = invalidate_feedback(state)
            for name, value in state.items():
                if name not in ('feedback_valid', 'credit_valid'):
                    torch.testing.assert_close(result[name], value, atol=0, rtol=0)
            self.assertFalse(bool(result['feedback_valid'].any()))
            self.assertFalse(bool(result['credit_valid'].any()))
            return result
        with patch('v07.evaluate.invalidate_feedback', side_effect=checked_invalidation):
            result = evaluate(model, TinyCorpus(), 'test', self.cfg, self.device,
                              Console(enabled=False), interventions=True)
        self.assertEqual(len(calls), self.cfg['test_windows'])
        self.assertTrue(all(int(count.max()) == self.cfg['prefix_tokens'] for count in calls))
        base = result['conditions']['credit_only_reset_after_prefix']['nll']
        for name in ('reset_fast_after_prefix', 'clear_slow_after_prefix'):
            self.assertAlmostEqual(result['contrasts'][name+'_vs_credit_only_cost'],
                                   result['conditions'][name]['nll']-base)

    def test_lexical_subsets_use_train_rarity_and_available_history_only(self):
        a = np.full((1, 69), 5, dtype=np.int64); b = np.full_like(a, 11)
        a[0, 0] = 7; a[0, 10] = 8; a[0, 65:] = [7, 8, 14, 15]
        corpus = SimpleNamespace(tokens={'train': np.array([5]*1000+[7]*2+[8]*2+[11]*200)})
        masks, protocol = history_subsets(corpus, dict(a=a, b=b), 64, 4)
        self.assertEqual(masks['remote_rare_repeat'].tolist(), [[True, True, False, False]])
        self.assertEqual(masks['no_remote_token_match'].tolist(), [[False, False, True, True]])
        # Future targets may affect their own offline scoring mask, never a preceding mask.
        changed = a.copy(); changed[0, 67:] = [8, 7]
        other, _ = history_subsets(corpus, dict(a=changed, b=b), 64, 4)
        np.testing.assert_array_equal(masks['remote_rare_repeat'][:, :2], other['remote_rare_repeat'][:, :2])
        b[0, 0] = 7
        excluded, _ = history_subsets(corpus, dict(a=a, b=b), 64, 4)
        self.assertFalse(excluded['remote_repeat'][0, 0])
        self.assertIn('not ground-truth', protocol['interpretation'])
        unavailable, info = history_subsets(TinyCorpus(), dict(a=a, b=b), 64, 4)
        self.assertNotIn('remote_rare_repeat', unavailable); self.assertTrue(info['unavailable'])

    def test_reset_each_token_actually_starts_fresh(self):
        model = RecordingModel()
        logits, state = feed(model, np.array([3, 4, 5, 6]), self.device, 3,
                             reset_each_token=True, diagnostics=[])
        self.assertEqual([call[1] for call in model.calls], [0, 0, 0, 0])
        self.assertEqual(state, 1); self.assertEqual(logits.shape, (4, 32))

    def test_targets_are_not_visible_before_their_own_position(self):
        a = RecordingModel(); b = RecordingModel()
        first, _ = feed(a, np.array([3, 4, 5, 6]), self.device, 1, diagnostics=[])
        second, _ = feed(b, np.array([3, 4, 21, 22]), self.device, 1, diagnostics=[])
        torch.testing.assert_close(first[:2], second[:2])
        self.assertEqual([x[0] for x in a.calls], [3, 4, 5, 6])

    def test_local_probe_suppresses_prefix_and_targets_and_restores_on_error(self):
        model = RecordingModel().train(); config_before = vars(model.cfg).copy()
        with runtime_intervention(model, 'no_local'):
            _, state = feed(model, np.array([3, 4, 5]), self.device, 2)
            feed(model, np.array([6, 7]), self.device, 2, state)
        self.assertTrue(all(call[2] is False for call in model.calls))
        self.assertTrue(model.runtime_local_enabled); self.assertEqual(vars(model.cfg), config_before)
        model.fail_local = True
        with self.assertRaisesRegex(RuntimeError, 'deliberate'):
            evaluate(model, TinyCorpus(), 'test', self.cfg, self.device, Console(enabled=False), interventions=True)
        self.assertTrue(model.training); self.assertTrue(model.runtime_local_enabled)
        self.assertEqual(vars(model.cfg), config_before)

    def test_unsupported_probes_are_explicit_and_hash_is_architecture_independent(self):
        graph = evaluate(RecordingModel(), TinyCorpus(), 'test', self.cfg, self.device, Console(enabled=False), interventions=True)
        other = evaluate(RecordingModel('stateless', local=False), TinyCorpus(), 'test', self.cfg,
                         self.device, Console(enabled=False), interventions=True)
        self.assertEqual(graph['evaluation_sha256'], other['evaluation_sha256'])
        self.assertEqual(other['intervention_applicability']['no_local']['status'], 'not_applicable')
        self.assertNotIn('no_local', other['conditions'])
        bypass = evaluate(RecordingModel('bypass'), TinyCorpus(), 'test', self.cfg,
                          self.device, Console(enabled=False), interventions=True)
        self.assertIn('head_only', bypass['conditions']); self.assertNotIn('head_only', graph['conditions'])

    def test_real_architectures_keep_weights_frozen_and_local_probe_is_inactive(self):
        cfg = self.cfg | dict(prefix_tokens=3, target_tokens=3, test_windows=1)
        hashes = []
        for variant, cls in [('full', Network), ('bypass', Network), ('transformer', Transformer), ('stateless', Stateless)]:
            model = cls(Config(vocab_size=64, variant=variant, width=16, matrix_size=4, modes=2,
                               internal_steps=2, transformer_width=16, transformer_ff=32))
            initialize(model, 71); model.train()
            before = {name: value.clone() for name, value in model.state_dict().items()}
            result = evaluate(model, TinyCorpus(), 'test', cfg, self.device, Console(enabled=False), interventions=True)
            hashes.append(result['evaluation_sha256'])
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, before[name], atol=0, rtol=0)
            if cls is Network:
                self.assertEqual(result['conditions']['no_local']['diagnostics']['local_update_rms']['maximum'], 0.)
                self.assertEqual(result['conditions']['no_local']['diagnostics']['slow_rms']['maximum'], 0.)
                self.assertTrue(model.runtime_local_enabled); self.assertTrue(model.runtime_slow_enabled)
            self.assertTrue(model.training)
        self.assertEqual(len(set(hashes)), 1)


if __name__ == '__main__':
    unittest.main()
