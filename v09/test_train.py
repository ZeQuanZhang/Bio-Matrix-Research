"""Small CPU tests of masked supervision, fixed schedules and real resume."""
from pathlib import Path
import json
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from v07.common import file_hash, save
from v07.data import ContinuousStream
from .model import Config, Network, initialize
from .train import (DEFAULTS, capture_rng, checkpoint_payload, curriculum, evaluate,
                    crossed_milestones, learning_rate_scale, parameter_groups, restore_checkpoint,
                    select_targets, settings, train, update, validation_data)


class TinyCorpus:
    def __init__(self):
        docs = [[1, *[3+(i+j)%29 for i in range(40)], 2] for j in range(4)]
        values = np.asarray(sum(docs, []), dtype=np.int64)
        offsets = np.arange(0, len(values)+1, 42, dtype=np.int64)
        self.tokens = {'train': values, 'validation': values[::-1].copy()}
        self.offsets = {'train': offsets, 'validation': offsets}
        self.manifest = dict(vocab_size=32, fixture='tiny-public-protocol-test')
        self.sampled = []

    def sample(self, split, rng, batch, length):
        if split != 'validation':
            raise AssertionError('Training uses the real ContinuousStream; never sample test')
        self.sampled.append(split)
        docs = rng.integers(0, 4, size=batch)
        starts = docs*42+rng.integers(0, 42-length, size=batch)
        values = np.stack([self.tokens[split][start:start+length+1] for start in starts])
        return values, docs.tolist(), starts.tolist()


def small_model():
    model = Network(Config(vocab_size=32, nodes=16, matrix_size=4, width=16,
                bands=2, modes=2, internal_steps=2, block_size=4, group_size=2,
                credit_horizon=5, active_fraction=.5, route_mode='cyclic'))
    initialize(model, 119)
    return model


class TrainingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_masked_ce_never_uses_source_metadata_and_only_resets_chunk_start(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.tensor([.2, -.5, 1., .7]))
            def forward(self, tokens, state=None, reset_mask=None):
                self.reset_seen = reset_mask.clone()
                return self.weight[None, None].expand(*tokens.shape, -1), dict(x=self.weight[None]*2), {}
        model = Tiny(); optimizer = torch.optim.SGD(model.parameters(), lr=0.)
        values = np.asarray([[0, 1, 2, 3], [2, 3, 1, 0]])
        mask = np.asarray([[True, False, True], [False, True, False]])
        reset = np.zeros_like(mask); reset[0, 0] = True
        expected = F.cross_entropy(model.weight[None].expand(3, -1), torch.tensor([1, 3, 1]))
        row, state = update(model, optimizer, values, mask, reset, None, torch.device('cpu'))
        self.assertAlmostEqual(row['loss'], float(expected.detach()), places=6)
        self.assertEqual(row['targets'], 3)
        self.assertEqual(row['masked_targets'], 3)
        torch.testing.assert_close(model.reset_seen, torch.tensor([True, False]))
        self.assertFalse(state['x'].requires_grad)
        reset[0, 1] = True
        with self.assertRaises(ValueError):
            update(model, optimizer, values, mask, reset, None, torch.device('cpu'))
        selected = select_targets(mask, 2)
        self.assertEqual(int(selected.sum()), 2)
        self.assertEqual(int(mask.sum()), 3)
        self.assertTrue(bool((selected <= mask).all()))

    def test_parameter_groups_have_distinct_rates_and_no_decay_for_raw_norm_bias(self):
        model = small_model(); groups = parameter_groups(model, DEFAULTS)
        mapping = {id(parameter): group for group in groups for parameter in group['params']}
        self.assertEqual(len(mapping), len(list(model.parameters())))
        parameters = dict(model.named_parameters())
        for name, rate in (('content_projection.weight', 3e-4), ('controls.0.weight', 1e-4),
                           ('raw_attention_temperature', 1e-4), ('raw_tau', 3e-5)):
            self.assertEqual(mapping[id(parameters[name])]['base_lr'], rate)
        for name, parameter in parameters.items():
            if parameter.ndim < 2 or name.startswith('raw_') or name.endswith('.bias'):
                self.assertEqual(mapping[id(parameter)]['weight_decay'], 0., name)
        self.assertEqual(mapping[id(parameters['embedding.weight'])]['weight_decay'], .01)
        from .benchmark import build
        transformer, _ = build(dict(condition='transformer', nodes=32), 32)
        groups = parameter_groups(transformer, DEFAULTS)
        mapping = {id(parameter): group for group in groups for parameter in group['params']}
        for name, parameter in transformer.named_parameters():
            if name.startswith('blocks.'):
                self.assertEqual(mapping[id(parameter)]['base_lr'], 3e-4, name)

    def test_schedule_and_resume_settings_keep_the_original_final_budget(self):
        self.assertEqual(learning_rate_scale(0, 100, .1, .1), 0.)
        self.assertEqual(learning_rate_scale(5, 100, .1, .1), .5)
        self.assertEqual(learning_rate_scale(10, 100, .1, .1), 1.)
        self.assertAlmostEqual(learning_rate_scale(100, 100, .1, .1), .1)
        payload = dict(protocol=dict(training=DEFAULTS | dict(tokens=100)))
        self.assertEqual(settings(SimpleNamespace(), payload)['tokens'], 100)
        self.assertEqual(settings(SimpleNamespace())['context_limit'], 4096)
        with self.assertRaises(ValueError):
            settings(SimpleNamespace(tokens=101), payload)
        model = small_model()
        curriculum(model, 50, 100)
        self.assertEqual(model.active_fraction, model.cfg.active_fraction)
        self.assertEqual(model.gate_blend, 1.)

    def test_checkpoint_restores_rng_next_continuous_batch_state_and_optimizer(self):
        random.seed(7); np.random.seed(7); torch.manual_seed(7)
        corpus = TinyCorpus(); model = small_model().double()
        optimizer = torch.optim.AdamW(parameter_groups(model, DEFAULTS))
        stream = ContinuousStream(corpus, 2, 8, 7101, 11, session_tokens=32,
                                  intrusion_tokens=2, intrusion_interval=12, reset_policy='session')
        values, mask, reset, _ = stream.next()
        _, state = update(model, optimizer, values, mask, reset, None, torch.device('cpu'))
        protocol = dict(test='checkpoint-protocol')
        payload = checkpoint_payload(model, optimizer, state, stream, protocol,
                                     dict(step=1, completed_tokens=int(mask.sum())), torch.device('cpu'))
        with TemporaryDirectory() as folder:
            save(Path(folder)/'last.pt', payload)
            frozen = torch.load(Path(folder)/'last.pt', weights_only=True)
        expected_random = (torch.rand(3), random.random(), np.random.random())
        next_values, next_mask, next_reset, next_sample = stream.next()
        expected, end = update(model, optimizer, next_values, next_mask, next_reset, state, torch.device('cpu'))
        recovered = small_model().double()
        resumed_optimizer = torch.optim.AdamW(parameter_groups(recovered, DEFAULTS))
        resumed_stream = ContinuousStream(corpus, 2, 8, 7101, 11, session_tokens=32,
                                         intrusion_tokens=2, intrusion_interval=12, reset_policy='session')
        recovered_state, counters = restore_checkpoint(frozen, recovered, resumed_optimizer,
                                                       resumed_stream, protocol, torch.device('cpu'))
        torch.testing.assert_close(torch.rand(3), expected_random[0], atol=0, rtol=0)
        self.assertEqual(random.random(), expected_random[1]); self.assertEqual(np.random.random(), expected_random[2])
        actual_values, actual_mask, actual_reset, actual_sample = resumed_stream.next()
        np.testing.assert_array_equal(actual_values, next_values)
        np.testing.assert_array_equal(actual_mask, next_mask)
        np.testing.assert_array_equal(actual_reset, next_reset)
        self.assertEqual(actual_sample, next_sample)
        actual, recovered_end = update(recovered, resumed_optimizer, actual_values, actual_mask, actual_reset,
                                       recovered_state, torch.device('cpu'))
        self.assertEqual(actual['loss'], expected['loss'])
        self.assertEqual(counters['step'], 1)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, dict(recovered.named_parameters())[name], atol=0, rtol=0, msg=name)
        for name in end:
            torch.testing.assert_close(end[name], recovered_end[name], atol=0, rtol=0, msg=name)
        with self.assertRaises(ValueError):
            restore_checkpoint(frozen, recovered, resumed_optimizer, resumed_stream,
                               protocol | {'changed': True}, torch.device('cpu'))

    def test_fixed_development_windows_are_cold_and_preserve_training_rng(self):
        corpus = TinyCorpus(); config = DEFAULTS | dict(sequence_tokens=8, validation_windows=3)
        values, sample = validation_data(corpus, config)
        other, other_sample = validation_data(corpus, config)
        np.testing.assert_array_equal(values, other); self.assertEqual(sample, other_sample)
        model = small_model().train()
        before = capture_rng(torch.device('cpu'))
        progress = []
        score = evaluate(model, values, torch.device('cpu'), 2, progress_callback=progress.append)
        after = capture_rng(torch.device('cpu'))
        self.assertEqual(corpus.sampled, ['validation', 'validation'])
        self.assertEqual(score['targets'], 24)
        self.assertTrue(score['cold_state'] and model.training)
        torch.testing.assert_close(before['torch']['cpu'], after['torch']['cpu'], atol=0, rtol=0)
        self.assertEqual(before['python'], after['python']); self.assertEqual(before['numpy'], after['numpy'])
        self.assertEqual([value['completed'] for value in progress], [2, 3])
        self.assertTrue(all(value['total'] == 3 and np.isfinite(value['nll']) for value in progress))

    def test_milestones_use_fixed_nominal_thresholds_and_exact_actual_counts(self):
        self.assertEqual(crossed_milestones(249984, 250112, 12000000), (250000,))
        self.assertEqual(crossed_milestones(999936, 1000064, 3000000), (1000000,))
        self.assertEqual(crossed_milestones(3000000, 3000128, 12000000), ())
        self.assertEqual(crossed_milestones(0, 12000000, 3000000), (250000, 1000000, 3000000))

    def test_real_training_entry_aborts_safely_and_resumes_identical_trajectory(self):
        import v09.train as implementation
        corpus = TinyCorpus()
        source = {'v09/train.py': file_hash(Path(implementation.__file__))}
        base = dict(device='cpu', threads=1, quiet=True, resume=None, data_root=Path('/tiny-fixture'),
                    condition='v09_dense', nodes=32, tokens=24, batch=1, sequence_tokens=16,
                    session_tokens=32, validation_windows=2, eval_every=12, checkpoint_every=1,
                    intrusion_tokens=2, intrusion_interval=12)
        def construct(*_, **kwargs):
            self.assertEqual(kwargs['context_limit'], 4096)
            model = small_model()
            return model, model.cfg
        class StopAfterOne:
            requested = False
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass
        stop = StopAfterOne()
        original_update = implementation.update
        def one_update(*args, **kwargs):
            output = original_update(*args, **kwargs); stop.requested = True
            return output
        with TemporaryDirectory() as folder, patch.object(implementation, 'frozen_corpus', return_value=corpus), \
                patch.object(implementation, 'build', side_effect=construct), \
                patch.object(implementation, 'training_sources', return_value=source), \
                patch.object(implementation, 'MILESTONES', (8, 16, 24)):
            full = Path(folder)/'full'; resumed = Path(folder)/'resumed'
            self.assertEqual(train(SimpleNamespace(**base, output=full))[1], 'completed')
            with patch.object(implementation, 'StopRequest', return_value=stop), \
                    patch.object(implementation, 'update', side_effect=one_update):
                self.assertEqual(train(SimpleNamespace(**base, output=resumed))[1], 'aborted')
            checkpoint = resumed/'last.pt'
            saved = torch.load(checkpoint, weights_only=True)
            self.assertLess(saved['counters']['completed_tokens'], 24)
            resume_args = SimpleNamespace(device='cpu', threads=1, quiet=True, resume=checkpoint,
                                          output=None, data_root=None)
            self.assertEqual(train(resume_args)[1], 'completed')
            left, right = (torch.load(path/'last.pt', weights_only=True) for path in (full, resumed))
            self.assertEqual(right['counters']['completed_tokens'], 24)
            self.assertEqual(left['stream'], right['stream'])
            for name in left['model']:
                torch.testing.assert_close(left['model'][name], right['model'][name], atol=0, rtol=0, msg=name)
            for name in left['runtime_state']:
                torch.testing.assert_close(left['runtime_state'][name], right['runtime_state'][name], atol=0, rtol=0, msg=name)
            self.assertEqual(left['counters']['best_step'], right['counters']['best_step'])
            self.assertEqual(left['counters']['best_nll'], right['counters']['best_nll'])
            live = json.loads((resumed/'live.json').read_text())
            self.assertEqual((live['status'], live['phase'], live['completed_tokens']), ('completed', 'completed', 24))
            self.assertTrue(np.isfinite(live['loss']) and live['tokens_per_second'] > 0)
            snapshots = sorted(resumed.glob('milestone_*_at_*.pt'))
            self.assertEqual(len(snapshots), 3)
            for path in snapshots:
                snapshot = torch.load(path, weights_only=True)
                self.assertGreaterEqual(snapshot['milestone']['actual_tokens'], snapshot['milestone']['nominal_tokens'])
                self.assertEqual(snapshot['milestone']['actual_tokens'], snapshot['counters']['completed_tokens'])
                self.assertIn('best_snapshot', snapshot)
            expected = [json.loads(line)['lr_scale'] for line in (full/'metrics.jsonl').read_text().splitlines()]
            actual = [json.loads(line)['lr_scale'] for line in (resumed/'metrics.jsonl').read_text().splitlines()]
            self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
