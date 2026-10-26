"""Small CPU validation fixtures; no test split or trained GPU model is read."""
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from v07.common import digest, file_hash
from .evaluate import (document_summary, feed_aligned, run_evaluation, score_segment,
                       select_windows, validation_corpus)
from .model import Config, Network, initialize


class OnlyValidation(dict):
    def __getitem__(self, name):
        if name != 'validation':
            raise AssertionError('evaluation must never access train or test arrays')
        return super().__getitem__(name)


def fixture(documents=4, length=160):
    tokens = np.concatenate([(np.arange(length)+doc*7)%29+3 for doc in range(documents)]).astype(np.int64)
    return SimpleNamespace(tokens=OnlyValidation(validation=tokens),
               offsets=OnlyValidation(validation=np.arange(documents+1)*length), manifest=dict(vocab_size=32, fixture=True))


class EvaluationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_plan_is_deterministic_nonoverlapping_and_exhausts_finitely(self):
        corpus = fixture()
        plan = select_windows(corpus, 3, history=16, recent=16, target=16, intrusion=16)
        same = select_windows(corpus, 3, history=16, recent=16, target=16, intrusion=16)
        self.assertEqual(plan, same)
        self.assertEqual(plan['coverage']['duplicate_scored_positions'], 0)
        for row in plan['windows']:
            self.assertNotEqual(row['document'], row['U0']['document'])
            self.assertNotEqual(row['document'], row['B']['document'])
            self.assertEqual(len(row['segment']), 17)
            self.assertEqual(row['target_start'], row['anchor']['position']+1)
            self.assertFalse(row['anchor']['scored'])
        short = select_windows(fixture(1, 48), 64, history=16, recent=16, target=16, intrusion=16)
        self.assertEqual(short['coverage']['status'], 'insufficient_windows')
        self.assertEqual(short['windows'], [])

    def test_only_validation_loader_never_opens_test_or_train(self):
        with TemporaryDirectory() as temp:
            root = Path(temp); corpus = fixture()
            (root/'manifest.json').write_text(json.dumps(corpus.manifest))
            np.save(root/'validation_tokens.npy', corpus.tokens['validation'])
            np.save(root/'validation_offsets.npy', corpus.offsets['validation'])
            loaded = validation_corpus(root)
            self.assertEqual(set(loaded.tokens), {'validation'})
            select_windows(loaded, 2, history=16, recent=16, target=16, intrusion=16)
            manifest = corpus.manifest | {'prepared_hashes': {name:file_hash(root/name) for name in
                              ('validation_tokens.npy', 'validation_offsets.npy')}}
            (root/'manifest.json').write_text(json.dumps(manifest))
            validation_corpus(root)
            with (root/'validation_tokens.npy').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'validation_tokens.npy'):
                validation_corpus(root)

    def test_anchor_shift_and_complete_blocks_are_causal(self):
        class Toy:
            cfg = SimpleNamespace(context_limit=64)
            def __init__(self):
                self.calls = []
            def __call__(self, values, state=None):
                self.calls.append(values.shape[1])
                logits = F.one_hot(values, 4).float()*2
                return logits, {}, {}
        model = Toy(); segment = np.arange(17)%4
        losses = score_segment(model, segment, {}, 'cpu')
        reference = F.cross_entropy(F.one_hot(torch.tensor(segment[:-1]), 4).float()*2,
                                    torch.tensor(segment[1:]), reduction='none').numpy()
        np.testing.assert_array_equal(losses, reference)
        self.assertEqual(model.calls, [16])
        changed = segment.copy(); changed[-1] = (changed[-1]+1)%4
        other = score_segment(model, changed, {}, 'cpu')
        np.testing.assert_array_equal(losses[:-1], other[:-1])
        with self.assertRaises(ValueError):
            feed_aligned(model, [1], 'cpu')

    def test_document_bootstrap_and_early_metrics_pool_every_window(self):
        rows = []
        for doc in (0, 1):
            for window in (doc*2, doc*2+1):
                base = [1. if window%2 == 0 else 3.]*16+[9.]*16
                rows.extend([dict(document=doc, window=window, condition='A0R', token_nll=base),
                             dict(document=doc, window=window, condition='U0R', token_nll=(np.asarray(base)+1).tolist())])
        summary = document_summary(rows, repetitions=100)
        self.assertEqual(summary['conditions']['A0R']['first_16_nll'], 2.)
        self.assertEqual(summary['conditions']['A0R']['first_16_tokens'], 64)
        self.assertEqual(summary['contrasts']['history_gain']['nll_difference'], 1.)
        self.assertEqual(summary['contrasts']['history_gain']['ci95'], [1., 1.])
        self.assertIn('not training-seed', summary['contrasts']['history_gain']['interpretation'])

    def test_checkpoint_evaluation_preserves_weights_logs_and_completed_output(self):
        cfg = Config(vocab_size=32, nodes=16, matrix_size=4, width=16, bands=2, modes=2,
                     block_size=16, group_size=2, credit_horizon=8, context_limit=64)
        trained = Network(cfg); initialize(trained, 11)
        corpus = fixture(); rebuilt = []

        def small_build(record, vocab, group_size, route_mode, context_limit):
            self.assertEqual(context_limit, 64)
            model = Network(cfg); rebuilt.append(model)
            return model, cfg

        payload = dict(format='v09-continuous-training-1', model=trained.state_dict(), counters=dict(completed_tokens=1234),
            protocol=dict(training=dict(condition='v09_dense', nodes=16, seed=11, group_size=2,
                          route_mode='summary', context_limit=64), model=asdict(cfg),
                          data_manifest_sha256=digest(corpus.manifest), source_hashes={'old.py':'abc'}))
        with TemporaryDirectory() as temp:
            checkpoint = Path(temp)/'last.pt'; output = Path(temp)/'evaluation'
            torch.save(payload, checkpoint); before = file_hash(checkpoint)
            with patch('v09.benchmark.build', side_effect=small_build):
                result = run_evaluation(checkpoint, output, 'cpu', windows=2, corpus=corpus,
                           history_tokens=16, recent_tokens=16, target_tokens=16, intrusion_tokens=16,
                           bootstrap_repetitions=100)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['coverage']['unique_scored_positions'], 32)
            raw = [json.loads(line) for line in (output/'raw.jsonl').read_text().splitlines()]
            self.assertEqual(len(raw), 16)
            self.assertTrue(all(not row['first_target_pre_intervention'] for row in raw))
            self.assertEqual(file_hash(checkpoint), before)
            for name, value in rebuilt[0].state_dict().items():
                torch.testing.assert_close(value, payload['model'][name], atol=0, rtol=0)
            self.assertTrue(all(parameter.grad is None for parameter in rebuilt[0].parameters()))
            live = json.loads((output/'live.json').read_text())
            self.assertEqual(live['completed_windows'], 2)
            self.assertEqual(live['progress'], 1.)
            self.assertEqual(live['completed'], 16)
            self.assertEqual(live['total'], 16)
            self.assertEqual(live['nll'], live['last_nll'])
            config = json.loads((output/'config.json').read_text())
            self.assertEqual(config['checkpoint_source_hashes'], {'old.py':'abc'})
            self.assertIn('v09/evaluate.py', config['evaluation_source_hashes'])
            with self.assertRaises(FileExistsError):
                run_evaluation(checkpoint, output, 'cpu', windows=2, corpus=corpus)

    def test_content_only_checkpoint_without_initial_state_evaluates_all_targets(self):
        from .baselines import Config as ContentConfig, ContentOnly, initialize as initialize_content
        cfg = ContentConfig(vocab_size=32, context_limit=64)
        model = ContentOnly(cfg); initialize_content(model, 11)
        corpus = fixture()
        payload = dict(model=model.state_dict(), protocol=dict(
            training=dict(condition='content_only', nodes=32, seed=11, group_size=16,
                          route_mode='summary', context_limit=64), model=asdict(cfg),
            data_manifest_sha256=digest(corpus.manifest), source_hashes={}))
        before_threads = torch.get_num_threads()
        with TemporaryDirectory() as temp:
            checkpoint = Path(temp)/'last.pt'; torch.save(payload, checkpoint)
            result = run_evaluation(checkpoint, Path(temp)/'evaluation', 'cpu', windows=1, corpus=corpus,
                           history_tokens=16, recent_tokens=16, target_tokens=16, intrusion_tokens=16,
                           bootstrap_repetitions=100)
            self.assertEqual(result['status'], 'completed')
            for name in ('credit_only', 'clear_slow', 'base_only'):
                self.assertEqual(result['applicability'][name], 'not_applicable')
            self.assertEqual(result['coverage']['unique_scored_positions'], 16)
            self.assertEqual(result['contrasts']['history_gain']['nll_difference'], 0.)
            raw = [json.loads(line) for line in (Path(temp)/'evaluation/raw.jsonl').read_text().splitlines()]
            self.assertEqual(len(raw), 5)
        self.assertEqual(torch.get_num_threads(), before_threads)


if __name__ == '__main__':
    unittest.main()
