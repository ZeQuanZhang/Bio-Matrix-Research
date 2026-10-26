"""Implementation invariants and failure modes, not tests of scientific success."""
from dataclasses import replace
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
import numpy as np
import torch
from torch.nn import functional as F
from .data import bank, collate, inputs, quartet, training_batch, validate, PROFILES
from .model import Config, Network, gaussian_update
from .run import RunLock, model_config, train_update
from unittest.mock import patch


class Correctness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg = json.loads((Path(__file__).resolve().parents[1]/'configs/v04/base.json').read_text())

    def network(self, variant='full', **kw):
        torch.manual_seed(21)
        return Network(Config(variant=variant, internal_steps=2, **kw))

    def batch(self, profile='return'):
        return collate(training_batch(np.random.default_rng(81), profile=profile))

    def test_data_balance_and_hidden_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            result = validate(folder, self.cfg | {'development_quartets': 1, 'test_quartets': 1})
            self.assertEqual(result['question_only_accuracy_exact'], .25)
        for profile in PROFILES:
            rows = quartet(np.random.default_rng(31), profile)
            batch = collate(rows)
            self.assertNotIn('world', inputs(batch, 'cpu'))
            self.assertNotIn('target', inputs(batch, 'cpu'))
            self.assertEqual(set(batch['target'][batch['query']].tolist()), {0, 1, 2, 3})
        independent = quartet(np.random.default_rng(19), 'independent')[0]
        self.assertFalse(independent['events'][-1]['combine'])
        self.assertTrue(independent['events'][-1]['observe'])

    def test_pycharm_entry_without_arguments_or_project_cwd(self):
        entry = Path(__file__).resolve().parents[1]/'run_biomatrix_v04.py'
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as folder, patch.object(sys, 'argv', [str(entry)]), patch('v04.run.execute') as execute:
            try:
                os.chdir(folder)
                namespace = runpy.run_path(str(entry))
                namespace['main']()
            finally:
                os.chdir(previous)
        config, root = execute.call_args.args
        self.assertEqual(config['nodes'], [32, 128, 512])
        self.assertEqual(config['read_paths'], ['shared', 'separated'])
        self.assertTrue(root.is_absolute())

    def test_forward_backward_all_variants(self):
        batch = self.batch()
        for variant in ('graph', 'full', 'matrix', 'probability', 'fourier'):
            model = self.network(variant)
            lp, state, details = model(**inputs(batch, 'cpu'))
            loss = F.nll_loss(lp[batch['query']], batch['target'][batch['query']])+.05*details['predictive'].mean()
            loss.backward()
            self.assertTrue(bool(torch.isfinite(lp).all()), variant)
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    self.assertTrue(bool(torch.isfinite(parameter.grad).all()), (variant, name))
            if model.fourier:
                torch.testing.assert_close(state['z'].square().sum(-1), torch.ones_like(state['z'][..., 0]), atol=1e-5, rtol=0)

    def test_chunk_equivalence_causality_and_isolation(self):
        model = self.network(); batch = self.batch('history'); x = inputs(batch, 'cpu')
        with torch.no_grad():
            whole, end, _ = model(**x)
            a, state, _ = model(**{k: v[:, :5] for k, v in x.items()})
            b, split_end, _ = model(**{k: v[:, 5:] for k, v in x.items()}, state=state)
            torch.testing.assert_close(whole, torch.cat((a, b), 1))
            for k in end:
                torch.testing.assert_close(end[k], split_end[k])
            changed = {k: v.clone() for k, v in x.items()}; changed['value'][:, 6:] += 3
            future, _, _ = model(**changed)
            torch.testing.assert_close(whole[:, :6], future[:, :6])
            alone, _, _ = model(**{k: v[:1] for k, v in x.items()})
            torch.testing.assert_close(whole[:1], alone, atol=2e-6, rtol=1e-5)

    def test_joseph_reference_and_positive_covariance(self):
        torch.manual_seed(4); d = 4
        m = torch.randn(3, d, d, dtype=torch.float64)
        a = torch.randn(3, d, d, dtype=torch.float64); p = a@a.transpose(-1, -2)+torch.eye(d)
        k = torch.randn(3, d, dtype=torch.float64); v = torch.randn(3, d, dtype=torch.float64)
        r = torch.tensor([.1, .5, 1.], dtype=torch.float64)
        updated, posterior = gaussian_update(m, p, k, v, r, torch.ones(3), torch.zeros(3))
        gain = (p@k[..., None]).squeeze(-1)/((k[..., None, :]@p@k[..., None]).flatten()+r)[:, None]
        expected = m+(v-(m@k[..., None]).squeeze(-1))[..., :, None]*gain[..., None, :]
        torch.testing.assert_close(updated, expected)
        self.assertTrue(bool((torch.linalg.eigvalsh(posterior) > 0).all()))

    def test_no_double_counting_or_internal_evidence_replay(self):
        model = self.network(); batch = self.batch('duplicate'); x = inputs(batch, 'cpu')
        event = {k: v[:, :1] for k, v in x.items()}
        with torch.no_grad():
            _, state, _ = model(**event)
            _, repeated, details = model(**event, state=state)
            self.assertFalse(bool(details['fresh'].any()))
            torch.testing.assert_close(state['m'], repeated['m'])
            torch.testing.assert_close(state['p'], repeated['p'])
            other = Network(replace(model.cfg, internal_steps=6)); other.load_state_dict(model.state_dict())
            _, extra_ticks, _ = other(**event)
            torch.testing.assert_close(state['m'], extra_ticks['m'])
            torch.testing.assert_close(state['p'], extra_ticks['p'])

    def test_padding_and_node_state_budget(self):
        batch = self.batch('history'); model = self.network(); x = inputs(batch, 'cpu')
        with torch.no_grad():
            _, state, _ = model(**x)
            empty = {k: v[:, :1].clone() for k, v in x.items()}; empty['valid'].zero_()
            _, padded, _ = model(**empty, state=state)
            for k in state:
                torch.testing.assert_close(state[k], padded[k])
        for n in (32, 128, 512):
            full = self.network(nodes=n)
            baseline_cfg = model_config(self.cfg | {'internal_steps': 2}, {'variant': 'graph', 'nodes': n})
            baseline = Network(baseline_cfg)
            self.assertEqual(full.ledger()['state_floats_per_stream'], baseline.ledger()['state_floats_per_stream'])
            a, b = full.ledger()['parameters'], baseline.ledger()['parameters']
            self.assertLess(abs(a-b)/a, .05)
            torch.testing.assert_close(full.neighbors, baseline.neighbors)
            torch.testing.assert_close(full.write_ports, baseline.write_ports)
            torch.testing.assert_close(full.read_ports, baseline.read_ports)
            with torch.no_grad():
                lp, _, _ = full(**{k: v[:, :1] for k, v in x.items()})
            self.assertTrue(bool(torch.isfinite(lp).all()))

    def test_separated_ports_and_graph_cut(self):
        model = self.network(read_path='separated', topology='cut')
        self.assertFalse(bool((model.write_ports*model.read_ports).any()))
        self.assertTrue(bool((model.write_ports[:, 16:] == 0).all()))
        self.assertTrue(bool((model.read_ports[:, :16] == 0).all()))
        self.assertTrue(bool((model.neighbors[:16] < 16).all()))
        with torch.no_grad():
            model.edge_query.weight.zero_()
            feature = model.features(model.initial_state(1))
            count = (model.edge_weights(feature) > 0).sum(-1)
            self.assertTrue(bool((count == model.cfg.active_neighbors).all()))

    def test_training_and_run_lock(self):
        model = self.network(); optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        metrics = train_update(model, optimizer, self.batch('partial'), self.cfg, torch.device('cpu'))
        self.assertTrue(np.isfinite(metrics['nll']))
        with tempfile.TemporaryDirectory() as folder:
            with RunLock(Path(folder)):
                with self.assertRaises(RuntimeError):
                    with RunLock(Path(folder)):
                        pass
            self.assertFalse((Path(folder)/'.trainer.lock').exists())

    def test_activation_checkpoint_gradients(self):
        plain = self.network(); saved = Network(replace(plain.cfg, checkpoint_ticks=True))
        saved.load_state_dict(plain.state_dict()); batch = self.batch('history')
        for model in (plain, saved):
            p, _, _ = model(**inputs(batch, 'cpu'))
            F.nll_loss(p[batch['query']], batch['target'][batch['query']]).backward()
        for (name, a), (_, b) in zip(plain.named_parameters(), saved.named_parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=1e-5, msg=name)

    def test_atomic_resume_replays_data_and_weights(self):
        from .run import train
        cfg = self.cfg | dict(train_steps=2, internal_steps=1, checkpoint_ticks=False,
                              eval_every=2, checkpoint_every=1, development_quartets=1, test_quartets=1)
        case = dict(variant='full', nodes=32, seed=31, id='resume_test')
        original_bank = bank
        small_bank = lambda split, count, keys: original_bank(split, 1, keys, profiles=('history',))
        with tempfile.TemporaryDirectory() as folder, patch('v04.run.bank', small_bank):
            root = Path(folder); interrupted, continuous = root/'interrupted', root/'continuous'
            flag = [False]
            original_update = train_update
            def stop_after_update(*args, **kwargs):
                value = original_update(*args, **kwargs); flag[0] = True
                return value
            with patch('v04.run.train_update', stop_after_update):
                self.assertIsNone(train(case, cfg, interrupted, torch.device('cpu'), flag, progress=False))
            status = json.loads((interrupted/'conditions.json').read_text())['resume_test']
            self.assertEqual(status['status'], 'aborted'); self.assertEqual(status['safe_checkpoint_step'], 1)
            train(case, cfg, interrupted, torch.device('cpu'), [False], progress=False)
            train(case, cfg, continuous, torch.device('cpu'), [False], progress=False)
            a = torch.load(interrupted/'resume_test/last.pt', weights_only=True)
            b = torch.load(continuous/'resume_test/last.pt', weights_only=True)
            self.assertEqual(a['batches'], b['batches'])
            for name in a['model']:
                torch.testing.assert_close(a['model'][name], b['model'][name], atol=0, rtol=0)
            # A source/protocol mismatch must preserve the previous run.
            before = (interrupted/'resume_test/last.pt').read_bytes()
            with self.assertRaises(ValueError):
                train(case, cfg | {'learning_rate': .002}, interrupted, torch.device('cpu'), [False], progress=False)
            self.assertEqual(before, (interrupted/'resume_test/last.pt').read_bytes())


if __name__ == '__main__':
    unittest.main()
