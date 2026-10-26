import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from .data import INPUT_FIELDS, PROFILES, bank, collate, inputs, quartet, training_batch, validate
from .model import Config, Network, matchings, pair_rotate
from .run import cases, model_config, train, train_update

PROJECT = Path(__file__).resolve().parents[1]


class Correctness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg = json.loads((PROJECT/'configs/v05/base.json').read_text())

    def model(self, **kw):
        torch.manual_seed(21)
        return Network(Config(internal_steps=2, checkpoint_events=False, **kw))

    def batch(self, profile='consolidation'):
        return collate(training_batch(np.random.default_rng(81), profile=profile))

    def test_data_balance_and_no_hidden_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            result = validate(folder, self.cfg | {'development_quartets': 1, 'test_quartets': 1})
            self.assertEqual(result['question_only_accuracy_exact'], .25)
        for profile in PROFILES:
            rows = quartet(np.random.default_rng(31), profile)
            batch = collate(rows)
            for t in batch['query'][0].nonzero().flatten():
                self.assertEqual(sorted(batch['target'][:, t].tolist()), [0, 1, 2, 3])
            self.assertFalse({'target', 'world', 'stage', 'feedback', 'truth', 'query'} & set(INPUT_FIELDS))

    def test_matchings_cover_every_edge_once(self):
        for n in (32, 128, 512):
            model = self.model(nodes=n)
            pairs = matchings(model.neighbors)
            all_edges = []
            for group in pairs:
                self.assertEqual(len(group.flatten()), len(set(group.flatten().tolist())))
                all_edges += [tuple(x) for x in group.tolist()]
            expected = {tuple(sorted((i, j))) for i, row in enumerate(model.neighbors.tolist()) for j in row}
            self.assertEqual(set(all_edges), expected)
            self.assertEqual(len(all_edges), len(expected))

    def test_unverified_claim_cannot_reveal_truth_or_require_guessing(self):
        rows = quartet(np.random.default_rng(9), 'repetition')
        batch = collate(rows)
        end = max(i for i, e in enumerate(rows[0]['events']) if e['stage'] == 'repeated_claim')+1
        for name, value in inputs(batch, 'cpu').items():
            torch.testing.assert_close(value[:, :end], value[:1, :end].expand_as(value[:, :end]), msg=name)
        self.assertFalse(bool(batch['query'][:, :end].any()))
        self.assertTrue(bool(batch['request'][:, :end].any()))
        self.assertEqual(sorted(batch['target'][:, -1].tolist()), [0, 1, 2, 3])

    def test_outlier_counterfactual_changes_history_not_current_claim(self):
        rows = quartet(np.random.default_rng(13), 'outlier')
        for row in rows[1:]:
            self.assertEqual(rows[0]['events'][-2]['value'], row['events'][-2]['value'])
        self.assertEqual(sorted(r['events'][-1]['target'] for r in rows), [0, 1, 2, 3])

    def test_blind_delay_and_long_split(self):
        dev, test = bank('dev', 2), bank('test', 2)
        self.assertNotIn('long', {r['profile'] for r in dev})
        self.assertIn('long', {r['profile'] for r in test})
        def lags(rows):
            result = set()
            for row in rows:
                if row['profile'] != 'delayed':
                    continue
                last_query = None
                for t, e in enumerate(row['events']):
                    if e['request']:
                        last_query = t
                    if e['stage'] == 'delayed_feedback':
                        result.add(t-last_query-1)
            return result
        self.assertTrue(lags(dev) <= {2, 4, 8})
        self.assertTrue(lags(test) <= {16, 32})
        self.assertFalse(lags(dev) & lags(test))

    def test_independent_reference_detects_wrong_scoring_target(self):
        from .audit_data import reference_check
        rows = quartet(np.random.default_rng(21), 'history')
        self.assertEqual(reference_check(rows)['errors'], 0)
        row = rows[0] | {'events': [dict(e) for e in rows[0]['events']]}
        row['events'][-1]['target'] = (row['events'][-1]['target']+1) % 4
        self.assertEqual(reference_check([row])['errors'], 1)

    def test_pair_rotation_norm_and_nonzero_transmission(self):
        x = torch.randn(2, 3, 8, 2, 4, 4, dtype=torch.float64, requires_grad=True)
        pairs = torch.tensor([[0, 7], [1, 6], [2, 5]])
        angle = torch.randn(2, 3, 3, dtype=torch.float64, requires_grad=True)
        out = pair_rotate(x, pairs, angle)
        torch.testing.assert_close(out.square().sum(), x.square().sum(), atol=1e-10, rtol=1e-12)
        out[:, :, 7].sum().backward()
        self.assertGreater(float(x.grad[:, :, 0].norm()), 0)
        self.assertGreater(float(angle.grad.norm()), 0)

    def test_complete_carrier_substep_norm_and_damping(self):
        model = self.model(); state = model.initial_state(2)
        state['x'] = torch.randn_like(state['x'])
        norm = state['x'].square().sum()
        with torch.no_grad():
            for index in range(model.matching_count):
                changed = model.tick(state, index, damping=False)
                torch.testing.assert_close(changed['x'].square().sum(), norm, rtol=2e-6, atol=.01)
            changed = model.tick(state, 0)
            self.assertLess(float(changed['x'].square().sum()), float(norm))

    def test_forward_backward_all_variants(self):
        batch = self.batch()
        for variant in self.cfg['variants']:
            model = self.model(variant=variant)
            p, state, details = model(**inputs(batch, 'cpu'))
            loss = F.nll_loss(p[batch['query']], batch['target'][batch['query']])+.05*details['predictive'].mean()
            loss.backward()
            self.assertTrue(bool(torch.isfinite(p).all()), variant)
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    self.assertTrue(bool(torch.isfinite(parameter.grad).all()), (variant, name))
            if variant in ('full', 'message'):
                self.assertGreater(float(model.coefficient.weight.grad.norm()), 0.)
            if model.fourier:
                torch.testing.assert_close(state['z'].square().sum(-1), torch.ones_like(state['z'][..., 0]), rtol=1e-5, atol=1e-5)

    def test_causality_chunk_equivalence_and_batch_isolation(self):
        model = self.model(); batch = self.batch(); x = inputs(batch, 'cpu')
        with torch.no_grad():
            whole, end, _ = model(**x)
            first, state, _ = model(**{k: v[:, :7] for k, v in x.items()})
            second, split, _ = model(**{k: v[:, 7:] for k, v in x.items()}, state=state)
            torch.testing.assert_close(whole, torch.cat((first, second), 1))
            for key in end:
                torch.testing.assert_close(end[key], split[key])
            changed = {k: v.clone() for k, v in x.items()}; changed['value'][:, 8:] += 3
            future, _, _ = model(**changed)
            torch.testing.assert_close(whole[:, :8], future[:, :8])
            alone, _, _ = model(**{k: v[:1] for k, v in x.items()})
            torch.testing.assert_close(whole[:1], alone, atol=2e-6, rtol=1e-5)

    def test_router_cannot_broadcast_answer_values(self):
        model = self.model(); batch = self.batch('history'); x = inputs(batch, 'cpu')
        altered = {k: v.clone() for k, v in x.items()}; altered['value'] = altered['value'].roll(1, -1)
        with torch.no_grad():
            _, a, da = model(**x); _, b, db = model(**altered)
        for name in ('cue', 'mass', 'rho'):
            torch.testing.assert_close(a[name], b[name])
        torch.testing.assert_close(da['routing'], db['routing'])

    def test_duplicate_evidence_does_not_consolidate(self):
        model = self.model(); batch = self.batch('history'); x = inputs(batch, 'cpu')
        with torch.no_grad():
            _, state, _ = model(**{k: v[:, :6] for k, v in x.items()})
            duplicate = {k: v[:, 4:5].clone() for k, v in x.items()}
            _, end, details = model(**duplicate, state=state)
        self.assertFalse(bool(details['fresh'].any()))
        for name in ('m', 'p', 'slow', 'support', 'protect'):
            torch.testing.assert_close(state[name], end[name])

    def test_short_reset_preserves_weights_and_their_function(self):
        model = self.model(); batch = self.batch(); x = inputs(batch, 'cpu')
        with torch.no_grad():
            _, state, _ = model(**{k: v[:, :18] for k, v in x.items()})
            clean = model.reset_fast(state, torch.ones(4, dtype=torch.bool))
            torch.testing.assert_close(clean['slow'], state['slow'])
            self.assertEqual(float(clean['m'].norm()), 0.)
            self.assertEqual(float(clean['x'].norm()), 0.)
            self.assertEqual(float(clean['trace'].norm()), 0.)
            self.assertGreater(float(clean['slow'].norm()), 0.)
            qa, qb = batch['query_a'][:, 0], batch['query_b'][:, 0]
            a = model.read(clean, qa, qb, torch.zeros(4, dtype=torch.bool))
            b = model.read(clean, qa, qb, torch.zeros(4, dtype=torch.bool), 'clear_slow')
            self.assertGreater(float((a-b).abs().max()), 1e-7)

    def test_delayed_credit_uses_trace_not_future_label(self):
        batch = self.batch('delayed'); norms = {}
        for variant in ('full', 'no_traces', 'no_consolidation'):
            with torch.no_grad():
                _, state, _ = self.model(variant=variant)(**inputs(batch, 'cpu'))
            norms[variant] = float(state['slow'].norm())
        self.assertGreater(norms['full'], 0.)
        self.assertEqual(norms['no_traces'], 0.)
        self.assertEqual(norms['no_consolidation'], 0.)
        with torch.no_grad():
            _, state, _ = self.model()(**inputs(batch, 'cpu'), intervention='clear_slow')
        self.assertEqual(float(state['slow'].norm()), 0.)

    def test_conflicts_reopen_protection_and_reduce_applicability(self):
        model = self.model(); batch = self.batch('history'); x = inputs(batch, 'cpu')
        with torch.no_grad():
            _, state, _ = model(**{k: v[:, :6] for k, v in x.items()})
            state['protect'].fill_(10); state['app'].fill_(1.)
            event = {k: v[:, 4:5].clone() for k, v in x.items()}
            event['evidence_id'] += 100000000; event['value'] += 3.
            _, changed, _ = model(**event, state=state)
        self.assertLess(float(changed['protect'].sum()), float(state['protect'].sum()))
        self.assertLess(float(changed['app'].sum()), float(state['app'].sum()))

    def test_ports_cut_budget_and_exact_capacity(self):
        cfg = self.cfg | dict(stage='capacity', seeds=[11])
        totals = []
        for case in cases(cfg):
            model = Network(model_config(cfg, case)); ledger = model.ledger(); totals.append(ledger['state_floats_per_stream'])
            self.assertFalse(bool((model.write_ports*model.read_ports).any()))
            full = Network(replace(model.cfg, variant='full'))
            self.assertEqual(ledger['state_floats_per_stream'], full.ledger()['state_floats_per_stream'])
            self.assertLess(abs(ledger['parameters']-full.ledger()['parameters'])/full.ledger()['parameters'], .05)
        self.assertEqual(len(set(totals)), 1)
        model = self.model(read_path='separated', topology='cut')
        self.assertFalse(bool((model.neighbors[:16] >= 16).any()))
        with torch.no_grad():
            _, state, _ = model(**inputs(self.batch('history'), 'cpu'))
        self.assertEqual(float(state['x'][:, :, 16:].norm()), 0.)
        self.assertEqual(float(state['slow'][:, :, 16:].norm()), 0.)

    def test_padding_and_single_observation_not_replayed_by_ticks(self):
        batch = self.batch(); x = {k: v[:, :1] for k, v in inputs(batch, 'cpu').items()}
        model = self.model(); other = Network(replace(model.cfg, internal_steps=6)); other.load_state_dict(model.state_dict())
        with torch.no_grad():
            _, state, _ = model(**x); _, longer, _ = other(**x)
            for name in ('m', 'p', 'slow', 'support'):
                torch.testing.assert_close(state[name], longer[name])
            empty = {k: v.clone() for k, v in x.items()}; empty['valid'].zero_()
            _, padded, _ = model(**empty, state=state)
            for name in state:
                torch.testing.assert_close(state[name], padded[name])

    def test_checkpoint_gradients_and_full_credit(self):
        batch = self.batch('history'); a = self.model(); b = Network(replace(a.cfg, checkpoint_events=True))
        b.load_state_dict(a.state_dict())
        for model in (a, b):
            p, _, _ = model(**inputs(batch, 'cpu'))
            F.nll_loss(p[batch['query']], batch['target'][batch['query']]).backward()
        for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=1e-5, msg=name)
        x = inputs(batch, 'cpu'); x['value'] = x['value'].clone().requires_grad_(True)
        p, _, _ = a(**x); (-p[:, -1, 0].sum()).backward()
        self.assertGreater(float(x['value'].grad[:, :2].norm()), 0.)

    def test_pycharm_entry_uses_absolute_project_paths(self):
        entry = PROJECT/'run_biomatrix_v05.py'; previous = Path.cwd()
        with tempfile.TemporaryDirectory() as folder, patch.object(sys, 'argv', [str(entry)]), patch('v05.run.execute') as execute:
            try:
                os.chdir(folder); scope = runpy.run_path(str(entry)); scope['main']()
            finally:
                os.chdir(previous)
        cfg, root = execute.call_args.args
        self.assertEqual(cfg['nodes'], [32, 128, 512]); self.assertTrue(root.is_absolute())
        self.assertEqual(cfg['credit'], 'full')

    def test_full_credit_reaches_early_input_and_truncation_does_not(self):
        norms = {}
        for credit in ('full', 'truncated'):
            model = self.model(); batch = self.batch('history')
            batch['query'].zero_(); batch['query'][:, -1] = True
            batch['value'].requires_grad_(True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            train_update(model, optimizer, batch, self.cfg | {'credit': credit, 'bptt_steps': 8, 'predictive_loss_weight': 0}, torch.device('cpu'))
            norms[credit] = float(batch['value'].grad[:, :2].norm())
        self.assertGreater(norms['full'], 0.)
        self.assertEqual(norms['truncated'], 0.)

    def test_cpu_resume_replays_exact_weights_and_data(self):
        cfg = self.cfg | dict(stage='core', variants=['full'], nodes=[32], seeds=[31], read_paths=['shared'],
                              train_steps=2, internal_steps=1, checkpoint_events=False, eval_every=2,
                              checkpoint_every=1, development_quartets=1, test_quartets=1)
        case = cases(cfg)[0]
        original = train_update
        small_bank = lambda split, count, keys: bank(split, 1, keys, ('history',))
        with tempfile.TemporaryDirectory() as folder, patch('v05.run.bank', small_bank):
            interrupted, continuous = Path(folder)/'interrupted', Path(folder)/'continuous'; stop = [False]
            def update(*args, **kwargs):
                metrics = original(*args, **kwargs); stop[0] = True; return metrics
            with patch('v05.run.train_update', update):
                self.assertIsNone(train(case, cfg, interrupted, torch.device('cpu'), stop, False))
            train(case, cfg, interrupted, torch.device('cpu'), [False], False)
            train(case, cfg, continuous, torch.device('cpu'), [False], False)
            a = torch.load(interrupted/case['id']/'last.pt', weights_only=True)
            b = torch.load(continuous/case['id']/'last.pt', weights_only=True)
            self.assertEqual(a['batches'], b['batches'])
            for name in a['model']:
                torch.testing.assert_close(a['model'][name], b['model'][name], atol=0, rtol=0)
            before = (interrupted/case['id']/'last.pt').read_bytes()
            with self.assertRaises(ValueError):
                train(case, cfg | {'learning_rate': .002}, interrupted, torch.device('cpu'), [False], False)
            self.assertEqual(before, (interrupted/case['id']/'last.pt').read_bytes())


if __name__ == '__main__':
    unittest.main()
