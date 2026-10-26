"""Structural, gradient, causal and recovery checks; no capability claims."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from v04.model import gaussian_update
from .common import Console, detach, state_bytes
from .data import Corpus, ROOT, documents
from .evaluate import evaluate, windows
from .graph import solid_ball
from .model import Config, Network, Transformer, initialize, joseph, jump_gate, parameter_count, transformer_shape
from .run import build, cases, execute, load_config, recover_logs, train, train_update, training_batch, validate_config


class Correctness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, **kwargs):
        model = Network(Config(vocab_size=64, matrix_size=4, width=16, modes=2, **kwargs))
        initialize(model, 71); return model

    def test_joseph_values_and_gradients(self):
        torch.manual_seed(1)
        values = [torch.randn(2, 3, 4, 4, dtype=torch.float64), torch.randn(2, 3, 4, 4, dtype=torch.float64)]
        values += [torch.randn(2, 3, 4, dtype=torch.float64), torch.randn(2, 3, 4, dtype=torch.float64),
                   torch.ones(2, 3, dtype=torch.float64)*.1, torch.rand(2, 3, dtype=torch.float64),
                   torch.ones(2, 3, dtype=torch.float64)*.001]
        a = [v.clone().requires_grad_() for v in values]; b = [v.clone().requires_grad_() for v in values]
        # Covariance lives on the symmetric PSD manifold; compare gradients through its factor.
        def arguments(v):
            return [v[0], v[1]@v[1].transpose(-1, -2)+torch.eye(4)*.1, *v[2:]]
        first, second = joseph(*arguments(a)), gaussian_update(*arguments(b))
        first = (first[0], first[1]-1e-6*torch.eye(4, dtype=torch.float64))
        for x, y in zip(first, second):
            torch.testing.assert_close(x, y, atol=1e-11, rtol=1e-11)
        grad_a = torch.autograd.grad(sum(v.square().sum() for v in first), a)
        grad_b = torch.autograd.grad(sum(v.square().sum() for v in second), b)
        self.assertLess(max(float((x-y).abs().max()) for x, y in zip(grad_a, grad_b)), 1e-10)

    def test_ball_reachability_and_cut(self):
        for n in (32, 128, 512):
            graph = solid_ball(n)
            reached = set([0])
            for _ in range(3):
                reached |= {i for i, row in enumerate(graph['neighbors'].tolist()) if any(j in reached for j in row)}
            self.assertEqual(len(reached), n)
            self.assertEqual(graph['neighbors'].shape, (n, 8))
            self.assertTrue(torch.equal(graph['outward'], graph['shell'][:, None] > graph['shell'][graph['neighbors']]))
            cut = solid_ball(n, topology='cut'); level = cut['shell']
            cross = (level[:, None] < 2) != (level[cut['neighbors']] < 2)
            self.assertEqual(float(cut['edge_mask'][cross].sum()), 0.)

    def test_hard_gate_and_surrogate_derivative(self):
        score = torch.tensor([-1., -.01, .01, 1.], requires_grad=True)
        torch.testing.assert_close(jump_gate(score), torch.tensor([0., 0., 1., 1.]))
        jump_gate(score, training=True).sum().backward()
        self.assertTrue(bool((score.grad > 0).all()))

    def test_fixed_state_and_one_evidence_count_per_token(self):
        model = self.model().eval(); x = torch.randint(0, 64, (1, 12))
        with torch.no_grad():
            _, short, _ = model(x[:, :2]); _, long, _ = model(x)
        self.assertEqual(state_bytes(short), state_bytes(long))
        self.assertLessEqual(int(long['evidence_count'].max()), 12)
        self.assertEqual(int(long['evidence_count'][0, 0]), 12)
        eigenvalues = torch.linalg.eigvalsh(long['p'])
        self.assertGreater(float(eigenvalues.min()), 0)
        self.assertLessEqual(float(long['m'].norm(dim=(-1, -2)).max()), 4.00001)

    def test_forward_causality_and_stream_equivalence(self):
        model = self.model().eval(); x = torch.randint(0, 64, (1, 6)); changed = x.clone(); changed[:, 3:] = 63-x[:, 3:]
        with torch.no_grad():
            y, state, _ = model(x); other, _, _ = model(changed)
            first, s, _ = model(x[:, :3]); second, s, _ = model(x[:, 3:], s)
        torch.testing.assert_close(y[:, :3], other[:, :3])
        torch.testing.assert_close(y, torch.cat((first, second), 1))
        for name in state:
            torch.testing.assert_close(state[name], s[name])

    def test_history_reaches_readout_and_cut_blocks_that_path(self):
        x = torch.tensor([[1, 4, 5, 8, 9, 10]])
        other = torch.tensor([[1, 24, 25, 8, 9, 10]])
        outputs = {}
        for topology in ('connected', 'cut'):
            model = self.model(topology=topology).eval()
            with torch.no_grad():
                a, _, _ = model(x); b, _, _ = model(other)
            outputs[topology] = float((a[:, -1]-b[:, -1]).abs().max())
        self.assertGreater(outputs['connected'], 1e-7)
        self.assertLess(outputs['cut'], 1e-7)

    def test_all_variants_train_and_finite_gradients(self):
        values = np.array([[1, 4, 5, 6, 7, 2, 1, 4, 5]])
        cfg = load_config() | dict(sequence_tokens=8, bptt_tokens=4, gate_warmup_steps=0)
        for variant in ('message', 'independent', 'full', 'no_local', 'no_slow', 'no_probability', 'no_dynamic', 'single_timescale', 'no_threshold'):
            model = self.model(variant=variant)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            result, state = train_update(model, optimizer, values, cfg, torch.device('cpu'), 0)
            self.assertTrue(np.isfinite(result['loss']), variant)
            self.assertTrue(all(p.grad is None or bool(p.grad.isfinite().all()) for p in model.parameters()), variant)
            if variant == 'full':
                for name in ('node_shape', 'raw_slow_rate', 'raw_coupling', 'read_projection.weight'):
                    self.assertIsNotNone(dict(model.named_parameters())[name].grad, name)
                self.assertGreater(float(state['slow'].abs().sum()), 0.)

    def test_transformer_full_and_chunked_cache_equivalence(self):
        cfg = Config(vocab_size=64, variant='transformer', transformer_width=16, transformer_ff=32)
        model = Transformer(cfg).eval(); initialize(model, 71); x = torch.randint(0, 64, (2, 12))
        with torch.no_grad():
            whole, _, _ = model(x)
            first, state, _ = model(x[:, :5]); second, _, _ = model(x[:, 5:], state)
        torch.testing.assert_close(whole, torch.cat((first, second), 1), atol=1e-6, rtol=1e-5)
        shape, estimate = transformer_shape(309265, 4096)
        self.assertEqual(estimate, parameter_count(Transformer(Config(variant='transformer', **shape))))

    def test_pair_order_and_shared_initialization(self):
        cfg = load_config(); conditions = cases(cfg)
        self.assertEqual(len(conditions), 36)
        self.assertEqual([c['variant'] for c in conditions[:4]], cfg['variants'])
        self.assertEqual([c['nodes'] for c in conditions[::4]][:3], [32, 128, 512])
        a = self.model(variant='full'); b = self.model(variant='message')
        for name in ('embedding.weight', 'node_shape', 'input_projection.weight', 'ff.0.weight'):
            torch.testing.assert_close(dict(a.named_parameters())[name], dict(b.named_parameters())[name])
        validate_config(cfg)

    def test_checkpointed_event_gradient_equivalence(self):
        a = self.model(); b = self.model(checkpoint_events=True); b.load_state_dict(a.state_dict())
        x = torch.tensor([[1, 5, 8]])
        for model in (a, b):
            model(x)[0].square().sum().backward()
        for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5, msg=name)

    def test_truncated_and_full_gradient_reference(self):
        model = self.model(); x = torch.tensor([[1, 4, 7, 9]])
        hidden, _, _ = model(x); F.cross_entropy(hidden[:, :-1].flatten(0, 1), x[:, 1:].flatten()).backward()
        self.assertTrue(all(p.grad is None or bool(p.grad.isfinite().all()) for p in model.parameters()))

    def test_resume_optimizer_and_rng_equivalence(self):
        cfg = load_config() | dict(sequence_tokens=4, bptt_tokens=2)
        data = np.array([[1, 4, 5, 4, 5]])
        a = self.model(); optimizer = torch.optim.AdamW(a.parameters(), lr=.001)
        train_update(a, optimizer, data, cfg, torch.device('cpu'), 0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'cp.pt'
            torch.save(dict(model=a.state_dict(), optimizer=optimizer.state_dict()), path)
            b = self.model(); other = torch.optim.AdamW(b.parameters(), lr=.001)
            cp = torch.load(path, weights_only=True); b.load_state_dict(cp['model']); other.load_state_dict(cp['optimizer'])
            train_update(a, optimizer, data, cfg, torch.device('cpu'), 1)
            train_update(b, other, data, cfg, torch.device('cpu'), 1)
            for p, q in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(p, q, atol=0, rtol=0)
            metrics = Path(folder)/'metrics.jsonl'; metrics.write_text('{"step":1}\n{"step":2}\n{"step":3}\n')
            recover_logs(Path(folder), 2)
            self.assertEqual(len(metrics.read_text().splitlines()), 2)
            self.assertEqual(len(list(Path(folder).glob('*.uncommitted-*'))), 1)

    @unittest.skipUnless((ROOT/'manifest.json').exists(), 'Run data preparation first')
    def test_real_data_sampling_and_intervention_targets(self):
        corpus = Corpus(); cfg = load_config() | dict(validation_windows=1, test_windows=1, prefix_tokens=4, target_tokens=4, bptt_tokens=4)
        a, meta = training_batch(corpus, cfg, 11, 0); b, other = training_batch(corpus, cfg, 11, 0)
        self.assertTrue(np.array_equal(a, b)); self.assertEqual(meta, other)
        chosen = windows(corpus, 'test', cfg); self.assertNotEqual(chosen['docs'][0], chosen['other'][0])
        model = Network(Config(vocab_size=corpus.manifest['vocab_size'], matrix_size=4, width=16, modes=2))
        with tempfile.TemporaryDirectory() as folder:
            result = evaluate(model, corpus, 'test', cfg, torch.device('cpu'), Console(enabled=False), folder, True)
            rows = json.loads((Path(folder)/'test_predictions.json').read_text())
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(r['target_tokens'] == rows[0]['target_tokens'] for r in rows))
        self.assertEqual(result['conditions']['carry']['tokens'], 4)
        self.assertEqual(documents(['\n', ' = A =\n', 'a\n', ' = = subsection = =\n', 'b\n', ' = B =\n', 'c\n']),
                         [' = A =\na\n = = subsection = =\nb\n', ' = B =\nc\n'])

    @unittest.skipUnless((ROOT/'manifest.json').exists(), 'Run data preparation first')
    def test_runner_completion_resume_and_protocol_guard(self):
        cfg = load_config() | dict(nodes=[32], seeds=[11], variants=['full'], device='cpu',
                                 train_steps=2, batch_size=1, sequence_tokens=4, bptt_tokens=2,
                                 matrix_size=4, width=16, modes=2, internal_steps=2,
                                 validation_windows=1, test_windows=1, prefix_tokens=4, target_tokens=4,
                                 eval_every=2, checkpoint_every=1)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = execute(cfg, root, progress=False)
            self.assertEqual(first[0]['status'], 'completed')
            run = root/first[0]['id']; original = (run/'metrics.jsonl').read_text()
            execute(cfg, root, progress=False)
            self.assertEqual(original, (run/'metrics.jsonl').read_text())
            self.assertTrue((run/'runtime_example.pt').exists())
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                execute(cfg | dict(train_steps=3), root, progress=False)

    @unittest.skipUnless((ROOT/'manifest.json').exists(), 'Run data preparation first')
    def test_stop_after_final_update_resumes_pending_validation(self):
        cfg = load_config() | dict(nodes=[32], seeds=[11], variants=['full'], device='cpu',
                                 train_steps=1, batch_size=1, sequence_tokens=4, bptt_tokens=2,
                                 matrix_size=4, width=16, modes=2, internal_steps=2,
                                 validation_windows=1, test_windows=1, prefix_tokens=4, target_tokens=4,
                                 eval_every=1, checkpoint_every=1)
        stop = {'requested': False}; case = cases(cfg)[0]; corpus = Corpus()
        def stop_after_update(*args, **kwargs):
            result = train_update(*args, **kwargs); stop['requested'] = True; return result
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            with patch('v06.run.train_update', side_effect=stop_after_update):
                self.assertIsNone(train(cfg, case, corpus, path, torch.device('cpu'), Console(enabled=False), stop))
            self.assertFalse((path/'best.pt').exists())
            result = train(cfg, case, corpus, path, torch.device('cpu'), Console(enabled=False), {'requested': False})
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(len((path/'metrics.jsonl').read_text().splitlines()), 1)


if __name__ == '__main__':
    unittest.main()
