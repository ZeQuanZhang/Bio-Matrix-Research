"""Small CPU checks for paired initialization, task credit and safe restoration."""
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from v07.model import Config as LegacyConfig, Transformer
from .model import Config, Network
from .train import (DEFAULTS, build_models, build_teacher, evaluate, forward_segments,
                    objective, optimizer_for, restore_checkpoint, save_checkpoint,
                    settings, source_hashes, train, update)


def fixture():
    torch.manual_seed(18)
    old = LegacyConfig(vocab_size=32, transformer_width=16, transformer_layers=1,
                       transformer_ff=32, transformer_heads=2, context_limit=32)
    backbone = Transformer(old)
    payload = dict(protocol=dict(model=asdict(old), fixture=True), model=backbone.state_dict())
    config = DEFAULTS | dict(batch=1, segments=2, segment_tokens=4, context_limit=8,
                              memory_heads=2, memory_dim=4, bands=2, nodes=4,
                              internal_steps=1, full_dynamics=False, tokens=16,
                              validation_windows=2, threads=1)
    return payload, config


class TrainingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_parameter_matched_adapter_and_shared_phase_ablation(self):
        payload, config = fixture()
        models, cfg, counts = build_models(payload, config)
        self.assertEqual(sum(p.numel() for p in models['transformer'].parameters() if p.requires_grad), 0)
        self.assertLessEqual(abs(counts['adapter']['trainable_parameter_difference']), 2*cfg.width*cfg.layers)
        full, ablation = models['bio_full'].state_dict(), models['bio_no_phase'].state_dict()
        for name, value in ablation.items():
            if name in full and full[name].shape == value.shape:
                torch.testing.assert_close(value, full[name], atol=0, rtol=0, msg=name)
        tokens = torch.tensor([[1, 4, 7, 2]])
        baseline, _, _ = models['transformer'](tokens)
        adapter, _, _ = models['adapter'](tokens)
        torch.testing.assert_close(baseline, adapter, atol=0, rtol=0)

    def test_future_suffix_loss_trains_earlier_write_without_detaching(self):
        payload, config = fixture()
        model = build_models(payload, config)[0]['bio_full']
        first, state, _ = model(torch.tensor([[1, 5, 6, 7]]))
        second, _, _ = model(torch.tensor([[8, 9, 10, 11]]), state)
        F.cross_entropy(second.flatten(0, 1), torch.tensor([9, 10, 11, 12])).backward()
        gradient = model.memories[0].write_gate.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for name, p in model.named_parameters() if not name.startswith('memories.')))

    def test_objective_masks_tail_and_kd_uses_causal_teacher_targets(self):
        _, config = fixture()
        logits = torch.randn(1, 3, 6, requires_grad=True)
        teacher = torch.randn_like(logits)
        target = torch.tensor([[1, 2, 3]])
        mask = torch.tensor([[True, False, False]])
        loss, ce, kd = objective(logits, target, teacher, mask, config)
        expected = F.cross_entropy(logits[:, 0], target[:, 0])
        torch.testing.assert_close(ce, expected)
        self.assertGreaterEqual(float(kd.detach()), -1e-6)
        loss.backward()
        self.assertEqual(float(logits.grad[:, 1:].abs().sum()), 0.)

    def test_one_multisegment_update_is_exact_budget_and_keeps_backbone_frozen(self):
        payload, config = fixture()
        model = build_models(payload, config)[0]['bio_full']
        teacher = build_teacher(payload, torch.device('cpu'))
        optimizer = optimizer_for(model, config)
        before = {name: value.clone() for name, value in model.state_dict().items() if not name.startswith('memories.')}
        values = np.asarray([[1, 4, 5, 6, 7, 8, 9, 10, 11]])
        row, state = update(model, teacher, optimizer, values, config, torch.device('cpu'), remaining=5)
        self.assertEqual(row['targets'], 5)
        self.assertEqual(row['visible_inputs'], 8)
        self.assertTrue(np.isfinite(row['loss']))
        self.assertFalse(state['memory'][0].requires_grad)
        for name, value in before.items():
            torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0, msg=name)

    def test_checkpoint_restores_sampling_rng_optimizer_and_rejects_protocol_changes(self):
        payload, config = fixture()
        models, _, _ = build_models(payload, config)
        model = models['bio_full']
        teacher = build_teacher(payload, torch.device('cpu'))
        optimizer = optimizer_for(model, config)
        rng = np.random.default_rng(991)
        values = np.asarray([[1, 4, 5, 6, 7, 8, 9, 10, 11]])
        _, state = update(model, teacher, optimizer, values, config, torch.device('cpu'), 8)
        protocol = dict(source_hashes=source_hashes(), fixture=True)
        counters = dict(updates=1, completed_tokens=8, consolidations=0)
        with TemporaryDirectory() as folder:
            path = Path(folder)/'last.pt'
            save_checkpoint(path, model, optimizer, state, counters, rng, protocol, torch.device('cpu'))
            expected = rng.integers(10000, size=8)
            recovered = build_models(payload, config)[0]['bio_full']
            other_optimizer = optimizer_for(recovered, config)
            other_rng = np.random.default_rng(1)
            actual = restore_checkpoint(path, recovered, other_optimizer, other_rng, protocol, torch.device('cpu'))
            self.assertEqual(actual, counters)
            np.testing.assert_array_equal(other_rng.integers(10000, size=8), expected)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(recovered.state_dict()[name], value, atol=0, rtol=0, msg=name)
            self.assertEqual(len(other_optimizer.state), len(optimizer.state))
            with self.assertRaises(ValueError):
                restore_checkpoint(path, recovered, other_optimizer, other_rng,
                                   protocol | dict(changed=True), torch.device('cpu'))

    def test_history_evaluation_is_paired_and_leaves_parameters_fixed(self):
        payload, config = fixture()
        model = build_models(payload, config)[0]['bio_full']
        before = {name: value.clone() for name, value in model.state_dict().items()}
        values = np.asarray([[1, 4, 5, 6, 7, 8, 9, 10, 11], [1, 12, 13, 14, 15, 16, 17, 18, 19]])
        result = evaluate(model, values, config, torch.device('cpu'))
        self.assertTrue(np.isfinite(result['warm_nll']))
        self.assertTrue(np.isfinite(result['cold_nll']))
        self.assertEqual(result['conditions']['unrelated_history']['status'], 'not_applicable')
        for name, value in before.items():
            torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0, msg=name)

    def test_settings_and_overwrite_guards(self):
        result = settings(SimpleNamespace())
        self.assertTrue(result['full_dynamics'])
        self.assertEqual(result['segments'], 8)
        with self.assertRaises(ValueError):
            settings(SimpleNamespace(segments=1))
        with self.assertRaises(ValueError):
            settings(SimpleNamespace(tokens=result['tokens']+1), result)
        with TemporaryDirectory() as folder:
            (Path(folder)/'kept.txt').write_text('keep')
            with self.assertRaises(FileExistsError):
                train(SimpleNamespace(device='cpu', output=folder), verify_first=False)
            self.assertEqual((Path(folder)/'kept.txt').read_text(), 'keep')

    def test_fixture_pilot_records_reference_and_three_real_adaptations_then_resumes(self):
        payload, config = fixture()
        class TinyCorpus:
            manifest = dict(vocab_size=32, fixture='train-run-contract')
            def sample(self, split, rng, batch, length):
                values = rng.integers(1, 31, size=(batch, length+1), dtype=np.int64)
                return values, list(range(batch)), [0]*batch
        with TemporaryDirectory() as folder:
            root = Path(folder)
            backbone = root/'backbone.pt'
            torch.save(payload, backbone)
            args = SimpleNamespace(**(config | dict(backbone=str(backbone), output=root/'pilot',
                                                    device='cpu', quiet=True, resume=None)))
            with patch('v1.train.load_corpus', return_value=TinyCorpus()):
                output, status = train(args, verify_first=False)
                self.assertEqual(status, 'completed')
                import json
                result = json.loads((output/'summary.json').read_text())
                reference = result['conditions']['transformer']
                self.assertFalse(reference['training_performed'])
                self.assertEqual(reference['completed_tokens'], 0)
                for name in ('adapter', 'bio_no_phase', 'bio_full'):
                    self.assertEqual(result['conditions'][name]['completed_tokens'], config['tokens'])
                    self.assertEqual(result['conditions'][name]['updates'], 2)
                    self.assertTrue((output/name/'last.pt').is_file())
                resumed = SimpleNamespace(resume=output, device='cpu', quiet=True)
                self.assertEqual(train(resumed, verify_first=False), (output, 'completed'))

    def test_batch_two_full_dynamics_pilot_reaches_final_consolidation_boundary(self):
        payload, config = fixture()
        config.update(batch=2, full_dynamics=True, nodes=8)
        class TinyCorpus:
            manifest = dict(vocab_size=32, fixture='full-multilane-contract')
            def sample(self, split, rng, batch, length):
                return rng.integers(1, 31, size=(batch, length+1), dtype=np.int64), list(range(batch)), [0]*batch
        with TemporaryDirectory() as folder:
            root = Path(folder)
            backbone = root/'backbone.pt'
            torch.save(payload, backbone)
            args = SimpleNamespace(**(config | dict(backbone=str(backbone), output=root/'pilot',
                                                    device='cpu', quiet=True, resume=None)))
            with patch('v1.train.load_corpus', return_value=TinyCorpus()):
                output, status = train(args, verify_first=False)
            self.assertEqual(status, 'completed')
            import json
            result = json.loads((output/'summary.json').read_text())
            for name in ('bio_no_phase', 'bio_full'):
                self.assertEqual(result['conditions'][name]['completed_tokens'], 16)
                self.assertEqual(result['conditions'][name]['updates'], 1)
                self.assertEqual(result['conditions'][name]['consolidation_attempts'], 1)
                records = [json.loads(line) for line in (output/name/'consolidation.jsonl').read_text().splitlines()]
                self.assertEqual(records[0]['consolidation_sessions'], 2)
                checkpoint = torch.load(output/name/'last.pt', map_location='cpu', weights_only=True)
                self.assertEqual(checkpoint['counters']['consolidation_attempts'], 1)

    def test_stop_after_last_update_stays_aborted_until_final_evaluation_resumes(self):
        payload, config = fixture()
        config.update(tokens=8)

        class TinyCorpus:
            manifest = dict(vocab_size=32, fixture='stop-boundary')
            def sample(self, split, rng, batch, length):
                return rng.integers(1, 31, size=(batch, length+1), dtype=np.int64), list(range(batch)), [0]*batch

        class Stop:
            requested = False
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        stop = Stop()
        def interrupted_update(*args, **kwargs):
            result = update(*args, **kwargs)
            stop.requested = True
            return result

        with TemporaryDirectory() as folder:
            root = Path(folder)
            backbone = root/'backbone.pt'
            torch.save(payload, backbone)
            args = SimpleNamespace(**(config | dict(backbone=str(backbone), output=root/'pilot',
                                                    device='cpu', quiet=True, resume=None)))
            with patch('v1.train.load_corpus', return_value=TinyCorpus()), \
                 patch('v1.train.StopRequest', return_value=stop), \
                 patch('v1.train.update', side_effect=interrupted_update):
                output, status = train(args, verify_first=False)
            self.assertEqual(status, 'aborted')
            import json
            result = json.loads((output/'summary.json').read_text())
            self.assertEqual(result['conditions']['adapter']['status'], 'aborted')
            self.assertEqual(result['conditions']['adapter']['completed_tokens'], 8)
            self.assertNotIn('validation', result['conditions']['adapter'])
            with patch('v1.train.load_corpus', return_value=TinyCorpus()):
                _, resumed = train(SimpleNamespace(resume=output, device='cpu', quiet=True), verify_first=False)
            self.assertEqual(resumed, 'completed')
            result = json.loads((output/'summary.json').read_text())
            self.assertIn('validation', result['conditions']['adapter'])

    def test_failed_resume_preflight_preserves_original_registry(self):
        payload, config = fixture()

        class TinyCorpus:
            manifest = dict(vocab_size=32, fixture='resume-ownership')
            def sample(self, split, rng, batch, length):
                return rng.integers(1, 31, size=(batch, length+1), dtype=np.int64), list(range(batch)), [0]*batch

        with TemporaryDirectory() as folder:
            root = Path(folder)
            backbone = root/'backbone.pt'
            torch.save(payload, backbone)
            args = SimpleNamespace(**(config | dict(backbone=str(backbone), output=root/'pilot',
                                                    device='cpu', quiet=True, resume=None)))
            with patch('v1.train.load_corpus', return_value=TinyCorpus()):
                output, _ = train(args, verify_first=False)
            original = (output/'registry.json').read_bytes()
            resumed = SimpleNamespace(resume=output, device='cpu', quiet=True)
            with patch('v1.train.load_corpus', side_effect=ValueError('damaged data')):
                with self.assertRaises(ValueError):
                    train(resumed, verify_first=False)
            self.assertEqual((output/'registry.json').read_bytes(), original)
            with patch('v1.train.RunLock', side_effect=RuntimeError('busy')):
                with self.assertRaises(RuntimeError):
                    train(resumed, verify_first=False)
            self.assertEqual((output/'registry.json').read_bytes(), original)
            self.assertEqual(len(list((output/'resume_attempts').glob('failure_*.txt'))), 2)

    def test_incomplete_masked_window_never_consolidates_unscored_observations(self):
        payload, config = fixture()
        config.update(batch=2, full_dynamics=True, nodes=8, tokens=9)

        class TinyCorpus:
            manifest = dict(vocab_size=32, fixture='masked-consolidation')
            def sample(self, split, rng, batch, length):
                return rng.integers(1, 31, size=(batch, length+1), dtype=np.int64), list(range(batch)), [0]*batch

        with TemporaryDirectory() as folder:
            root = Path(folder)
            backbone = root/'backbone.pt'
            torch.save(payload, backbone)
            args = SimpleNamespace(**(config | dict(backbone=str(backbone), output=root/'pilot',
                                                    device='cpu', quiet=True, resume=None)))
            with patch('v1.train.load_corpus', return_value=TinyCorpus()):
                output, status = train(args, verify_first=False)
            self.assertEqual(status, 'completed')
            import json
            result = json.loads((output/'summary.json').read_text())
            for condition in ('bio_no_phase', 'bio_full'):
                record = result['conditions'][condition]
                self.assertEqual(record['completed_tokens'], 9)
                self.assertEqual(record['consolidation_attempts'], 0)
                self.assertEqual(record['unscored_visible_tail'], 7)

    def test_numerical_dependency_hashes_cover_joseph_and_topology(self):
        sources = source_hashes()
        for dependency in ('v06/model.py', 'v07/graph.py', 'v03/graph.py',
                           'v04/model.py', 'v05/model.py', 'v08/recurrence.py'):
            self.assertIn(dependency, sources)


if __name__ == '__main__':
    unittest.main()
