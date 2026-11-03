"""CPU contracts for the staged experiment, pairing, credit, and safe resume."""
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

from v07.model import Config as LegacyConfig, Transformer
from .train import (DEFAULTS, build_models, build_teacher, objective, optimizer_for,
                    restore_checkpoint, save_checkpoint, settings, source_hashes, train, update)


def fixture():
    torch.manual_seed(18)
    old = LegacyConfig(vocab_size=32, transformer_width=16, transformer_layers=1,
                       transformer_ff=32, transformer_heads=2, context_limit=32)
    payload = dict(protocol=dict(model=asdict(old), fixture=True), model=Transformer(old).state_dict())
    config = DEFAULTS | dict(batch=2, segments=4, segment_tokens=4, context_limit=4,
                              bands=2, nodes=8, internal_steps=3, cpt_tokens=32, pt_updates=1,
                              validation_windows=2, threads=1)
    return payload, config


class TinyCorpus:
    manifest = dict(vocab_size=32, fixture='staged-run')
    def sample(self, split, rng, batch, length):
        return rng.integers(1, 31, size=(batch, length+1), dtype=np.int64), list(range(batch)), [0]*batch


class TinyQA:
    manifest = dict(fixture='history-answer-mask')
    def sample(self, split, rng, batch):
        values = rng.integers(1, 31, size=(batch, 17), dtype=np.int64)
        mask = np.zeros((batch, 16), dtype=bool); mask[:, 12:14] = True
        first = np.zeros_like(mask); first[:, 12] = True
        return values, mask, dict(first_answer_mask=first, question_ids=[f'{split}-{i}' for i in range(batch)])


class TrainingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_paired_summary_ordered_and_zero_adapter_initialization(self):
        payload, cfg = fixture(); models, architecture, counts = build_models(payload, cfg)
        self.assertEqual(architecture.memory_dim, 8)
        self.assertTrue(architecture.phase_enabled)
        self.assertEqual(counts['bio_summary'], counts['bio_ordered'])
        self.assertEqual(counts['transformer']['trainable_parameters'], 0)
        self.assertLessEqual(abs(counts['adapter']['trainable_parameter_difference']), 2*architecture.width*architecture.layers)
        for name, tensor in models['bio_ordered'].state_dict().items():
            torch.testing.assert_close(tensor, models['bio_summary'].state_dict()[name], atol=0, rtol=0)
        tokens = torch.tensor([[1, 4, 7, 2]])
        torch.testing.assert_close(models['transformer'](tokens)[0], models['adapter'](tokens)[0], atol=0, rtol=0)

    def test_stage_objectives_and_answer_only_gradient(self):
        _, cfg = fixture()
        logits = torch.randn(1, 4, 6, requires_grad=True); teacher = torch.randn_like(logits)
        targets = torch.tensor([[1, 2, 3, 4]]); mask = torch.tensor([[False, False, True, False]])
        loss, ce, kd = objective(logits, targets, None, mask, cfg, 'cpt')
        torch.testing.assert_close(loss, ce); self.assertEqual(float(kd), 0.)
        pt, pt_ce, pt_kd = objective(logits, targets, teacher, mask, cfg, 'pt')
        torch.testing.assert_close(pt, pt_ce+cfg['pt_kd_weight']*pt_kd)
        pt.backward()
        self.assertEqual(float(logits.grad[:, :2].abs().sum()+logits.grad[:, 3:].abs().sum()), 0.)
        with self.assertRaises(ValueError): objective(logits, targets, None, mask, cfg, 'pt')

    def test_answer_loss_reaches_earlier_writes_with_frozen_backbone(self):
        payload, cfg = fixture(); model = build_models(payload, cfg)[0]['bio_ordered']
        _, state, _ = model(torch.tensor([[1, 5, 6, 7]]))
        logits, _, _ = model(torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15]]), state)
        F.cross_entropy(logits[:, -1], torch.tensor([9])).backward()
        self.assertGreater(float(model.memories[0].write_gate.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for name, p in model.named_parameters() if not name.startswith('memories.')))

    def test_real_staged_update_finite_and_backbone_unchanged(self):
        payload, cfg = fixture(); model = build_models(payload, cfg)[0]['bio_ordered']
        teacher = build_teacher(payload, torch.device('cpu')); optimizer = optimizer_for(model, cfg)
        before = {n: v.clone() for n, v in model.state_dict().items() if not n.startswith('memories.')}
        values, mask, _ = TinyQA().sample('train', np.random.default_rng(1), 2)
        row, state = update(model, teacher, optimizer, values, mask, cfg, torch.device('cpu'), 'pt')
        self.assertEqual(row['targets'], 4); self.assertTrue(np.isfinite(row['loss']))
        self.assertFalse(state['memory'][0].requires_grad)
        for n, v in before.items(): torch.testing.assert_close(v, model.state_dict()[n], atol=0, rtol=0)

    def test_checkpoint_restores_stage_rng_optimizer_and_rejects_drift(self):
        payload, cfg = fixture(); model = build_models(payload, cfg)[0]['bio_ordered']
        optimizer = optimizer_for(model, cfg); rng = np.random.default_rng(9)
        protocol = dict(source_hashes=source_hashes(), fixture=True)
        counters = dict(updates=2, cpt_tokens=32, pt_updates=1, answer_targets=4)
        with TemporaryDirectory() as folder:
            path = Path(folder)/'last.pt'
            save_checkpoint(path, model, optimizer, None, counters, rng, protocol, torch.device('cpu'))
            expected = rng.integers(1000, size=8)
            recovered = build_models(payload, cfg)[0]['bio_ordered']; opt = optimizer_for(recovered, cfg)
            other = np.random.default_rng(8)
            self.assertEqual(restore_checkpoint(path, recovered, opt, other, protocol, torch.device('cpu')), counters)
            np.testing.assert_array_equal(expected, other.integers(1000, size=8))
            with self.assertRaises(ValueError):
                restore_checkpoint(path, recovered, opt, other, protocol | dict(changed=True), torch.device('cpu'))

    def run_fixture(self, root, config=None, stop=None):
        payload, default = fixture(); cfg = default | (config or {})
        backbone = root/'backbone.pt'; torch.save(payload, backbone)
        args = SimpleNamespace(**(cfg | dict(backbone=str(backbone), output=root/'pilot', device='cpu', quiet=True)))
        with patch('v11.train.load_corpus', return_value=TinyCorpus()), patch('v11.train.prepare_qa', return_value=TinyQA()):
            if stop is None: return train(args, verify_first=False)
            original = update
            def interrupted(*args, **kwargs):
                result = original(*args, **kwargs); stop.requested = True; return result
            with patch('v11.train.StopRequest', return_value=stop), patch('v11.train.update', side_effect=interrupted):
                return train(args, verify_first=False)

    def test_runner_saves_initial_two_stages_paired_samples_and_resume(self):
        with TemporaryDirectory() as folder:
            output, status = self.run_fixture(Path(folder)); self.assertEqual(status, 'completed')
            summary = json.loads((output/'summary.json').read_text())
            self.assertFalse(summary['test_evaluated'])
            paired = []
            for name in ('adapter', 'bio_summary', 'bio_ordered'):
                self.assertTrue((output/name/'initial_validation.json').exists())
                self.assertTrue((output/name/'cpt_validation.json').exists())
                record = summary['conditions'][name]
                self.assertEqual(record['cpt_tokens'], 32); self.assertEqual(record['pt_updates'], 1)
                self.assertEqual(record['answer_targets'], 4)
                rows = [json.loads(line) for line in (output/name/'metrics.jsonl').read_text().splitlines()]
                self.assertEqual([r['stage'] for r in rows], ['cpt', 'pt'])
                paired.append([r['sample_sha256'] for r in rows])
            self.assertEqual(paired[0], paired[1]); self.assertEqual(paired[1], paired[2])
            self.assertFalse(summary['conditions']['transformer']['training_performed'])
            with patch('v11.train.load_corpus', return_value=TinyCorpus()), patch('v11.train.prepare_qa', return_value=TinyQA()):
                self.assertEqual(train(SimpleNamespace(resume=output, device='cpu', quiet=True), verify_first=False), (output, 'completed'))

    def test_stop_boundary_aborts_and_resumes_remaining_stages(self):
        class Stop:
            requested = False
            def __enter__(self): return self
            def __exit__(self, *args): pass
        with TemporaryDirectory() as folder:
            output, status = self.run_fixture(Path(folder), stop=Stop())
            self.assertEqual(status, 'aborted')
            first = json.loads((output/'summary.json').read_text())['conditions']
            self.assertEqual(first['adapter']['status'], 'aborted')
            self.assertEqual(first['bio_ordered']['status'], 'not_executed')
            with patch('v11.train.load_corpus', return_value=TinyCorpus()), patch('v11.train.prepare_qa', return_value=TinyQA()):
                _, status = train(SimpleNamespace(resume=output, device='cpu', quiet=True), verify_first=False)
            self.assertEqual(status, 'completed')
            self.assertTrue((output/'adapter/cpt_validation.json').exists())

    def test_failed_resume_preserves_registry_and_overwrite_rejected(self):
        with TemporaryDirectory() as folder:
            output, _ = self.run_fixture(Path(folder)); before = (output/'registry.json').read_bytes()
            with patch('v11.train.load_corpus', side_effect=ValueError('hash mismatch')):
                with self.assertRaises(ValueError): train(SimpleNamespace(resume=output, device='cpu', quiet=True), verify_first=False)
            self.assertEqual((output/'registry.json').read_bytes(), before)
            with self.assertRaises(FileExistsError): train(SimpleNamespace(output=output, device='cpu'), verify_first=False)

    def test_settings_reject_short_history_and_changes(self):
        cfg = settings(SimpleNamespace()); self.assertEqual(cfg['internal_steps'], 3)
        with self.assertRaises(ValueError): settings(SimpleNamespace(segments=2))
        with self.assertRaises(ValueError): settings(SimpleNamespace(pt_updates=0))
        with self.assertRaises(ValueError): settings(SimpleNamespace(cpt_tokens=1), cfg)
        for dependency in ('v11/dynamics.py', 'v1/train.py', 'v07/model.py', 'v06/model.py'):
            self.assertIn(dependency, source_hashes())

    def test_resume_never_silently_restarts_a_progressed_condition(self):
        class Stop:
            requested = False
            def __enter__(self): return self
            def __exit__(self, *args): pass
        with TemporaryDirectory() as folder:
            output, _ = self.run_fixture(Path(folder), stop=Stop())
            (output/'adapter/last.pt').unlink()
            before = (output/'registry.json').read_bytes()
            with patch('v11.train.load_corpus', return_value=TinyCorpus()), patch('v11.train.prepare_qa', return_value=TinyQA()):
                with self.assertRaises(FileNotFoundError):
                    train(SimpleNamespace(resume=output, device='cpu', quiet=True), verify_first=False)
            self.assertEqual((output/'registry.json').read_bytes(), before)

    def test_missing_initial_score_is_not_replaced_with_a_trained_score(self):
        class Stop:
            requested = False
            def __enter__(self): return self
            def __exit__(self, *args): pass
        with TemporaryDirectory() as folder:
            output, _ = self.run_fixture(Path(folder), stop=Stop())
            (output/'adapter/initial_validation.json').unlink()
            before = (output/'registry.json').read_bytes()
            with patch('v11.train.load_corpus', return_value=TinyCorpus()), patch('v11.train.prepare_qa', return_value=TinyQA()):
                with self.assertRaises(FileNotFoundError):
                    train(SimpleNamespace(resume=output, device='cpu', quiet=True), verify_first=False)
            self.assertEqual((output/'registry.json').read_bytes(), before)

    def test_partial_cpt_budget_and_padded_pt_never_consolidate(self):
        with TemporaryDirectory() as folder:
            output, status = self.run_fixture(Path(folder), dict(cpt_tokens=33))
            self.assertEqual(status, 'completed')
            summary = json.loads((output/'summary.json').read_text())
            for name in ('bio_summary', 'bio_ordered'):
                record = summary['conditions'][name]
                self.assertEqual(record['cpt_tokens'], 33)
                self.assertEqual(record['pt_updates'], 1)
                self.assertEqual(record['consolidation_attempts'], 0)


if __name__ == '__main__': unittest.main()
