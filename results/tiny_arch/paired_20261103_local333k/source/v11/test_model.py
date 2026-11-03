"""Projection transfer, causality and task-credit checks for v1.1."""
from types import SimpleNamespace
from dataclasses import replace
import unittest

import torch
from torch.nn import functional as F

from v07.model import Transformer
from .model import Config, Network


def config(**changes):
    values = dict(vocab_size=31, width=16, layers=2, heads=2, ff=32,
                  context_limit=8, segment_tokens=4, memory_heads=2,
                  memory_dim=8, bands=2, full_dynamics=False)
    return Config(**(values | changes))


def assert_tree_close(test, actual, expected, atol=2e-9, rtol=2e-9):
    if isinstance(actual, dict):
        test.assertEqual(actual.keys(), expected.keys())
        for name in actual:
            assert_tree_close(test, actual[name], expected[name], atol, rtol)
    elif isinstance(actual, (tuple, list)):
        test.assertEqual(len(actual), len(expected))
        for first, second in zip(actual, expected):
            assert_tree_close(test, first, second, atol, rtol)
    elif isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    else:
        test.assertEqual(actual, expected)


class ModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(19)

    def test_disabled_is_original_transformer(self):
        cfg = config(memory_enabled=False)
        old = Transformer(SimpleNamespace(vocab_size=cfg.vocab_size,
                          transformer_width=cfg.width, transformer_heads=cfg.heads,
                          transformer_ff=cfg.ff, transformer_layers=cfg.layers,
                          context_limit=cfg.context_limit)).double()
        model = Network(cfg).double().load_backbone({'model': old.state_dict()}, freeze=False)
        tokens = torch.randint(cfg.vocab_size, (2, 7))
        actual, state, diagnostics = model(tokens)
        expected, old_state, _ = old(tokens)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for name in ('kv', 'valid', 'position'):
            assert_tree_close(self, state[name], old_state[name], atol=0, rtol=0)
        self.assertEqual(state['memory'], [])
        self.assertEqual(diagnostics, {})

    def test_exact_head_geometry_and_tensor_transfer_from_real_backbone_shape(self):
        cfg = config(width=56, heads=4, memory_heads=None, memory_dim=None, layers=1,
                     memory_init='backbone', phase_enabled=False)
        self.assertEqual((cfg.memory_heads, cfg.memory_dim), (4, 14))
        old = Transformer(SimpleNamespace(vocab_size=cfg.vocab_size,
                          transformer_width=cfg.width, transformer_heads=cfg.heads,
                          transformer_ff=cfg.ff, transformer_layers=cfg.layers,
                          context_limit=cfg.context_limit)).double()
        model = Network(cfg).double().load_backbone({'model': old.state_dict()})
        for source, memory in zip(old.blocks, model.memories):
            for target, weight in zip((memory.query, memory.key, memory.value), source.qkv.weight.chunk(3)):
                torch.testing.assert_close(target.weight, weight, atol=0, rtol=0)
                self.assertNotEqual(target.weight.data_ptr(), weight.data_ptr())
                self.assertTrue(target.weight.requires_grad)
            torch.testing.assert_close(memory.output.weight, source.out.weight, atol=0, rtol=0)
            self.assertNotEqual(memory.output.weight.data_ptr(), source.out.weight.data_ptr())
            self.assertIsInstance(memory.norm, torch.nn.Identity)
        self.assertFalse(model.blocks[0].qkv.weight.requires_grad)
        self.assertEqual(model.memory_initialization['mode'], 'backbone')

    def test_inherited_input_and_output_maps_preserve_head_coordinates(self):
        model = Network(config(phase_enabled=False, layers=1)).double()
        block, memory = model.blocks[0], model.memories[0]
        hidden = block.norm1(torch.randn(2, 5, 16, dtype=torch.float64))
        projected = block.qkv(hidden).reshape(2, 5, 3, 2, 8)
        controls = memory.prepare(hidden, memory.initial_state(2), torch.arange(5)[None].expand(2, -1))
        expected_q = F.normalize(projected[:, :, 0], dim=-1, eps=1e-6)
        expected_k = F.normalize(projected[:, :, 1], dim=-1, eps=1e-6)
        expected_v = projected[:, :, 2].tanh()
        for name, expected in [('query', expected_q), ('key', expected_k), ('value', expected_v)]:
            torch.testing.assert_close(controls[name], expected[:, :, None].expand_as(controls[name]),
                                       atol=2e-12, rtol=2e-12)
        read = torch.randn(2, 5, 2, 8, dtype=torch.float64).flatten(-2)
        torch.testing.assert_close(memory.output(read), block.out(read), atol=0, rtol=0)

    def test_random_initialization_is_an_explicit_non_copy_control(self):
        cfg = config(memory_init='random', layers=1)
        model = Network(cfg).double()
        initial = {name: value.clone() for name, value in model.memories[0].state_dict().items()}
        source = Network(config(memory_enabled=False, layers=1)).double()
        model.load_backbone(source.state_dict())
        for name, value in initial.items():
            torch.testing.assert_close(model.memories[0].state_dict()[name], value, atol=0, rtol=0)
        self.assertFalse(torch.equal(model.memories[0].query.weight, model.blocks[0].qkv.weight[:cfg.width]))
        self.assertEqual(model.memory_initialization['mode'], 'random')

    def test_unknown_projection_geometry_is_rejected_before_loading(self):
        with self.assertRaisesRegex(ValueError, 'exact backbone head geometry'):
            config(memory_dim=4)
        with self.assertRaisesRegex(ValueError, 'exact backbone head geometry'):
            config(memory_heads=1, memory_dim=16)
        with self.assertRaises(ValueError):
            config(memory_init='truncate')
        with self.assertRaises(ValueError):
            config(field_write_mode='unbounded')

    def test_future_tokens_never_change_earlier_outputs(self):
        model = Network(config()).double()
        tokens = torch.randint(31, (2, 13))
        other = tokens.clone()
        other[:, 7:] = (other[:, 7:]+3) % 31
        first, _, _ = model(tokens)
        second, _, _ = model(other)
        torch.testing.assert_close(first[:, :7], second[:, :7], atol=0, rtol=0)

    def test_arbitrary_calls_match_whole_sequence_and_token_calls(self):
        model = Network(config()).double()
        tokens = torch.randint(31, (2, 17))
        expected, expected_state, _ = model(tokens)
        for cuts in ((1,)*17, (3, 2, 7, 1, 4), (8, 9)):
            state, outputs, offset = None, [], 0
            for length in cuts:
                output, state, _ = model(tokens[:, offset:offset+length], state)
                outputs.append(output)
                offset += length
            torch.testing.assert_close(torch.cat(outputs, 1), expected, atol=2e-9, rtol=2e-9)
            assert_tree_close(self, state, expected_state)

    def test_snapshot_changes_only_at_fixed_segment_boundaries(self):
        model = Network(config()).double()
        _, state, _ = model(torch.tensor([[1, 2, 3]]))
        self.assertEqual(int(state['phase']), 3)
        self.assertTrue(all(torch.count_nonzero(value) == 0 for value in state['memory']))
        self.assertTrue(all(torch.count_nonzero(value) > 0 for value in state['pending']))
        _, committed, _ = model(torch.tensor([[4]]), state)
        self.assertEqual(int(committed['phase']), 0)
        self.assertTrue(all(torch.count_nonzero(value) > 0 for value in committed['memory']))
        assert_tree_close(self, committed['memory'], committed['pending'], atol=0, rtol=0)

    def test_future_segment_loss_trains_earlier_writes_and_phase(self):
        model = Network(config()).double().freeze_backbone()
        tokens = torch.randint(31, (2, 8))
        _, state, _ = model(tokens[:, :4])
        self.assertTrue(all(value.grad_fn is not None for value in state['memory']))
        logits, _, _ = model(tokens[:, 4:], state)
        F.cross_entropy(logits.flatten(0, 1), tokens[:, :4].flatten()).backward()
        for name in ('query.weight', 'key.weight', 'value.weight', 'write_gate.weight',
                     'log_tau', 'raw_frequency'):
            parameter = dict(model.memories[0].named_parameters())[name]
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().max()), 0., name)
        self.assertTrue(all(parameter.grad is None for name, parameter in model.named_parameters()
                            if not name.startswith('memories.')))

    def test_mixed_lane_reset_clears_history_without_affecting_other_lane(self):
        model = Network(config()).double()
        prefix = torch.randint(31, (2, 6))
        suffix = torch.randint(31, (2, 7))
        _, warm, _ = model(prefix)
        actual, final, _ = model(suffix, warm, reset_mask=torch.tensor([True, False]))
        fresh, fresh_state, _ = model(suffix[:1])
        _, second_state, _ = model(prefix[1:])
        retained, retained_state, _ = model(suffix[1:], second_state)
        torch.testing.assert_close(actual[:1], fresh, atol=2e-9, rtol=2e-9)
        torch.testing.assert_close(actual[1:], retained, atol=2e-9, rtol=2e-9)
        for name in ('memory', 'pending'):
            for combined, first, second in zip(final[name], fresh_state[name], retained_state[name]):
                torch.testing.assert_close(combined[:1], first, atol=2e-9, rtol=2e-9)
                torch.testing.assert_close(combined[1:], second, atol=2e-9, rtol=2e-9)
        torch.testing.assert_close(final['position'], torch.tensor([7, 13]))

    def test_read_disabled_matches_same_backbone_and_write_disabled_freezes_state(self):
        cfg = config()
        model = Network(cfg).double()
        baseline = Network(config(memory_enabled=False)).double().load_backbone(model.state_dict(), freeze=False)
        tokens = torch.randint(31, (2, 12))
        actual, state, _ = model(tokens, memory_read=False, memory_write=False)
        expected, _, _ = baseline(tokens)
        torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
        self.assertTrue(all(torch.count_nonzero(value) == 0 for value in state['memory']+state['pending']))
        _, warm, _ = model(tokens[:, :6])
        old_memory, old_pending = warm['memory'], warm['pending']
        _, frozen, _ = model(tokens[:, 6:], warm, memory_write=False)
        assert_tree_close(self, frozen['memory'], old_memory, atol=0, rtol=0)
        assert_tree_close(self, frozen['pending'], old_pending, atol=0, rtol=0)

    def test_state_capacity_does_not_grow_with_history(self):
        model = Network(config())
        initial = model.initial_state(2)
        expected = sum(value.numel() for value in initial['memory']+initial['pending'])
        with torch.no_grad():
            _, state, _ = model(torch.randint(31, (2, 49)))
        self.assertEqual(sum(value.numel() for value in state['memory']+state['pending']), expected)
        self.assertLessEqual(state['valid'].shape[-1], model.cfg.context_limit)
        self.assertTrue(all(value.dtype == torch.float32 for value in state['memory']))

    def test_phase_is_norm_preserving_and_no_phase_is_explicit(self):
        model = Network(config()).double()
        branch = model.memories[0]
        vector = torch.randn(2, 11, 2, 2, 8, dtype=torch.float64)
        positions = torch.arange(11)[None].expand(2, -1)
        actual, error = branch._rotate(vector, positions)
        torch.testing.assert_close(actual.square().sum(-1), vector.square().sum(-1), atol=2e-12, rtol=2e-12)
        self.assertLess(float(error.detach()), 1e-12)
        no_phase = Network(config(phase_enabled=False)).double()
        rotated, error = no_phase.memories[0]._rotate(vector, positions)
        torch.testing.assert_close(rotated, vector, atol=0, rtol=0)
        self.assertIsNone(no_phase.memories[0].raw_frequency)

    def test_invalid_geometry_or_partial_backbone_load_is_rejected(self):
        for changes in (dict(memory_dim=3), dict(width=15), dict(segment_tokens=9), dict(bands=0)):
            with self.assertRaises(ValueError):
                config(**changes)
        model = Network(config())
        with self.assertRaises(ValueError):
            model.load_backbone({'embedding.weight': model.embedding.weight})
        state = model.initial_state(2)
        state['phase'] += 1
        with self.assertRaises(ValueError):
            model(torch.ones(2, 1, dtype=torch.long), state)

    def test_full_field_streaming_is_causal_and_split_invariant(self):
        model = Network(config(layers=1, full_dynamics=True, nodes=8)).double()
        tokens = torch.randint(31, (2, 13))
        expected, whole_state, diag = model(tokens)
        modified = tokens.clone()
        modified[:, 9:] = (modified[:, 9:]+5) % 31
        causal, _, _ = model(modified)
        torch.testing.assert_close(causal[:, :9], expected[:, :9], atol=0, rtol=0)
        state, parts = None, []
        for at in range(tokens.shape[1]):
            output, state, _ = model(tokens[:, at:at+1], state)
            parts.append(output)
        torch.testing.assert_close(torch.cat(parts, 1), expected, atol=2e-9, rtol=2e-9)
        assert_tree_close(self, state, whole_state)
        self.assertIn('field_effective_nodes', diag)
        self.assertEqual(int(state['field_summary'][0]['count'][0]), 1)
        phase = state['field'][0]['phase']
        torch.testing.assert_close(phase.square().sum(-1), torch.ones_like(phase[..., 0]), atol=1e-12, rtol=1e-12)

    def test_full_field_future_loss_reaches_history_mechanisms(self):
        model = Network(config(layers=1, full_dynamics=True, nodes=8)).double().freeze_backbone()
        tokens = torch.randint(31, (2, 12))
        _, state, _ = model(tokens[:, :8])
        logits, _, _ = model(tokens[:, 8:], state)
        F.cross_entropy(logits.flatten(0, 1), tokens[:, :4].flatten()).backward()
        field = model.memories[0].field
        for name in ('node_shape', 'input_rank', 'raw_variance', 'raw_tau', 'raw_frequency', 'raw_coupling'):
            parameter = dict(field.named_parameters())[name]
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().max()), 0., name)

    def test_full_field_reset_and_no_phase_include_all_state(self):
        model = Network(config(layers=1, full_dynamics=True, nodes=8, phase_enabled=False)).double()
        self.assertFalse(model.memories[0].field.phase_enabled)
        prefix, suffix = torch.randint(31, (2, 6)), torch.randint(31, (2, 5))
        _, warm, _ = model(prefix)
        actual, state, _ = model(suffix, warm, reset_mask=torch.tensor([True, True]))
        expected, fresh, _ = model(suffix)
        torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
        for name in ('field', 'field_summary', 'memory', 'pending'):
            assert_tree_close(self, state[name], fresh[name])

    def test_ordered_and_summary_have_equal_state_budget_and_real_distinct_writes(self):
        cfg = config(layers=1, full_dynamics=True, nodes=8, field_write_mode='summary')
        summary = Network(cfg).double()
        ordered = Network(replace(cfg, field_write_mode='ordered')).double()
        ordered.load_state_dict(summary.state_dict())
        tokens = torch.randint(31, (2, 13))
        outputs, states = [], []
        for model in (summary, ordered):
            output, whole, _ = model(tokens)
            split, parts, offset = None, [], 0
            for length in (2, 1, 6, 4):
                part, split, _ = model(tokens[:, offset:offset+length], split)
                parts.append(part)
                offset += length
            torch.testing.assert_close(torch.cat(parts, 1), output, atol=2e-9, rtol=2e-9)
            assert_tree_close(self, split, whole)
            self.assertEqual(whole['field_summary'][0]['tokens_key'].shape[1], cfg.segment_tokens)
            outputs.append(output)
            states.append(whole)
        torch.testing.assert_close(outputs[0][:, :4], outputs[1][:, :4], atol=0, rtol=0)
        for name in ('field', 'field_summary', 'memory', 'pending'):
            for first, second in zip(states[0][name], states[1][name]):
                if isinstance(first, dict):
                    self.assertEqual({key: value.shape for key, value in first.items()},
                                     {key: value.shape for key, value in second.items()})
                else:
                    self.assertEqual(first.shape, second.shape)
        difference = (states[0]['field'][0]['association']-states[1]['field'][0]['association']).abs().max()
        self.assertGreater(float(difference.detach()), 1e-7)


if __name__ == '__main__':
    unittest.main()
