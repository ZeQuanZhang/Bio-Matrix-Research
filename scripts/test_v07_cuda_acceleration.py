"""CPU adapter references. No CUDA work or formal training is performed."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts import v07_cuda_acceleration as adapter
from v07.model import Config, Network, Stateless, Transformer, VARIANTS, initialize


class AccelerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def tearDown(self):
        adapter.uninstall()
        torch._dynamo.reset()

    def setUp(self):
        self.previous_cache_policy = {name:getattr(torch._dynamo.config,name) for name in adapter.CACHE_POLICY}
        self.addCleanup(lambda: [setattr(torch._dynamo.config,name,value) for name,value in self.previous_cache_policy.items()])

    def model(self):
        model = Network(Config(vocab_size=32, matrix_size=4, width=16, modes=2))
        initialize(model, 71)
        return model

    def test_lazy_idempotent_install_and_serializable_gate(self):
        original_tf, original_stateless = Transformer.forward, Stateless.forward
        with patch('torch.compile', side_effect=lambda fn, **kwargs: fn) as compile_mock:
            adapter.install(device='cpu', backend='eager')
            model = self.model()
            keys = list(model.state_dict())
            self.assertEqual(compile_mock.call_count, 0)
            model.gate_blend = .125
            model(torch.tensor([[1,2]]))
            self.assertEqual(compile_mock.call_count, 1)
            self.assertEqual(model._v07_acceleration['cache_policy'], dict(recompile_limit=32,accumulated_recompile_limit=256))
            self.assertEqual(torch._dynamo.config.recompile_limit,32)
            self.assertEqual(torch._dynamo.config.accumulated_recompile_limit,256)
            self.assertEqual(set(compile_mock.call_args.kwargs),{'backend','mode','fullgraph','dynamic'})
            scalar = model._v07_acceleration_control['blend']
            version = scalar._version
            model(torch.tensor([[3]]))
            self.assertEqual(compile_mock.call_count, 1)
            self.assertIs(scalar, model._v07_acceleration_control['blend'])
            self.assertEqual(scalar._version, version)
            model.gate_blend = .75
            model(torch.tensor([[4]]))
            self.assertEqual(compile_mock.call_count, 1)
            self.assertEqual(float(scalar), .75)
            self.assertEqual(model.gate_blend, .75)
            json.dumps(dict(gate_blend=model.gate_blend))
            self.assertEqual(keys, list(model.state_dict()))
            self.assertIs(Transformer.forward, original_tf)
            self.assertIs(Stateless.forward, original_stateless)
            direct = self.model()
            adapter.attach_compiled_step(direct, device='cpu', backend='eager')
            direct(torch.tensor([[1]]))
            self.assertEqual(compile_mock.call_count, 2)

    def test_fullgraph_aot_eager_midchunk_resets_then_one_backward(self):
        eager = self.model().train()
        candidate = copy.deepcopy(eager)
        adapter.attach_compiled_step(candidate, device='cpu', backend='aot_eager')
        tokens = torch.tensor([[1,2,3,4]])
        reference = []
        for model in (eager, candidate):
            model.gate_blend = .125
            first, state, _ = model(tokens[:,:2])
            second, state, _ = model(tokens[:,2:], state, reset_mask=torch.tensor([True]))
            loss = first.square().sum()+second.square().sum()
            loss.backward()
            reference.append((first.detach(),second.detach(),state,
                              {name: None if p.grad is None else p.grad.detach().clone() for name,p in model.named_parameters()}))
        for left, right in zip(reference[0][:2],reference[1][:2]):
            torch.testing.assert_close(left,right,atol=2e-6,rtol=2e-5)
        for name,left in reference[0][2].items():
            torch.testing.assert_close(left, reference[1][2][name],atol=2e-6,rtol=2e-5)
        for name,left in reference[0][3].items():
            right = reference[1][3][name]
            if left is None:
                self.assertIsNone(right)
            else:
                torch.testing.assert_close(left,right,atol=2e-5,rtol=2e-4)
        self.assertEqual(candidate.gate_blend,.125)

    def test_failures_propagate_and_metadata_is_saved(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'acceleration.json'
            model=self.model()
            metadata=adapter.attach_compiled_step(model,device='cpu',backend='eager',record_path=path)
            with patch('torch.compile',side_effect=RuntimeError('compile exploded')):
                with self.assertRaisesRegex(RuntimeError,'compile exploded'):
                    model(torch.tensor([[1]]))
            self.assertEqual(metadata['status'],'failed')
            self.assertEqual(json.loads(path.read_text())['status'],'failed')
            self.assertIsInstance(model.gate_blend,float)
        model=self.model()
        metadata=adapter.attach_compiled_step(model,device='cpu',backend='eager')
        def cache_exhausted(*args,**kwargs):
            raise RuntimeError('recompile_limit=32 reached')
        with patch('torch.compile',return_value=cache_exhausted) as compile_mock:
            with self.assertRaisesRegex(RuntimeError,'recompile_limit=32 reached'):
                model(torch.tensor([[1]]))
        self.assertEqual(compile_mock.call_count,1)
        self.assertEqual(metadata['status'],'failed')
        self.assertIs(model.step,cache_exhausted)  # No substitution of the eager step.
        model=self.model().double()
        adapter.attach_compiled_step(model,device='cpu',backend='eager')
        with self.assertRaisesRegex(ValueError,'FP32'):
            model(torch.tensor([[1]]))

    def test_checkpoint_events_rejected_and_no_baseline_patch(self):
        self.assertEqual(set(adapter.BIO_VARIANTS), set(VARIANTS)-{'stateless','transformer'})
        model=self.model();model.cfg.checkpoint_events=True
        adapter.attach_compiled_step(model,device='cpu',backend='eager')
        with self.assertRaisesRegex(ValueError,'checkpoint_events=false'):
            model(torch.tensor([[1]]))
        with self.assertRaises(TypeError):
            adapter.attach_compiled_step(Stateless(Config(vocab_size=32,width=16)),backend='eager')


if __name__=='__main__':
    unittest.main()
