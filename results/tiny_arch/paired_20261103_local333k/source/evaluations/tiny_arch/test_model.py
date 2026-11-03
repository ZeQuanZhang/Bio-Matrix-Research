import copy
import unittest
import torch

from v3.train import DEFAULTS
from .model import Model,metadata,CONDITIONS


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(4)

    def test_budget_tying_and_hash_experts(self):
        for name in CONDITIONS:
            model=Model(name,11,DEFAULTS);info=metadata(model)
            self.assertLess(abs(info['total_parameters']-333136)/333136,.001)
            self.assertIs(model.model.get_input_embeddings().weight,model.model.get_output_embeddings().weight)
            if name=='deepseek4_tiny':
                rows=model.model.model.layers[0].mlp.gate.tid2eid
                self.assertTrue(bool((rows[:,0]!=rows[:,1]).all()));self.assertEqual(len(rows.unique()),4)

    def test_causality_prefix_read_and_no_mutation(self):
        for name in CONDITIONS:
            model=Model(name,11,DEFAULTS).eval();x=torch.randint(3,4096,(1,256));other=x.clone();other[:,128:]=torch.randint(3,4096,(1,128))
            with torch.no_grad():
                expected=model(x)[0];changed=model(other)[0]
                torch.testing.assert_close(expected[:,:128],changed[:,:128],atol=3e-5,rtol=3e-4)
                _,state,_=model(x[:,:192]);saved=state['tokens'].clone()
                suffix=model(x[:,192:],state)[0]
                torch.testing.assert_close(expected[:,192:],suffix,atol=3e-5,rtol=3e-4)
                torch.testing.assert_close(state['tokens'],saved,atol=0,rtol=0)

    def test_native_mps_backward_uses_same_embedded_data(self):
        if not torch.backends.mps.is_available():self.skipTest('MPS unavailable')
        for name in CONDITIONS:
            model=Model(name,11,DEFAULTS).to('mps');x=torch.randint(3,4096,(1,32),device='mps')
            logits,_,diag=model(x);loss=logits.square().mean()
            if 'native_auxiliary' in diag:loss=loss+.001*diag['native_auxiliary']
            loss.backward()
            self.assertTrue(bool(torch.isfinite(loss)));self.assertTrue(bool(torch.isfinite(model.model.get_input_embeddings().weight.grad).all()))


if __name__=='__main__':unittest.main()
