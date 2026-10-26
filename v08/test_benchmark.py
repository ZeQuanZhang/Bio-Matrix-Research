"""Protocol checks independent of hardware speed."""
import unittest

import torch

from v07.model import Config, Transformer, parameter_count
from .benchmark import matched_transformer, plan, training_update


class BenchmarkProtocol(unittest.TestCase):
    def test_plan_keeps_impl_and_dynamics_comparisons_separate(self):
        rows = plan([32,128,512],64)
        self.assertEqual(len(rows),36)
        self.assertEqual(len({r['id'] for r in rows}),36)
        self.assertTrue(all(r['status']=='not_executed' for r in rows))
        for n in (32,128,512):
            methods = {r['method'] for r in rows if r['kind']=='model' and r['family']=='v08' and r['nodes']==n and r['block']==16}
            self.assertEqual(methods,{'sequential','parallel','triangular'})

    def test_transformer_parameter_formula_matches_actual_model(self):
        cfg = matched_transformer(Config(vocab_size=4096),350000)
        w=cfg.transformer_width
        ff=cfg.transformer_ff
        expected=cfg.vocab_size*w+cfg.transformer_layers*(4*w*w+2*w*ff+ff+5*w)+2*w
        self.assertEqual(parameter_count(Transformer(cfg)),expected)

    def test_timed_update_really_updates_parameters_and_detaches_state(self):
        torch.set_num_threads(1); torch.manual_seed(11)
        cfg=Config(vocab_size=64,transformer_width=16,transformer_ff=64,transformer_layers=1)
        model=Transformer(cfg)
        before=model.embedding.weight.detach().clone()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        values=torch.randint(0,64,(2,9))
        result,state=training_update(model,optimizer,values,None,torch.device('cpu'),True)
        self.assertFalse(torch.equal(before,model.embedding.weight))
        self.assertEqual(result['targets'],16)
        self.assertGreater(result['tokens_per_second'],0)
        self.assertIn('backward',result['phase_seconds'])
        self.assertFalse(state['kv'][0][0].requires_grad)


if __name__=='__main__':
    unittest.main()
