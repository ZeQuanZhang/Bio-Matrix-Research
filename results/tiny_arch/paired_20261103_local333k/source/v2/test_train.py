"""Paired training, parameter-matched dense controls and checkpoint contracts."""
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from v11.test_train import fixture as old_fixture, TinyCorpus, TinyQA
from .train import (DEFAULTS,CONDITIONS,build_models,optimizer_for,restore_checkpoint,
                    save_checkpoint,settings,source_hashes,train,update,validation_data,evaluate)


def fixture():
    payload,_=old_fixture()
    cfg=DEFAULTS | dict(seeds='11',batch=2,segments=4,segment_tokens=4,context_limit=4,
                         nodes=4,phase_bands=3,cpt_tokens=32,pt_updates=1,validation_windows=2,threads=1)
    return payload,cfg


class TrainingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_plain_transformers_match_parameter_budget_and_pretrained_function(self):
        payload,c=fixture();models,cfg,counts=build_models(payload,c,11)
        self.assertEqual(counts['v2_static']['total_parameters'],counts['v2_dynamic']['total_parameters'])
        for model in models.values():self.assertTrue(all(p.requires_grad for p in model.parameters()))
        for name in ('trans_local','trans_full'):
            self.assertLessEqual(abs(counts[name]['parameter_difference']),cfg.layers*(2*cfg.width+1))
            self.assertIsInstance(models[name].blocks[0].ff,torch.nn.Sequential)
        for name,value in models['v2_static'].state_dict().items():
            torch.testing.assert_close(value,models['v2_dynamic'].state_dict()[name],atol=0,rtol=0)
        for name,value in models['trans_local'].state_dict().items():
            torch.testing.assert_close(value,models['trans_full'].state_dict()[name],atol=0,rtol=0)
        from .model import Network
        base=Network(replace(cfg,memory_enabled=False)).load_backbone(payload)
        x=torch.tensor([[1,4,5,6]])
        torch.testing.assert_close(base(x)[0],models['trans_local'](x)[0],atol=1e-7,rtol=1e-6)

    def test_answer_mask_and_all_backbone_parameters_really_train(self):
        payload,c=fixture();values,mask,_=TinyQA().sample('train',np.random.default_rng(1),2)
        for name,model in build_models(payload,c,11)[0].items():
            old=model.embedding.weight.detach().clone()
            row,state=update(model,optimizer_for(model,c),values,mask,c,torch.device('cpu'))
            self.assertEqual(row['targets'],4);self.assertEqual(row['inputs'],32)
            self.assertTrue(np.isfinite(row['ce']))
            self.assertGreater(float((model.embedding.weight.detach()-old).abs().sum()),0)
            self.assertFalse(any(v.requires_grad for v in (state['memory'][0].values() if state['memory'] else [])))

    def test_checkpoint_restores_rng_optimizer_and_rejects_protocol_drift(self):
        payload,c=fixture();model=build_models(payload,c,11)[0]['v2_dynamic']
        optimizer=optimizer_for(model,c);rng=np.random.default_rng(1)
        values,mask,_=TinyQA().sample('train',rng,2)
        update(model,optimizer,values,mask,c,torch.device('cpu'))
        protocol=dict(source_hashes=source_hashes(),fixture=True);counters=dict(updates=1,cpt_tokens=0,pt_updates=1)
        with TemporaryDirectory() as folder:
            path=Path(folder)/'last.pt'
            save_checkpoint(path,model,optimizer,counters,rng,protocol,torch.device('cpu'))
            expected=rng.integers(10000,size=8)
            other=build_models(payload,c,11)[0]['v2_dynamic'];otheropt=optimizer_for(other,c);otherrng=np.random.default_rng(9)
            self.assertEqual(restore_checkpoint(path,other,otheropt,otherrng,protocol,torch.device('cpu')),counters)
            np.testing.assert_array_equal(expected,otherrng.integers(10000,size=8))
            self.assertEqual(len(otheropt.state),len(optimizer.state))
            for n,v in model.state_dict().items():torch.testing.assert_close(v,other.state_dict()[n],atol=0,rtol=0)
            with self.assertRaises(ValueError):restore_checkpoint(path,other,otheropt,otherrng,protocol|dict(changed=True),torch.device('cpu'))

    def run_fixture(self,root,changes=None,stop=None):
        payload,c=fixture();c.update(changes or {})
        backbone=root/'base.pt';torch.save(payload,backbone)
        args=SimpleNamespace(**(c | dict(backbone=str(backbone),output=root/'paired',device='cpu',quiet=True)))
        with patch('v2.train.load_corpus',return_value=TinyCorpus()),patch('v2.train.prepare_qa',return_value=TinyQA()):
            if stop is None:return train(args,verify_first=False)
            def interrupted(*args,**kwargs):
                result=update(*args,**kwargs);stop.requested=True;return result
            with patch('v2.train.StopRequest',return_value=stop),patch('v2.train.update',side_effect=interrupted):
                return train(args,verify_first=False)

    def test_real_two_seed_runner_pairs_samples_and_saves_one_final_intervention(self):
        with TemporaryDirectory() as folder:
            root,status=self.run_fixture(Path(folder),dict(seeds='11,23'))
            self.assertEqual(status,'completed');summary=json.loads((root/'summary.json').read_text())
            self.assertEqual(len(summary['conditions']),8);self.assertFalse(summary['test_evaluated'])
            comparison=json.loads((root/'comparison.json').read_text())
            self.assertEqual(comparison['aggregate']['v2_dynamic']['completed_seeds'],2)
            self.assertEqual(len(comparison['paired_differences']['v2_dynamic-minus-trans_local']),2)
            self.assertTrue((root/'COMPARISON.md').exists())
            across=[]
            for seed in (11,23):
                shas=[]
                for name in CONDITIONS:
                    path=root/f'{name}_s{seed}'
                    record=summary['conditions'][path.name]
                    self.assertEqual(record['updates'],2);self.assertEqual(record['answer_targets'],4)
                    self.assertEqual(record['cpt_tokens'],32);self.assertEqual(record['pt_updates'],1)
                    self.assertTrue((path/'initial_validation.json').exists())
                    self.assertTrue((path/'cpt_validation.json').exists())
                    metrics=[json.loads(x) for x in (path/'metrics.jsonl').read_text().splitlines()]
                    shas.append([x['sample_sha256'] for x in metrics])
                    validations=[json.loads(x) for x in (path/'validation.jsonl').read_text().splitlines()]
                    self.assertEqual([v['stage'] for v in validations],['initial','cpt_end','final'])
                    self.assertEqual(validations[-1]['qa']['conditions']['remote_history_reencoded']['status'],'completed' if name!='trans_full' else 'not_applicable')
                self.assertTrue(all(x==shas[0] for x in shas));across.append(shas[0])
            self.assertNotEqual(across[0],across[1])
            with patch('v2.train.load_corpus',return_value=TinyCorpus()),patch('v2.train.prepare_qa',return_value=TinyQA()):
                self.assertEqual(train(SimpleNamespace(resume=root,device='cpu',quiet=True),verify_first=False),(root,'completed'))

    def test_warm_only_and_full_score_agree_without_parameter_mutation(self):
        payload,c=fixture();model=build_models(payload,c,11)[0]['v2_dynamic']
        data,_=validation_data(TinyCorpus(),TinyQA(),c)
        before={n:v.detach().clone() for n,v in model.state_dict().items()}
        warm=evaluate(model,data,c,torch.device('cpu'))
        full=evaluate(model,data,c,torch.device('cpu'),full=True)
        for task in ('lm','qa'):
            self.assertAlmostEqual(warm[task]['warm_nll'],full[task]['warm_nll'],places=6)
        for n,v in before.items():torch.testing.assert_close(v,model.state_dict()[n],atol=0,rtol=0)

    def test_stop_resumes_at_safe_boundary_and_runs_missing_cpt_score(self):
        class Stop:
            requested=False
            def __enter__(self):return self
            def __exit__(self,*args):pass
        with TemporaryDirectory() as folder:
            root,status=self.run_fixture(Path(folder),stop=Stop())
            self.assertEqual(status,'aborted')
            summary=json.loads((root/'summary.json').read_text())
            self.assertEqual(summary['conditions']['trans_local_s11']['status'],'aborted')
            self.assertEqual(summary['conditions']['v2_dynamic_s11']['status'],'not_executed')
            with patch('v2.train.load_corpus',return_value=TinyCorpus()),patch('v2.train.prepare_qa',return_value=TinyQA()):
                _,status=train(SimpleNamespace(resume=root,device='cpu',quiet=True),verify_first=False)
            self.assertEqual(status,'completed');self.assertTrue((root/'trans_local_s11/cpt_validation.json').exists())

    def test_resume_missing_checkpoint_or_initial_score_does_not_restart(self):
        class Stop:
            requested=False
            def __enter__(self):return self
            def __exit__(self,*args):pass
        for missing in ('last.pt','initial_validation.json'):
            with self.subTest(missing=missing),TemporaryDirectory() as folder:
                root,_=self.run_fixture(Path(folder),stop=Stop())
                (root/'trans_local_s11'/missing).unlink();before=(root/'registry.json').read_bytes()
                with patch('v2.train.load_corpus',return_value=TinyCorpus()),patch('v2.train.prepare_qa',return_value=TinyQA()):
                    with self.assertRaises(FileNotFoundError):train(SimpleNamespace(resume=root,device='cpu',quiet=True),verify_first=False)
                self.assertEqual((root/'registry.json').read_bytes(),before)

    def test_preflight_failure_preserves_registry_and_existing_output(self):
        with TemporaryDirectory() as folder:
            root,_=self.run_fixture(Path(folder));before=(root/'registry.json').read_bytes()
            with patch('v2.train.load_corpus',side_effect=ValueError('data changed')):
                with self.assertRaises(ValueError):train(SimpleNamespace(resume=root,device='cpu',quiet=True),verify_first=False)
            self.assertEqual((root/'registry.json').read_bytes(),before)
            with self.assertRaises(FileExistsError):train(SimpleNamespace(output=root,device='cpu'),verify_first=False)

    def test_settings_reject_duplicate_seeds_and_changed_budgets(self):
        c=settings(SimpleNamespace())
        for change in (dict(seeds='11,11'),dict(cpt_tokens=0),dict(batch=0),dict(segments=1)):
            with self.assertRaises(ValueError):settings(SimpleNamespace(**change))
        with self.assertRaises(ValueError):settings(SimpleNamespace(pt_updates=1),c)


if __name__=='__main__':unittest.main()
