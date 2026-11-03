"""Scorer parity, label isolation, ragged batching, state clocks and generation credit."""
import ast
from collections import Counter
import copy
from pathlib import Path
import string
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import regex
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer,models,pre_tokenizers,decoders,trainers

from v3.model import Config,Network
from v2.model import Config as TFConfig,Network as Transformer
from . import scoring
from .data import history_text,encode_question,encoded_samples,prepare
from .report import aggregate
from .run import map_tree,stack_states,prefill,answer_batch


class Tokens:
    def decode(self,ids,skip_special_tokens=True):return ' '.join(str(i) for i in ids if i not in (0,1,2))


def cfg():return Config(vocab_size=32,width=16,layers=1,heads=2,ff=32,nodes=4,phase_bands=2,segment_tokens=4,context_limit=8)


def mini_tokenizer():
    t=Tokenizer(models.BPE(unk_token='[UNK]'));t.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder=decoders.ByteLevel()
    t.train_from_iterator(['Date: 2020\nAlice: old memory.\nQuestion: Where?\nAnswer: Paris.'],
        trainers.BpeTrainer(vocab_size=280,show_progress=False,initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                           special_tokens=['[PAD]','[BOS]','[EOS]','[UNK]']))
    return t


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):torch.manual_seed(17)

    def test_official_f1_reference_parity(self):
        source=Path(__file__).resolve().parents[2]/'data/locomo/evaluation.py'
        if not source.exists():self.skipTest('Official pinned scorer checked by dataset preparation')
        original=ast.parse(source.read_text());names={'normalize_answer','f1_score','f1','eval_question_answering'}
        functions=ast.Module(body=[n for n in original.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
        scope=dict(regex=regex,string=string,Counter=Counter,np=np,ps=scoring.ps,print=lambda *a,**k:None)
        exec(compile(functions,'official-scorer','exec'),scope)
        probes=[('cities and houses','city, house',1),('May 2020','in May, 2020',2),
                ('Paris','Paris; France',3),('the running dog','a running dog',4),
                ('not mentioned','',5),('a guess','',5),('','',4)]
        for prediction,answer,category in probes:
            qa=dict(category=category,answer=answer,prediction=prediction,evidence=[])
            expected=scope['eval_question_answering']([qa])[0][0]
            self.assertAlmostEqual(scoring.score(prediction,qa),float(expected),places=12)

    def test_serialization_uses_only_chronological_raw_dialogue(self):
        sample=dict(conversation={'session_10_date_time':'later','session_10':[dict(dia_id='D10:1',speaker='B',text='second')],
            'session_2_date_time':'earlier','session_2':[dict(dia_id='D2:1',speaker='A',text='first',blip_caption='image')],
            'speaker_a':'A','speaker_b':'B'},qa=[dict(answer='SECRET_LABEL')],observation='PRIVATE_SUMMARY')
        s=history_text(sample)
        self.assertLess(s.index('first'),s.index('second'));self.assertIn('Image caption: image',s)
        self.assertNotIn('SECRET_LABEL',s);self.assertNotIn('PRIVATE_SUMMARY',s)
        self.assertEqual(history_text(sample,['D10:1']),'Date: later\n[D10:1] B: second')
        with self.assertRaises(ValueError):history_text(sample,['D404:1'])

    def test_query_is_independent_of_answer_and_oracle_labels(self):
        t=mini_tokenizer();a=dict(category=4,question='Where?',answer='Paris',evidence=['D1:1'])
        b=a|dict(answer='Tokyo',evidence=['D99:1'])
        self.assertEqual(encode_question(t,a)[0],encode_question(t,b)[0])
        query,answer=encode_question(t,a)
        self.assertEqual(t.decode(answer).strip(),'Paris');self.assertNotIn('Paris',t.decode(query))

    def test_missing_evidence_does_not_remove_main_question(self):
        s=dict(sample_id='test',conversation={'session_1_date_time':'2020','session_1':[
            dict(dia_id='D1:1',speaker='A',text='Paris.')], 'speaker_a':'A','speaker_b':'B'},qa=[
            dict(category=4,question='Where?',answer='Paris',evidence=['D404:1']),
            dict(category=4,question='Where?',answer='Paris',evidence=[])])
        rows=encoded_samples([s],mini_tokenizer())[0]['cases']
        self.assertEqual(len(rows),2);self.assertTrue(all(c['query'] and c['answer_ids'] for c in rows))
        self.assertEqual([r['oracle_reason'] for r in rows],['unknown_evidence_id','no_evidence_annotation'])

    def test_tampered_official_data_is_rejected(self):
        with TemporaryDirectory() as folder:
            Path(folder,'locomo10.json').write_text('[]')
            with self.assertRaises(ValueError):prepare(folder)

    def compare_batch(self,model,device):
        model=model.to(device).eval();cases=[];prepared={};saved=[]
        with torch.inference_mode():
            for index,ids in enumerate(([1,3,4],[1,5,6,7,8,9],[1,10,11,12,13])):
                logits,state,_=model(torch.tensor([ids],device=device))
                case=dict(id=str(index),qa=dict(category=4,answer='7 8'),answer_ids=[7,8,9][:index+1])
                cases.append(case);prepared[case['id']]=(logits[0,-1],state)
                saved.append(map_tree(state,lambda x:x.clone()))
            together=answer_batch(model,cases,prepared,Tokens(),device,5)
            for case,before in zip(cases,saved):
                alone=answer_batch(model,[case],{case['id']:prepared[case['id']]},Tokens(),device,5)[case['id']]
                self.assertEqual(together[case['id']]['generated_ids'],alone['generated_ids'])
                self.assertAlmostEqual(together[case['id']]['answer_nll'],alone['answer_nll'],places=5)
                a=map_tree(prepared[case['id']][1],lambda x:x.cpu());b=map_tree(before,lambda x:x.cpu())
                def equal(x,y):
                    if isinstance(x,dict):
                        for k in x:equal(x[k],y[k])
                    elif isinstance(x,(list,tuple)):
                        for p,q in zip(x,y):equal(p,q)
                    elif x is not None:torch.testing.assert_close(x,y,atol=0,rtol=0)
                equal(a,b)
            altered=[case|dict(answer_ids=[20,21,22,23]) for case in cases]
            rerun=answer_batch(model,altered,prepared,Tokens(),device,5)
            self.assertEqual([together[c['id']]['generated_ids'] for c in cases],[rerun[c['id']]['generated_ids'] for c in cases])

    def test_mixed_clock_bio_batch_matches_independent_generation(self):
        self.compare_batch(Network(cfg()).double(),torch.device('cpu'))

    def test_ragged_transformer_kv_matches_independent_generation(self):
        c=cfg();self.compare_batch(Transformer(TFConfig(**vars(c),memory_enabled=False)).double(),torch.device('cpu'))

    @unittest.skipUnless(torch.backends.mps.is_available(),'Native MPS unavailable')
    def test_native_mps_ragged_batching(self):
        self.compare_batch(Network(cfg()),torch.device('mps'))
        self.compare_batch(Transformer(TFConfig(**vars(cfg()),memory_enabled=False)),torch.device('mps'))

    def test_prefill_buckets_match_individual_prefixes(self):
        model=Network(cfg()).double().eval()
        with torch.inference_mode():
            _,base,_=model(torch.tensor([[1,4,5,6,7]]));cases=[dict(id='a',query=[8,9]),dict(id='b',query=[10,11,12]),dict(id='c',query=[13,14])]
            actual=prefill(model,cases,base,torch.device('cpu'),16)
            for case in cases:
                expected=model(torch.tensor([case['query']]),base)[0][0,-1]
                torch.testing.assert_close(actual[case['id']][0],expected,atol=1e-12,rtol=1e-10)

    def test_official_main_excludes_adversarial_and_weights_nll_by_tokens(self):
        base=dict(status='completed',f1=.5,first_hit=False,first_answer_nll=2.,prediction='x',generated_tokens=5,hit_token_limit=False)
        rows=[base|dict(category=4,answer_loss_sum=4.,answer_tokens=2),
              base|dict(category=1,f1=1.,answer_loss_sum=12.,answer_tokens=3),
              base|dict(category=5,f1=1.,answer_loss_sum=0.,answer_tokens=0)]
        result=aggregate(rows)
        self.assertEqual(result['main_questions'],2);self.assertEqual(result['main_f1'],.75)
        self.assertEqual(result['answer_nll'],16/5);self.assertEqual(result['adversarial_refusal'],1.)


if __name__=='__main__':unittest.main()
