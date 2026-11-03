"""Finite acceptance for operator equivalence, data boundaries and paired recovery."""
from dataclasses import asdict
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer,models,pre_tokenizers,trainers,decoders

from v3.model import Config,Network
from v07.common import digest,file_hash,write
from .model import Bio,Transformer,build,inspect,CONDITIONS
from . import data,engine,run


def config():return Config(vocab_size=32,width=16,layers=2,heads=2,ff=32,nodes=4,phase_bands=2,segment_tokens=4,context_limit=8)


def tokenizer():
    t=Tokenizer(models.BPE(unk_token='[UNK]'));t.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False);t.decoder=decoders.ByteLevel()
    t.train_from_iterator(['The answer is Paris.\nQuestion: Where?\nAnswer: Paris.'],trainers.BpeTrainer(
        vocab_size=280,show_progress=False,initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),special_tokens=['[UNK]']))
    return t


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):torch.manual_seed(71)

    def test_compact_checkpointed_bio_matches_frozen_outputs_and_gradients(self):
        cfg=config();old=Network(cfg).double();new=Bio(cfg,True).double();new.load_state_dict(old.state_dict())
        x=torch.randint(0,32,(2,13));target=torch.randint(0,32,(2,13))
        a=old(x)[0];b=new(x)[0];torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-10)
        for output in (a,b):torch.nn.functional.cross_entropy(output.flatten(0,1),target.flatten()).backward()
        for (name,p),(other,q) in zip(old.named_parameters(),new.named_parameters()):
            self.assertEqual(name,other);torch.testing.assert_close(p.grad,q.grad,atol=1e-10,rtol=1e-8,msg=name)
        with torch.no_grad():
            first,state,_=new(x[:,:3]);second,_,_=new(x[:,3:],state)
            torch.testing.assert_close(b,torch.cat((first,second),1),atol=1e-12,rtol=1e-10)

    def test_full_sdpa_matches_existing_causal_block_and_blocks_future(self):
        model=Transformer(config(),False).double();x=torch.randint(0,32,(2,8));hidden=model.embedding(x)
        actual,kv=model.full_block(model.blocks[0],hidden)
        expected,_=model.blocks[0](hidden,None,torch.zeros(2,dtype=torch.long),8)
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-10)
        changed=x.clone();changed[:,4:]=torch.randint(0,32,(2,4))
        torch.testing.assert_close(model(x)[0][:,:4],model(changed)[0][:,:4],atol=1e-12,rtol=1e-10)

    def test_known_clock_path_preserves_full_state_at_partial_and_complete_boundaries(self):
        old=Network(config()).double();new=Bio(config(),False).double();new.load_state_dict(old.state_dict())
        with torch.no_grad():
            for length in (1,4,8,13,16):
                x=torch.randint(0,32,(2,length));a,sa,da=old(x);b,sb,db=new(x)
                torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-10)
                torch.testing.assert_close(sa['position'],sb['position'])
                for field in ('memory','pending'):
                    for p,q in zip(sa[field],sb[field]):
                        for name in p:torch.testing.assert_close(p[name],q[name],atol=1e-12,rtol=1e-10)
                for name in da:torch.testing.assert_close(da[name],db[name],atol=1e-12,rtol=1e-10)

    def test_accumulation_weights_actual_targets_and_preserves_update(self):
        a=build('bio_dynamic',asdict(config()),11,True)
        b=build('bio_dynamic',asdict(config()),11,True);b.load_state_dict(a.state_dict())
        values=np.random.default_rng(5).integers(0,32,(3,9),dtype=np.int64)
        mask=np.ones((3,8),dtype=bool);mask[1,2:]=False;mask[2,5:]=False
        oa=engine.optimizer(a,1e-3,.01,torch.device('cpu'));ob=engine.optimizer(b,1e-3,.01,torch.device('cpu'))
        ra=engine.update(a,oa,values,mask,3,torch.device('cpu'),'fp32',1.)
        rb=engine.update(b,ob,values,mask,1,torch.device('cpu'),'fp32',1.)
        self.assertEqual(ra['targets'],15);self.assertAlmostEqual(ra['ce'],rb['ce'],places=5)
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,atol=3e-6,rtol=2e-4)

    def test_bf16_cpu_autocast_state_and_gradients_remain_finite(self):
        for name in CONDITIONS:
            model=build(name,asdict(config()),11,True);opt=engine.optimizer(model,1e-3,.01,torch.device('cpu'))
            values=np.random.default_rng(6).integers(0,32,(1,9),dtype=np.int64)
            result=engine.update(model,opt,values,np.ones((1,8),dtype=bool),1,torch.device('cpu'),'bf16',1.)
            self.assertTrue(np.isfinite(result['ce']))

    @unittest.skipUnless(torch.backends.mps.is_available(),'Native MPS unavailable')
    def test_native_mps_real_forward_backward_and_checkpointing(self):
        for name in CONDITIONS:
            model=build(name,asdict(config()),11,True).to('mps');opt=engine.optimizer(model,1e-3,.01,torch.device('mps'))
            values=np.random.default_rng(7).integers(0,32,(1,9),dtype=np.int64)
            result=engine.update(model,opt,values,np.ones((1,8),dtype=bool),1,torch.device('mps'),'fp32',1.)
            self.assertTrue(np.isfinite(result['ce']))

    def test_shared_initialization_and_production_parameter_match(self):
        spec=asdict(config());models_={n:build(n,spec,11,False) for n in CONDITIONS}
        for model in models_.values():torch.testing.assert_close(model.embedding.weight,models_['bio_dynamic'].embedding.weight)
        self.assertFalse(models_['bio_static'].cfg.dynamic_addresses)
        cfg=json.loads(run.DEFAULT_CONFIG.read_text());counts=inspect(cfg['model'])['parameters']
        self.assertEqual(counts['bio_dynamic'],133284672);self.assertEqual(counts['transformer'],133281168)

    def test_uint16_pack_cross_file_and_exact_partial_budget(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);files=[]
            for i,values in enumerate(([1,2,3,4],[5,6,7,8,9])):
                name=f'{i}.bin';np.array(values,dtype='<u2').tofile(root/name);files.append(dict(path=name,split='train'))
            stream=data.Tokens(root,dict(files=files),'train')
            np.testing.assert_array_equal(stream.read(2,5),[3,4,5,6,7])
            values,mask=run.lm_batch(stream,np.array([1,0]),0,2,4,6)
            np.testing.assert_array_equal(values,[[5,6,7,8,9],[1,2,3,4,5]]);self.assertEqual(mask.sum(),6)
            with self.assertRaises(IndexError):stream.read(8,2)

    def test_evidence_crop_and_answer_eos_mask(self):
        t=tokenizer();paragraph=dict(context='One long sentence. The answer is Paris. Another long sentence.')
        qa=dict(id='q1',question='Where?',answers=[dict(text='Paris',answer_start=paragraph['context'].index('Paris'))])
        row=data.qa_example(paragraph,qa,t,128);self.assertIsNotNone(row)
        self.assertIn('Paris',t.decode(row['values'][1:row['prompt_tokens']]))
        self.assertEqual(row['values'][row['end']],data.EOS)
        self.assertEqual(row['end']-row['start'],len(t.encode(' Paris').ids)+1)
        bad=qa|dict(answers=[dict(text='Paris',answer_start=0)])
        with self.assertRaises(ValueError):data.qa_example(paragraph,bad,t,128)

    def test_real_parquet_fixture_hash_split_and_bounded_packing(self):
        texts={}
        for i in range(50000):
            text=f'Original document number {i}. Good words and ordinary words.';texts.setdefault(data.document_split(text),text)
            if len(texts)==3:break
        with TemporaryDirectory() as folder:
            root=Path(folder);path=root/'source.parquet';pq.write_table(pa.table({'text':[texts['validation'],texts['test'],texts['train']]}),path)
            result=data.pack_shard(path,root,0,tokenizer(),10,dict(validation=20,test=20))
            self.assertEqual(result['counts'],dict(train=10,validation=20,test=20))
            for row in result['files']:self.assertEqual((root/row['path']).stat().st_size,2*row['tokens'])
            self.assertEqual(data.document_split(texts['train']),data.document_split(texts['train']))

    def test_bad_download_never_replaces_verified_target(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';source.write_bytes(b'wrong');target=root/'target'
            with self.assertRaises(ValueError):data.download(source.as_uri(),target,'0'*64,5)
            self.assertFalse(target.exists());self.assertTrue(target.with_suffix('.part').exists())

    def test_complete_partial_download_recovers_without_network(self):
        with TemporaryDirectory() as folder:
            target=Path(folder)/'target';target.with_suffix('.part').write_bytes(b'verified')
            data.download('https://invalid.example/unused',target,hashlib.sha256(b'verified').hexdigest(),8)
            self.assertEqual(target.read_bytes(),b'verified');self.assertFalse(target.with_suffix('.part').exists())

    def test_insufficient_disk_stops_before_any_dataset_download(self):
        with TemporaryDirectory() as folder,patch('scale.data.shutil.disk_usage',return_value=SimpleNamespace(free=1)),patch('scale.data.sources') as fetch:
            with self.assertRaisesRegex(OSError,'GiB'):data.prepare(folder,10_000_000_000,2048)
            fetch.assert_not_called()

    def fixture_data(self,root,c):
        root.mkdir();t=Tokenizer(models.WordLevel({'<unk>':0,**{str(i):i for i in range(1,32)}},unk_token='<unk>'));t.save(str(root/'tokenizer.json'))
        source=dict(fixture=True,tokenizer_sha256=file_hash(root/'tokenizer.json'));write(root/'sources.json',source)
        files=[]
        for split in ('train','validation','test'):
            path=root/f'{split}.bin';(np.arange(65)%31+1).astype('<u2').tofile(path)
            files.append(dict(path=path.name,split=split,tokens=65,sha256=file_hash(path)))
        qa=root/'qa';qa.mkdir();qfiles={}
        for split in ('train','validation'):
            np.save(qa/f'{split}.npy',np.array([[1,3,4,5,6,2,2,2,2],[1,7,8,9,10,2,2,2,2]],dtype=np.uint16))
            rows=[dict(question_id=f'q{i}',article_id=f'a{i}',start=3,end=5,prompt_tokens=4,reference_answers=['6'],evidence_tokens=2) for i in range(2)]
            (qa/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            for name in (f'{split}.npy',f'{split}.jsonl'):qfiles[name]=file_hash(qa/name)
        write(qa/'manifest.json',dict(files=qfiles,sequence=8,tokenizer_fingerprint=digest(json.loads(t.to_str()))))
        write(root/'manifest.json',dict(status='completed',source_manifest_sha256=digest(source),files=files,
            preparation=dict(tokens=c['pretrain_tokens'],sequence=c['sequence']),qa_manifest_sha256=file_hash(qa/'manifest.json')))

    def test_three_way_milestones_sft_and_stop_resume_match_uninterrupted(self):
        c=json.loads(run.DEFAULT_CONFIG.read_text())|dict(model=asdict(config()),pretrain_tokens=24,milestones=[8,16,24],sequence=8,
            microbatch=1,accumulation=2,warmup_tokens=8,sft_updates=2,validation_sequences=2,qa_eval_examples=2,checkpoint_every=1,eval_every=1)
        with TemporaryDirectory() as folder:
            root=Path(folder);d=root/'data';self.fixture_data(d,c)
            a=root/'a';b=root/'b';a.mkdir();b.mkdir();device=torch.device('cpu')
            def ui():
                records=[dict(id=f'{n}-{p}') for n in CONDITIONS for p in ('pretrain','sft')]
                return run.Progress(records,24,False)
            def execute(output,resume,stop):
                args=SimpleNamespace(data_root=d,resume=output if resume else None)
                run.train(args,c,device,'fp32',output,ui(),stop)
            with patch('scale.run.shutil.disk_usage',return_value=SimpleNamespace(free=100*2**30)):
                execute(a,False,SimpleNamespace(requested=False))
                stop=SimpleNamespace(requested=False);original=engine.update
                def once(*args,**kwargs):
                    result=original(*args,**kwargs);stop.requested=True;return result
                with patch('scale.run.engine.update',side_effect=once):
                    with self.assertRaises(KeyboardInterrupt):execute(b,False,stop)
                self.assertEqual(json.loads((b/'registry.json').read_text())['status'],'aborted')
                execute(b,True,SimpleNamespace(requested=False))
            for name in CONDITIONS:
                x=torch.load(a/name/'last.pt',weights_only=True);y=torch.load(b/name/'last.pt',weights_only=True)
                self.assertEqual(x['counters']['pretrain_tokens'],24);self.assertEqual(x['counters']['sft_updates'],2)
                for key,value in x['model'].items():torch.testing.assert_close(value,y['model'][key],atol=0,rtol=0,msg=key)
                logs=[json.loads(z) for z in (b/name/'metrics.jsonl').read_text().splitlines()]
                self.assertEqual(len(logs),5)
                for target in (8,16,24):self.assertTrue((b/name/f'tokens_{target}.pt').exists())
            paired=[]
            for name in CONDITIONS:paired.append([json.loads(x)['batch_sha256'] for x in (b/name/'metrics.jsonl').read_text().splitlines()])
            self.assertEqual(paired[0],paired[1]);self.assertEqual(paired[1],paired[2])
            self.assertEqual(json.loads((b/'summary.json').read_text())['status'],'completed')


if __name__=='__main__':unittest.main()
