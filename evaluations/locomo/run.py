"""Evaluate frozen V3 checkpoints on pinned LoCoMo; no optimizer or paid judge."""
import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
import traceback
import unittest

import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from v07.common import PROJECT,RunLock,device_for,digest,file_hash,save,state_bytes,sync,write
from v09.train import StopRequest
from v3.train import build_models
from scripts.v2_console import Progress as BaseProgress
from .data import prepare,encoded_samples
from .scoring import score,normalize_answer

FORMAT='biomatrix-v3-locomo-1'
GROUPS=(('D-main','v3_dynamic','warm'),('D-clear','v3_dynamic','clear'),
        ('D-donor','v3_dynamic','donor'),('D-oracle','v3_dynamic','oracle'),
        ('S-main','v3_static','warm'),('S-oracle','v3_static','oracle'),
        ('L128','trans_local','warm'),('L-oracle','trans_local','oracle'),
        ('F512','trans_full','warm'),('F-oracle','trans_full','oracle'))


class Progress(BaseProgress):label='V3 LoCoMo'


def map_tree(value,fn):
    if isinstance(value,torch.Tensor):return fn(value)
    if isinstance(value,dict):return {k:map_tree(v,fn) for k,v in value.items()}
    if isinstance(value,list):return [map_tree(v,fn) for v in value]
    if isinstance(value,tuple):return tuple(map_tree(v,fn) for v in value)
    if value is None:return None
    raise ValueError('Unexpected model state')


def stack_states(states):
    first=states[0]
    if first is None:
        if any(x is not None for x in states):raise ValueError('Mixed absent/present cache')
        return None
    if isinstance(first,dict):return {k:stack_states([s[k] for s in states]) for k in first}
    if isinstance(first,list):return [stack_states([s[i] for s in states]) for i in range(len(first))]
    if isinstance(first,tuple):return tuple(stack_states([s[i] for s in states]) for i in range(len(first)))
    if any(x.shape[1:]!=first.shape[1:] for x in states):
        # Left-pad ragged oracle KV; validity masks hide every padded key.
        if first.ndim==4:
            size=max(x.shape[2] for x in states)
            states=[F.pad(x,(0,0,size-x.shape[2],0)) for x in states]
        elif first.ndim==2 and first.dtype==torch.bool:
            size=max(x.shape[1] for x in states)
            states=[F.pad(x,(size-x.shape[1],0),value=False) for x in states]
        else:raise ValueError('Unsupported ragged state tensor')
    return torch.cat(states,0)


def parameter_digest(model):
    h=hashlib.sha256()
    for p in model.parameters():h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def ingest(model,tokens,device,progress=lambda done,total:None,stop=None):
    state=None
    for at in range(0,len(tokens),512):
        if stop is not None and stop.requested:raise KeyboardInterrupt
        _,state,_=model(torch.tensor([tokens[at:at+512]],device=device,dtype=torch.long),state)
        progress(min(at+512,len(tokens)),len(tokens))
    return map_tree(state,lambda x:x.cpu())


@torch.inference_mode()
def prefill(model,cases,base,device,batch,oracle=False,progress=lambda text:None,stop=None):
    buckets=defaultdict(list);prepared={}
    for case in cases:
        tokens=case['oracle'] if oracle else case['query']
        if tokens is not None:buckets[len(tokens)].append(case)
    for length,rows in sorted(buckets.items()):
        for begin in range(0,len(rows),batch):
            if stop is not None and stop.requested:raise KeyboardInterrupt
            group=rows[begin:begin+batch]
            tokens=[r['oracle'] if oracle else r['query'] for r in group]
            state=None if oracle else map_tree(base,lambda x:x.expand(len(group),*x.shape[1:]).to(device))
            logits,state,_=model(torch.tensor(tokens,device=device,dtype=torch.long),state)
            for index,case in enumerate(group):
                prepared[case['id']]=(logits[index,-1].clone(),map_tree(state,lambda x,i=index:x[i:i+1].clone()))
            progress(f'问题编码 {len(prepared)}/{sum(len(v) for v in buckets.values())}')
    return prepared


@torch.inference_mode()
def answer_batch(model,cases,prepared,tokenizer,device,max_new_tokens):
    actual=[c for c in cases if c['id'] in prepared];output={}
    if not actual:return output
    states=stack_states([prepared[c['id']][1] for c in actual])
    first=torch.stack([prepared[c['id']][0] for c in actual])
    metrics=[dict(answer_nll=None,answer_loss_sum=0.,answer_tokens=0,first_answer_nll=None,first_hit=None) for _ in actual]
    longest=max(len(c['answer_ids']) for c in actual)
    if longest:
        targets=np.zeros((len(actual),longest),dtype=np.int64)
        for index,case in enumerate(actual):
            ids=case['answer_ids'];targets[index,:len(ids)]=ids
        target=torch.as_tensor(targets,device=device)
        logits=first[:,None]
        if longest>1:logits=torch.cat((logits,model(target[:,:-1],states)[0]),1)
        losses=F.cross_entropy(logits.float().flatten(0,1),target.flatten(),reduction='none').reshape_as(target).cpu().numpy()
        first_predictions=first.argmax(-1).cpu().tolist()
        for index,case in enumerate(actual):
            count=len(case['answer_ids'])
            if count:
                values=losses[index,:count]
                if not bool(np.isfinite(values).all()):raise FloatingPointError('Nonfinite answer score')
                metrics[index]=dict(answer_nll=float(values.mean()),answer_loss_sum=float(values.sum()),
                    answer_tokens=count,first_answer_nll=float(values[0]),first_hit=first_predictions[index]==case['answer_ids'][0])
    # Functional state updates keep teacher-forced labels out of free generation.
    logits=first;generated=[[] for _ in actual];finished=[False]*len(actual)
    for at in range(max_new_tokens):
        tokens=logits.argmax(-1).cpu().tolist()
        for index,token in enumerate(tokens):
            if not finished[index]:generated[index].append(token)
            finished[index]=finished[index] or token==2
        if all(finished) or at+1==max_new_tokens:break
        following=torch.tensor([[2 if done else token] for done,token in zip(finished,tokens)],device=device,dtype=torch.long)
        logits,states,_=model(following,states);logits=logits[:,-1]
    for index,case in enumerate(actual):
        prediction=tokenizer.decode(generated[index],skip_special_tokens=True).strip()
        output[case['id']]=dict(status='completed',prediction=prediction,generated_ids=generated[index],
            generated_tokens=len(generated[index]),hit_token_limit=(len(generated[index])==max_new_tokens and generated[index][-1]!=2),
            f1=score(prediction,case['qa']),**metrics[index])
    return output


def sources():
    files=[*sorted((PROJECT/'evaluations/locomo').glob('*.py')),PROJECT/'evaluations/locomo/requirements.txt',
           PROJECT/'evaluations/locomo/LICENSE.locomo.txt',PROJECT/'run_biomatrix_v3_locomo.py',
           PROJECT/'.run/Bio-Matrix V3 LoCoMo.run.xml']
    return {str(p.relative_to(PROJECT)):file_hash(p) for p in files}


def verify(root):
    output=io.StringIO();suite=unittest.defaultTestLoader.discover(str(PROJECT/'evaluations/locomo'),pattern='test_*.py',top_level_dir=str(PROJECT))
    result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
    (root/'verification.txt').write_text(output.getvalue())
    write(root/'verification.json',dict(status='completed' if result.wasSuccessful() else 'failed',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors)))
    if not result.wasSuccessful():raise RuntimeError('LoCoMo adapter self-check failed')


def run(args):
    device=device_for(args.device)
    if device.type=='mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise RuntimeError('Native MPS required')
    root=Path(args.resume or args.output or PROJECT/'results/locomo'/datetime.now().strftime('v3_%Y%m%dT%H%M%S_%f')).resolve()
    if not args.resume and root.exists() and any(root.iterdir()):raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True)
    ui=Progress([dict(id=g[0]) for g in GROUPS],1,not args.quiet);ui.device=device
    registry={g[0]:dict(status='not_executed',completed_questions=0) for g in GROUPS};current=None
    try:
        torch.set_num_threads(4);ui.stage('LoCoMo 评分／批量状态自检');verify(root);torch.set_num_threads(4)
        with RunLock(PROJECT/'results/locomo/.evaluate.lock'),StopRequest() as stop:
            samples,manifest=prepare(args.data_root,ui.stage)
            runroot=Path(args.run_root).resolve();training=json.loads((runroot/'protocol.json').read_text())
            for name,sha in training['source_hashes'].items():
                if file_hash(PROJECT/name)!=sha:raise ValueError(f'Frozen model source changed: {name}')
            data_root=Path(training['training']['data_root']);tokenizer_path=data_root/'tokenizer.json'
            corpus_manifest=json.loads((data_root/'manifest.json').read_text())
            if digest(corpus_manifest)!=training['data_manifest_sha256'] or file_hash(tokenizer_path)!=corpus_manifest['prepared_hashes']['tokenizer.json']:
                raise ValueError('Frozen tokenizer identity changed')
            tokenizer=Tokenizer.from_file(str(tokenizer_path));encoded=encoded_samples(samples,tokenizer,ui.stage)
            total=sum(len(s['cases']) for s in encoded)
            for bar in ui.details.values():bar.total=total
            models,_,counts=build_models(tokenizer.get_vocab_size(),training['training'],args.seed)
            checkpoints={name:runroot/f'{name}_s{args.seed}/last.pt' for name in models}
            protocol=dict(format=FORMAT,seed=args.seed,source_run=str(runroot),device=str(device),torch=str(torch.__version__),
                evaluator_sources=sources(),model_sources=training['source_hashes'],dataset=manifest,
                tokenizer_sha256=file_hash(tokenizer_path),checkpoint_sha256={n:file_hash(p) for n,p in checkpoints.items()},
                batch=args.batch,max_new_tokens=args.max_new_tokens,decode='greedy, EOS stop; no answer text in generation',
                groups=GROUPS,history='chronological dates/speakers/turn IDs and supplied image captions; no summaries or QA annotations',
                main_metric='Official stemmed word F1, categories 1-4; category5 raw-refusal diagnostic separate. No LLM judge.',
                query='Each question branches from the same history snapshot; no previous answers retained. Native question/answer formatting.',
                oracle='Annotated original evidence only, no gold answer; invalid/missing evidence is not_applicable. Transformer oracle window is extended to fit all selected evidence; diagnostic only.',
                donor='Other conversation token stream repeated/truncated to identical token count; clocks exactly matched.',
                planned_conversations=len(encoded),planned_questions=total,parameter_counts=counts,
                training='No parameter updates; one predeclared seed, all official conversations, no tuning on this evaluation.')
            if args.resume:
                if json.loads((root/'protocol.json').read_text())!=json.loads(json.dumps(protocol)):
                    raise ValueError('Evaluation resume protocol mismatch')
                registry=json.loads((root/'registry.json').read_text())['conditions']
            else:
                write(root/'protocol.json',protocol)
                for relative in protocol['evaluator_sources']:
                    target=root/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((PROJECT/relative).read_bytes())
            write(root/'data_audit.json',dict(conversations=[dict(id=s['id'],history_tokens=len(s['tokens']),questions=len(s['cases']),
                oracle_eligible=sum(c['oracle'] is not None for c in s['cases'])) for s in encoded],
                main_answerable_questions=sum(c['qa']['category']!=5 for s in encoded for c in s['cases'])))
            loaded=None;model=None
            for identity,name,mode in GROUPS:
                current=identity
                if stop.requested:raise KeyboardInterrupt
                if registry[identity]['status']=='completed':
                    ui.details[identity].n=total;ui.total.update(1);continue
                if loaded!=name:
                    if model is not None:model.to('cpu')
                    model=models[name]
                    payload=torch.load(checkpoints[name],map_location='cpu',weights_only=True)
                    if payload['protocol']!=training or payload['counters']['updates']!=1024:raise ValueError('Unexpected checkpoint')
                    model.load_state_dict(payload['model']);model.to(device).eval();del payload
                    loaded=name
                original_cfg=model.cfg
                if mode=='oracle':
                    max_input=max(len(c['oracle'] or [])+len(c['answer_ids'])+args.max_new_tokens for s in encoded for c in s['cases'])
                    if name.startswith('trans_'):model.cfg=replace(model.cfg,context_limit=max(model.cfg.context_limit,max_input))
                before=parameter_digest(model)
                registry[identity].update(status='running');write(root/'registry.json',dict(status='running',conditions=registry))
                started=time.perf_counter();folder=root/identity;folder.mkdir(exist_ok=True)
                timings=[]
                for ci,conversation in enumerate(encoded):
                    cases=conversation['cases'];batch_files=[folder/f'c{ci:02d}_b{at//args.batch:03d}.json' for at in range(0,len(cases),args.batch)]
                    if all(p.exists() for p in batch_files):
                        ui.details[identity].update(len(cases));continue
                    ui.stage(f'{identity} · 对话 {ci+1}/10 · 写入历史')
                    sync(device);start=time.perf_counter();base=None
                    if mode!='oracle':
                        if mode=='clear':
                            base=model.initial_state(1);base['position'].fill_(len(conversation['tokens']))
                            base=map_tree(base,lambda x:x.cpu())
                        else:
                            history=conversation['tokens']
                            if mode=='donor':
                                donor=encoded[(ci+1)%len(encoded)]['tokens']
                                history=(donor*((len(history)+len(donor)-1)//len(donor)))[:len(history)]
                            cache=root/'history_cache'/f'{name}_{mode}_c{ci:02d}.pt'
                            if cache.exists():base=torch.load(cache,map_location='cpu',weights_only=True)
                            else:
                                base=ingest(model,history,device,
                                    lambda done,total:ui.stage(f'{identity} · 对话{ci+1}/10 · 历史 {done}/{total} token'),stop)
                                cache.parent.mkdir(parents=True,exist_ok=True);save(cache,base)
                    sync(device);history_seconds=time.perf_counter()-start
                    prepared=prefill(model,cases,base,device,args.batch,mode=='oracle',
                        lambda text:ui.stage(f'{identity} · 对话{ci+1}/10 · {text}'),stop)
                    for at,path in zip(range(0,len(cases),args.batch),batch_files):
                        group=cases[at:at+args.batch]
                        if path.exists():ui.details[identity].update(len(group));continue
                        if stop.requested:raise KeyboardInterrupt
                        ui.stage(f'{identity} · 对话{ci+1}/10 · 生成答案 {at+1}-{at+len(group)}/{len(cases)}')
                        sync(device);begin=time.perf_counter()
                        predictions=answer_batch(model,group,prepared,tokenizer,device,args.max_new_tokens)
                        sync(device);seconds=time.perf_counter()-begin;rows=[]
                        for case in group:
                            row=dict(id=case['id'],conversation=conversation['id'],index=case['index'],category=case['qa']['category'],
                                question_sha256=case['question_sha256'],oracle_eligible=case['oracle'] is not None,
                                history_tokens=len(conversation['tokens']),question=case['qa']['question'],
                                reference_answer=case['qa'].get('answer'),evidence=case['qa'].get('evidence',[]))
                            row.update(predictions.get(case['id'],dict(status='not_applicable',reason=case['oracle_reason'])))
                            rows.append(row)
                        write(path,dict(rows=rows,seconds=seconds));ui.details[identity].update(len(group))
                        completed=[r for r in rows if r['status']=='completed' and r['category']!=5]
                        if completed:ui.details[identity].set_postfix_str(f'QA={np.mean([r["f1"] for r in completed]):.4f}',refresh=False)
                        registry[identity]['completed_questions']=int(ui.details[identity].n)
                        write(root/'registry.json',dict(status='running',conditions=registry))
                    timings.append(dict(conversation=conversation['id'],history_seconds=history_seconds,
                                        runtime_state_bytes=state_bytes(base),oracle_context_limit=model.cfg.context_limit))
                    del prepared,base
                if parameter_digest(model)!=before or file_hash(checkpoints[name])!=protocol['checkpoint_sha256'][name]:
                    raise RuntimeError('Frozen model changed during evaluation')
                write(folder/'timing.json',dict(seconds=time.perf_counter()-started,conversations=timings))
                registry[identity].update(status='completed',completed_questions=total);ui.total.update(1)
                model.cfg=original_cfg
                write(root/'registry.json',dict(status='running',conditions=registry))
            write(root/'registry.json',dict(status='completed',conditions=registry))
            from .report import report
            report(root)
            return root,'completed'
    except KeyboardInterrupt:
        if current and registry[current]['status']=='running':registry[current]['status']='aborted'
        write(root/'registry.json',dict(status='aborted',conditions=registry));return root,'aborted'
    except BaseException:
        if current and registry[current]['status']=='running':registry[current]['status']='failed'
        write(root/'registry.json',dict(status='failed',conditions=registry))
        (root/'failure.txt').write_text(traceback.format_exc());raise
    finally:ui.close()


def main(arguments=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,default=PROJECT/'results/v3/paired_20261103T094249_111075')
    parser.add_argument('--data-root',type=Path,default=PROJECT/'data/locomo')
    parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--device',choices=('auto','mps','cpu','cuda'),default='auto')
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--max-new-tokens',type=int,default=50)
    parser.add_argument('--output',type=Path);parser.add_argument('--resume',type=Path);parser.add_argument('--quiet',action='store_true')
    args=parser.parse_args(arguments)
    if args.batch<1 or args.max_new_tokens<1 or args.seed<0:parser.error('Positive batch/token limit and nonnegative seed required')
    root,status=run(args);print(f'V3 LoCoMo {status}: {root}')


if __name__=='__main__':main()
