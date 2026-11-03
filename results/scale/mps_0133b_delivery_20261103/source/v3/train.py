"""V3 standalone vector-node Bio network versus from-scratch Transformers."""
import argparse
import io
import json
import math
import os
from pathlib import Path
import random
import time
import traceback
import unittest
from dataclasses import asdict, replace
from datetime import datetime

import numpy as np
import torch
from torch.nn import functional as F

from v07.common import PROJECT, RunLock, detach, device_for, digest, file_hash, save, state_bytes, sync, write
from v09.benchmark import append
from .console import Progress
from v09.train import StopRequest, capture_rng, restore_random, learning_rate_scale
from v1.train import load_corpus, optimizer_for, scalar_diagnostics
from v11.data import prepare_qa
from v11.evaluate import _aggregate, _scoring_masks
from .model import Config, Network
from .evaluate import evaluate_history

FORMAT = 'bio-matrix-v3-distributed-vectors-1'
CONDITIONS = ('trans_local','trans_full','v3_static','v3_dynamic')
DEFAULTS = dict(width=56,layers=2,heads=4,ff=272,
                data_root=str(PROJECT/'data/v06/wikitext2'),qa_root=str(PROJECT/'data/v11/squad_remote256'),
                seeds='11,23,37',cpt_tokens=1_048_576,pt_updates=512,batch=4,segments=8,
                segment_tokens=64,context_limit=128,nodes=32,phase_bands=8,
                data_seed=22001,evaluation_seed=22091,validation_windows=64,
                pt_eval_every=256,checkpoint_every=16,learning_rate=3e-4,
                weight_decay=.01,gradient_clip=1.,threads=4)


def seed_list(value):
    try: values=[int(x.strip()) for x in value.split(',')]
    except (ValueError,AttributeError): raise ValueError('seeds must be comma-separated integers') from None
    if not values or min(values)<0 or len(set(values))!=len(values):
        raise ValueError('Need distinct nonnegative seeds')
    return values


def settings(args, old=None):
    c={}
    for name,default in DEFAULTS.items():
        previous=default if old is None else old[name]
        supplied=getattr(args,name,None)
        if old is not None and supplied is not None and supplied!=previous:
            raise ValueError(f'Resume cannot change {name}')
        c[name]=previous if supplied is None else supplied
    for name in ('data_root','qa_root'): c[name]=str(Path(c[name]).resolve())
    seed_list(c['seeds'])
    for name in DEFAULTS:
        if type(DEFAULTS[name]) is int and (type(c[name]) is not int or c[name]<1):
            raise ValueError(f'{name} must be a positive integer')
    for name in ('learning_rate','weight_decay','gradient_clip'):
        if not math.isfinite(c[name]) or c[name]<0: raise ValueError(f'Invalid {name}')
    if min(c['learning_rate'],c['gradient_clip'])<=0: raise ValueError('Positive lr and clip required')
    if c['segments']<4 or c['segment_tokens']>c['context_limit']:
        raise ValueError('Need >=4 blocks with block length <= local window')
    return c


def source_hashes():
    paths=[*sorted((PROJECT/'v3').glob('*.py')),*sorted((PROJECT/'v2').glob('*.py')),*sorted((PROJECT/'v11').glob('*.py')),
           *sorted((PROJECT/'v1').glob('*.py')),PROJECT/'run_biomatrix_v3.py',
           PROJECT/'.run/Bio-Matrix V3.run.xml',PROJECT/'scripts/v2_console.py']
    paths += [PROJECT/p for p in ('v03/graph.py','v04/model.py','v05/model.py','v06/model.py',
              'v06/common.py','v06/graph.py','v07/common.py','v07/data.py','v07/model.py',
              'v07/run.py','v07/graph.py','v07/evaluate.py','v08/benchmark.py','v08/model.py',
              'v08/recurrence.py','v09/benchmark.py','v09/train.py')]
    return {str(p.relative_to(PROJECT)):file_hash(p) for p in paths if p.is_file()}


def build_models(vocab_size,c,seed):
    from v2.model import Config as TransformerConfig, Network as Transformer
    cfg=Config(vocab_size=vocab_size,**{k:c[k] for k in ('width','layers','heads','ff','context_limit',
               'segment_tokens','nodes','phase_bands')})
    torch.manual_seed(seed)
    dynamic=Network(cfg)
    static=Network(replace(cfg,dynamic_addresses=False))
    static.load_state_dict(dynamic.state_dict())
    target=sum(p.numel() for p in dynamic.parameters())
    base_cfg=TransformerConfig(**asdict(cfg),memory_enabled=False)
    torch.manual_seed(seed+1000)
    baseline=Transformer(base_cfg)
    count=sum(p.numel() for p in baseline.parameters())
    ff=max(1,cfg.ff+round((target-count)/(cfg.layers*(2*cfg.width+1))))
    base_cfg=replace(base_cfg,ff=ff)
    torch.manual_seed(seed+1000)
    local=Transformer(base_cfg)
    full=Transformer(replace(base_cfg,context_limit=c['segments']*c['segment_tokens']))
    with torch.no_grad():
        local.embedding.weight.copy_(dynamic.embedding.weight)
        local.norm.load_state_dict(dynamic.norm.state_dict())
        for source,dest in zip(dynamic.layers,local.blocks):
            dest.norm2.load_state_dict(source.norm.state_dict())
            shared=min(cfg.ff,ff)
            dest.ff[0].weight[:shared].copy_(source.ff[0].weight[:shared])
            dest.ff[0].bias[:shared].copy_(source.ff[0].bias[:shared])
            dest.ff[2].weight[:,:shared].copy_(source.ff[2].weight[:,:shared])
            dest.ff[2].bias.copy_(source.ff[2].bias)
            if ff>shared: dest.ff[2].weight[:,shared:].zero_()
    full.load_state_dict(local.state_dict())
    models=dict(trans_local=local,trans_full=full,v3_static=static,v3_dynamic=dynamic)
    counts={n:dict(total_parameters=sum(p.numel() for p in m.parameters()),
                   trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),
                   context_limit=m.cfg.context_limit,explicit_kv_tokens=(m.cfg.context_limit if n.startswith('trans_') else 0),
                   runtime_state='token KV' if n.startswith('trans_') else 'vector key/value/phase only',
                   parameter_difference=sum(p.numel() for p in m.parameters())-target) for n,m in models.items()}
    if any(row['total_parameters']>=1_000_000_000 for row in counts.values()):
        raise ValueError('All models must remain below 1B parameters')
    return models,cfg,counts


def update(model,optimizer,values,mask,c,device):
    x=torch.as_tensor(values[:,:-1].copy(),device=device,dtype=torch.long)
    y=torch.as_tensor(values[:,1:].copy(),device=device,dtype=torch.long)
    selected=torch.as_tensor(mask.copy(),device=device,dtype=torch.bool)
    if selected.shape!=y.shape or not bool(selected.any()): raise ValueError('Empty or invalid target mask')
    model.train();optimizer.zero_grad(set_to_none=True)
    sync(device);started=time.perf_counter()
    logits,state,diagnostics=model(x)
    loss=F.cross_entropy(logits.float().flatten(0,1),y.flatten(),reduction='none').reshape_as(y)[selected].mean()
    if not bool(loss.detach().isfinite()): raise FloatingPointError('Nonfinite training loss')
    loss.backward()
    gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip'],error_if_nonfinite=True)
    optimizer.step();state=detach(state)
    sync(device);elapsed=time.perf_counter()-started
    targets=int(selected.sum().cpu())
    return dict(ce=float(loss.detach().cpu()),gradient_norm=float(gradient.detach().cpu()),
                targets=targets,inputs=x.numel(),seconds=elapsed,targets_per_second=targets/max(elapsed,1e-9),
                input_tokens_per_second=x.numel()/max(elapsed,1e-9),diagnostics=scalar_diagnostics(diagnostics)),state


@torch.no_grad()
def warm_score(model,values,mask,first,c,device,progress=None):
    prefix=(c['segments']-2)*c['segment_tokens']
    masks,first_masks=_scoring_masks(values,prefix,mask,first)
    sums,counts,fsums,fcounts,correct=[],[],[],[],[]
    previous=model.training;model.eval();sync(device);started=time.perf_counter()
    try:
        for begin in range(0,len(values),c['batch']):
            rows=values[begin:begin+c['batch']]
            x=torch.as_tensor(rows.copy(),device=device,dtype=torch.long)
            _,state,_=model(x[:,:prefix])
            logits,_,_=model(x[:,prefix:-1],state)
            target=x[:,prefix+1:]
            losses=F.cross_entropy(logits.float().flatten(0,1),target.flatten(),reduction='none').reshape_as(target).cpu().numpy()
            hits=(logits.argmax(-1)==target).cpu().numpy()
            for i in range(len(rows)):
                m,f=masks[begin+i],first_masks[begin+i]
                if not np.isfinite(losses[i,m]).all(): raise FloatingPointError('Nonfinite validation')
                sums.append(float(losses[i,m].sum()));counts.append(int(m.sum()))
                fsums.append(float(losses[i,f].sum()));fcounts.append(int(f.sum()));correct.append(int(hits[i,f].sum()))
            if progress:progress(min(begin+c['batch'],len(values)),len(values))
        warm=_aggregate(sums,counts,fsums,fcounts,correct)
        sync(device)
        return dict(conditions={'warm':warm},warm_nll=warm['nll'],warm_ppl=warm['ppl'],
                    first_answer_nll=warm['first_answer_nll'],first_answer_accuracy=warm['first_answer_accuracy'],
                    seconds=time.perf_counter()-started,scope='warm_only')
    finally: model.train(previous)


def validation_data(corpus,qa,c):
    length=c['segments']*c['segment_tokens']
    lm,docs,starts=corpus.sample('validation',np.random.default_rng(c['evaluation_seed']),c['validation_windows'],length)
    q,mask,meta=qa.sample('validation',np.random.default_rng(c['evaluation_seed']),c['validation_windows'])
    return (lm,q,mask,meta['first_answer_mask']),dict(
        lm=dict(documents=docs,starts=starts,sha256=digest(lm.tolist())),
        qa={k:v for k,v in meta.items() if k!='first_answer_mask'})


def evaluate(model,data,c,device,full=False,progress=None):
    lm,qa,mask,first=data
    result={}
    for task,values,selected,initial in (('lm',lm,None,None),('qa',qa,mask,first)):
        if progress: progress(f'{task} '+('完整历史干预' if full else '正常历史评分'))
        if full:
            result[task]=evaluate_history(model,values,device,(c['segments']-2)*c['segment_tokens'],
                         batch=c['batch'],seed=c['evaluation_seed'],target_mask=selected,first_answer_mask=initial,
                         progress=(lambda stage,done,total,t=task:progress(f'{t}/{stage} {done}/{total}')) if progress else None)
        else: result[task]=warm_score(model,values,selected,initial,c,device,
                         progress=(lambda done,total,t=task:progress(f'{t}/正常历史 {done}/{total}')) if progress else None)
    return result


def verify(root):
    output=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(PROJECT/'v3'),pattern='test_*.py',top_level_dir=str(PROJECT))
    result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
    record=dict(status='completed' if result.wasSuccessful() else 'failed',tests=result.testsRun,
                failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped))
    (root/'verification.txt').write_text(output.getvalue());write(root/'verification.json',record)
    if not result.wasSuccessful(): raise RuntimeError('v3 self-check failed; see verification.txt')
    return record


def save_checkpoint(path,model,optimizer,counters,rng,protocol,device):
    if source_hashes()!=protocol['source_hashes']: raise RuntimeError('Sources changed; retain checkpoint and use a new run')
    save(path,dict(format=FORMAT,protocol=protocol,model=model.state_dict(),optimizer=optimizer.state_dict(),
                   counters=counters.copy(),data_rng=rng.bit_generator.state,rng=capture_rng(device)))


def restore_checkpoint(path,model,optimizer,rng,protocol,device):
    p=torch.load(path,map_location='cpu',weights_only=True)
    if p.get('format')!=FORMAT or p['protocol']!=protocol: raise ValueError('Resume protocol mismatch')
    model.load_state_dict(p['model']);optimizer.load_state_dict(p['optimizer'])
    rng.bit_generator.state=p['data_rng'];restore_random(p['rng'],device)
    return dict(p['counters'])


def train(args,verify_first=True):
    device=device_for(getattr(args,'device','auto'))
    if device.type=='mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':
        raise RuntimeError('Disable MPS fallback for native benchmarking')
    resume=Path(args.resume).resolve() if getattr(args,'resume',None) else None
    if resume and resume.suffix=='.pt':
        if resume.name!='last.pt': raise ValueError('Use the last safe checkpoint')
        resume=resume.parent.parent
    previous=json.loads((resume/'protocol.json').read_text()) if resume else None
    c=settings(args,previous['training'] if previous else None)
    if resume and getattr(args,'output',None) and Path(args.output).resolve()!=resume:
        raise ValueError('Resume output must remain the same')
    root=resume or Path(getattr(args,'output',None) or PROJECT/'results/v3'/datetime.now().strftime('paired_%Y%m%dT%H%M%S_%f')).resolve()
    if not resume and root.exists() and any(root.iterdir()): raise FileExistsError(f'Refusing to overwrite {root}')
    root.mkdir(parents=True,exist_ok=True)
    records=[dict(id=f'{n}_s{s}',condition=n,seed=s) for s in seed_list(c['seeds']) for n in CONDITIONS]
    work=c['cpt_tokens']+c['pt_updates']*c['batch']*c['segments']*c['segment_tokens']
    ui=Progress(records,work,not getattr(args,'quiet',False));ui.total.set_description_str('v3 已完成条件');ui.device=device
    registry={r['id']:dict(status='not_executed',completed_work=0) for r in records}
    if resume: registry=json.loads((root/'registry.json').read_text())['conditions']
    owns=False
    try:
        if verify_first:
            checks=root if not resume else root/'resume_attempts'/datetime.now().strftime('%Y%m%dT%H%M%S_%f')
            checks.mkdir(parents=True,exist_ok=True);ui.stage('v3 代码自检');verify(checks)
        with RunLock(PROJECT/'results/v3/.train.lock'),StopRequest() as stop:
            torch.set_num_threads(c['threads']);ui.stage('冻结公开数据与分词器校验')
            corpus=load_corpus(c['data_root'])
            length=c['segments']*c['segment_tokens']
            qa=prepare_qa(c['qa_root'],corpus,length=length,prefix=(c['segments']-2)*c['segment_tokens'],
                          context_limit=c['context_limit'],backbone_layers=c['layers'],allow_download=not bool(resume))
            models,cfg,counts=build_models(corpus.manifest['vocab_size'],c,seed_list(c['seeds'])[0])
            protocol=dict(format=FORMAT,training=c,model=asdict(cfg),parameter_counts=counts,
                          source_hashes=source_hashes(),data_manifest_sha256=digest(corpus.manifest),
                          qa_manifest_sha256=digest(qa.manifest),
                          device=str(device),torch=str(torch.__version__),dtype='float32',
                          comparison='All four models train from scratch. Shared token embeddings and compatible pointwise FFN tensors; same examples, masks and optimizer. Transformer FFN width matches parameter count, not runtime state or FLOPs.',
                          context='Bio has no token attention or KV cache. Transformer local window is the remote-evidence reference; full Transformer retains the complete sampled window.',
                          update='Block-snapshot joint-error vector gradients. Exclusive causal dual matrix products expose earlier value writes within each block. Keys and phase commit only at block boundaries. Normalize and commit at fixed block boundaries. No node-internal square matrix or free-direction matrix.',
                          capacity='Fixed node pool; each node has key/value vectors and phase pairs. Rank-one distributed readout. Reduced per-node capacity; no exact equivalence to arbitrary full matrices and no node birth.',
                          phase='Included in all Bio conditions. Dynamic keys and circular phase receive bounded joint-error directions; static freezes runtime keys and phase but trains their anchors. No autonomous rotation.',
                          evaluation='Warm-only initial/CPT/mid-PT, paired final interventions once. Remote replacement uses the same reference span across Bio and local Transformer. Reserved test untouched.',
                          budget='CPT progress counts supervised targets; PT progress counts input slots including PAD. Actual answer targets logged separately.')
            if previous is not None and previous!=protocol: raise ValueError('Changed protocol; choose a new output')
            data,sample=validation_data(corpus,qa,c)
            if resume:
                for name,row in registry.items():
                    folder=root/name
                    if row['status']!='completed' and (row.get('updates',0)>0 or (folder/'metrics.jsonl').exists()):
                        if not (folder/'last.pt').exists() or not (folder/'initial_validation.json').exists():
                            raise FileNotFoundError(f'Missing safe checkpoint or t=0 score for {name}')
            if previous is None:
                write(root/'protocol.json',protocol);write(root/'validation_sample.json',sample)
                for relative in protocol['source_hashes']:
                    dst=root/'source'/relative;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((PROJECT/relative).read_bytes())
            owns=True
            current_seed=seed_list(c['seeds'])[0]
            for record in records:
                identity,condition,seed=record['id'],record['condition'],record['seed']
                if stop.requested: break
                if registry[identity]['status']=='completed':
                    ui.details[identity].update(registry[identity]['completed_work']);ui.total.update(1);continue
                if seed!=current_seed:
                    models,_,_=build_models(corpus.manifest['vocab_size'],c,seed);current_seed=seed
                model=models[condition].to(device);optimizer=optimizer_for(model,c)
                rng=np.random.default_rng(c['data_seed']+seed)
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                counters=dict(updates=0,cpt_tokens=0,pt_updates=0,answer_targets=0,completed_work=0,
                              input_tokens=0,train_seconds=0.,validation_seconds=0.)
                folder=root/identity;folder.mkdir(exist_ok=True);last=folder/'last.pt'
                if resume and last.exists():
                    counters=restore_checkpoint(last,model,optimizer,rng,protocol,device)
                    for name in ('metrics.jsonl','validation.jsonl'):
                        f=folder/name
                        if f.exists():
                            rows=[json.loads(x) for x in f.read_text().splitlines()]
                            kept=[row for row in rows if row['updates']<=counters['updates']]
                            if len(kept)!=len(rows):
                                f.rename(folder/f'{name}.pre_resume-{time.time_ns()}')
                                f.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in kept))
                ui.details[identity].update(counters['completed_work'])
                registry[identity]=dict(status='running',**counters)
                write(root/'registry.json',dict(status='running',conditions=registry))
                def assess(stage,full=False):
                    def progress(text):ui.stage(f'v3 {identity} · {stage} · {text}')
                    scores=evaluate(model,data,c,device,full,progress)
                    counters['validation_seconds']+=sum(v['seconds'] for v in scores.values())
                    append(folder/'validation.jsonl',dict(**counters,stage=stage,**scores))
                    return scores
                save_checkpoint(last,model,optimizer,counters,rng,protocol,device)
                if not (folder/'initial_validation.json').exists():
                    write(folder/'initial_validation.json',assess('initial'))
                while counters['completed_work']<work and not stop.requested:
                    stage='cpt' if counters['cpt_tokens']<c['cpt_tokens'] else 'pt'
                    if stage=='pt' and counters['pt_updates']==0 and not (folder/'cpt_validation.json').exists():
                        write(folder/'cpt_validation.json',assess('cpt_end'))
                    ui.stage(f'v3 {identity} · {stage.upper()} · 从零配对训练')
                    if stage=='cpt':
                        values,docs,starts=corpus.sample('train',rng,c['batch'],length)
                        mask=np.arange(c['batch']*length).reshape(c['batch'],length)<c['cpt_tokens']-counters['cpt_tokens']
                        meta=dict(documents=docs,starts=starts)
                        done=counters['cpt_tokens']+int(mask.sum());budget=c['cpt_tokens']
                    else:
                        values,mask,meta=qa.sample('train',rng,c['batch']);meta={k:v for k,v in meta.items() if k!='first_answer_mask'}
                        done=counters['pt_updates']+1;budget=c['pt_updates']
                    lr=c['learning_rate']*learning_rate_scale(done,budget)
                    for group in optimizer.param_groups:group['lr']=lr
                    row,state=update(model,optimizer,values,mask,c,device)
                    counters['updates']+=1;counters['input_tokens']+=row['inputs'];counters['train_seconds']+=row['seconds']
                    if stage=='cpt':
                        counters['cpt_tokens']+=row['targets'];advanced=row['targets']
                        stage_end=counters['cpt_tokens']==c['cpt_tokens']
                    else:
                        counters['pt_updates']+=1;counters['answer_targets']+=row['targets'];advanced=row['inputs']
                        stage_end=counters['pt_updates']==c['pt_updates']
                    counters['completed_work']+=advanced
                    row.update(counters,stage=stage,learning_rate=lr,runtime_bytes=state_bytes(state),sample=meta,
                               sample_sha256=digest(dict(values=values.tolist(),mask=mask.tolist())))
                    append(folder/'metrics.jsonl',row)
                    ui.details[identity].update(advanced)
                    ui.details[identity].set_postfix_str(f'{stage.upper()} CE={row["ce"]:.4f} 输入/s={row["input_tokens_per_second"]:.0f} 答案目标={counters["answer_targets"]}',refresh=False)
                    due=stage_end or (stage=='pt' and counters['pt_updates']%c['pt_eval_every']==0)
                    if stage=='cpt' and stage_end and not stop.requested:
                        write(folder/'cpt_validation.json',assess('cpt_end'))
                    elif stage=='pt' and due and not stage_end and not stop.requested:assess('pt_mid')
                    registry[identity]=dict(status='running',**counters,stage=stage,ce=row['ce'])
                    write(root/'live.json',dict(condition=identity,**registry[identity]))
                    if counters['updates']%c['checkpoint_every']==0 or due or stop.requested:
                        save_checkpoint(last,model,optimizer,counters,rng,protocol,device)
                        write(root/'registry.json',dict(status='running',conditions=registry))
                finished=counters['completed_work']==work and not stop.requested
                result=dict(status='completed' if finished else 'aborted',**counters,**counts[condition],seed=seed,condition=condition)
                if finished:
                    result['validation']=assess('final',full=True)
                    result['validation_seconds']=counters['validation_seconds']
                save_checkpoint(last,model,optimizer,counters,rng,protocol,device)
                write(folder/'result.json',result);registry[identity]=result
                ui.total.update(1);model.to('cpu')
                write(root/'registry.json',dict(status='running',conditions=registry))
                if not finished:break
            status='completed' if all(row['status']=='completed' for row in registry.values()) else 'aborted'
            summary=dict(status=status,conditions=registry,test_evaluated=False,protocol='protocol.json')
            write(root/'registry.json',summary);write(root/'summary.json',summary)
            from .report import report
            report(root)
            return root,status
    except BaseException:
        if resume and not owns:
            attempts=root/'resume_attempts';attempts.mkdir(exist_ok=True)
            (attempts/f'failure_{time.time_ns()}.txt').write_text(traceback.format_exc())
        else:
            (root/'failure.txt').write_text(traceback.format_exc())
            for row in registry.values():
                if row['status']=='running':row['status']='failed'
            write(root/'registry.json',dict(status='failed',conditions=registry))
        raise
    finally:ui.close()


def main(arguments=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('train','verify'),default='train')
    parser.add_argument('--device',choices=('auto','cpu','mps','cuda'),default='auto')
    parser.add_argument('--output',type=Path);parser.add_argument('--resume',type=Path)
    parser.add_argument('--quiet',action='store_true')
    for name,default in DEFAULTS.items():parser.add_argument('--'+name.replace('_','-'),type=type(default),default=None)
    args=parser.parse_args(arguments)
    if args.mode=='verify':
        root=Path(args.output or PROJECT/'results/v3'/datetime.now().strftime('verify_%Y%m%dT%H%M%S_%f'))
        if root.exists() and any(root.iterdir()):raise FileExistsError(root)
        root.mkdir(parents=True,exist_ok=True);record=verify(root)
        print(f'v3 自检 {record["status"]}: {root}')
    else:
        root,status=train(args);print(f'v3 配对实验 {status}: {root}')


if __name__=='__main__':main()
