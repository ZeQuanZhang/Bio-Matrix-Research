"""Replay the original local 333K data/budgets with public Qwen and DS architectures."""
import argparse
from datetime import datetime
from importlib.metadata import version
import inspect
import io
import json
from pathlib import Path
import time
import traceback
import unittest

import numpy as np
import torch
from torch.nn import functional as F
from transformers.utils import logging

from scripts.v2_console import Progress as BaseProgress
from v07.common import PROJECT,RunLock,device_for,digest,file_hash,save,state_bytes,sync,write
from v1.train import load_corpus,optimizer_for
from v11.data import prepare_qa
from v3.train import validation_data,evaluate,seed_list,source_hashes as old_sources
from v09.train import StopRequest,learning_rate_scale,capture_rng,restore_random
from .model import Model,metadata,CONDITIONS

OLD=PROJECT/'results/v3/paired_20261103T094249_111075'


class Progress(BaseProgress):label='Tiny Qwen/DS · 333K'


def sources():
    files=[*sorted((PROJECT/'evaluations/tiny_arch').glob('*.py')),PROJECT/'evaluations/tiny_arch/requirements.txt',PROJECT/'run_biomatrix_tiny_arch.py']
    return old_sources()|{str(p.relative_to(PROJECT)):file_hash(p) for p in files}


def append(path,row):
    with Path(path).open('a') as handle:handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def update(model,opt,values,mask,c,device):
    x=torch.as_tensor(values[:,:-1].copy(),device=device);y=torch.as_tensor(values[:,1:].copy(),device=device)
    selected=torch.as_tensor(mask.copy(),device=device)
    if not bool(selected.any()):raise ValueError('Empty objective')
    model.train();opt.zero_grad(set_to_none=True);sync(device);start=time.perf_counter()
    logits,state,diag=model(x);ce=F.cross_entropy(logits.float().flatten(0,1),y.flatten(),reduction='none').reshape_as(y)[selected].mean()
    aux=diag.get('native_auxiliary',ce.new_zeros(()));loss=ce+model.auxiliary_weight()*aux
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite loss')
    loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip'],error_if_nonfinite=True)
    opt.step();sync(device);seconds=time.perf_counter()-start
    return dict(ce=float(ce.detach().cpu()),objective=float(loss.detach().cpu()),native_auxiliary=float(aux.detach().cpu()),
        gradient_norm=float(gradient.detach().cpu()),targets=int(mask.sum()),inputs=int(x.numel()),seconds=seconds,
        input_tokens_per_second=x.numel()/seconds),state


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',default='auto',choices=('auto','mps','cpu'))
    parser.add_argument('--seeds',default='11,23,37');parser.add_argument('--output',type=Path);parser.add_argument('--resume',type=Path)
    parser.add_argument('--quiet',action='store_true');args=parser.parse_args(argv);logging.set_verbosity_error()
    device=device_for(args.device);original=json.loads((OLD/'protocol.json').read_text());c=original['training'];seeds=seed_list(args.seeds)
    if any(seed not in seed_list(c['seeds']) for seed in seeds):raise ValueError('Use an original paired seed')
    if old_sources()!=original['source_hashes']:raise ValueError('Original V3 source identity changed')
    torch.set_num_threads(c['threads'])
    root=(args.resume or args.output or PROJECT/'results/tiny_arch'/datetime.now().strftime('paired_%Y%m%dT%H%M%S_%f')).resolve()
    if not args.resume and root.exists() and any(root.iterdir()):raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True)
    records=[dict(id=f'{name}_s{seed}') for seed in seeds for name in CONDITIONS];work=2_097_152
    ui=Progress(records,work,not args.quiet);ui.device=device
    registry={r['id']:dict(status='not_executed',completed_work=0) for r in records};active=None
    try:
        ui.stage('官方实现、因果性、参数与原生MPS自检')
        output=io.StringIO();checks=unittest.TextTestRunner(stream=output,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('evaluations.tiny_arch.test_model'))
        (root/'verification.txt').write_text(output.getvalue());write(root/'verification.json',dict(status='completed' if checks.wasSuccessful() else 'failed',tests=checks.testsRun,failures=len(checks.failures),errors=len(checks.errors)))
        if not checks.wasSuccessful():raise RuntimeError('Preflight failed')
        with RunLock(PROJECT/'results/tiny_arch/.train.lock'),StopRequest() as stop:
            corpus=load_corpus(c['data_root']);qa=prepare_qa(c['qa_root'],corpus,length=512,prefix=384,context_limit=128,backbone_layers=2,allow_download=False)
            if digest(corpus.manifest)!=original['data_manifest_sha256'] or digest(qa.manifest)!=original['qa_manifest_sha256']:raise ValueError('Original data changed')
            data,meta=validation_data(corpus,qa,c)
            inspect_models={name:Model(name,seeds[0],c) for name in CONDITIONS}
            model_info={name:metadata(model) for name,model in inspect_models.items()}
            external={name:dict(module=type(model.model).__module__,sha256=file_hash(inspect.getfile(type(model.model)))) for name,model in inspect_models.items()}
            del inspect_models
            protocol=dict(format='tiny-public-architectures-1',source_run=str(OLD),training=c,seeds=seeds,
                sources=sources(),external=external,packages={k:version(k) for k in ('torch','transformers','numpy','tokenizers')},
                data_manifest_sha256=digest(corpus.manifest),qa_manifest_sha256=digest(qa.manifest),evaluation=meta,
                models=model_info,device=str(device),dtype='float32',
                pairing='Every one of 1024 input/mask hashes must equal original trans_full at the same seed/update.',
                objective='Original CE and masks, including the original answer-only/no-EOS QA supervision. DS native balance auxiliary .001 reported separately.',
                scope='Local 333K prototypes, four-layer reduced public architectures; same initial embedding, vocabulary and budgets. No pretrained weights. Not flagship or equal-FLOP comparisons.',
                kernels='FP32 public eager/reference PyTorch on Mac, no CUDA/Fused GDN/MoE kernels. Cache-free complete-prefix validation.')
            if args.resume:
                if json.loads((root/'protocol.json').read_text())!=protocol:raise ValueError('Resume identity mismatch')
                registry=json.loads((root/'registry.json').read_text())['conditions']
            else:
                write(root/'protocol.json',protocol)
                for relative in protocol['sources']:
                    target=root/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((PROJECT/relative).read_bytes())
            write(root/'registry.json',dict(status='running',conditions=registry))
            for seed in seeds:
                expected=[json.loads(line)['sample_sha256'] for line in (OLD/f'trans_full_s{seed}/metrics.jsonl').read_text().splitlines()]
                if len(expected)!=1024:raise ValueError('Original training trace length mismatch')
                for name in CONDITIONS:
                    identity=f'{name}_s{seed}';active=identity;folder=root/identity;folder.mkdir(exist_ok=True)
                    if registry[identity]['status']=='completed':ui.details[identity].n=work;ui.total.update(1);continue
                    model=Model(name,seed,c).to(device);opt=optimizer_for(model,c);rng=np.random.default_rng(c['data_seed']+seed)
                    counters=dict(updates=0,cpt_tokens=0,pt_updates=0,answer_targets=0,completed_work=0,train_seconds=0.)
                    checkpoint=folder/'last.pt'
                    if checkpoint.exists():
                        p=torch.load(checkpoint,map_location='cpu',weights_only=True)
                        if p['protocol_sha256']!=digest(protocol):raise ValueError('Checkpoint protocol mismatch')
                        model.load_state_dict(p['model']);opt.load_state_dict(p['optimizer']);counters=p['counters'];rng.bit_generator.state=p['data_rng'];restore_random(p['rng'],device)
                        for f in (folder/'metrics.jsonl',folder/'validation.jsonl'):
                            if f.exists():
                                lines=[json.loads(x) for x in f.read_text().splitlines()];kept=[x for x in lines if x['updates']<=counters['updates']]
                                if len(kept)!=len(lines):f.rename(f.with_suffix(f'.uncommitted_{time.time_ns()}.jsonl'));f.write_text(''.join(json.dumps(x)+'\n' for x in kept))
                    def persist():
                        if sources()!=protocol['sources']:raise ValueError('Source changed during training')
                        save(checkpoint,dict(protocol_sha256=digest(protocol),model=model.state_dict(),optimizer=opt.state_dict(),
                            counters=counters.copy(),data_rng=rng.bit_generator.state,rng=capture_rng(device)))
                    def assess(stage):
                        ui.stage(f'{identity} · {stage}固定验证')
                        score=evaluate(model,data,c,device,full=False)
                        append(folder/'validation.jsonl',dict(**counters,stage=stage,**score));return score
                    registry[identity]=dict(status='running',**counters);write(root/'registry.json',dict(status='running',conditions=registry));persist()
                    if not (folder/'initial_validation.json').exists():write(folder/'initial_validation.json',assess('initial'))
                    while counters['updates']<1024 and not stop.requested:
                        stage='cpt' if counters['cpt_tokens']<c['cpt_tokens'] else 'pt'
                        if stage=='cpt':
                            values,docs,starts=corpus.sample('train',rng,4,512);mask=np.ones((4,512),dtype=bool)
                            done=counters['cpt_tokens']+2048;budget=c['cpt_tokens']
                        else:
                            values,mask,_=qa.sample('train',rng,4);done=counters['pt_updates']+1;budget=512
                        sha=digest(dict(values=values.tolist(),mask=mask.tolist()))
                        if sha!=expected[counters['updates']]:raise ValueError('Input/mask diverged from original paired trace')
                        rate=c['learning_rate']*learning_rate_scale(done,budget)
                        for group in opt.param_groups:group['lr']=rate
                        row,state=update(model,opt,values,mask,c,device);counters['updates']+=1;counters['train_seconds']+=row['seconds']
                        if stage=='cpt':counters['cpt_tokens']+=row['targets'];counters['completed_work']+=row['targets']
                        else:counters['pt_updates']+=1;counters['answer_targets']+=row['targets'];counters['completed_work']+=row['inputs']
                        row.update(counters,stage=stage,learning_rate=rate,sample_sha256=sha,runtime_bytes=state_bytes(state),input_pairing='matched')
                        append(folder/'metrics.jsonl',row);ui.details[identity].n=counters['completed_work'];ui.details[identity].postfix=f'CE={row["ce"]:.4f}'
                        ui.stage(f'{identity} · {stage} · {counters["updates"]}/1024 · CE {row["ce"]:.4f} · {row["input_tokens_per_second"]:,.0f} 输入/s')
                        registry[identity]=dict(status='running',**counters,ce=row['ce'])
                        if counters['updates']%16==0 or stop.requested:persist();write(root/'registry.json',dict(status='running',conditions=registry))
                        if counters['updates'] in (512,768):write(folder/('cpt_validation.json' if counters['updates']==512 else 'pt_mid_validation.json'),assess('cpt_end' if counters['updates']==512 else 'pt_mid'))
                    persist();finished=counters['updates']==1024 and not stop.requested
                    result=dict(status='completed' if finished else 'aborted',**counters,**model_info[name],seed=seed,condition=name)
                    if finished:result['validation']=assess('final')
                    write(folder/'result.json',result);registry[identity]=result
                    write(root/'registry.json',dict(status='running' if finished else 'aborted',conditions=registry))
                    model.to('cpu');del model,opt;torch.mps.empty_cache() if device.type=='mps' else None
                    if not finished:raise KeyboardInterrupt
                    ui.total.update(1)
            write(root/'registry.json',dict(status='completed',conditions=registry));write(root/'summary.json',dict(status='completed',conditions=registry))
    except BaseException as error:
        status='aborted' if isinstance(error,KeyboardInterrupt) else 'failed'
        if active:registry[active]['status']=status
        write(root/'registry.json',dict(status=status,conditions=registry));(root/'traceback.private.txt').write_text(traceback.format_exc())
        raise
    finally:ui.close()
    print(f'Tiny Qwen/DS {root}')


if __name__=='__main__':main()
