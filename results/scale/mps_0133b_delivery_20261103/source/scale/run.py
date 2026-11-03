"""Single-GPU 0.133B paired experiment, independent of historical V3 runs."""
import argparse
from datetime import datetime
import gc
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
from tokenizers import Tokenizer

from scripts.v2_console import Progress as BaseProgress
from v07.common import PROJECT,RunLock,device_for,digest,file_hash,save,sync,write
from v09.train import StopRequest,capture_rng,restore_random
from v3.train import source_hashes as historical_sources
from . import data,engine
from .model import CONDITIONS,build,inspect

DEFAULT_CONFIG=PROJECT/'configs/scale/v3_0133b.json'


class Progress(BaseProgress):
    label='V3-Scale 0.133B'
    def resources(self):
        text=super().resources()
        if self.device is not None and self.device.type=='cuda':
            text+=f' | GPU alloc {torch.cuda.memory_allocated()/2**30:.2f} / reserve {torch.cuda.memory_reserved()/2**30:.2f} GiB'
        return text


def settings(path):
    c=json.loads(Path(path).read_text())
    if c['format']!='bio-v3-scale-1':raise ValueError('Unknown scale protocol')
    positive=('pretrain_tokens','sequence','microbatch','accumulation','warmup_tokens','checkpoint_every','eval_every',
              'validation_sequences','sft_updates','qa_eval_examples','threads')
    if any(type(c[k]) is not int or c[k]<1 for k in positive):raise ValueError('Positive integer budgets required')
    if type(c['seed']) is not int or c['seed']<0 or type(c['data_seed']) is not int or c['data_seed']<0:raise ValueError('Invalid seed')
    for k in ('learning_rate','weight_decay','minimum_lr_fraction','gradient_clip','sft_learning_rate','qa_gate_f1'):
        if not math.isfinite(c[k]) or c[k]<0:raise ValueError(f'Invalid {k}')
    if min(c['learning_rate'],c['sft_learning_rate'],c['gradient_clip'])<=0:raise ValueError('Positive optimizer rates required')
    if not 0<c['warmup_tokens']<c['pretrain_tokens'] or not 0<=c['minimum_lr_fraction']<=1 or not 0<c['qa_gate_f1']<=1:raise ValueError('Invalid schedule or QA gate')
    if c['sequence']!=c['model']['context_limit'] or c['sequence']%c['model']['segment_tokens']:raise ValueError('Sequence must match full context and cover complete Bio blocks')
    if c['milestones']!=sorted(set(c['milestones'])) or c['milestones'][-1]!=c['pretrain_tokens'] or min(c['milestones'])<1:raise ValueError('Invalid milestones')
    if type(c['activation_checkpoint']) is not bool:raise ValueError('Checkpoint flag must be bool')
    if not all(100_000_000<=n<1_000_000_000 for n in inspect(c['model'])['parameters'].values()):
        raise ValueError('Production configuration must be 0.1B–1B')
    return c


def sources():
    paths=[*sorted((PROJECT/'scale').glob('*.py')),PROJECT/'scale/requirements.txt',PROJECT/'run_biomatrix_scale.py',PROJECT/'.run/Bio-Matrix V3 Scale.run.xml']
    return historical_sources()|{str(p.relative_to(PROJECT)):file_hash(p) for p in paths}


def precision_for(device,requested):
    if requested=='auto':requested='bf16' if device.type=='cuda' else 'fp32'
    if requested=='bf16' and device.type=='cuda' and not torch.cuda.is_bf16_supported():raise ValueError('CUDA GPU lacks native BF16; select fp32 explicitly or use a BF16-capable GPU')
    if requested=='bf16' and device.type=='mps':raise ValueError('Mac acceptance uses native FP32')
    return requested


def append(path,row):
    with Path(path).open('a') as handle:handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def clean_log(path,updates):
    if not path.exists():return
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    keep=[r for r in rows if r['updates']<=updates]
    if len(keep)!=len(rows):
        path.with_suffix(f'.uncommitted_{time.time_ns()}.jsonl').write_bytes(path.read_bytes())
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in keep))


def lm_batch(stream,order,offset,batch,sequence,remaining):
    count=min(batch,math.ceil(remaining/sequence));indices=order[offset:offset+count]
    if len(indices)!=count:raise ValueError('Prepared training blocks exhausted')
    values=np.stack([stream.read(int(i)*sequence,sequence+1) for i in indices])
    mask=np.arange(count*sequence).reshape(count,sequence)<remaining
    return values,mask


def checkpoint(root,model,opt,protocol,counters,phase,device):
    if sources()!=protocol['sources']:raise ValueError('Frozen source changed; checkpoint refused')
    save(root/'last.pt',dict(format='bio-v3-scale-checkpoint-1',protocol_sha256=digest(protocol),
        model=model.state_dict(),optimizer=opt.state_dict(),optimizer_phase=phase,counters=counters.copy(),rng=capture_rng(device)))
    write(root/'progress.json',dict(status='running',**counters))


def release(device):
    gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    if device.type=='mps':torch.mps.empty_cache()


def train(args,c,device,precision,root,ui,stop):
    if device.type!='cuda' and max(inspect(c['model'])['parameters'].values())>1_000_000:
        raise ValueError('完整10B训练请在CUDA GPU启动；本机使用inspect/verify/profile。')
    if shutil.disk_usage(root).free<12*2**30:raise OSError('至少保留12 GiB检查点空间，建议使用100 GiB以上数据盘')
    ui.stage('逐文件校验固定训练数据');manifest=data.validate(args.data_root)
    if manifest['preparation']['tokens']!=c['pretrain_tokens'] or manifest['preparation']['sequence']!=c['sequence']:raise ValueError('Prepared data budget or sequence differs from config')
    protocol=dict(format=c['format'],settings=c,device=str(device),precision=precision,torch=str(torch.__version__),
        data_manifest_sha256=digest(manifest),data_sources=json.loads((args.data_root/'sources.json').read_text()),
        packages={k:version(k) for k in ('torch','numpy','tokenizers','pyarrow')},
        sources=sources(),parameter_counts=inspect(c['model']),
        comparison='Three independently trained models; shared token order/masks/schedule, matched parameters, same initialization for embeddings and compatible FFN. FLOPs and runtime state are not matched.',
        training='10B effective pretraining targets per model, followed by answer+EOS SFT. No LoCoMo training. No pretrained model weights.',
        state='Reset per packed 2048-token sequence; full differentiation across its 16 Bio blocks. Longer continuous training not included.',
        baseline='Full causal SDPA over the complete sampled sequence; is_causal=True. Streaming KV only for generation.',
        gate='SQuAD heldout article free-generation F1 >= configured threshold before a later LoCoMo evaluation. Gate is an experimental choice.',
        hardware='Single CUDA GPU; no DDP, no expired server access.',
        gpu=dict(name=torch.cuda.get_device_name(),bytes=torch.cuda.get_device_properties(0).total_memory) if device.type=='cuda' else None)
    protocol_path=root/'protocol.json'
    if args.resume:
        if json.loads(protocol_path.read_text())!=protocol:raise ValueError('Resume protocol/source/data/device differs')
        registry=json.loads((root/'registry.json').read_text());registry['status']='running'
    else:
        write(protocol_path,protocol);registry=dict(status='running',conditions={n:dict(pretrain='not_executed',sft='not_executed') for n in CONDITIONS})
        for relative in protocol['sources']:
            destination=root/'source'/relative;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes((PROJECT/relative).read_bytes())
    write(root/'registry.json',registry)
    stream=data.Tokens(args.data_root,manifest,'train');val=data.Tokens(args.data_root,manifest,'validation')
    qa_train=data.QA(args.data_root,'train');qa_val=data.QA(args.data_root,'validation')
    tokenizer=Tokenizer.from_file(str(args.data_root/'tokenizer.json'))
    if tokenizer.get_vocab_size()!=c['model']['vocab_size']:raise ValueError('Tokenizer/model vocabulary mismatch')
    order=np.random.default_rng(c['data_seed']).permutation((stream.total-1)//c['sequence'])
    global_batch=c['microbatch']*c['accumulation'];active=None
    try:
        stages=[('pretrain',target,name) for target in c['milestones'] for name in CONDITIONS]
        stages += [('sft',c['sft_updates'],name) for name in CONDITIONS]
        for phase,target,name in stages:
            active=(name,phase);entry=registry['conditions'][name];folder=root/name;folder.mkdir(exist_ok=True)
            counters=dict(pretrain_tokens=0,blocks=0,sft_updates=0,sft_examples=0,answer_targets=0,updates=0,train_seconds=0.)
            payload=torch.load(folder/'last.pt',map_location='cpu',weights_only=True) if (folder/'last.pt').exists() else None
            if payload is not None:
                if payload['protocol_sha256']!=digest(protocol):raise ValueError('Checkpoint identity mismatch')
                counters=payload['counters']
            field='pretrain_tokens' if phase=='pretrain' else 'sft_updates'
            bar=ui.details[f'{name}-{phase}'];bar.n=counters[field]
            artifacts=([folder/f'tokens_{target}.json',folder/f'tokens_{target}.pt'] if phase=='pretrain' else [folder/'result.json'])
            if counters[field]>=target and all(p.exists() for p in artifacts):
                if payload is not None:del payload
                if counters[field]>=bar.total:entry[phase]='completed'
                continue
            if counters[field]>target:raise ValueError('A previous milestone artifact is missing; preserve this run for inspection')
            ui.stage(f'{name} · {phase} · 加载模型与优化器');entry[phase]='running';write(root/'registry.json',registry)
            model=build(name,c['model'],c['seed'],c['activation_checkpoint']).to(device)
            opt=engine.optimizer(model,c['learning_rate'] if phase=='pretrain' else c['sft_learning_rate'],c['weight_decay'],device)
            if payload is not None:
                model.load_state_dict(payload['model'])
                if payload['optimizer_phase']==phase:opt.load_state_dict(payload['optimizer'])
                restore_random(payload['rng'],device);del payload
            clean_log(folder/'metrics.jsonl',counters['updates'])
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
            try:
                while counters[field]<target:
                    if stop.requested:raise KeyboardInterrupt
                    if phase=='pretrain':
                        values,mask=lm_batch(stream,order,counters['blocks'],global_batch,c['sequence'],target-counters[field])
                        completed=counters['pretrain_tokens']+int(mask.sum())
                        rate=c['learning_rate']*engine.rate_at(completed,c['pretrain_tokens'],c['warmup_tokens'],c['minimum_lr_fraction'])
                    else:
                        indices=np.random.default_rng(c['data_seed']+1+counters['sft_updates']).integers(0,len(qa_train.rows),global_batch)
                        values,mask=qa_train.sample(indices)
                        rate=c['sft_learning_rate']*engine.rate_at(counters['sft_updates']+1,c['sft_updates'],max(1,c['sft_updates']//20),.1)
                    for group in opt.param_groups:group['lr']=rate
                    row=engine.update(model,opt,values,mask,c['microbatch'],device,precision,c['gradient_clip'])
                    counters['updates']+=1;counters['train_seconds']+=row['seconds']
                    if phase=='pretrain':counters['pretrain_tokens']+=row['targets'];counters['blocks']+=len(values)
                    else:
                        counters['sft_updates']+=1;counters['sft_examples']+=len(values);counters['answer_targets']+=row['targets']
                    row.update(counters,phase=phase,learning_rate=rate,batch_sha256=digest(dict(values=values.tolist(),mask=mask.tolist())),
                        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
                        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if device.type=='cuda' else None)
                    append(folder/'metrics.jsonl',row);bar.n=counters[field];bar.postfix=f'CE={row["ce"]:.4f}'
                    ui.stage(f'{name} · {phase} · {counters[field]:,}/{target:,} · CE {row["ce"]:.4f} · {row["targets_per_second"]:,.0f} target/s · LR {rate:.2g}')
                    if counters['updates']%c['checkpoint_every']==0:checkpoint(folder,model,opt,protocol,counters,phase,device)
                    if counters['updates']%c['eval_every']==0:
                        ui.stage(f'{name} · 固定验证');score=engine.evaluate_lm(model,val,c['sequence'],c['validation_sequences'],device,precision)
                        append(folder/'validation.jsonl',dict(**counters,phase=phase,lm=score,
                            qa=engine.evaluate_qa_nll(model,qa_val,min(32,c['qa_eval_examples']),device,precision)))
                checkpoint(folder,model,opt,protocol,counters,phase,device)
                if phase=='pretrain':
                    snapshot=folder/f'tokens_{target}.pt'
                    if not snapshot.exists():save(snapshot,dict(format='bio-v3-scale-weights-1',protocol_sha256=digest(protocol),model=model.state_dict(),counters=counters.copy()))
                    ui.stage(f'{name} · {target:,} token检查点验证')
                    write(folder/f'tokens_{target}.json',dict(status='completed',**counters,
                        lm=engine.evaluate_lm(model,val,c['sequence'],c['validation_sequences'],device,precision)))
                    entry[phase]='completed' if target==c['pretrain_tokens'] else 'partial'
                else:
                    ui.stage(f'{name} · 独立逐题生成与问答评分')
                    score,predictions=engine.generate_qa(model,qa_val,tokenizer,c['qa_eval_examples'],device,precision)
                    write(folder/'qa_predictions.private.json',predictions)
                    write(folder/'result.json',dict(status='completed',**counters,qa=score,
                        qa_gate_passed=score['f1']>=c['qa_gate_f1'],qa_gate_threshold=c['qa_gate_f1'],
                        lm=engine.evaluate_lm(model,val,c['sequence'],c['validation_sequences'],device,precision)))
                    entry[phase]='completed'
            except BaseException:
                # A stop between updates can save exactly; failures retain the previous safe checkpoint.
                if stop.requested:checkpoint(folder,model,opt,protocol,counters,phase,device)
                raise
            finally:
                del model,opt;release(device)
            write(root/'registry.json',registry)
            ui.total.n=sum(v=='completed' for e in registry['conditions'].values() for v in e.values())
        registry['status']='completed';write(root/'registry.json',registry)
        write(root/'summary.json',dict(status='completed',conditions={n:json.loads((root/n/'result.json').read_text()) for n in CONDITIONS},
            seed=c['seed'],training_targets_per_model=c['pretrain_tokens'],sft_updates=c['sft_updates'],
            locomo='not_executed',official_squad_dev='not_executed',multiple_training_seeds='not_executed'))
    except BaseException as error:
        status='aborted' if isinstance(error,KeyboardInterrupt) else 'failed';registry['status']=status
        if active:registry['conditions'][active[0]][active[1]]=status
        write(root/'registry.json',registry)
        if status=='failed':(root/'traceback.private.txt').write_text(traceback.format_exc())
        raise


def profile(args,c,device,precision,root,ui):
    seq=args.profile_sequence or c['sequence'];batch=args.profile_batch or c['microbatch']
    if not 1<=seq<=c['sequence'] or batch<1:raise ValueError('Invalid bounded profiling geometry')
    values=np.random.default_rng(4).integers(0,c['model']['vocab_size'],(batch,seq+1),dtype=np.int64);mask=np.ones((batch,seq),dtype=bool)
    result=dict(status='running',device=str(device),precision=precision,sequence=seq,batch=batch,synthetic=True,
                sources=sources(),settings=c,
                full_training='not_executed',cuda_validation='completed' if device.type=='cuda' else 'not_executed',conditions={})
    for relative in result['sources']:
        target=root/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((PROJECT/relative).read_bytes())
    for name in CONDITIONS:
        ui.stage(f'{name} · 有限工程验收 {batch}×{seq}');model=build(name,c['model'],c['seed'],c['activation_checkpoint']).to(device)
        opt=engine.optimizer(model,c['learning_rate'],c['weight_decay'],device);rows=[]
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        for step in range(3):
            rows.append(engine.update(model,opt,values,mask,batch,device,precision,c['gradient_clip']))
            ui.details[f'{name}-profile'].n=step+1;ui.details[f'{name}-profile'].postfix=f'CE={rows[-1]["ce"]:.4f}'
        elapsed=sum(r['seconds'] for r in rows[1:])/2
        result['conditions'][name]=dict(status='completed',parameters=sum(p.numel() for p in model.parameters()),updates=rows,
            seconds_per_update=elapsed,targets_per_second=batch*seq/elapsed,
            note='Two measured updates after one warmup; synthetic operator acceptance. Full-run timing and gradient accumulation require GPU measurement.',
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if device.type=='cuda' else None)
        del model,opt;release(device);ui.total.update(1);write(root/'profile.json',result)
    result['status']='completed';write(root/'profile.json',result)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('inspect','verify','prepare','profile','train'),default='inspect')
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--data-root',type=Path,default=PROJECT/'data/scale/fineweb_edu_100bt_gpt2')
    parser.add_argument('--output',type=Path);parser.add_argument('--resume',type=Path)
    parser.add_argument('--device',choices=('auto','cpu','mps','cuda'),default='auto')
    parser.add_argument('--precision',choices=('auto','bf16','fp32'),default='auto')
    parser.add_argument('--profile-sequence',type=int);parser.add_argument('--profile-batch',type=int)
    parser.add_argument('--quiet',action='store_true');args=parser.parse_args(argv)
    if args.resume and args.mode!='train':parser.error('Resume applies to training only')
    c=settings(args.config);torch.set_num_threads(c['threads']);args.data_root=args.data_root.resolve()
    root=(args.resume or args.output or PROJECT/'results/scale'/datetime.now().strftime(args.mode+'_%Y%m%dT%H%M%S_%f')).resolve()
    if not args.resume and root.exists() and any(root.iterdir()):raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True)
    records=[dict(id=f'{n}-{phase}') for phase in (('pretrain','sft') if args.mode=='train' else (args.mode,)) for n in CONDITIONS]
    ui=Progress(records,3,not args.quiet);device=device_for(args.device);ui.device=device
    try:
        if device.type=='mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('Mac验收要求原生MPS，请移除CPU fallback设置')
        if args.mode=='inspect':
            ui.stage('参数核算完成；数据准备、GPU测速和正式训练尚未执行')
            write(root/'capacity.json',dict(status='completed',**inspect(c['model']),settings=c,
                full_training='not_executed',cuda_profile='not_executed',data_preparation='not_executed'))
        elif args.mode=='verify':
            import unittest
            with (root/'verification.txt').open('w') as handle:
                result=unittest.TextTestRunner(stream=handle,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('scale.test_scale'))
            write(root/'verification.json',dict(status='completed' if result.wasSuccessful() else 'failed',tests=result.testsRun,
                failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped)))
            if not result.wasSuccessful():raise RuntimeError(f'Acceptance failed; see {root}/verification.txt')
        else:
            precision=precision_for(device,args.precision)
            with RunLock(PROJECT/'results/scale/.train.lock'),StopRequest() as stop:
                if args.mode=='prepare':
                    manifest=data.prepare(args.data_root,c['pretrain_tokens'],c['sequence'],ui.stage,stop)
                    write(root/'preparation.json',dict(status='completed',manifest_sha256=digest(manifest),totals=manifest['totals']))
                elif args.mode=='profile':profile(args,c,device,precision,root,ui)
                else:
                    for n in CONDITIONS:
                        ui.details[f'{n}-pretrain'].total=c['pretrain_tokens'];ui.details[f'{n}-sft'].total=c['sft_updates']
                    train(args,c,device,precision,root,ui,stop)
    except BaseException as error:
        write(root/'execution.json',dict(status='aborted' if isinstance(error,KeyboardInterrupt) else 'failed',mode=args.mode,error=str(error)))
        raise
    finally:ui.close()
    write(root/'execution.json',dict(status='completed',mode=args.mode,full_training='completed' if args.mode=='train' else 'not_executed'))
    print(f'V3 Scale {args.mode} 完成：{root}')


if __name__=='__main__':main()
