"""Isolated eager/compiled-step numerical and CUDA engineering probe.

No core source edits, queue locks, installation, or formal training. Default uses
a fixed-address tensor for the changing gate blend, preserving the 400-step gate
schedule. --blend-mode hard is an explicit post-warmup-only alternative.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--config', type=Path, default=PROJECT/'configs/v07/base.json')
    parser.add_argument('--data-root', type=Path, default=PROJECT/'data/v06/wikitext2')
    parser.add_argument('--nodes', type=int, default=32)
    parser.add_argument('--variant', choices=['full','message','independent'], default='full')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--target', choices=['step', 'tick'], default='step')
    parser.add_argument('--blend-mode', choices=['tensor', 'hard'], default='tensor')
    parser.add_argument('--backend', default='inductor')
    parser.add_argument('--mode', default='default')
    parser.add_argument('--warmup-updates', type=int, default=1)
    parser.add_argument('--probe-updates', type=int, default=2)
    parser.add_argument('--atol', type=float, default=3e-5)
    parser.add_argument('--rtol', type=float, default=3e-3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--self-test', action='store_true', help='CPU/Dynamo-eager interface check on B1/T8; no CUDA speed claim')
    args = parser.parse_args()
    if min(args.warmup_updates, args.probe_updates) < 1 or min(args.atol, args.rtol) < 0:
        parser.error('Positive update counts and nonnegative tolerances required')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root = args.root or PROJECT/'results/v07/cuda_compile_probe'/stamp
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        parser.error('Output root must be empty; previous probes are preserved')
    result = dict(status='running', formal_training=False, model_variant=args.variant, requested_nodes=args.nodes,
                  target=args.target, blend_mode=args.blend_mode, fullgraph=True, dynamic=False,
                  backend=args.backend, compile_mode=args.mode, self_test=args.self_test,
                  atol=args.atol, rtol=args.rtol,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    stage = 'imports'
    started_total = time.perf_counter()
    try:
        import torch
        from torch.nn import functional as F
        from torch._dynamo.utils import counters
        from v07.common import detach, digest, source_hashes, state_bytes
        from v07.data import Corpus, prepare
        from v07.model import initialize
        from v07.run import build, load_config, make_training_stream, model_config, move_state, train_update, validate_config

        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        if args.self_test:
            args.device='cpu';args.backend='eager';args.mode='default';args.nodes=32
        device=torch.device(args.device)
        if not args.self_test and (device.type!='cuda' or not torch.cuda.is_available()):
            raise RuntimeError('Real engineering probe requires CUDA; use --self-test for an explicit CPU mock')
        if device.type=='cuda':
            torch.cuda.set_device(device)
            torch.backends.cuda.matmul.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        cfg=load_config(args.config)|dict(device=device.type,cpu_threads=1,batch_size=2,sequence_tokens=128,
                                       bptt_tokens=128,nodes=[args.nodes],variants=[args.variant],seeds=[args.seed])
        if args.self_test:
            cfg.update(batch_size=1,sequence_tokens=8,bptt_tokens=8,matrix_size=4,width=16,modes=2)
        if args.blend_mode=='hard':
            cfg['gate_warmup_steps']=1
        validate_config(cfg)
        if not (args.data_root/'manifest.json').is_file():
            raise FileNotFoundError('Frozen prepared dataset required; probe never downloads or refits')
        prepare(args.data_root,cfg['vocab_size']);corpus=Corpus(args.data_root)
        cfg['vocab_size']=corpus.manifest['vocab_size']
        source=source_hashes()
        stream=make_training_stream(corpus,cfg,args.seed)
        batches=[stream.next() for _ in range(args.warmup_updates+args.probe_updates)]
        case=dict(stage='compile_probe',nodes=args.nodes,variant=args.variant,seed=args.seed)
        mc=model_config(cfg,case)
        torch.manual_seed(args.seed)
        eager=build(mc);initialize(eager,args.seed)
        initial={name:value.detach().clone() for name,value in eager.state_dict().items()}
        adapted=build(mc);adapted.load_state_dict(initial)
        compiled=build(mc);compiled.load_state_dict(initial)
        models={'core_eager':eager.to(device),'tensor_eager':adapted.to(device),'compiled':compiled.to(device)}

        def synchronize():
            if device.type=='cuda':torch.cuda.synchronize(device)

        def counter_snapshot():
            return {str(group):{str(key):int(value) for key,value in values.items() if isinstance(value,int)}
                    for group,values in counters.items() if values}

        def tensorize_blend(model):
            original_forward=model.forward
            blend_tensor=torch.ones((),device=device,dtype=model.embedding.weight.dtype)
            def forward(tokens,state=None,reset_mask=None,diagnostic_mask=None):
                value=model.gate_blend
                if isinstance(value,torch.Tensor):
                    if value is not blend_tensor:blend_tensor.copy_(value)
                else:
                    blend_tensor.fill_(float(value))
                model.gate_blend=blend_tensor
                return original_forward(tokens,state,reset_mask,diagnostic_mask)
            model.forward=forward
            return blend_tensor

        if args.blend_mode=='tensor':
            tensorize_blend(adapted);tensorize_blend(compiled)
        original_target=getattr(compiled,args.target)
        compile_ctor_started=time.perf_counter()
        setattr(compiled,args.target,torch.compile(original_target,backend=args.backend,mode=args.mode,
                                                 fullgraph=True,dynamic=False))
        compile_ctor_seconds=time.perf_counter()-compile_ctor_started
        result.update(config=cfg,config_sha256=digest(cfg),
                      config_file_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                      source_hashes=source,data_manifest_sha256=digest(corpus.manifest),model=asdict(mc),
                      device=str(device),cuda_device=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                      torch=str(torch.__version__),backend=args.backend,compile_mode=args.mode,
                      compile_constructor_seconds=compile_ctor_seconds,
                      gate_schedule='Original scheduled blend, tensorized outside the compiled graph' if args.blend_mode=='tensor' else 'Fixed hard gate=1; warmup schedule is NOT covered',
                      warmup_schedule_covered=args.blend_mode=='tensor',
                      input_content_sha256=[batch[3]['content_sha256'] for batch in batches])
        write(root/'protocol.json',result)

        def snapshot(model,values,mask,reset):
            model.train();model.zero_grad(set_to_none=True)
            model.gate_blend=min(1.,1/max(1,cfg['gate_warmup_steps']))
            x=torch.tensor(values[:,:-1].copy(),device=device,dtype=torch.long)
            y=torch.tensor(values[:,1:].copy(),device=device,dtype=torch.long)
            valid=torch.as_tensor(mask,device=device,dtype=torch.bool)
            reset_tensor=torch.as_tensor(reset[:,0],device=device,dtype=torch.bool)
            assert not reset[:,1:].any(), 'Initial parity window unexpectedly resets mid-chunk'
            synchronize();started=time.perf_counter()
            logits,state,diag=model(x,reset_mask=reset_tensor,diagnostic_mask=valid)
            loss=(F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none')*valid.flatten()).sum()/valid.sum()
            loss.backward();synchronize();elapsed=time.perf_counter()-started
            return dict(logits=logits.detach().cpu(),loss=loss.detach().cpu(),
                        gradients={name:None if param.grad is None else param.grad.detach().cpu().clone()
                                   for name,param in model.named_parameters()},
                        state=move_state(detach(state),'cpu'),
                        diagnostics={name:value.detach().cpu() for name,value in diag.items()},seconds=elapsed)

        def compare_tensors(left,right):
            if left is None or right is None:return dict(close=left is None and right is None,reference_none=left is None,candidate_none=right is None)
            if left.shape!=right.shape or left.dtype!=right.dtype:return dict(close=False,reference_shape=list(left.shape),candidate_shape=list(right.shape))
            if not left.is_floating_point():
                mismatches=int((left!=right).sum())
                return dict(close=mismatches==0,mismatches=mismatches,dtype=str(left.dtype))
            finite=bool(left.isfinite().all()&right.isfinite().all())
            if not finite:return dict(close=False,finite=False)
            delta=(left-right).abs();allowed=args.atol+args.rtol*left.abs()
            return dict(close=bool((delta<=allowed).all()),finite=True,max_abs=float(delta.max()) if delta.numel() else 0.,
                        rms=float(delta.square().mean().sqrt()) if delta.numel() else 0.,
                        max_tolerance_ratio=float((delta/allowed.clamp_min(1e-30)).max()) if delta.numel() else 0.,
                        mismatches=int((delta>allowed).sum()))

        def comparison(reference,candidate):
            categories={name:compare_tensors(reference[name],candidate[name]) for name in ('logits','loss')}
            for name in ('gradients','state','diagnostics'):
                assert reference[name].keys()==candidate[name].keys()
                categories[name]={key:compare_tensors(reference[name][key],candidate[name][key]) for key in reference[name]}
            close=all(categories[name]['close'] for name in ('logits','loss'))
            close &= all(row['close'] for name in ('gradients','state','diagnostics') for row in categories[name].values())
            return dict(close=bool(close),categories=categories)

        snapshots={}
        for name,model in models.items():
            stage='parity_'+name
            print(json.dumps(dict(stage=stage,status='starting')),flush=True)
            before=counter_snapshot()
            values,mask,reset,_=batches[0]
            snapshots[name]=snapshot(model,values,mask,reset)
            result.setdefault('parity_execution',{})[name]=dict(seconds=snapshots[name]['seconds'],
                                      dynamo_before=before,dynamo_after=counter_snapshot())
            write(root/'progress.json',dict(stage=stage,status='completed',seconds=snapshots[name]['seconds']))
            print(json.dumps(dict(stage=stage,status='completed',seconds=snapshots[name]['seconds'])),flush=True)
        parity=dict(core_vs_tensor=comparison(snapshots['core_eager'],snapshots['tensor_eager']),
                    tensor_vs_compiled=comparison(snapshots['tensor_eager'],snapshots['compiled']),
                    core_vs_compiled=comparison(snapshots['core_eager'],snapshots['compiled']))
        result['parity']=parity
        write(root/'parity.json',parity)
        torch.save(snapshots,root/'parity_snapshots_cpu.pt')
        # Preserve tensor snapshots for review, then release the extra adapter model.
        del snapshots,adapted;models.pop('tensor_eager');gc.collect()
        trajectories={}
        for name,model in models.items():
            model.load_state_dict(initial);model.zero_grad(set_to_none=True)
            trajectories[name]=dict(state=None,optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']),
                                    warmup=[],steady=[])
        for step,(values,mask,reset,meta) in enumerate(batches):
            for name in (['core_eager','compiled'] if step%2==0 else ['compiled','core_eager']):
                stage=f'update_{step}_{name}'
                model=models[name];track=trajectories[name]
                if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
                before=counter_snapshot();synchronize();started=time.perf_counter()
                metrics,state=train_update(model,track['optimizer'],values,cfg,device,step,state=track['state'],reset_mask=reset,loss_mask=mask)
                synchronize();seconds=time.perf_counter()-started;track['state']=state
                row=dict(step=step,seconds=seconds,valid_targets=metrics['valid_targets'],
                         tokens_per_second=metrics['valid_targets']/seconds,metrics=metrics,
                         dynamo_before=before,dynamo_after=counter_snapshot(),content_sha256=meta['content_sha256'])
                row['new_unique_graphs']=row['dynamo_after'].get('stats',{}).get('unique_graphs',0)-before.get('stats',{}).get('unique_graphs',0)
                if device.type=='cuda':
                    row.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                               cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
                track['warmup' if step<args.warmup_updates else 'steady'].append(row)
                print(json.dumps(dict(stage=stage,seconds=seconds,loss=metrics['loss'],valid_targets=metrics['valid_targets'])),flush=True)
                write(root/'timing_partial.json',{key:{field:value[field] for field in ('warmup','steady')} for key,value in trajectories.items()})
        timing={}
        for name,track in trajectories.items():
            rows=track['steady'];targets=sum(row['valid_targets'] for row in rows);seconds=sum(row['seconds'] for row in rows)
            timing[name]=dict(warmup=track['warmup'],steady=rows,steady_seconds=seconds,
                              steady_valid_targets=targets,steady_tokens_per_second=targets/seconds,
                              steady_new_unique_graphs=sum(row['new_unique_graphs'] for row in rows),
                              runtime_state_bytes=state_bytes(track['state']),
                              parameters_finite=all(bool(parameter.detach().isfinite().all()) for parameter in models[name].parameters()))
        result.update(timing=timing,speedup=timing['core_eager']['steady_seconds']/timing['compiled']['steady_seconds'],
                      memory_note='Peak memory is sampled in one process with both compared models resident; reserved memory includes allocator history, not isolated-model capacity',
                      dynamo_final=counter_snapshot(),
                      final_parameters={name:compare_tensors(left.detach().cpu(),dict(compiled.named_parameters())[name].detach().cpu())
                                        for name,left in eager.named_parameters()},
                      final_state={name:compare_tensors(value.detach().cpu(),trajectories['compiled']['state'][name].detach().cpu())
                                   for name,value in trajectories['core_eager']['state'].items()})
        assert source==source_hashes(),'Source changed during engineering probe'
        assert result['config_file_sha256']==hashlib.sha256(args.config.read_bytes()).hexdigest(),'Config changed during probe'
        parity_close=all(row['close'] for row in parity.values())
        trajectory_close=all(row['close'] for field in ('final_parameters','final_state') for row in result[field].values())
        all_close=parity_close and trajectory_close
        finite=all(row['parameters_finite'] for row in timing.values())
        result.update(status='completed' if all_close and finite else 'numerical_mismatch',
                      finite=finite,numerically_accepted=all_close,parity_accepted=parity_close,
                      optimizer_trajectory_accepted=trajectory_close,elapsed_seconds=time.perf_counter()-started_total)
        if args.self_test:
            assert all_close and finite,'CPU fullgraph eager-backend probe failed numerical equivalence'
            result['self_test_passed']=True
    except BaseException:
        result.update(status='failed',failed_stage=stage,error=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started_total)
        (root/'error.txt').write_text(result['error'])
        print(result['error'],file=sys.stderr,flush=True)
    write(root/'result.json',result)
    print(json.dumps({key:result.get(key) for key in ('status','failed_stage','numerically_accepted','speedup','elapsed_seconds')}),flush=True)
    # Failure is an inspectable engineering result, never an implicit change to formal execution.
    return 0 if result['status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
