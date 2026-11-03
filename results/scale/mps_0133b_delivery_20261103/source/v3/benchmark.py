"""Save native-device correctness and matched end-to-end update timings."""
import argparse
import copy
from dataclasses import asdict
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import torch
from torch.nn import functional as F

from v07.common import device_for,write,state_bytes,sync
from v1.train import optimizer_for
from .model import Config,Network
from .train import DEFAULTS,build_models,update,source_hashes


def correctness(device):
    torch.manual_seed(51)
    cfg=Config(vocab_size=32,width=16,layers=2,heads=2,ff=32,nodes=4,phase_bands=3,segment_tokens=4)
    cpu=Network(cfg);native=copy.deepcopy(cpu).to(device)
    x=torch.randint(0,32,(2,17));target=torch.randint(0,32,(2,17))
    a=cpu(x)[0];b=native(x.to(device))[0]
    torch.testing.assert_close(a,b.cpu(),atol=3e-5,rtol=2e-4)
    F.cross_entropy(a.flatten(0,1),target.flatten()).backward()
    F.cross_entropy(b.flatten(0,1),target.to(device).flatten()).backward()
    differences=[]
    for (name,p),(other,q) in zip(cpu.named_parameters(),native.named_parameters()):
        assert name==other
        torch.testing.assert_close(p.grad,q.grad.cpu(),atol=3e-5,rtol=3e-3,msg=name)
        differences.append(float((p.grad-q.grad.cpu()).abs().max()))
    with torch.no_grad():
        parts=[];state=None
        for start,end in ((0,1),(1,3),(3,8),(8,9),(9,17)):
            out,state,_=native(x[:,start:end].to(device),state);parts.append(out)
        streaming=torch.cat(parts,1)
        torch.testing.assert_close(b,streaming,atol=3e-5,rtol=2e-4)
    return dict(status='completed',config=asdict(cfg),logits_max_abs=float((a-b.cpu()).detach().abs().max()),
                gradients_max_abs=max(differences),gradient_tensors=len(differences),
                streaming_max_abs=float((b-streaming).detach().abs().max()))


def benchmark(root,device,repeats=5):
    root=Path(root)
    if root.exists() and any(root.iterdir()):raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True)
    if device.type=='mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':
        raise RuntimeError('Native MPS required; remove fallback environment variable')
    torch.set_num_threads(4)
    result=dict(status='running',device=str(device),torch=str(torch.__version__),machine=platform.machine(),
                platform=platform.platform(),source_hashes=source_hashes(),correctness=correctness(device),
                batch=4,input_tokens=512,repeats=repeats,conditions={})
    write(root/'benchmark.json',result)
    c=DEFAULTS.copy();models,_,counts=build_models(4096,c,11)
    values=np.random.default_rng(3).integers(0,4096,size=(4,513),dtype=np.int64)
    mask=np.ones((4,512),dtype=bool)
    for name,model in models.items():
        model.to(device);optimizer=optimizer_for(model,c);rows=[]
        for step in range(repeats+2):
            row,state=update(model,optimizer,values,mask,c,device)
            if step>=2:rows.append(row)
        forward=[];model.eval()
        with torch.no_grad():
            x=torch.as_tensor(values[:,:-1].copy(),device=device)
            for _ in range(repeats+2):
                sync(device);started=time.perf_counter();_,state,_=model(x);sync(device)
                forward.append(time.perf_counter()-started)
        median=statistics.median(r['seconds'] for r in rows)
        result['conditions'][name]=dict(**counts[name],updates=rows,training_seconds_median=median,
            input_tokens_per_second=2048/median,forward_seconds_median=statistics.median(forward[2:]),
            persistent_runtime_bytes=state_bytes(state),
            note='Includes cross-entropy, backward, gradient clip, AdamW and device synchronization. Two warmups excluded. All models from scratch; runtime state capacity differs.')
        model.to('cpu');del optimizer,state
        write(root/'benchmark.json',result)
    baseline=result['conditions']['trans_local']['training_seconds_median']
    for row in result['conditions'].values():row['training_time_ratio_to_local']=row['training_seconds_median']/baseline
    result['status']='completed';write(root/'benchmark.json',result)
    lines=['# V3 原生设备工程测速','',f'设备：{device}，批量4，序列512，FP32，排除2次预热，计时{repeats}次。',
           '','| 条件 | 参数 | 更新秒数中位数 | 输入token/s | 相对局部Transformer耗时 | 持久状态字节 |',
           '|---|---:|---:|---:|---:|---:|']
    for name,row in result['conditions'].items():
        lines.append(f'| {name} | {row["total_parameters"]} | {row["training_seconds_median"]:.6f} | {row["input_tokens_per_second"]:.0f} | {row["training_time_ratio_to_local"]:.2f} | {row["persistent_runtime_bytes"]} |')
    lines+=['','这是随机输入的计算测速，无法说明语言或检索能力。持久状态不包含训练激活及优化器。',
            'V3使用向量节点，容量与旧版完整矩阵不同；相对旧版计时不能视作等容量算法提速。',
            '',f'CPU/原生设备及流式一致性：{result["correctness"]}']
    (root/'REPORT.md').write_text('\n'.join(lines))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='auto',choices=('auto','cpu','mps','cuda'))
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args()
    if args.repeats<1:parser.error('--repeats must be positive')
    result=benchmark(args.output,device_for(args.device),args.repeats)
    print(f'V3 工程测速 {result["status"]}: {args.output}')
