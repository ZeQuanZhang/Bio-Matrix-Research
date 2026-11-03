"""Token-weighted accumulation, finite AdamW updates and fixed validation."""
from collections import Counter
from contextlib import nullcontext
import math
import re
import string
import time

import numpy as np
import torch
from torch.nn import functional as F

from v07.common import sync
from .data import EOS


def amp(device,precision):
    if precision=='bf16':
        if device.type not in ('cuda','cpu'):raise ValueError('BF16 acceptance requires CUDA or explicit CPU testing')
        return torch.autocast(device.type,dtype=torch.bfloat16)
    if precision!='fp32':raise ValueError('Supported precision: fp32 or bf16')
    return nullcontext()


def optimizer(model,rate,decay,device):
    groups={False:[],True:[]}
    for name,p in model.named_parameters():groups[p.ndim>1 and not name.endswith('.bias')].append(p)
    return torch.optim.AdamW([dict(params=params,weight_decay=decay if key else 0.) for key,params in groups.items() if params],
                             lr=rate,betas=(.9,.95),eps=1e-8,fused=device.type=='cuda')


def update(model,opt,values,mask,microbatch,device,precision,clip):
    if values.dtype!=np.int64 or mask.dtype!=np.bool_ or mask.shape!=(len(values),values.shape[1]-1):raise ValueError('Invalid token/mask batch')
    targets=int(mask.sum())
    if targets<1:raise ValueError('Empty objective')
    model.train();opt.zero_grad(set_to_none=True);sync(device);started=time.perf_counter()
    loss_sum=torch.zeros((),device=device);state=None
    for start in range(0,len(values),microbatch):
        part=values[start:start+microbatch];selected=torch.as_tensor(mask[start:start+microbatch],device=device)
        if not bool(selected.any()):continue
        x=torch.as_tensor(part[:,:-1].copy(),device=device);y=torch.as_tensor(part[:,1:].copy(),device=device)
        with amp(device,precision):
            logits,state,_=model(x)
            summed=F.cross_entropy(logits[selected].float(),y[selected],reduction='sum')
        (summed/targets).backward();loss_sum+=summed.detach()
        del logits,state
    gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),clip,error_if_nonfinite=True)
    if not bool(torch.isfinite(loss_sum)):raise FloatingPointError('Nonfinite loss; optimizer not advanced')
    opt.step();sync(device);seconds=time.perf_counter()-started
    return dict(ce=float(loss_sum.cpu())/targets,gradient_norm=float(gradient.cpu()),targets=targets,
                input_slots=int(values[:,:-1].size),seconds=seconds,targets_per_second=targets/max(seconds,1e-9))


def rate_at(tokens,total,warmup,minimum):
    if not 0<=tokens<=total or not 0<warmup<total or not 0<=minimum<=1:raise ValueError('Invalid schedule')
    if tokens<warmup:return max(1,tokens)/warmup
    return minimum+(1-minimum)*.5*(1+math.cos(math.pi*(tokens-warmup)/(total-warmup)))


@torch.inference_mode()
def evaluate_lm(model,stream,sequence,windows,device,precision):
    if stream.total<sequence+1:raise ValueError('Validation split too short')
    model.eval();total=0.;count=0
    starts=np.linspace(0,stream.total-sequence-1,windows,dtype=np.int64)
    if len(set(starts.tolist()))!=windows:raise ValueError('Validation windows duplicated')
    for start in starts:
        tokens=torch.as_tensor(stream.read(int(start),sequence+1)[None],device=device)
        with amp(device,precision):logits,_,_=model(tokens[:,:-1])
        loss=F.cross_entropy(logits.float().flatten(0,1),tokens[:,1:].flatten(),reduction='sum')
        total+=float(loss.cpu());count+=sequence
    return dict(nll=total/count,ppl=math.exp(total/count),targets=count,windows=windows,reset='each sampled sequence')


@torch.inference_mode()
def evaluate_qa_nll(model,qa,count,device,precision):
    model.eval();total=0.;targets=0
    for i in range(min(count,len(qa.rows))):
        values,mask=qa.sample([i]);x=torch.as_tensor(values[:,:-1],device=device);y=torch.as_tensor(values[:,1:],device=device)
        selected=torch.as_tensor(mask,device=device)
        with amp(device,precision):logits,_,_=model(x)
        total+=float(F.cross_entropy(logits[selected].float(),y[selected],reduction='sum').cpu());targets+=int(mask.sum())
    return dict(nll=total/targets,targets=targets,questions=min(count,len(qa.rows)))


def normalize(text):
    text=''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',text).split())


def qa_score(prediction,references):
    pred=normalize(prediction);best_f1=0.;best_em=0.
    for reference in references:
        gold=normalize(reference);best_em=max(best_em,float(pred==gold));p,g=pred.split(),gold.split()
        common=sum((Counter(p)&Counter(g)).values())
        f1=float(p==g) if not p or not g else 2*common/(len(p)+len(g))
        best_f1=max(best_f1,f1)
    return best_em,best_f1


@torch.inference_mode()
def generate_qa(model,qa,tokenizer,count,device,precision,max_tokens=64):
    model.eval();rows=[]
    for index in range(min(count,len(qa.rows))):
        row=qa.rows[index];prefix=np.asarray(qa.values[index,:row['prompt_tokens']],dtype=np.int64)
        with amp(device,precision):logits,state,_=model(torch.as_tensor(prefix[None],device=device))
        generated=[]
        for _ in range(max_tokens):
            token=int(logits[0,-1].argmax().cpu());generated.append(token)
            if token==EOS:break
            with amp(device,precision):logits,state,_=model(torch.tensor([[token]],device=device),state)
        prediction=tokenizer.decode(generated,skip_special_tokens=True).strip()
        em,f1=qa_score(prediction,row['reference_answers'])
        rows.append(dict(question_id=row['question_id'],article_id=row['article_id'],evidence_tokens=row['evidence_tokens'],
            prediction=prediction,reference_answers=row['reference_answers'],em=em,f1=f1,
            generated_tokens=len(generated),hit_limit=len(generated)==max_tokens and generated[-1]!=EOS))
    count=len(rows)
    return dict(questions=count,em=sum(r['em'] for r in rows)/count,f1=sum(r['f1'] for r in rows)/count,
        distinct_predictions=len({r['prediction'] for r in rows}),token_limit_fraction=sum(r['hit_limit'] for r in rows)/count),rows
