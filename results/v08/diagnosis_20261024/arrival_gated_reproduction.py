"""Same seed/data/optimizer as first failed C16 case, after arrival gating."""
from pathlib import Path
from dataclasses import asdict
import json,math
import numpy as np
import torch
from torch.nn import functional as F
from v07.data import Corpus
from v07.common import detach,file_hash
from v07.model import invalidate_feedback
from v08.benchmark import build
from v08.model import initialize

torch.set_num_threads(4);seed=11;torch.manual_seed(seed);corpus=Corpus()
model,cfg=build(dict(family='v08',nodes=32,block=16,method='triangular'),4096);initialize(model,seed)
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
values,docs,starts=corpus.sample('train',np.random.default_rng(seed),2,8*64)
tokens=torch.tensor(values,dtype=torch.long);state=None;records=[]
for step in range(3):
 optimizer.zero_grad(set_to_none=True);batch=tokens[:,step*64:step*64+65]
 logits,state,diag=model(batch[:,:-1],state);loss=F.cross_entropy(logits.flatten(0,1),batch[:,1:].flatten());loss.backward()
 stats=[]
 for name,p in model.named_parameters():
  if p.grad is None:continue
  g=p.grad.detach();stats.append(dict(name=name,finite=bool(g.isfinite().all()),norm64=float(g.double().norm()),max_abs=float(g.abs().max()),norm32=float(g.norm())))
 stats.sort(key=lambda x:x['norm64'],reverse=True);norm64=math.sqrt(sum(s['norm64']**2 for s in stats))
 norm32=float(torch.stack([p.grad.norm().square() for p in model.parameters() if p.grad is not None]).sum().sqrt())
 metrics=dict(step=step,loss=float(loss.detach()),gradient_norm64=norm64,gradient_norm32=norm32,all_elements_finite=all(s['finite'] for s in stats),top=stats[:12],content_projection_norm64=next(s['norm64'] for s in stats if s['name']=='content_projection.weight'),reached=float(diag['reached'].detach()))
 print(json.dumps(metrics),flush=True);records.append(metrics)
 torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();state=invalidate_feedback(detach(state))
root=Path('results/v08/diagnosis_20261024')
(root/'arrival_gated_reproduction.json').write_text(json.dumps(dict(condition='arrival-gated-observation',device='cpu',dtype='float32',config=asdict(cfg),seed=seed,documents=docs,starts=starts,model_sha256=file_hash(Path('v08/model.py')),steps=records),indent=2,allow_nan=False))
