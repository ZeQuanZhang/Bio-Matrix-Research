from pathlib import Path
import inspect,json,math
import numpy as np
import torch
from torch.nn import functional as F
from v07.data import Corpus
from v07.common import detach
from v07.model import invalidate_feedback
# Pin probes to the immutable first-round source, including original line numbers.
import importlib.util, sys
snapshot = Path('results/v08/mps_20261024_c16_c32/source/v08/model.py')
spec = importlib.util.spec_from_file_location('v08.model', snapshot)
original_model = importlib.util.module_from_spec(spec)
sys.modules['v08.model'] = original_model
spec.loader.exec_module(original_model)

from v08.benchmark import build
from v08.model import initialize

torch.set_num_threads(4);seed=11;torch.manual_seed(seed);corpus=Corpus();model,cfg=build(dict(family='v08',nodes=32,block=16,method='triangular'),4096);initialize(model,seed)
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
values,docs,starts=corpus.sample('train',np.random.default_rng(seed),2,8*64);tokens=torch.tensor(values,dtype=torch.long)
state=None;original=F.normalize;records=[];blocknum=[0]
block_original=model.block

def block(*args):
 blocknum[0]+=1;return block_original(*args)
model.block=block

def normalize(x,*args,**kwargs):
 y=original(x,*args,**kwargs)
 if x.requires_grad:
  frame=inspect.currentframe().f_back;site=f'{Path(frame.f_code.co_filename).name}:{frame.f_lineno}'
  dim=kwargs.get('dim',1);eps=kwargs.get('eps',1e-12);norm=x.detach().norm(dim=dim)
  row=dict(index=len(records),block=blocknum[0],site=site,shape=list(x.shape),epsilon=eps,norm_min=float(norm.min()),norm_max=float(norm.max()),norm_median=float(norm.median()),below_eps=int((norm<eps).sum()),input_grad_norm=None,output_grad_norm=None)
  def ih(g):
   row['input_grad_norm']=float(g.detach().double().norm());row['input_grad_max_abs']=float(g.detach().abs().max())
   return g
  def oh(g):
   row['output_grad_norm']=float(g.detach().double().norm());row['output_grad_max_abs']=float(g.detach().abs().max())
   return g
  x.register_hook(ih);y.register_hook(oh);records.append(row)
 return y
F.normalize=normalize
all_steps=[]
for step in range(2):
 records=[];blocknum[0]=0
 optimizer.zero_grad(set_to_none=True);batch=tokens[:,step*64:step*64+65]
 logits,state,_=model(batch[:,:-1],state);loss=F.cross_entropy(logits.flatten(0,1),batch[:,1:].flatten());loss.backward()
 for row in records:
  row['gradient_gain']=(row['input_grad_norm'] or 0)/max(row['output_grad_norm'] or 0,1e-300)
 data=dict(step=step,loss=float(loss.detach()),normalizations=records)
 all_steps.append(data)
 top=sorted(records,key=lambda row:row['input_grad_norm'] or 0,reverse=True)[:12]
 print('STEP',step,'TOP_INPUT_GRAD',json.dumps(top))
 print('GAIN',json.dumps(sorted(records,key=lambda row:row['gradient_gain'],reverse=True)[:8]))
 if step==0:
  torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();state=invalidate_feedback(detach(state))
root=Path('results/v08/diagnosis_20261024');(root/'normalization_hooks.json').write_text(json.dumps(all_steps,indent=2))
