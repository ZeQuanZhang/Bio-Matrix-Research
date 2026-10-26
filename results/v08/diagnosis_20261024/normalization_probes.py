"""Diagnostic monkey patches only: original v08 source/checkpoints are unchanged."""
from pathlib import Path
import inspect,json,math,time
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

torch.set_num_threads(4);seed=11;corpus=Corpus()
values,docs,starts=corpus.sample('train',np.random.default_rng(seed),2,8*64)
tokens=torch.tensor(values,dtype=torch.long);original=F.normalize
root=Path('results/v08/diagnosis_20261024');output=[]
conditions=[('baseline',0.),('detach_final',0.),('mask_final',0.)]
conditions += [(group,epsilon) for group in ('final','local','local_delta','delta') for epsilon in (1e-4,1e-3,1e-2)]
for group,epsilon in conditions:
 torch.manual_seed(seed);model,cfg=build(dict(family='v08',nodes=32,block=16,method='triangular'),4096);initialize(model,seed)
 optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01);state=None;steps=[]
 def normalize(x,*args,**kwargs):
  frame=inspect.currentframe().f_back;file=Path(frame.f_code.co_filename);line=frame.f_lineno
  isv08=file.parent.name=='v08' and file.name=='model.py'
  isv07=file.parent.name=='v07' and file.name=='model.py'
  final=isv08 and line==152
  local=(isv08 and line in (128,152)) or (isv07 and line==220)
  delta=isv08 and line==156
  selected=(group=='final' and final) or (group=='local' and local) or (group=='local_delta' and (local or delta)) or (group=='delta' and delta)
  if selected:
   dim=kwargs.get('dim',1);return x/torch.sqrt(x.square().sum(dim,keepdim=True)+epsilon**2)
  result=original(x,*args,**kwargs)
  if group=='detach_final' and final:return result.detach()
  if group=='mask_final' and final:
   arrived=frame.f_locals['drive'][...,-1]>1e-4
   return torch.where(arrived[...,None],result,torch.zeros_like(result))
  return result
 F.normalize=normalize
 try:
  for step in range(3):
   optimizer.zero_grad(set_to_none=True);batch=tokens[:,step*64:step*64+65]
   y,state,diag=model(batch[:,:-1],state);loss=F.cross_entropy(y.flatten(0,1),batch[:,1:].flatten());loss.backward()
   grads=[(n,p.grad.detach()) for n,p in model.named_parameters() if p.grad is not None]
   norms={n:float(g.double().norm()) for n,g in grads}
   norm64=math.sqrt(sum(value**2 for value in norms.values()))
   norm32=float(torch.stack([g.norm().square() for _,g in grads]).sum().sqrt())
   metrics=dict(step=step,loss=float(loss.detach()),norm64=norm64,norm32=norm32,
     all_elements_finite=all(bool(g.isfinite().all()) for _,g in grads),
     input_projection_bias_grad=norms['input_projection.bias'],content_projection_grad=norms['content_projection.weight'],
     raw_content_gain_grad=norms['raw_content_gain'],read_projection_grad=norms['read_projection.weight'],
     output_token_variation=float((y[:,1:]-y[:,:-1]).detach().norm()),
     reached=float(diag['reached'].detach()),slow_rms=float(diag['slow_rms'].detach()))
   steps.append(metrics)
   if not math.isfinite(norm32):break
   torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();state=invalidate_feedback(detach(state))
  status='completed' if len(steps)==3 and math.isfinite(steps[-1]['norm32']) else 'failed'
  record=dict(group=group,epsilon=epsilon,status=status,steps=steps)
 except Exception as exc:record=dict(group=group,epsilon=epsilon,status='failed',error=repr(exc),steps=steps)
 finally:F.normalize=original
 output.append(record);(root/'normalization_probes.json').write_text(json.dumps(dict(seed=seed,documents=docs,starts=starts,conditions=output),indent=2,allow_nan=True))
 print(json.dumps(record),flush=True)
