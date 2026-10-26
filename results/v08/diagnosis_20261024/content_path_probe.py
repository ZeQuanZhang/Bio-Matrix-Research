"""Current-input intervention after diagnostic smoothing; no production edits."""
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

torch.set_num_threads(4);seed=11;corpus=Corpus();values,docs,starts=corpus.sample('train',np.random.default_rng(seed),2,8*64);tokens=torch.tensor(values,dtype=torch.long);original=F.normalize
results=[]
for group,epsilon in [('mask_final',0.),('local',.01),('local_delta',.001)]:
 torch.manual_seed(seed);model,cfg=build(dict(family='v08',nodes=32,block=16,method='triangular'),4096);initialize(model,seed)
 optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01);state=None;steps=[]
 def normalize(x,*args,**kwargs):
  frame=inspect.currentframe().f_back;file=Path(frame.f_code.co_filename);line=frame.f_lineno
  v8=file.parent.name=='v08' and file.name=='model.py';v7=file.parent.name=='v07' and file.name=='model.py'
  local=(v8 and line in (128,152)) or (v7 and line==220)
  delta=v8 and line==156
  if (group=='local' and local) or (group=='local_delta' and (local or delta)):
   return x/torch.sqrt(x.square().sum(kwargs.get('dim',1),keepdim=True)+epsilon**2)
  result=original(x,*args,**kwargs)
  if group=='mask_final' and v8 and line==152:
   return torch.where((frame.f_locals['drive'][...,-1]>1e-4)[...,None],result,torch.zeros_like(result))
  return result
 F.normalize=normalize
 for step in range(3):
  optimizer.zero_grad(set_to_none=True);batch=tokens[:,step*64:step*64+65];y,state,diag=model(batch[:,:-1],state);loss=F.cross_entropy(y.flatten(0,1),batch[:,1:].flatten());loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();state=invalidate_feedback(detach(state))
  # Freeze identical old memory and remove pending feedback before input interventions.
  probe=tokens[:,(step+1)*64:(step+1)*64+16].clone();changed=probe.clone();changed[:,8]=(changed[:,8]+139)%4096
  with torch.no_grad():
   normal,_,_=model(probe,state)
   perturbed,_,_=model(changed,state)
   coeff=model.block_coefficients(probe,state)
   changed_coeff=model.block_coefficients(changed,state)
   delta=(coeff['transport']-changed_coeff['transport']).reshape(2,16,32,8,8)
   read_transport=(delta*model.read_ports[None,None,:,None,None]).sum(2)
   hook=model.content_projection.register_forward_hook(lambda module,args,output:torch.zeros_like(output))
   erased,_,_=model(probe,state);hook.remove()
  steps.append(dict(step=step,gradient_norm=float(norm),loss=float(loss.detach()),reached=float(diag['reached'].detach()),
      single_token_intervention_logit_max=float((normal-perturbed).abs().max()),single_token_read_port_transport_change=float(read_transport.norm()),
      erase_fresh_content_logit_max=float((normal-erased).abs().max())))
 F.normalize=original;record=dict(group=group,epsilon=epsilon,steps=steps);results.append(record);print(json.dumps(record),flush=True)
Path('results/v08/diagnosis_20261024/content_path_probe.json').write_text(json.dumps(results,indent=2))
