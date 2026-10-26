"""Post-hoc inference only. Does not edit model sources or acquire the trainer lock."""
import json, math, os
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from v06.common import write, source_hashes, digest, PROJECT
from v06.data import Corpus
from v06.evaluate import windows
from v06.model import Config
from v06.run import build

torch.set_num_threads(1)
root=PROJECT/'results/v06/public_lm'
out=PROJECT/'results/v06/partial_review_20261023'
registered=json.loads((root/'conditions.json').read_text())
corpus=Corpus(); cfg=registered['config']; selected=windows(corpus,'test',cfg)
prefix,target=cfg['prefix_tokens'],cfg['target_tokens']
x=torch.tensor(selected['a'][:,prefix:prefix+target].copy(),dtype=torch.long).flatten()
y=torch.tensor(selected['a'][:,prefix+1:prefix+target+1].copy(),dtype=torch.long).flatten()
results=[]
for case in registered['conditions']:
    if case['status']!='completed': continue
    folder=root/case['id']; protocol=json.loads((folder/'protocol.json').read_text())
    model=build(Config(**protocol['model'])); checkpoint=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['model']); model.eval()
    result=json.loads((folder/'result.json').read_text())
    assert selected['sha256']==result['test']['windows_sha256']
    fresh_losses=[]; head_losses=[]
    with torch.no_grad():
        for start in range(0,len(x),16):
            current=x[start:start+16]; truth=y[start:start+16]
            # Batch positions are independent streams: all runtime state is fresh for every token.
            logits,_,_=model(current[:,None]); logits=logits[:,0]
            fresh_losses.extend(F.cross_entropy(logits,truth,reduction='none').tolist())
            if case['variant']!='transformer':
                e=model.embedding(current)
                h=model.output_norm(e+model.ff(model.norm(e)))
                direct=F.linear(h,model.embedding.weight)
                head_losses.extend(F.cross_entropy(direct,truth,reduction='none').tolist())
    carry=result['test']['conditions']['carry']['nll']
    row=dict(id=case['id'],variant=case['variant'],nodes=case['nodes'],seed=case['seed'],
             carry_nll=carry,carry_ppl=math.exp(carry),
             reset_each_token_nll=float(np.mean(fresh_losses)),
             reset_each_token_ppl=math.exp(float(np.mean(fresh_losses))),
             reset_each_token_delta_nll=float(np.mean(fresh_losses))-carry,
             reset_each_token_per_window_nll=np.array(fresh_losses).reshape(-1,target).mean(-1).tolist(),
             head_only_nll=float(np.mean(head_losses)) if head_losses else None,
             head_only_ppl=math.exp(float(np.mean(head_losses))) if head_losses else None,
             head_only_delta_nll=float(np.mean(head_losses))-carry if head_losses else None,
             head_only_per_window_nll=np.array(head_losses).reshape(-1,target).mean(-1).tolist() if head_losses else None)
    results.append(row)
    write(out/'posthoc_inference.json',dict(source_hashes=source_hashes(),evaluation_sha256=selected['sha256'],
        conditions=results,notes='CPU inference only; checkpoint weights fixed; no retraining. Head-only deletes graph readout in the already trained graph model. These are functional-dependence diagnostics, not independently trained baselines.'))
    print(json.dumps({k:v for k,v in row.items() if 'per_window' not in k}),flush=True)
