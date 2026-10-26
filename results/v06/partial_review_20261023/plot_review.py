import json,os
from pathlib import Path
out=Path('results/v06/partial_review_20261023')
os.environ.setdefault('MPLCONFIGDIR',str(out/'.matplotlib'))
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
root=Path('results/v06/public_lm')
post=json.loads((out/'posthoc_inference.json').read_text())
full=next(r for r in post['conditions'] if r['id']=='core_n32_full_s11')
tf=next(r for r in post['conditions'] if r['id']=='core_n32_transformer_s11')
fig,axes=plt.subplots(1,2,figsize=(11.5,4.6),layout='constrained')
colors={'message':'#2876a8','independent':'#59a14f','full':'#e27b21','transformer':'#9467bd'}
for variant in colors:
    path=root/f'core_n32_{variant}_s11'/'validation.jsonl'
    data=[json.loads(x) for x in path.read_text().splitlines()]
    axes[0].plot([r['step'] for r in data],[r['conditions']['carry']['nll'] for r in data],
                 label=variant,color=colors[variant],linewidth=2 if variant in ('full','transformer') else 1.5,
                 linestyle={'message':'--','independent':':','full':'-','transformer':'-'}[variant])
axes[0].set(title='Validation: the graph variants almost overlap',xlabel='Optimizer update',ylabel='BPE token NLL')
axes[0].legend(fontsize=9);axes[0].grid(alpha=.2)
values=[full['carry_ppl'],full['reset_each_token_ppl'],full['head_only_ppl'],tf['carry_ppl']]
labels=['Full: normal state','Full: fresh state every token','Full: remove graph readout','Transformer: normal cache']
for y,(value,label) in enumerate(zip(values,labels)):
    axes[1].scatter(value,y,s=75,color='#9467bd' if y==3 else '#e27b21',marker='o' if y in (0,3) else 'D')
    axes[1].text(value+2,y,f'{value:.2f}',va='center',fontsize=10)
axes[1].set_yticks(range(4),labels);axes[1].invert_yaxis();axes[1].set_xlim(400,555)
axes[1].set(title='Inference ablations: most quality survives',xlabel='Test BPE perplexity (lower is better)')
axes[1].grid(axis='x',alpha=.2)
fig.suptitle('Bio-Matrix v0.6 interim review | N=32, seed=11 | 24 test windows',fontsize=12)
fig.savefig(out/'interim_review.png',dpi=170);plt.close(fig)
print(out/'interim_review.png')
