"""Saved-result analysis only; no model inference, optimizer update or test scoring."""
from pathlib import Path
import json
import statistics

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent
SEEDS=(11,23,37)
NAMES=('trans_local','trans_full','v2_static','v2_dynamic')
RESULTS={(name,seed):json.loads((ROOT/f'{name}_s{seed}'/'result.json').read_text()) for name in NAMES for seed in SEEDS}
SAMPLE=json.loads((ROOT/'validation_sample.json').read_text())
SUMMARY=json.loads((ROOT/'comparison.json').read_text())
RNG=np.random.default_rng(20261103)
REPEATS=30_000


def compare(a,b=None,task='qa',condition='warm',first=False):
    b=a if b is None else b
    field='first_answer_nll_per_window' if first else 'per_window_nll'
    count='first_answer_targets_per_window' if first else 'per_window_targets'
    differences=[]
    for seed in SEEDS:
        one=RESULTS[a,seed]['validation'][task]['conditions'][condition]
        two=RESULTS[b,seed]['validation'][task]['conditions']['warm']
        assert one['status']==two['status']=='completed'
        assert one[count]==two[count]
        differences.append(np.asarray(one[field])-np.asarray(two[field]))
    differences=np.stack(differences);weights=np.asarray(one[count])
    groups=SAMPLE[task]['documents'];unique=list(dict.fromkeys(groups))
    numerator=np.stack([np.sum(differences[:,np.array(groups)==g]*weights[np.array(groups)==g],axis=1) for g in unique],axis=1)
    denominator=np.array([weights[np.array(groups)==g].sum() for g in unique])
    si=RNG.integers(0,len(SEEDS),size=(REPEATS,len(SEEDS)))
    gi=RNG.integers(0,len(unique),size=(REPEATS,len(unique)))
    draws=numerator[si[:,:,None],gi[:,None,:]].sum((1,2))/len(SEEDS)/denominator[gi].sum(1)
    point=np.sum(differences*weights,axis=1)/weights.sum()
    return dict(mean_difference=float(point.mean()),per_seed=point.tolist(),
                crossed_seed_and_source_percentile_95ci=np.quantile(draws,[.025,.975]).tolist(),
                clusters=len(unique),rows=len(groups),targets_per_seed=int(weights.sum()),
                per_question_mean=differences.mean(0).tolist(),
                note='Descriptive crossed bootstrap over 3 observed seeds and fixed source clusters; limited seed population and validation sample.')


comparisons={}
for task in ('qa','lm'):
    for other in ('trans_local','trans_full','v2_static'):
        comparisons[f'{task}/dynamic-minus-{other}']=compare('v2_dynamic',other,task)
for name in ('v2_static','v2_dynamic'):
    for condition in ('empty_memory','memory_read_off_all','unrelated_history','shuffled_history','remote_history_reencoded'):
        comparisons[f'qa/{name}/{condition}-minus-warm']=compare(name,condition=condition)
    comparisons[f'first/{name}/remote-minus-warm']=compare(name,condition='remote_history_reencoded',first=True)

audits={};trajectories={};retention={};accuracy={}
for seed in SEEDS:
    samples=[];budgets=[]
    for name in NAMES:
        folder=ROOT/f'{name}_s{seed}'
        rows=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
        assert len(rows)==576
        assert [x['stage'] for x in rows]==['cpt']*64+['pt']*512
        samples.append([x['sample_sha256'] for x in rows])
        result=RESULTS[name,seed]
        assert result['cpt_tokens']==131072 and result['pt_updates']==512
        budgets.append(result['answer_targets'])
        val=[json.loads(line) for line in (folder/'validation.jsonl').read_text().splitlines()]
        assert [x['stage'] for x in val]==['initial','cpt_end','pt_mid','final']
        trajectories[f'{name}_s{seed}']=[dict(update=x['updates'],stage=x['stage'],qa_nll=x['qa']['warm_nll'],lm_ppl=x['lm']['warm_ppl']) for x in val]
        before=val[0]['lm']['warm_ppl'];after=result['validation']['lm']['warm_ppl']
        retention[f'{name}_s{seed}']=dict(initial_ppl=before,final_ppl=after,increase_percent=100*(after/before-1),last_diagnostics=rows[-1]['diagnostics'])
        if name.startswith('v2'):
            cond=result['validation']['qa']['conditions']
            accuracy[f'{name}_s{seed}']={c:[i for i,x in enumerate(cond[c]['first_answer_accuracy_per_window']) if x]
                                       for c in ('warm','memory_read_off_all','remote_history_reencoded')}
    assert all(x==samples[0] for x in samples) and len(set(budgets))==1
    audits[str(seed)]=dict(updates=576,cpt_targets=131072,pt_updates=512,answer_targets=budgets[0],all_samples_and_masks_paired=True)

payload=dict(status='completed',aggregate=SUMMARY['aggregate'],comparisons=comparisons,paired_audit=audits,
             trajectories=trajectories,retention=retention,first_correct_indices=accuracy,
             qa_rows=len(SAMPLE['qa']['documents']),qa_articles=len(set(SAMPLE['qa']['documents'])),
             unique_qa_records=len(set(SAMPLE['qa']['record_ids'])),
             total_train_seconds=sum(r['train_seconds'] for r in RESULTS.values()),
             total_validation_seconds=sum(r['validation_seconds'] for r in RESULTS.values()),
             test_evaluated=False,model_executed=False,bootstrap_repeats=REPEATS)
(OUT/'statistics.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))

colors=dict(trans_local='#D28526',trans_full='#343B45',v2_static='#25999B',v2_dynamic='#7451AF')
labels=dict(trans_local='Local Transformer',trans_full='Full-window Transformer',v2_static='v2 fixed addresses',v2_dynamic='v2 dynamic addresses')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(13,8.2),constrained_layout=True)
for name in NAMES:
    allrows=[trajectories[f'{name}_s{s}'] for s in SEEDS]
    x=[v['update'] for v in allrows[0]]
    for ax,key,title in ((axes[0,0],'qa_nll','Answer NLL across training'),(axes[0,1],'lm_ppl','Language-model PPL across training')):
        values=np.array([[r[key] for r in rows] for rows in allrows]);mean=values.mean(0);sd=values.std(0,ddof=1)
        ax.plot(x,mean,'-o',ms=4,color=colors[name],label=labels[name]);ax.fill_between(x,mean-sd,mean+sd,color=colors[name],alpha=.12)
        ax.set(title=title,xlabel='Optimizer updates (CPT ends at 64)',ylabel=key)
        ax.axvline(64,ls=':',color='gray');ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=8)
ax=axes[1,0];keys=[f'qa/{name}/remote_history_reencoded-minus-warm' for name in ('v2_static','v2_dynamic')]+[f'first/{name}/remote-minus-warm' for name in ('v2_static','v2_dynamic')]
for i,key in enumerate(keys):
    v=comparisons[key];lo,hi=v['crossed_seed_and_source_percentile_95ci'];point=v['mean_difference']
    ax.errorbar(point,i,xerr=[[max(0,point-lo)],[max(0,hi-point)]],fmt='o',capsize=4,color=colors['v2_dynamic' if 'dynamic' in key else 'v2_static'])
ax.axvline(0,color='black',lw=.8)
ax.set(title='Remote evidence replacement: loss increase',xlabel='NLL(changed history) - NLL(correct history)',yticks=range(4),yticklabels=['Fixed: answer','Dynamic: answer','Fixed: first token','Dynamic: first token'])
ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
ax=axes[1,1]
base=SUMMARY['aggregate']['trans_local']['train_seconds']['mean']
times=[SUMMARY['aggregate'][n]['train_seconds']['mean'] for n in NAMES]
bars=ax.bar(range(4),times,color=[colors[n] for n in NAMES])
for bar,t in zip(bars,times):ax.text(bar.get_x()+bar.get_width()/2,t+2,f'{t:.1f}s\n{t/base:.2f}x',ha='center',fontsize=9)
ax.set(title='Pure training time per seed (576 updates)',ylabel='Seconds',xticks=range(4),xticklabels=['Local TF','Full TF','Fixed v2','Dynamic v2'],ylim=(0,125));ax.grid(axis='y',alpha=.2)
fig.suptitle('Bio-Matrix v2 | 3 training seeds | fixed 64-row validation | no final test',fontsize=14)
fig.savefig(OUT/'comparison.png',dpi=160);fig.savefig(OUT/'comparison.svg')
print(json.dumps(dict(qa_rows=payload['qa_rows'],qa_articles=payload['qa_articles'],unique_qa_records=payload['unique_qa_records'],
                     total_train_seconds=payload['total_train_seconds'],total_validation_seconds=payload['total_validation_seconds'],
                     selected={k:comparisons[k] for k in ('qa/dynamic-minus-v2_static','qa/v2_dynamic/remote_history_reencoded-minus-warm','first/v2_dynamic/remote-minus-warm')}),ensure_ascii=False,indent=2))
