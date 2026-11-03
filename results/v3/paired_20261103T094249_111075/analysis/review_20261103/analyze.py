"""Audit saved V3 results; replay four existing validation prefixes for state diagnostics only."""
from pathlib import Path
import hashlib
import json
import math
import statistics
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent
PROJECT=ROOT.parents[2]
sys.path.insert(0,str(PROJECT))
from v3.train import build_models
from v11.data import QACorpus

SEEDS=(11,23,37)
NAMES=('trans_local','trans_full','v3_static','v3_dynamic')
PROTO=json.loads((ROOT/'protocol.json').read_text())
SAMPLE=json.loads((ROOT/'validation_sample.json').read_text())
SUMMARY=json.loads((ROOT/'summary.json').read_text())
RESULT={(name,seed):json.loads((ROOT/f'{name}_s{seed}/result.json').read_text()) for name in NAMES for seed in SEEDS}
RNG=np.random.default_rng(20261103)


def compare(a,b=None,task='qa',condition='warm',first=False):
    b=a if b is None else b
    field='first_answer_nll_per_window' if first else 'per_window_nll'
    count='first_answer_targets_per_window' if first else 'per_window_targets'
    differences=[]
    for seed in SEEDS:
        x=RESULT[a,seed]['validation'][task]['conditions'][condition]
        y=RESULT[b,seed]['validation'][task]['conditions']['warm']
        assert x['status']==y['status']=='completed' and x[count]==y[count]
        differences.append(np.asarray(x[field])-np.asarray(y[field]))
    differences=np.stack(differences);weights=np.asarray(x[count])
    groups=SAMPLE[task]['documents'];unique=list(dict.fromkeys(groups))
    numerator=np.stack([np.sum(differences[:,np.array(groups)==g]*weights[np.array(groups)==g],axis=1) for g in unique],axis=1)
    denominator=np.array([weights[np.array(groups)==g].sum() for g in unique])
    si=RNG.integers(0,len(SEEDS),size=(30_000,len(SEEDS)))
    gi=RNG.integers(0,len(unique),size=(30_000,len(unique)))
    draws=numerator[si[:,:,None],gi[:,None,:]].sum((1,2))/len(SEEDS)/denominator[gi].sum(1)
    per_seed=np.sum(differences*weights,axis=1)/weights.sum()
    return dict(mean_difference=float(per_seed.mean()),per_seed=per_seed.tolist(),
        crossed_seed_and_article_95ci=np.quantile(draws,[.025,.975]).tolist(),clusters=len(unique),
        windows=len(groups),targets_per_seed=int(weights.sum()),note='Exploratory crossed bootstrap: three observed seeds and fixed validation articles; shared questions across seeds.')


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    torch.set_num_threads(4)
    changed=[p for p,h in PROTO['source_hashes'].items() if hashlib.sha256((PROJECT/p).read_bytes()).hexdigest()!=h]
    assert not changed and SUMMARY['status']=='completed' and not SUMMARY['test_evaluated']
    comparisons={}
    for task in ('qa','lm'):
        for other in ('trans_local','trans_full','v3_static'):
            comparisons[f'{task}/dynamic-minus-{other}']=compare('v3_dynamic',other,task)
    comparisons['first/dynamic-minus-static']=compare('v3_dynamic','v3_static',first=True)
    for name in ('v3_static','v3_dynamic'):
        for condition in ('empty_memory','memory_read_off_all','unrelated_history','shuffled_history','remote_history_reencoded'):
            for task in ('qa','lm'):
                comparisons[f'{task}/{name}/{condition}-minus-warm']=compare(name,task=task,condition=condition)
        comparisons[f'first/{name}/remote-minus-warm']=compare(name,condition='remote_history_reencoded',first=True)
    audit={};stages={};training={};runtime={};snapshots={}
    qa=QACorpus(PROTO['training']['qa_root'])
    values,_,meta=qa.sample('validation',np.random.default_rng(PROTO['training']['evaluation_seed']),64)
    assert meta['sha256']==SAMPLE['qa']['sha256']
    for seed in SEEDS:
        models,_,_=build_models(4096,PROTO['training'],seed)
        paired=[]
        for name in NAMES:
            identity=f'{name}_s{seed}';folder=ROOT/identity
            metrics=[json.loads(x) for x in (folder/'metrics.jsonl').read_text().splitlines()]
            assert len(metrics)==1024 and [x['stage'] for x in metrics]==['cpt']*512+['pt']*512
            assert all(math.isfinite(x['ce']) and math.isfinite(x['gradient_norm']) for x in metrics)
            paired.append([x['sample_sha256'] for x in metrics])
            payload=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
            assert payload['protocol']==PROTO and payload['counters']['updates']==1024
            model=models[name]
            embedding_delta=float((model.embedding.weight.detach()-payload['model']['embedding.weight']).abs().sum())
            assert embedding_delta>0
            audit[identity]=dict(status=RESULT[name,seed]['status'],updates=len(metrics),
                cpt_targets=metrics[-1]['cpt_tokens'],pt_updates=metrics[-1]['pt_updates'],
                answer_targets=metrics[-1]['answer_targets'],finite_losses_and_gradients=True,
                embedding_changed_l1=embedding_delta,optimizer_states=len(payload['optimizer']['state']),
                runtime_bytes=metrics[-1]['runtime_bytes'])
            training[identity]={stage:dict(first32_ce=statistics.mean(x['ce'] for x in metrics[begin:begin+32]),
                last32_ce=statistics.mean(x['ce'] for x in metrics[end-32:end]),
                max_gradient_norm=max(x['gradient_norm'] for x in metrics[begin:end]))
                for stage,begin,end in (('cpt',0,512),('pt',512,1024))}
            stages[identity]=[{k:row[k] for k in ('stage','updates')}|dict(
                lm_ppl=row['lm']['warm_ppl'],qa_nll=row['qa']['warm_nll'],first_answer_nll=row['qa']['first_answer_nll'],
                first_answer_accuracy=row['qa']['first_answer_accuracy']) for row in
                (json.loads(x) for x in (folder/'validation.jsonl').read_text().splitlines())]
            if name.startswith('v3'):
                model.load_state_dict(payload['model']);model.eval()
                before=model.initial_state(4)
                with torch.no_grad():_,after,_=model(torch.as_tensor(values[:4,:384].copy(),dtype=torch.long))
                layers=[]
                for old,new in zip(before['memory'],after['memory']):
                    p=old['phase'].detach().reshape(4,4,32,8,2);q=new['phase'].reshape_as(p)
                    angle=torch.atan2(p[...,0]*q[...,1]-p[...,1]*q[...,0],(p*q).sum(-1))
                    layers.append(dict(phase_rms_radians=float(angle.square().mean().sqrt()),
                        key_rms_delta=float((new['key']-old['key'].detach()).square().mean().sqrt()),
                        value_rms_delta=float((new['value']-old['value'].detach()).square().mean().sqrt())))
                runtime[identity]=dict(scope='Four existing validation history prefixes, 384 tokens; state diagnostics only, no new score.',layers=layers)
                snapshots[identity]={k:dict(value=p.detach().tolist()) for k,p in model.named_parameters()
                    if k.endswith(('raw_address_rate','raw_write_rate','raw_temperature','log_tau'))}
        assert all(s==paired[0] for s in paired)
    stats=dict(comparisons=comparisons,stages=stages,training=training,runtime_replay=runtime,
        learned_controls=snapshots,aggregate=json.loads((ROOT/'comparison.json').read_text())['aggregate'],
        training_seconds=sum(r['train_seconds'] for r in RESULT.values()),
        validation_seconds=sum(r['validation_seconds'] for r in RESULT.values()),
        test_evaluated=False,validation=dict(qa_questions=64,qa_articles=20,answer_targets=475,lm_windows=64,lm_articles=28),
        note='Read saved results and checkpoints; replayed four existing validation prefixes only. No training or final test scoring.')
    save(OUT/'statistics.json',stats)
    save(OUT/'audit.json',dict(status='completed',source_files=len(PROTO['source_hashes']),changed_sources=changed,
        all_12_completed=True,paired_samples_equal_for_all_1024_steps=True,conditions=audit,test_evaluated=False,
        extra_inference='Four existing validation prefixes per Bio condition/seed for state diagnostics only.'))
    colors=dict(zip(NAMES,('#60758b','#263b54','#d18b24','#bf4562')))
    labels=dict(zip(NAMES,('Local Transformer','Full Transformer','Bio static','Bio dynamic')))
    fig,axes=plt.subplots(2,2,figsize=(12,8.6),constrained_layout=True)
    for name in NAMES:
        rows=[stages[f'{name}_s{s}'] for s in SEEDS];x=[r['updates'] for r in rows[0]]
        for ax,field in ((axes[0,0],'qa_nll'),(axes[0,1],'lm_ppl')):
            y=np.array([[r[field] for r in row] for row in rows])
            ax.plot(x,y.mean(0),marker='o',color=colors[name],label=labels[name])
            ax.fill_between(x,y.min(0),y.max(0),color=colors[name],alpha=.10)
    axes[0,0].set(title='Answer NLL during training',xlabel='Optimizer updates',ylabel='NLL (lower is better)')
    axes[0,1].set(title='Language PPL: CPT then QA adaptation',xlabel='Optimizer updates',ylabel='Validation PPL')
    for ax in axes[0]:
        ax.axvline(512,color='#999999',linestyle='--',alpha=.7);ax.legend(fontsize=8)
    selected=['qa/v3_static/remote_history_reencoded-minus-warm','qa/v3_dynamic/remote_history_reencoded-minus-warm',
              'first/v3_static/remote-minus-warm','first/v3_dynamic/remote-minus-warm']
    for at,key in enumerate(selected):
        row=comparisons[key];mean=row['mean_difference'];low,high=row['crossed_seed_and_article_95ci']
        axes[1,0].errorbar(mean,at,xerr=[[mean-low],[high-mean]],fmt='o',capsize=4,color='#bf4562')
    axes[1,0].axvline(0,color='#777777',linewidth=1)
    axes[1,0].set(yticks=range(4),yticklabels=['QA: static','QA: dynamic','First: static','First: dynamic'],
        title='Remote replacement cost, exploratory 95% CI',xlabel='NLL: remote replaced - correct history')
    axes[1,0].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    axes[1,0].ticklabel_format(axis='x',style='sci',scilimits=(0,0))
    seconds=[statistics.mean(RESULT[name,s]['train_seconds'] for s in SEEDS) for name in NAMES]
    axes[1,1].bar(range(4),seconds,color=[colors[n] for n in NAMES])
    axes[1,1].set(xticks=range(4),xticklabels=['Local TF','Full TF','Bio static','Bio dynamic'],
                  ylabel='Seconds per seed',title='Pure training time, 1024 updates')
    for at,value in enumerate(seconds):axes[1,1].text(at,value+3,f'{value:.1f}',ha='center',fontsize=9)
    for ax in axes.flat:ax.grid(axis='y',alpha=.15)
    fig.suptitle('Bio-Matrix V3: 3 seeds, fixed 64 QA questions; all first-token hits = 0',fontsize=13)
    fig.savefig(OUT/'comparison.png',dpi=180);fig.savefig(OUT/'comparison.svg');plt.close(fig)
    print(json.dumps(dict(comparisons=comparisons,runtime=runtime),ensure_ascii=False))


if __name__=='__main__':main()
