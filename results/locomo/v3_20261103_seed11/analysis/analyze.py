"""Audit the completed evaluation and plot scores; no model inference or training."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[2]
OUT=Path(__file__).parent
sys.path.insert(0,str(PROJECT))
from evaluations.locomo.scoring import normalize_answer


def main():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    summary=json.loads((ROOT/'summary.json').read_text())
    registry=json.loads((ROOT/'registry.json').read_text())
    assert summary['status']==registry['status']=='completed'
    for key in ('model_sources','evaluator_sources'):
        assert all(hashlib.sha256((PROJECT/p).read_bytes()).hexdigest()==h for p,h in protocol[key].items())
    for name,sha in protocol['checkpoint_sha256'].items():
        path=Path(protocol['source_run'])/f'{name}_s{protocol["seed"]}/last.pt'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
    details={};records={};loss_pairs={};main_ids=None
    for name,row in summary['conditions'].items():
        rows=[r for p in sorted((ROOT/name).glob('c*_b*.json')) for r in json.loads(p.read_text())['rows']]
        assert len(rows)==1986 and len({r['id'] for r in rows})==1986
        records[name]={r['id']:r for r in rows if r['status']=='completed' and r['category']!=5}
        if 'oracle' not in name:
            ids=set(records[name]);assert len(ids)==1540
            if main_ids is None:main_ids=ids
            else:assert ids==main_ids
        else:assert len(records[name])==1527
        done=[r for r in rows if r['status']=='completed']
        main=list(records[name].values())
        frequencies=Counter(r['prediction'] for r in done)
        common_text,common_count=frequencies.most_common(1)[0]
        exact=0;nonempty=0
        for r in main:
            target=str(r['reference_answer'])
            if r['category']==3:target=target.split(';')[0].strip()
            gold=normalize_answer(target)
            if gold:
                nonempty+=1;exact+=normalize_answer(r['prediction'])==gold
        timing=json.loads((ROOT/name/'timing.json').read_text())
        details[name]=dict(**row,seconds=timing['seconds'],
            f1_positive_questions=sum(r['f1']>0 for r in main),normalized_exact_matches=exact,
            nonempty_normalized_references=nonempty,
            most_common_prediction_count=common_count,most_common_prediction_sha256=hashlib.sha256(common_text.encode()).hexdigest(),
            most_common_generated_word=Counter(common_text.split()).most_common(1)[0][0] if common_text.split() else '',
            main_first_hits=sum(bool(r['first_hit']) for r in main),
            history_state_bytes=[v['runtime_state_bytes'] for v in timing['conversations']] if 'oracle' not in name else None)
    for other in ('D-clear','D-donor','D-oracle','S-main','L128','F512'):
        a,b=records['D-main'],records[other];ids=sorted(set(a)&set(b));groups=sorted({a[i]['conversation'] for i in ids})
        sums=np.array([sum(a[i]['answer_loss_sum']-b[i]['answer_loss_sum'] for i in ids if a[i]['conversation']==g) for g in groups])
        counts=np.array([sum(a[i]['answer_tokens'] for i in ids if a[i]['conversation']==g) for g in groups])
        assert all(a[i]['answer_tokens']==b[i]['answer_tokens'] for i in ids)
        rng=np.random.default_rng(1103);index=rng.integers(0,len(groups),size=(20000,len(groups)))
        samples=sums[index].sum(1)/counts[index].sum(1)
        loss_pairs[f'D-main-minus-{other}']=dict(questions=len(ids),answer_nll_difference=float(sums.sum()/counts.sum()),
            conversation_bootstrap_95ci=np.quantile(samples,[.025,.975]).tolist())
    audit=dict(status='completed',conditions=10,raw_question_records=sum(v['processed'] for v in details.values()),
        model_sources_unchanged=len(protocol['model_sources']),evaluator_sources_unchanged=len(protocol['evaluator_sources']),
        checkpoints_unchanged=4,model_parameter_runtime_digest_checks='passed for all ten conditions',
        full_main_questions_per_condition=1540,oracle_questions_per_condition=1527,
        main_missing_evidence_not_removed=True,training_performed=False,
        extra_model_inference_in_analysis=False,seed=11,judge_used=False)
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    stats=dict(conditions=details,paired_answer_nll=loss_pairs,
               total_group_seconds=sum(v['seconds'] for v in details.values()),seed=11,
               note='Official word F1, one frozen seed; all ten conditions. Analysis reads saved predictions only.')
    (OUT/'statistics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
    names=list(details);f1=[details[n]['main_f1']*100 for n in names];nll=[details[n]['answer_nll'] for n in names]
    colors=['#c98231' if 'oracle' in n else '#9aa4ad' if n in ('D-clear','D-donor') else '#4b7396' for n in names]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    axes[0].bar(names,f1,color=colors);axes[0].set(ylabel='Official word F1 (%)',title='Free-generation answer scores',ylim=(0,max(5,max(f1)*1.3)))
    for at,value in enumerate(f1):axes[0].text(at,value+.08,f'{value:.2f}',ha='center',fontsize=8)
    axes[1].bar(names,nll,color=colors);axes[1].set(ylabel='Token-weighted answer NLL',title='Teacher-forced diagnostic (lower is better)')
    for ax in axes:
        ax.tick_params(axis='x',rotation=55);ax.grid(axis='y',alpha=.15)
    fig.suptitle('V3 LoCoMo | seed 11 | main 1540 QA, oracle 1527 QA | greedy, 50-token cap',fontsize=12)
    fig.savefig(OUT/'comparison.png',dpi=180);fig.savefig(OUT/'comparison.svg');plt.close(fig)
    print(json.dumps(dict(audit=audit,statistics=stats),ensure_ascii=False))


if __name__=='__main__':main()
