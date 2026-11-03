"""Read-only result analysis; no model forward, training, or test-set scoring."""
from pathlib import Path
import json
import math
import statistics

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
NAMES = ('transformer', 'adapter', 'bio_summary', 'bio_ordered')
RESULTS = {name: json.loads((ROOT/name/'result.json').read_text()) for name in NAMES}
SAMPLE = json.loads((ROOT/'validation_sample.json').read_text())
RNG = np.random.default_rng(20261026)
REPEATS = 50_000


def compare(a, b, task='qa', a_condition='warm', b_condition='warm', first=False):
    one = RESULTS[a]['validation'][task]['conditions'][a_condition]
    two = RESULTS[b]['validation'][task]['conditions'][b_condition]
    field = 'first_answer_nll_per_window' if first else 'per_window_nll'
    weight_field = 'first_answer_targets_per_window' if first else 'per_window_targets'
    values = np.asarray(one[field])-np.asarray(two[field])
    weight = np.asarray(one[weight_field])
    assert np.array_equal(weight, np.asarray(two[weight_field]))
    groups = (SAMPLE['qa']['documents'] if task == 'qa' else SAMPLE['lm']['documents'])
    ids = list(dict.fromkeys(groups))
    numerator = np.asarray([sum(values[i]*weight[i] for i,g in enumerate(groups) if g == group) for group in ids])
    denominator = np.asarray([sum(weight[i] for i,g in enumerate(groups) if g == group) for group in ids])
    indices = RNG.integers(0, len(ids), size=(REPEATS, len(ids)))
    draws = numerator[indices].sum(1)/denominator[indices].sum(1)
    return dict(difference_a_minus_b=float((values*weight).sum()/weight.sum()),
                article_or_document_bootstrap_95ci=np.quantile(draws, [.025,.975]).tolist(),
                unique_clusters=len(ids), rows=len(values), effective_targets=int(weight.sum()),
                per_window_difference=values.tolist(), a_lower=int((values < 0).sum()),
                a_higher=int((values > 0).sum()), ties=int((values == 0).sum()),
                note='Conditional uncertainty across this fixed validation sample; no training-seed uncertainty.')


table, comparisons = {}, {}
for name, result in RESULTS.items():
    validation = result['validation']
    record = dict(lm_nll=validation['lm']['warm_nll'], lm_ppl=validation['lm']['warm_ppl'],
                  qa_nll=validation['qa']['warm_nll'], first_answer_nll=validation['qa']['first_answer_nll'],
                  first_answer_accuracy=validation['qa']['first_answer_accuracy'],
                  qa_window_mean_nll=validation['qa']['conditions']['warm']['window_mean_nll'],
                  train_seconds=result.get('train_seconds'), answer_targets=result.get('answer_targets'),
                  trainable_parameters=result['trainable_parameters'])
    f = ROOT/name/'validation.jsonl'
    if f.exists():
        records = [json.loads(line) for line in f.read_text().splitlines()]
        recorded = sum(row['lm']['seconds']+row['qa']['seconds'] for row in records)
        final = sum(validation[task]['seconds'] for task in ('lm', 'qa'))
        record.update(recorded_periodic_and_initial_validation_seconds=recorded,
                      additional_final_validation_seconds=final,
                      total_recorded_validation_seconds=recorded+final,
                      validation_fraction_of_train_and_validation=(recorded+final)/(recorded+final+result['train_seconds']))
    table[name] = record
for task in ('qa', 'lm'):
    for a,b in (('bio_ordered','transformer'),('bio_summary','transformer'),('adapter','transformer'),
                ('bio_ordered','bio_summary'),('bio_ordered','adapter')):
        comparisons[f'{task}/{a}-minus-{b}'] = compare(a,b,task)
for name in ('bio_summary','bio_ordered'):
    comparisons[f'qa_first/{name}-minus-transformer'] = compare(name,'transformer',first=True)
    for condition in ('empty_memory','memory_read_off','unrelated_history','shuffled_history',
                      'memory_read_off_all','remote_history_reencoded'):
        comparisons[f'qa/{name}/{condition}-minus-warm'] = compare(name,name,a_condition=condition)
        comparisons[f'qa_first/{name}/{condition}-minus-warm'] = compare(name,name,a_condition=condition,first=True)
derived = dict(
    ordered_over_summary_training_time=table['bio_ordered']['train_seconds']/table['bio_summary']['train_seconds'],
    summary_over_adapter_training_time=table['bio_summary']['train_seconds']/table['adapter']['train_seconds'],
    ordered_over_adapter_training_time=table['bio_ordered']['train_seconds']/table['adapter']['train_seconds'],
    ordered_lm_ppl_improvement_percent=100*(1-table['bio_ordered']['lm_ppl']/table['transformer']['lm_ppl']),
    adapter_lm_ppl_increase_percent=100*(table['adapter']['lm_ppl']/table['transformer']['lm_ppl']-1),
    ordered_qa_ppl_improvement_over_summary_percent=100*(1-math.exp(table['bio_ordered']['qa_nll']-table['bio_summary']['qa_nll'])))
payload = dict(status='completed', table=table, comparisons=comparisons, derived=derived,
               bootstrap_repetitions=REPEATS, seed=20261026,
               data_scope='Saved validation predictions only, 8 QA rows / 7 articles / 52 answer tokens. Reserved test untouched.')
(OUT/'paired_statistics.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))

colors = dict(adapter='#E58D2C', bio_summary='#278F91', bio_ordered='#694BB0', transformer='#555555')
labels = dict(adapter='FFN adapter', bio_summary='Bio summary', bio_ordered='Bio ordered', transformer='Frozen Transformer')
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
fig, axes = plt.subplots(2,2,figsize=(13,8.3),constrained_layout=True)
for name in NAMES[1:]:
    records = [json.loads(line) for line in (ROOT/name/'validation.jsonl').read_text().splitlines()]
    x = [row['updates'] for row in records]
    axes[0,0].plot(x,[row['qa']['warm_nll'] for row in records],'-o',ms=3,color=colors[name],label=labels[name])
    axes[0,1].plot(x,[row['qa']['first_answer_nll'] for row in records],'-o',ms=3,color=colors[name],label=labels[name])
for ax,key,title in ((axes[0,0],'qa_nll','Answer NLL (lower is better)'),(axes[0,1],'first_answer_nll','First answer token NLL (lower is better)')):
    ax.axhline(table['transformer'][key],color=colors['transformer'],ls='--',lw=1,label=labels['transformer'])
    ax.axvline(64,color='gray',ls=':',lw=1)
    ax.set(title=title,xlabel='Optimizer updates (CPT ends at 64)',ylabel='NLL / token')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
ax = axes[1,0]
for name in NAMES[2:]:
    data = comparisons[f'qa/{name}/remote_history_reencoded-minus-warm']
    ax.plot(np.arange(1,9),data['per_window_difference'],'-o',ms=4,color=colors[name],label=labels[name])
ax.axhline(0,color='black',lw=.8)
ax.set(title='Replace remote evidence, reencode: per-QA effect',xlabel='Fixed validation QA row',ylabel='NLL(changed history) - NLL(correct history)')
ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,-3));ax.grid(alpha=.2);ax.legend(fontsize=8)
ax = axes[1,1]
conditions = ('empty_memory','memory_read_off_all','unrelated_history','shuffled_history','remote_history_reencoded')
x=np.arange(len(conditions));width=.34
for offset,name in ((-.5,'bio_summary'),(.5,'bio_ordered')):
    values=[comparisons[f'qa/{name}/{c}-minus-warm']['difference_a_minus_b'] for c in conditions]
    bars=ax.bar(x+offset*width,values,width,color=colors[name],label=labels[name])
    for bar,value in zip(bars,values):
        if abs(value)<.005: ax.annotate(f'{value:+.1e}',(bar.get_x()+width/2,max(0,value)),xytext=(0,5 if name=='bio_ordered' else 18),textcoords='offset points',ha='center',fontsize=7,rotation=20)
ax.axhline(0,color='black',lw=.8)
ax.set(title='QA response to memory interventions',ylabel='NLL(intervention) - NLL(warm)',xticks=x,
       xticklabels=['Clear state','Read off\nwhole path','Wrong\nmemory','Shuffle\nmemory','Remote text\nreencoded'])
ax.set_ylim(-.006,.12);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Bio-Matrix v1.1 — validation-only, one seed, 8 QA rows / 7 articles',fontsize=14)
fig.savefig(OUT/'comparison.png',dpi=160)
fig.savefig(OUT/'comparison.svg')
print(json.dumps(dict(table=table,derived=derived,selected={k:comparisons[k] for k in (
    'qa/bio_ordered-minus-bio_summary','qa/bio_ordered/remote_history_reencoded-minus-warm',
    'qa/bio_ordered/unrelated_history-minus-warm','lm/bio_ordered-minus-transformer')}),ensure_ascii=False,indent=2))
