"""Plot completed holdout results and immutable raw learning traces."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
summary = json.loads((ROOT/'summary.json').read_text())
assert summary['status'] == 'completed'
stats = summary['statistics']; names = list(stats)
labels = ['TF128', 'TF512', 'Bio static', 'Bio dynamic', 'Qwen3.5', 'DS4 orig.', 'DS4 balanced']
colors = ['#a6adb4', '#626f7b', '#69a28c', '#216c56', '#d19132', '#6d79ba', '#344c8c']
seeds = [11, 23, 37]
fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout='constrained')

def mean_sd(key):
    values = np.array([[r[key] for r in stats[n]['per_seed']] for n in names])
    return values.mean(1), values.std(1, ddof=1)

means, sd = mean_sd('cpt_ppl')
axes[0, 0].bar(range(7), means, yerr=sd, color=colors, capsize=3)
unigram = np.mean(list(summary['unigram_same_cpt_targets_test_ppl'].values()))
axes[0, 0].axhline(unigram, color='#555555', ls='--', label='Train-only unigram')
bigram = json.loads((ROOT/'bigram_control.json').read_text())['ppl_mean']
axes[0, 0].axhline(bigram, color='#aa574b', ls=':', label='Count bigram (different state budget)')
axes[0, 0].set_title('Pretraining checkpoint: full held-out text PPL')
axes[0, 0].legend(fontsize=8)
x = np.arange(7)
for offset, key, label, alpha in [(-.18, 'f1', 'Correct context', 1.), (.18, 'unrelated_f1', 'Unrelated context', .45)]:
    m, s = mean_sd(key)
    axes[0, 1].bar(x+offset, m*100, .34, yerr=s*100, color=colors, alpha=alpha, capsize=2, label=label)
controls = json.loads((ROOT/'qa_prior_controls.json').read_text())['controls']
prior = max(c.get('same_generation_strata_weighted_f1', 0.) for c in controls.values())
axes[0, 1].axhline(prior*100, color='#555555', ls='--', label='Best constant-answer control')
axes[0, 1].set_title('Held-out answer generation: population-weighted F1 (%)')
axes[0, 1].legend(fontsize=8)
axes[1, 0].bar(x, [stats[n]['train_seconds_mean'] for n in names], color=colors)
axes[1, 0].set_title('Final CPT + QA: mean core training seconds / seed')
for offset, key, label in [(-.23, 'unrelated_answer_nll_effect', 'Replace whole context'),
                            (0., 'remote_answer_nll_effect_far_bin', 'Replace remote part, >256 bin'),
                            (.23, 'independent_prior_answer_nll_effect', 'Add unrelated prior to new task')]:
    m, s = mean_sd(key)
    axes[1, 1].bar(x+offset, m, .22, yerr=s, capsize=2, label=label)
axes[1, 1].axhline(0., color='#555555', lw=.8)
axes[1, 1].set_title('Paired answer-NLL change under history interventions')
axes[1, 1].legend(fontsize=8)
for ax in axes.flat:
    ax.set_xticks(x, labels, rotation=25, ha='right'); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
fig.suptitle('Corrected local 333K comparison | 3 final seeds | error bars: seed SD')
fig.savefig(OUT/'comparison.png', dpi=170); fig.savefig(OUT/'comparison.svg'); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(15, 5), layout='constrained'); budgets = {}
for name, label, color in zip(names, labels, colors):
    budgets[name] = {}
    for phase, ax in zip(('cpt', 'qa'), axes):
        traces = [[json.loads(x) for x in (ROOT/f'final_{name}_s{s}_{phase}/metrics.jsonl').read_text().splitlines()] for s in seeds]
        expected = 256 if phase == 'cpt' else 512
        assert all(len(trace) == expected for trace in traces)
        values = np.asarray([[r['ce'] for r in trace] for trace in traces])
        smoothed = np.asarray([np.convolve(row, np.ones(16)/16, mode='valid') for row in values])
        xx = np.arange(16, expected+1)
        if phase == 'cpt':
            xx = xx*4096/1e6
        mean, sd = smoothed.mean(0), smoothed.std(0, ddof=1)
        ax.plot(xx, mean, color=color, label=label); ax.fill_between(xx, mean-sd, mean+sd, color=color, alpha=.09)
        budgets[name][phase] = dict(final_core_seconds=sum(sum(r['seconds'] for r in t) for t in traces),
            final_supervised_targets=sum(sum(r['targets'] for r in t) for t in traces),
            final_input_tokens=sum(sum(r['inputs'] for r in t) for t in traces))
    for phase in ('cpt', 'qa'):
        results = [json.loads((ROOT/f'{phase}_search_{name}_r{i}/result.json').read_text()) for i in range(3)]
        budgets[name][phase]['search_core_seconds'] = sum(r['train_seconds'] for r in results)
axes[0].set_title('Text next-token training CE'); axes[0].set_xlabel('Supervised text targets (millions)')
axes[1].set_title('Answer + EOS training CE'); axes[1].set_xlabel('QA optimizer updates')
for ax in axes:
    ax.grid(alpha=.2); ax.legend(fontsize=8); ax.set_ylabel('Cross-entropy')
fig.suptitle('Raw training traces | 16-update moving average | mean +/- seed SD')
fig.savefig(OUT/'learning_curves.png', dpi=170); fig.savefig(OUT/'learning_curves.svg'); plt.close(fig)
(OUT/'compute_budget.json').write_text(json.dumps(budgets, ensure_ascii=False, indent=2)+'\n')

history = json.loads((ROOT/'lm_history_controls.json').read_text())['conditions']
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), layout='constrained')
modes = [('remote128_replaced', 'Replace farthest 128 inputs'),
         ('prefix383_replaced_current_token_kept', 'Replace history; keep current token'),
         ('prefix384_replaced', 'Replace history + first current token')]
for ax, (mode, title) in zip(axes, modes):
    for i, (name, color) in enumerate(zip(names, colors)):
        point = history[name]['effects'][mode]
        lo, hi = point['paired_seed_document_95ci']
        ax.vlines(i, lo, hi, color=color, lw=2); ax.scatter(i, point['nll_increase'], color=color, s=40)
    ax.axhline(0, color='#555555', ls='--', lw=.8); ax.grid(axis='y', alpha=.2)
    ax.set_xticks(x, labels, rotation=30, ha='right'); ax.set_title(title); ax.set_ylabel('Paired text-NLL increase')
fig.suptitle('Pretraining checkpoints | 128 fixed text windows | exploratory seed/document 95% intervals')
fig.savefig(OUT/'text_history.png', dpi=170); fig.savefig(OUT/'text_history.svg'); plt.close(fig)
print('Saved completed holdout plots, learning curves and measured compute budgets.')
