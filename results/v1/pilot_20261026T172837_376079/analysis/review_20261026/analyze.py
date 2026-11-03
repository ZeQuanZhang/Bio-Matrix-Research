"""Read-only paired analysis; never builds a model or runs training/inference."""
from pathlib import Path
import hashlib
import json
import math
import os

import numpy as np

OUT = Path(__file__).resolve().parent
RUN = OUT.parent.parent
os.environ.setdefault('MPLCONFIGDIR', str(OUT/'.mplconfig'))
os.environ['FONTCONFIG_FILE'] = '<HOME>/.config/codex-fonts/fonts.conf'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

summary = json.loads((RUN/'summary.json').read_text())
protocol = json.loads((RUN/'protocol.json').read_text())
sample = json.loads((RUN/'validation_sample.json').read_text())
conditions = summary['conditions']
assert summary['status'] == 'completed'
assert len(set(sample['documents'])) == 8
assert all(hashlib.sha256((RUN/'source'/p).read_bytes()).hexdigest() == sha
           for p, sha in protocol['source_hashes'].items())

names = ['transformer', 'adapter', 'bio_no_phase', 'bio_full']
logs = {}
for name in names[1:]:
    rows = [json.loads(line) for line in (RUN/name/'metrics.jsonl').read_text().splitlines()]
    assert [row['updates'] for row in rows] == list(range(1, 65))
    assert sum(row['targets'] for row in rows) == 65536
    logs[name] = rows
assert all([r['sample_sha256'] for r in logs[name]] == [r['sample_sha256'] for r in logs['adapter']]
           for name in names[1:])

rng = np.random.default_rng(20261026)
draws = rng.integers(0, 8, size=(50000, 8))

def paired(first, second):
    difference = np.asarray(first)-np.asarray(second)
    means = difference[draws].mean(1)
    interval = np.quantile(means, [.025, .975]).tolist()
    return dict(mean=float(difference.mean()), interval_95=interval,
                positive_windows=int((difference > 0).sum()), negative_windows=int((difference < 0).sum()),
                per_document=difference.tolist(),
                uncertainty='Document-window bootstrap conditional on one trained seed; not training-seed uncertainty')

def window(name, condition='warm'):
    return conditions[name]['validation']['conditions'][condition]['per_window_nll']

baseline = conditions['transformer']['validation']['warm_nll']
full = conditions['bio_full']['validation']
effects = {name+'_minus_transformer': paired(window(name), window('transformer')) for name in names[1:]}
effects['full_minus_no_phase'] = paired(window('bio_full'), window('bio_no_phase'))
for intervention in ['empty_memory', 'memory_read_off', 'unrelated_history', 'shuffled_history']:
    effects[intervention+'_minus_warm'] = paired(window('bio_full', intervention), window('bio_full'))

result = dict(status='completed', comparison='Fixed last checkpoint',
              run=RUN.name, source_archive_hashes_valid=True,
              training_samples_identical=True, validation_windows=8,
              validation_documents=sample['documents'], scored_targets=1024, training_seeds=1,
              effects=effects, full_ppl_improvement_percent=100*(1-math.exp(full['warm_nll']-baseline)),
              full_to_adapter_update_time=conditions['bio_full']['train_seconds']/conditions['adapter']['train_seconds'],
              phase_update_time_increase_percent=100*(conditions['bio_full']['train_seconds']/conditions['bio_no_phase']['train_seconds']-1),
              no_new_model_execution=True)
(OUT/'paired_statistics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')

fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.5), constrained_layout=True)
colors = dict(adapter='#777777', bio_no_phase='#d68a22', bio_full='#136eb0')
for name in names[1:]:
    rows = [json.loads(line) for line in (RUN/name/'validation.jsonl').read_text().splitlines()]
    axes[0, 0].plot([r['completed_tokens'] for r in rows], [r['warm_nll'] for r in rows],
                    '-o', color=colors[name], label=name)
axes[0, 0].axhline(baseline, color='#222222', linestyle='--', label='frozen Transformer')
axes[0, 0].set(title='Fixed-window validation trajectory', xlabel='Training targets', ylabel='Warm NLL (lower is better)')
axes[0, 0].legend(fontsize=8)

for index, name in enumerate(['bio_no_phase', 'bio_full']):
    row = effects[name+'_minus_transformer']
    center = -row['mean']
    lo, hi = -row['interval_95'][1], -row['interval_95'][0]
    axes[0, 1].errorbar(center, index, xerr=[[center-lo], [hi-center]], fmt='o', color=colors[name], capsize=5)
axes[0, 1].axvline(0, color='black', linewidth=.8)
axes[0, 1].set(yticks=[0, 1], yticklabels=['Bio no phase', 'Bio full'],
               title='Bio gain versus frozen Transformer', xlabel='NLL gain; positive = improvement (95% document CI)')
axes[0, 1].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

labels = ['Clear memory', 'Read off', 'Unrelated memory', 'Shuffled history']
keys = ['empty_memory', 'memory_read_off', 'unrelated_history', 'shuffled_history']
for index, key in enumerate(keys):
    row = effects[key+'_minus_warm']
    center = row['mean']; lo, hi = row['interval_95']
    axes[1, 0].errorbar(center, index, xerr=[[center-lo], [hi-center]], fmt='o', color='#136eb0', capsize=4)
axes[1, 0].axvline(0, color='black', linewidth=.8)
axes[1, 0].set(yticks=list(range(4)), yticklabels=labels,
               title='Bio full: direct memory interventions', xlabel='NLL increase from warm (95% document CI)')
axes[1, 0].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

times = [conditions[name]['train_seconds'] for name in names[1:]]
bars = axes[1, 1].bar(['Adapter', 'Bio no phase', 'Bio full'], times, color=[colors[n] for n in names[1:]])
axes[1, 1].bar_label(bars, labels=[f'{v:.2f}s' for v in times], padding=3)
axes[1, 1].set(title='Cumulative adaptation update time', ylabel='Seconds, including teacher forward')
axes[1, 1].set_ylim(0, max(times)*1.17)
for ax in axes.flat:
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.15, axis='x' if ax is axes[0, 1] or ax is axes[1, 0] else 'y')
fig.suptitle('Bio-Matrix v1.0: 65,536 targets, N32, one seed, 8 validation documents', fontsize=13)
fig.savefig(OUT/'comparison.png', dpi=170)
plt.close(fig)

table = '\n'.join(f'| {name} | {conditions[name]["validation"]["warm_nll"]:.9f} | '
                  f'{conditions[name]["validation"]["warm_ppl"]:.6f} | '
                  f'{conditions[name].get("train_seconds", 0):.3f} |' for name in names)
report = f'''# Bio-Matrix v1.0 首轮结果分析

**本轮完整Bio几乎保持冻结Transformer的原始预测，历史内容的增益极小；当前质量收益不足以支撑额外计算成本。**

三种适配各完成65,536有效目标、64次更新，逐步训练样本SHA完全相同。冻结Transformer只评分。44项自检通过，归档源码SHA匹配。原始记录见[summary](../../summary.json)和[protocol](../../protocol.json)。

## 固定末尾检查点

| 条件 | 暖NLL↓ | 暖PPL↓ | 核心适配秒 |
|---|---:|---:|---:|
{table}

完整Bio对冻结TF的平均NLL改善为{baseline-full['warm_nll']:.9f}，PPL改善{result['full_ppl_improvement_percent']:.5f}%。8个文档中只赢3个。full−TF配对区间为{effects['bio_full_minus_transformer']['interval_95']}，跨零。相位组对无相位组的NLL改善约0.000139，区间同样跨零。

完整组更新时间为Adapter的{result['full_to_adapter_update_time']:.2f}倍。两者同预算、额外可训练参数接近；计时包含教师前向，排除数据采样、验证和保存。冻结TF本轮没有训练，不能把这项比例当作相对TF训练速度。

## 记忆的直接作用

完整组清空Bio经验后NLL增加{full['memory_clear_cost']:.9f}；评分时关闭读取增加{full['read_off_cost']:.9f}；移植错配远史增加{full['unrelated_cost']:.9f}；打乱历史变化{effects['shuffled_history_minus_warm']['mean']:.9f}。各项文档区间均跨零。进入评分时注意力KV、位置和时钟相同。

正常暖状态相对冷启动的改善约{full['cold_nll']-full['warm_nll']:.6f} NLL；普通冻结Transformer也获得约{conditions['transformer']['validation']['cold_nll']-baseline:.6f}。这轮暖状态的大部分收益已有普通注意力上下文提供，Bio额外历史内容的贡献很小。

## 机制覆盖缺口

检查点只读统计见[分层状态统计](../field_dynamics_readonly_review.json)。默认J=2的传播使最外层18/32节点没有fresh观测支持，P保持初值、slow为零；完整相位组该层fast也近乎为零。相位本身在演化，但空间层没有充分参与新观测学习。需要补分层到达率验收，并修正默认传播覆盖。

每次更新从独立512-token文档窗开始。支持量与运行态不跨全部64次更新累积；参数巩固只在最后一步执行，主要涉及两层的中心节点。该轮尚无关闭巩固或同权重巩固前后评分，无法单独量化长期固化收益。

## 下一步优先级

先校准传播到达与实际读出幅度，记录每层各壳的有效观测/写入/输出贡献；再用有明确历史需求的监督和完整采样窗口教师训练写读。参数巩固需要多次触发与独立关闭对照。完成这些检查后，再决定增加训练预算和节点规模。

Adapter训练目标降低而固定验证变差。Top-64重归一化蒸馏会同时推动概率集中，其相对教师条件、真实CE和验证表现之间的关系需要单独审计；当前证据不能确认这是唯一原因。

## 范围

一个训练种子、8个不同验证文档、1,024评分目标；文档bootstrap只描述固定模型的抽样变动，不能代表训练种子不确定性。验证集曾被多轮研究使用，test未评估。没有执行新的模型前向、优化或长训练，原始结果未改动。

![结果对比](comparison.png)
'''
(OUT/'REPORT.md').write_text(report)
print(json.dumps(dict(report=str(OUT/'REPORT.md'), image=str(OUT/'comparison.png'),
                      full_gain=baseline-full['warm_nll'], full_ppl_percent=result['full_ppl_improvement_percent'],
                      full_vs_tf_ci=effects['bio_full_minus_transformer']['interval_95'],
                      full_vs_adapter_time=result['full_to_adapter_update_time']), ensure_ascii=False))
