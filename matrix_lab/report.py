"""Aggregate all registered seeds, including unfavorable and incomplete results."""
import json
import os
from pathlib import Path
import statistics

_fontconfig = Path.home()/'.config/codex-fonts/fonts.conf'
if _fontconfig.is_file():
    os.environ.setdefault('FONTCONFIG_FILE', str(_fontconfig))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PRIMARY = ('two_delay24', 'four_delay24')


def report(root=Path('results/matrix')):
    cfg = json.loads((root / 'config.json').read_text())
    rows = {}
    for variant in cfg['variants']:
        rows[variant] = [json.loads(p.read_text()) for seed in cfg['seeds']
                         if (p := root / f'{variant}_seed{seed}' / 'result.json').exists()]
    complete = all(len(v) == len(cfg['seeds']) for v in rows.values())
    for seed in cfg['seeds']:
        paired = [r for values in rows.values() for r in values if r['protocol']['seed'] == seed]
        assert len({r['data_schedule_sha256'] for r in paired}) <= 1, 'Different training event schedules'
        assert len({r['events_seen'] for r in paired}) <= 1
        assert all(r['protocol']['persistent_state_floats'] == cfg['state'] for r in paired)
    aggregate = {}
    lines = ['# Bio-Matrix 状态记忆实验', '', f'状态：{"全部完成" if complete else "部分完成，以下不是最终比较"}。', '',
             '无 Attention、无 KV Cache、无查询阶段的写入历史。所有模型只依靠显式递推状态。', '',
             '主指标预先固定为 two_delay24 和 four_delay24 的准确率与交叉熵等权平均。'
             '正式模型均训练相同事件序列与步数，状态容量相同；参数和实际计算量不完全相同。', '',
             '| 模型 | 种子数 | 参数 | 持久状态浮点数 | 主准确率（均值 ± SD） | 主 CE | 清零准确率 | 交换准确率 | 优化秒/次 | 事件/秒 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    primary = lambda r, condition, metric: statistics.mean(r['test'][name][condition][metric] for name in PRIMARY)
    for variant, values in rows.items():
        if not values:
            continue
        scores = [primary(r, 'carry', 'accuracy') for r in values]
        aggregate[variant] = {'runs': len(values), 'accuracy_mean': statistics.mean(scores),
            'accuracy_std': statistics.stdev(scores) if len(scores) > 1 else 0.,
            'loss_mean': statistics.mean(primary(r, 'carry', 'loss') for r in values),
            'reset_accuracy': statistics.mean(primary(r, 'reset', 'accuracy') for r in values),
            'swap_accuracy': statistics.mean(primary(r, 'swap', 'accuracy') for r in values),
            'parameters': values[0]['protocol']['parameters'], 'state_floats': cfg['state'],
            'seconds': statistics.mean(r['optimization_seconds'] for r in values),
            'events_per_second': statistics.mean(r['events_per_second'] for r in values), 'per_seed_accuracy': scores}
        a = aggregate[variant]
        lines.append(f"| {variant} | {len(values)} | {a['parameters']:,} | {a['state_floats']} | "
                     f"{a['accuracy_mean']:.1%} ± {a['accuracy_std']:.1%} | {a['loss_mean']:.4f} | "
                     f"{a['reset_accuracy']:.1%} | {a['swap_accuracy']:.1%} | {a['seconds']:.2f} | {a['events_per_second']:,.0f} |")
    comparisons = [('gated_add', 'linear_add'), ('reaction_add', 'gated_add'),
                   ('reaction_dynamic', 'gated_dynamic'), ('gru_dynamic', 'gru_add'),
                   ('reaction_dynamic', 'reaction_add'), ('reaction_dynamic', 'gru_dynamic')]
    lines += ['', '## 配对消融', '', '| 比较（前者减后者） | 主准确率差（百分点） | 每个种子的差值 |', '|---|---:|---|']
    effects = {}
    for left, right in comparisons:
        a = {r['protocol']['seed']: r for r in rows.get(left, [])}
        b = {r['protocol']['seed']: r for r in rows.get(right, [])}
        differences = [primary(a[s], 'carry', 'accuracy') - primary(b[s], 'carry', 'accuracy') for s in cfg['seeds'] if s in a and s in b]
        if differences:
            effects[f'{left}_minus_{right}'] = differences
            lines.append(f"| {left} − {right} | {statistics.mean(differences)*100:+.2f} | "
                         + ', '.join(f'{d*100:+.2f}' for d in differences) + ' |')
    lines += ['', '## 完整难度结果', '', '| 模型 | 条件 | 保留准确率 | 清零准确率 | 交换准确率 | 保留 CE |', '|---|---|---:|---:|---:|---:|']
    for variant, values in rows.items():
        if not values:
            continue
        for condition in values[0]['test']:
            acc = [statistics.mean(r['test'][condition][intervention]['accuracy'] for r in values) for intervention in ('carry', 'reset', 'swap')]
            loss = statistics.mean(r['test'][condition]['carry']['loss'] for r in values)
            lines.append(f'| {variant} | {condition} | {acc[0]:.1%} | {acc[1]:.1%} | {acc[2]:.1%} | {loss:.4f} |')
    lines += ['', '## 工程成本', '', '| 模型 | CPU 进程峰值 RSS GB | 采样 Metal 驱动 GB | 非查询事件稠密 MAC 估计 |', '|---|---:|---:|---:|']
    for variant, values in rows.items():
        if values:
            lines.append(f"| {variant} | {statistics.mean(r['process_peak_rss_bytes'] for r in values)/1e9:.3f} | "
                         f"{statistics.mean(r['sampled_metal_driver_bytes'] for r in values)/1e9:.3f} | {values[0]['dense_macs_per_nonquery_event_estimate']:,} |")
    lines += ['', 'MAC 只估计稠密线性运算，未包括查询读出、激活、优化器和反向传播，不作为完整 FLOPs。'
              'Metal 数值为驱动分配量采样，RSS 为独立训练进程的峰值；二者不能相加当作统一内存峰值。'
              '优化秒不含数据生成、验证、保存和测试；分批评测的状态交换只在当前批内滚动。', '',
              '## 解释边界', '', '四种值的理论机会准确率为 25%。正式比较有三个种子；标准差描述种子波动，'
              '不等于置信区间或统计显著性。所有预设条件和种子均报告，没有按照测试分数挑选模型。', '',
              '主测试处于训练过的难度范围；64/128 延迟和6/8键属于外推。'
              '多查询的4次查询超过训练时最多2次。输入为中文符号编码的合成事件，并非自然中文语言理解任务。', '',
              '这些实验验证特定状态单元与读出方式，不能外推为真实生物机制、持续学习或通用大模型优势。', '',
              '![训练和验证曲线](learning_curves.png)', '', '![状态干预](interventions.png)']
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    (root / 'summary.json').write_text(json.dumps({'complete': complete, 'aggregate': aggregate, 'paired_effects': effects}, indent=2))
    colors = dict(zip(cfg['variants'], plt.get_cmap('tab10').colors))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for variant in cfg['variants']:
        for number, seed in enumerate(cfg['seeds']):
            path = root / f'{variant}_seed{seed}' / 'metrics.jsonl'
            if not path.exists():
                continue
            entries = [json.loads(line) for line in path.read_text().splitlines()]
            for axis, key in zip(axes, ('train_loss', 'validation_accuracy')):
                chosen = [e for e in entries if key in e]
                axis.plot([e['step'] for e in chosen], [e[key] for e in chosen],
                          color=colors[variant], alpha=.55, label=variant if number == 0 else None)
    axes[0].set(title='Training query cross entropy', ylabel='Cross entropy')
    axes[1].set(title='Validation retrieval accuracy', ylabel='Accuracy', ylim=(0, 1.03))
    for axis in axes:
        axis.set_xlabel('Optimizer step'); axis.grid(alpha=.2); axis.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(root / 'learning_curves.png', dpi=180); plt.close(fig)
    fig, axis = plt.subplots(figsize=(11, 4.5))
    present = list(aggregate); x = np.arange(len(present))
    for i, (key, title) in enumerate((('accuracy_mean', 'Carry'), ('reset_accuracy', 'Reset'), ('swap_accuracy', 'Swap'))):
        error = [aggregate[v]['accuracy_std'] for v in present] if i == 0 else None
        axis.bar(x + (i - 1)*.25, [aggregate[v][key] for v in present], width=.25, label=title, yerr=error, capsize=3)
    axis.axhline(.25, color='black', linestyle='--', linewidth=1, label='Chance')
    axis.set_xticks(x, present, rotation=20, ha='right'); axis.set(title='Means; carry error bars show seed SD', ylabel='Primary retrieval accuracy', ylim=(0, 1.05))
    axis.legend(); fig.tight_layout(); fig.savefig(root / 'interventions.png', dpi=180); plt.close(fig)
    print(json.dumps({'complete': complete, 'aggregate': aggregate, 'paired_effects': effects}, indent=2))


if __name__ == '__main__':
    report()
