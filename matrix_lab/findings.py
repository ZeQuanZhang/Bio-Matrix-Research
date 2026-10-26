"""Build cross-experiment tables from completed raw results only."""
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

from .extension_run import VARIANTS as EXTENSIONS
from .optimizer_run import VARIANTS as OPTIMIZERS


def run():
    core = json.loads(Path('results/matrix/summary.json').read_text())
    assert core['complete']
    extension = {v: [json.loads((Path('results/matrix_extensions')/f'{v}_seed{s}'/'result.json').read_text())
                     for s in (42, 137, 2026)] for v in EXTENSIONS}
    signals = {v: [json.loads((Path('results/matrix_signals')/f'{v}_seed{s}.json').read_text())
                   for s in (42, 137, 2026)] for v in ('spectral_zero', 'spectral_fixed', 'spectral_learned')}
    functions = json.loads(Path('results/matrix_optimizer/functions.json').read_text())
    neural = json.loads(Path('results/matrix_optimizer/neural.json').read_text())
    for seed in (42, 137, 2026):
        rows = [r for group in extension.values() for r in group if r['protocol']['seed'] == seed]
        assert len({r['schedule_sha256'] for r in rows}) == 1
        assert len({r['events'] for r in rows}) == 1
    lines = ['# Bio-Matrix 完整首轮研究结果', '',
        '核心状态实验21次，概率/频率/容量控制24次，周期预测9次；全部使用MPS。'
        '另做CPU解析函数120次和固定小网络12次优化器实验。每项均保留配置与原始结果。', '',
        '## Q1：递推状态是否保存了可读取的信息？', '',
        '核心实验的2/4键延迟24事件为主指标，理论机会水平25%。以下为三个种子的均值：', '',
        '| 模型 | 保留 | 清零 | 交换 |', '|---|---:|---:|---:|']
    for v, row in core['aggregate'].items():
        lines.append(f"| {v} | {row['accuracy_mean']:.1%} | {row['reset_accuracy']:.1%} | {row['swap_accuracy']:.1%} |")
    lines += ['', '这项任务中模型没有Attention或KV缓存。干预发生在查询之前，'
              '查询只携带键而无值；因此保留与清零的差异直接检验历史状态的作用。', '',
              '## Q2/Q3：二阶反应与动态矩阵是否提供增量收益？', '',
              '| 配对比较（前者减后者） | 平均准确率差，百分点 | 三个种子的差值 |', '|---|---:|---|']
    for name, values in core['paired_effects'].items():
        lines.append(f"| {name} | {statistics.mean(values)*100:+.2f} | " + ', '.join(f'{v*100:+.2f}' for v in values)+' |')
    gains = core['paired_effects']['gru_dynamic_minus_gru_add']
    quadratic = core['paired_effects']['reaction_dynamic_minus_gated_dynamic']
    lines += ['', f"GRU增加动态矩阵后，{sum(v > 0 for v in gains)}/3个种子提高准确率，平均提高{statistics.mean(gains)*100:.2f}个百分点。"
              f"在动态读出下加入二阶反应项，{sum(v < 0 for v in quadratic)}/3个种子反而降低准确率。"
              '当前结果支持继续复测动态读出，但没有证明二阶反应的稳定增量价值。']
    lines += ['', 'Gated与Reaction的差别为二阶乘性项；Add与Dynamic有相同的参数数和非线性输出头。'
              'GRU和Reaction循环参数量不同，因此不把这些结果描述为严格等参数或等FLOPs胜利。', '',
              '## Q4：概率与频率扩展是否有额外收益？', '',
              '这部分全部重训2000步，25%训练样本为有噪声重复观测，不能与核心4000步分数直接混排。'
              '括号为种子样本标准差。64/96维确定性模型用于检查额外状态容量的影响。', '',
              '| 模型 | 参数 | 状态浮点数 | 正常2/4键延迟24 | 20%噪声准确率 | 35%噪声准确率 | 20%噪声清零 | 优化秒/次 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    ext_summary = {}
    for v, rows in extension.items():
        normal = [statistics.mean(r['test'][k]['carry']['accuracy'] for k in ('two_delay24', 'four_delay24')) for r in rows]
        noisy = [r['test']['noisy_p20']['carry']['accuracy'] for r in rows]
        noisy35 = [r['test']['noisy_p35']['carry']['accuracy'] for r in rows]
        a = {'normal_accuracy': statistics.mean(normal), 'normal_std': statistics.stdev(normal),
             'noisy20_accuracy': statistics.mean(noisy), 'noisy20_std': statistics.stdev(noisy),
             'noisy35_accuracy': statistics.mean(noisy35),
             'noisy20_reset': statistics.mean(r['test']['noisy_p20']['reset']['accuracy'] for r in rows),
             'seconds': statistics.mean(r['seconds'] for r in rows),
             'parameters': rows[0]['protocol']['parameters'], 'state_floats': rows[0]['protocol']['state_floats']}
        ext_summary[v] = a
        lines.append(f"| {v} | {a['parameters']:,} | {a['state_floats']} | {a['normal_accuracy']:.1%} ({a['normal_std']:.1%}) | "
                     f"{a['noisy20_accuracy']:.1%} ({a['noisy20_std']:.1%}) | {a['noisy35_accuracy']:.1%} | {a['noisy20_reset']:.1%} | {a['seconds']:.2f} |")
    delta = ext_summary['bayes']['noisy20_accuracy'] - ext_summary['reaction64']['noisy20_accuracy']
    ratio = ext_summary['bayes']['seconds']/ext_summary['reaction64']['seconds']
    lines += ['', f'概率版本相对64维普通状态，20%噪声准确率差为{delta*100:+.2f}个百分点，优化耗时为{ratio:.2f}倍。']
    lines += ['', 'Gaussian版本采用对角EKF线性化与学习得到的伪观测，尚未做概率校准；'
              '因此分数变化不能单独解释为“模型理解了知识的不确定性”。'
              '另一个限制是尚无保留同一伪观测结构、仅替换方差更新的点估计对照，不能把差异完全归因于不确定性。', '',
              '## 周期预测：频率机制的独立检验', '',
              '| 状态 | 同频率16步 MSE | 同频率64步 MSE | 未见频率16步 MSE | 观测噪声0.2 MSE | 同频率16步清零 MSE |',
              '|---|---:|---:|---:|---:|---:|']
    signal_summary = {}
    for v, rows in signals.items():
        a = {k: statistics.mean(r['test'][k]['carry']['mse'] for r in rows) for k in rows[0]['test']}
        a['reset'] = statistics.mean(r['test']['seen16']['reset']['mse'] for r in rows)
        signal_summary[v] = a
        lines.append(f"| {v} | {a['seen16']:.5f} | {a['seen64']:.5f} | {a['unseen16']:.5f} | {a['noise20']:.5f} | {a['reset']:.5f} |")
    if signal_summary['spectral_fixed']['seen16'] < signal_summary['spectral_zero']['seen16'] and signal_summary['spectral_fixed']['seen64'] > signal_summary['spectral_zero']['seen64']:
        lines += ['', '本轮振荡状态在16步预测上更好，但64步外推更差；短期周期拟合收益没有转化为更长预测优势。']
    lines += ['', '输入只提供前16个观测值，预测段全部缺失；成功也只支持周期结构利用，不能外推为语言记忆。', '',
              '## 独立优化器测试', '', '| 函数 | 方法 | 成功次数/10 | 最终损失均值 ± SD | 最好损失均值 | 秒/次 |', '|---|---|---:|---:|---:|---:|']
    optimizer_summary = {}
    for function in ('rosenbrock', 'rastrigin', 'ackley'):
        optimizer_summary[function] = {}
        for v in OPTIMIZERS:
            rows = [r for r in functions if r['function'] == function and r['variant'] == v]
            assert len(rows) == 10 and all(r['gradient_calls'] == 2000 for r in rows)
            scores = [r['loss'] for r in rows]
            a = {'successes': sum(r['success'] for r in rows), 'loss': statistics.mean(scores),
                 'std': statistics.stdev(scores), 'best': statistics.mean(r['best_loss'] for r in rows),
                 'seconds': statistics.mean(r['seconds'] for r in rows)}
            optimizer_summary[function][v] = a
            lines.append(f"| {function} | {v} | {a['successes']} | {a['loss']:.5g} ± {a['std']:.5g} | {a['best']:.5g} | {a['seconds']:.3f} |")
    lines += ['', '| 固定XOR小网络的方法 | 测试准确率 | CE | 秒/次 |', '|---|---:|---:|---:|']
    for v in OPTIMIZERS:
        rows = [r for r in neural if r['variant'] == v]; assert len(rows) == 3
        lines.append(f"| {v} | {statistics.mean(r['accuracy'] for r in rows):.1%} | {statistics.mean(r['loss'] for r in rows):.5f} | {statistics.mean(r['seconds'] for r in rows):.3f} |")
    lines += ['', '滤波和周期方法具有额外状态与计算；每个方法的梯度/函数调用预算相同，'
              '但没有声称等时间。所有解析函数只用2维测试，结果不能外推到大模型优化或保证全局最优。']
    ackley = optimizer_summary['ackley']
    lines += ['', f"二维Ackley上，余弦Adam的最终损失均值为{ackley['adam_cosine']['loss']:.4f}，"
              f"加入梯度滤波后为{ackley['gaussian_cosine']['loss']:.4f}；"
              '但未达到预设成功阈值，且其他函数与小网络没有给出一致的通用改进证据。'
              '周期探索相对单独滤波没有显示稳定的额外收益。']
    diagnostics = json.loads(Path('results/matrix/state_diagnostics.json').read_text())['runs']
    lines += ['', '## 事后状态诊断', '',
              '在额外生成的64条2键/24干扰样本上检查状态，未据此修改模型。更新门和饱和度取写入后的25步平均，'
              '包括干扰和查询；局部Jacobian仅取最后一次干扰的4条样本。', '',
              '| 模型 | 平均更新门 | 状态绝对值>0.95比例 | 局部Jacobian最大奇异值 | 二阶/线性项RMS比 |', '|---|---:|---:|---:|---:|']
    for v in core['aggregate']:
        group = [d for d in diagnostics if d['variant'] == v]
        avg = lambda key: statistics.mean(d[key] for d in group)
        ratio = f"{avg('quadratic_to_linear_rms_ratio'):.3f}" if group[0]['quadratic_to_linear_rms_ratio'] is not None else '—'
        lines.append(f"| {v} | {avg('update_gate_mean'):.4f} | {avg('state_abs_gt_095_fraction'):.1%} | {avg('local_jacobian_max_singular_mean'):.3f} | {ratio} |")
    lines += ['', '较小的更新门与较低的状态饱和可以提示保留方式的差异，但不能单独确定性能差距的原因。'
              '单步Jacobian最大奇异值大于1，也不等于整条序列已经梯度爆炸。']
    lines += ['', '## 下一轮优先事项', '',
              '优先复测简单保留门与动态读出的组合，并把写入阶段和保持阶段的更新规则作为独立变量。'
              '在相同数据与超参数搜索预算下检查结果是否稳定，再判断是否保留二阶项。'
              '当前不建议直接扩大到Transformer：先解决稳定保留和机制增量价值，比增加状态复杂度更有判别力。']
    lines += ['', '## 可复现材料', '', '- 核心完整表格、曲线：`results/matrix/report.md`。',
              '- 概率/频率原始指标、状态干预与权重：`results/matrix_extensions/`。',
              '- 周期预测逐样本MSE与权重：`results/matrix_signals/`。',
              '- 优化器轨迹、成功步数与小网络权重：`results/matrix_optimizer/`。', '',
              '三个种子只能提供探索性证据；本轮没有学习率网格搜索，也没有严格等参数、等FLOPs或等墙钟预算重训。'
              '扩大到Transformer前，应先复现所关注机制相对容量控制的稳定增益。', '',
              '[核心完整报告](../../results/matrix/report.md) · '
              '[扩展完整报告](../../results/matrix_extensions/report.md) · '
              '[周期预测曲线](../../results/matrix_signals/curves.png) · '
              '[优化器曲线](../../results/matrix_optimizer/curves.png)', '',
              '![核心状态干预](../../results/matrix/interventions.png)', '',
              '![扩展准确率](../../results/matrix_extensions/memory_accuracy.png)']
    Path('matrix_lab/docs/FINDINGS.md').write_text('\n'.join(lines)+'\n')
    summary = {'complete': True, 'core': core, 'extensions': ext_summary, 'signals': signal_summary, 'optimizers': optimizer_summary}
    Path('results/matrix_all_summary.json').write_text(json.dumps(summary, indent=2))
    ext_lines = ['# 概率与频率扩展：全部难度和状态干预', '',
                 '| 模型 | 条件 | 保留准确率 | 清零准确率 | 交换准确率 | 保留CE |', '|---|---|---:|---:|---:|---:|']
    for variant, rows in extension.items():
        for condition in rows[0]['test']:
            acc = [statistics.mean(r['test'][condition][i]['accuracy'] for r in rows) for i in ('carry', 'reset', 'swap')]
            loss = statistics.mean(r['test'][condition]['carry']['loss'] for r in rows)
            ext_lines.append(f'| {variant} | {condition} | {acc[0]:.1%} | {acc[1]:.1%} | {acc[2]:.1%} | {loss:.4f} |')
    ext_lines += ['', '![学习曲线](learning_curves.png)', '', '![记忆准确率](memory_accuracy.png)']
    Path('results/matrix_extensions/report.md').write_text('\n'.join(ext_lines)+'\n')
    fig, axis = plt.subplots(figsize=(10, 4.5))
    for number, variant in enumerate(EXTENSIONS):
        for index, row in enumerate(extension[variant]):
            path = Path('results/matrix_extensions')/f"{variant}_seed{row['protocol']['seed']}"/'metrics.jsonl'
            trace = [json.loads(line) for line in path.read_text().splitlines()]
            trace = [t for t in trace if 'train_loss' in t]
            axis.plot([t['step'] for t in trace], [t['train_loss'] for t in trace],
                      color=plt.get_cmap('tab10')(number), alpha=.5, label=variant if index == 0 else None)
    axis.set(xlabel='Optimizer step', ylabel='Query cross entropy', title='Separate extension experiment: 2000 steps')
    axis.legend(fontsize=7); axis.grid(alpha=.2); fig.tight_layout()
    fig.savefig('results/matrix_extensions/learning_curves.png', dpi=180); plt.close(fig)
    fig, axis = plt.subplots(figsize=(12, 4.8)); x = np.arange(len(EXTENSIONS))
    for i, (metric, error, label) in enumerate((('normal_accuracy', 'normal_std', 'Normal'), ('noisy20_accuracy', 'noisy20_std', '20% observation noise'))):
        axis.bar(x+(i-.5)*.35, [ext_summary[v][metric] for v in EXTENSIONS], width=.35,
                 yerr=[ext_summary[v][error] for v in EXTENSIONS], capsize=3, label=label)
    axis.axhline(.25, color='black', linestyle='--', label='Chance')
    axis.set_xticks(x, EXTENSIONS, rotation=20, ha='right'); axis.set(ylabel='Accuracy', ylim=(0, 1.02), title='Means and seed SD; state capacity differs')
    axis.legend(); fig.tight_layout(); fig.savefig('results/matrix_extensions/memory_accuracy.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for number, (variant, rows) in enumerate(signals.items()):
        for i, row in enumerate(rows):
            trace = [t for t in row['trace'] if 'validation_mse' in t]
            axes[0].plot([t['step'] for t in trace], [t['validation_mse'] for t in trace],
                         color=plt.get_cmap('tab10')(number), alpha=.6, label=variant if i == 0 else None)
        axes[1].plot([16, 64], [signal_summary[variant]['seen16'], signal_summary[variant]['seen64']], marker='o', label=variant)
    axes[0].set(xlabel='Optimizer step', ylabel='Validation MSE', title='Periodic forecasting validation')
    axes[1].set(xlabel='Unobserved forecast steps', ylabel='Test MSE', title='Forecast length extrapolation')
    for axis in axes:
        axis.grid(alpha=.2); axis.legend(fontsize=7)
    fig.tight_layout(); fig.savefig('results/matrix_signals/curves.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for axis, function in zip(axes, ('rosenbrock', 'rastrigin', 'ackley')):
        for v in OPTIMIZERS:
            rows = [r for r in functions if r['function'] == function and r['variant'] == v]
            trace = np.array([[p['best'] for p in r['trace']] for r in rows])
            axis.plot([p['step'] for p in rows[0]['trace']], np.maximum(trace.mean(0), 1e-8), label=v)
        axis.set(title=function, xlabel='Gradient calls', ylabel='Mean best objective', yscale='log'); axis.grid(alpha=.2)
    axes[-1].legend(fontsize=6); fig.tight_layout(); fig.savefig('results/matrix_optimizer/curves.png', dpi=180); plt.close(fig)
    print('All registered experiments complete; findings written.')


if __name__ == '__main__':
    run()
