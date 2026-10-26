"""Generate figures and a Chinese report exclusively from completed raw conditions."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
_fontconfig = Path.home()/'.config/codex-fonts/fonts.conf'
if _fontconfig.is_file():
    os.environ.setdefault('FONTCONFIG_FILE', str(_fontconfig))
os.environ.setdefault('MPLCONFIGDIR', str(Path('results/v03/mplconfig').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .run import ROOT, write
from .baselines import identifier as baseline_identifier
from .repeat import plan as repeat_plan

COLORS = {'C1': '#2878b5', 'C2': '#e59b24', 'C3': '#c83e4d'}
NODES = (32, 128, 512)


def score(result, ood=False, key=None, value=None, metric='correct'):
    rows = result['test']['episode_results']
    if ood:
        rows = [e for e in rows if e['objects'] == 64 or e['depth'] == 4 or e['holdout']]
    if key:
        rows = [e for e in rows if e[key] == value]
    return float(np.mean([e[metric] for e in rows])) if rows else float('nan')


def stats(values):
    a = np.asarray(values, dtype=float); a = a[np.isfinite(a)]
    if not len(a):
        return {'n': 0, 'mean': None, 'sd': None, 'ci95_half': None}
    sd = float(a.std(ddof=1)) if len(a) > 1 else None
    # Student-t intervals across independent training seeds, not question counts.
    critical = {2: 12.7062, 3: 4.30265, 4: 3.18245, 5: 2.77645}.get(len(a), 1.96)
    return {'n': len(a), 'mean': float(a.mean()), 'sd': sd,
            'ci95_half': critical*sd/math.sqrt(len(a)) if sd is not None else None, 'seed_values': a.tolist()}


def formatted(s, percentage=True):
    if not s['n']:
        return '未执行'
    factor = 100 if percentage else 1
    text = f"{factor*s['mean']:.2f}" + ('%' if percentage else '')
    return text + (f" ± {factor*s['sd']:.2f} (n={s['n']})" if s['sd'] is not None else ' (n=1)')


def savefig(path):
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', default=str(ROOT)); p.add_argument('--output', default='docs/V03_FINDINGS.md'); args = p.parse_args()
    root = Path(args.root); figures = root/'figures'; figures.mkdir(parents=True, exist_ok=True)
    statuses = json.loads((root/'conditions.json').read_text()); runs = []; byid = {}; diagnostics = {}
    for path in root.glob('*/result.json'):
        r = json.loads(path.read_text())
        if r.get('status') == 'completed' and 'protocol' in r:
            r['_folder'] = str(path.parent); runs.append(r); byid[r['protocol']['condition']['id']] = r
            dpath = path.parent/'diagnostics.json'
            if dpath.exists():
                diagnostics[r['protocol']['condition']['id']] = json.loads(dpath.read_text())
    mains = [r for r in runs if r['protocol']['condition']['stage'] == 'main']
    groups = defaultdict(list)
    for r in mains:
        c = r['protocol']['condition']; groups[(c['model'], c['nodes'])].append(r)
    summary = {'generated_utc': datetime.now(timezone.utc).isoformat(), 'status_counts': dict(Counter(v['status'] for v in statuses.values() if 'condition' in v)),
               'auxiliary_status_counts': dict(Counter(v['status'] for v in statuses.values() if 'condition' not in v)),
               'main': {}, 'paired': {}, 'scale_interactions': {}, 'stages': {}, 'breakdowns': {}}
    for key, rs in groups.items():
        kind, n = key
        summary['main'][f'{kind}_N{n}'] = {'overall': stats([score(r) for r in rs]), 'ood': stats([score(r, True) for r in rs]),
                                        'loss': stats([score(r, metric='loss') for r in rs]),
                                        'optimization_seconds': stats([r['optimization_seconds'] for r in rs])}
    for reference in ('C1', 'C2'):
        for ood in (False, True):
            differences = {}
            for n in NODES:
                a = {r['protocol']['condition']['seed']: r for r in groups[('C3', n)]}
                b = {r['protocol']['condition']['seed']: r for r in groups[(reference, n)]}
                differences[n] = {s: score(a[s], ood)-score(b[s], ood) for s in sorted(a.keys() & b.keys())}
                summary['paired'][f'C3-{reference}_N{n}_'+('ood' if ood else 'overall')] = stats(list(differences[n].values()))
            common = differences[32].keys() & differences[512].keys()
            summary['scale_interactions'][f'C3-{reference}_'+('ood' if ood else 'overall')] = stats([differences[512][s]-differences[32][s] for s in sorted(common)])
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, ood in zip(axs, (False, True)):
        for kind in COLORS:
            means = []; intervals = []
            for n in NODES:
                s = stats([score(r, ood) for r in groups[(kind, n)]])
                means.append(np.nan if not s['n'] else 100*s['mean']); intervals.append(0 if s['ci95_half'] is None else 100*s['ci95_half'])
                for r in groups[(kind, n)]:
                    seed = r['protocol']['condition']['seed']; offset = {11: -.045, 23: 0, 37: .045}.get(seed, 0)
                    ax.scatter(np.log2(n)+offset, 100*score(r, ood), color=COLORS[kind], s=14, alpha=.65)
            ax.errorbar(np.log2(NODES), means, yerr=intervals, marker='o', label=kind, color=COLORS[kind], capsize=3)
        ax.set_xticks(np.log2(NODES), NODES); ax.set_xlabel('Nodes N'); ax.set_ylabel('Accuracy (%)')
        ax.set_title(('OOD slices' if ood else 'All registered slices')+'; seed t-95% CI'); ax.set_ylim(0, 100); ax.legend()
    savefig(figures/'scale.png')
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, label in zip(axs, ('overall', 'ood')):
        for reference in ('C1', 'C2'):
            ss = [summary['paired'].get(f'C3-{reference}_N{n}_{label}', stats([])) for n in NODES]
            ax.errorbar(np.log2(NODES), [100*s['mean'] if s['n'] else np.nan for s in ss],
                        yerr=[100*s['ci95_half'] if s['ci95_half'] is not None else 0 for s in ss], marker='o', capsize=3, label=f'C3-{reference}')
        ax.axhline(0, color='gray', lw=.8); ax.axhline(2, color='gray', ls=':', lw=.8)
        ax.set_xticks(np.log2(NODES), NODES); ax.set_xlabel('Nodes N'); ax.set_ylabel('Paired difference (percentage points)'); ax.set_title(label); ax.legend()
        low, high = ax.get_ylim(); ax.set_ylim(min(-2.5, low), max(2.5, high))
    savefig(figures/'paired_gains.png')
    plt.figure(figsize=(6, 4))
    for kind in COLORS:
        values = [stats([score(r) for r in runs if r['protocol']['condition']['stage'] == 'capacity'
                         and r['protocol']['condition']['model'] == kind and r['protocol']['condition']['nodes'] == n]) for n in NODES]
        if any(s['n'] for s in values):
            plt.errorbar(np.log2(NODES), [100*s['mean'] if s['n'] else np.nan for s in values],
                         yerr=[100*s['ci95_half'] if s['ci95_half'] is not None else 0 for s in values],
                         marker='o', capsize=3, label=kind, color=COLORS[kind])
    plt.xticks(np.log2(NODES), NODES); plt.xlabel('Nodes N'); plt.ylabel('Accuracy (%)'); plt.ylim(0, 100); plt.title('8192 state floats; seed means/t-CI; parameters differ')
    if any(r['protocol']['condition']['stage'] == 'capacity' for r in runs):
        plt.legend()
    savefig(figures/'fixed_capacity.png')
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    for col, n in enumerate((128, 512)):
        for kind in COLORS:
            by_j = defaultdict(list)
            for r in runs:
                c = r['protocol']['condition']
                if c['stage'] in ('iterations', 'main') and c['nodes'] == n and c['model'] == kind:
                    by_j[c['iterations']].append(r)
            if by_j:
                for ax, metric, factor in ((axs[0, col], lambda r: score(r), 100), (axs[1, col], lambda r: r['optimization_seconds'], 1)):
                    ss = [stats([metric(r) for r in by_j[j]]) for j in (2, 8, 16, 32)]
                    ax.errorbar((2, 8, 16, 32), [factor*s['mean'] if s['n'] else np.nan for s in ss],
                                yerr=[factor*s['ci95_half'] if s['ci95_half'] is not None else 0 for s in ss],
                                marker='o', capsize=3, label=kind, color=COLORS[kind])
        axs[0, col].set_title(f'N={n}; retrained at J; seed t-95% CI'); axs[0, col].set_ylabel('Accuracy (%)'); axs[0, col].set_ylim(0, 100); axs[1, col].set_ylabel('2048-update training seconds')
        for ax in axs[:, col]:
            ax.set_xticks((2, 8, 16, 32)); ax.set_xlabel('Internal iterations J')
            if ax.lines:
                ax.legend()
    savefig(figures/'trained_iterations.png')
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, values in zip(axs, ('family', 'objects', 'depth'),
                               (('retrieve', 'trace', 'compare', 'count', 'rule'), (8, 16, 32, 64), (0, 1, 2, 3, 4))):
        for kind in COLORS:
            means = []
            for v in values:
                ss = stats([score(r, key=key, value=v) for r in groups[(kind, 512)]])
                summary['breakdowns'][f'{kind}_N512_{key}_{v}'] = ss
                means.append(100*ss['mean'] if ss['n'] else np.nan)
            ax.plot(range(len(values)), means, 'o-', color=COLORS[kind], label=kind)
        ax.set_xticks(range(len(values)), values, rotation=25 if key == 'family' else 0); ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0, 100); ax.set_title(f'N512 by {key}'); ax.legend()
    savefig(figures/'breakdowns.png')
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    phase_count = 0
    for identifier, diagnostic in diagnostics.items():
        r = byid[identifier]; c = r['protocol']['condition']
        if 'phase_behavior' in diagnostic:
            for metric, ax in zip(('R1', 'edge_PLV_last8'), axs):
                rows = diagnostic['phase_behavior']
                for family_index, family in enumerate(sorted({row['family'] for row in rows})):
                    subset = [row for row in rows if row['family'] == family]
                    ax.scatter(np.mean([row[metric] for row in subset]), 100*np.mean([row['correct'] for row in subset]),
                               color=COLORS[c['model']], marker={32: 'o', 128: 's', 512: '^'}[c['nodes']], alpha=.7,
                               label=f"{c['model']} N{c['nodes']}" if family_index == 0 else None)
                ax.set_xlabel(metric); ax.set_ylabel('Task-family accuracy (%)'); ax.set_ylim(0, 100)
            phase_count += len(diagnostic['phase_behavior'])
    axs[0].set_title('Fixed offline samples; association only'); axs[1].set_title('C2/C3; no claim from synchronization alone')
    if phase_count:
        axs[0].legend(fontsize=8)
    savefig(figures/'phase_behavior.png')
    plt.figure(figsize=(8, 4))
    intervention_rows = []
    for identifier, diagnostic in diagnostics.items():
        c = byid[identifier]['protocol']['condition']
        if c['model'] == 'C3' and c['nodes'] in (128, 512) and 'interventions' in diagnostic:
            conditions = diagnostic['interventions']['conditions']; names = ('rotate', 'phase_swap', 'phase_scramble', 'content_matched', 'cut_graph')
            values = [100*(conditions[f'{name}_4']['accuracy']-conditions['carry_4']['accuracy']) for name in names]
            plt.plot(range(len(names)), values, 'o-', label=f"C3 N{c['nodes']} seed11")
            intervention_rows.append({'id': identifier, 'conditions': {k: {m: v[m] for m in ('accuracy', 'loss')} for k, v in conditions.items()},
                                      'energy': diagnostic['interventions']['mean_squared_perturbation_energy']})
    plt.axhline(0, color='gray', lw=.8); plt.xticks(range(5), ('rotation', 'phase swap', 'phase scramble', 'content matched', 'graph cut'), rotation=15)
    plt.ylabel('Change vs carry after 4 extra updates (pp)'); plt.title('Matched checkpoint/input; single-seed causal diagnostics')
    if intervention_rows:
        plt.legend()
    savefig(figures/'interventions.png')
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for kind in COLORS:
        rows = [r for r in mains if r['protocol']['condition']['model'] == kind]
        for ax, cost in zip(axs, ('training', 'inference')):
            xs = []; ys = []
            for r in rows:
                identifier = r['protocol']['condition']['id']; d = diagnostics.get(identifier)
                if cost == 'inference' and not d:
                    continue
                xs.append(r['optimization_seconds'] if cost == 'training' else 1000*d['median_inference_batch_seconds']); ys.append(100*score(r))
            if xs:
                ax.scatter(xs, ys, label=kind, color=COLORS[kind])
            ax.set_xlabel('Optimization seconds' if cost == 'training' else 'Fixed-batch GPU latency (ms)'); ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0, 100); ax.set_title('Measured cost / performance; all seeds')
            if ax.collections:
                ax.legend()
    savefig(figures/'cost_performance.png')
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for kind in COLORS:
        for stage, marker in (('main', 'o'), ('capacity', 's')):
            rows = [r for r in runs if r['protocol']['condition']['model'] == kind and r['protocol']['condition']['stage'] == stage]
            if not rows:
                continue
            for ax, field, label in zip(axs, ('current_state_floats_per_sample', 'trainable_parameters'), ('Current-state floats / sample', 'Trainable parameters')):
                ax.scatter([r['ledger'][field] for r in rows], [100*score(r) for r in rows], marker=marker,
                           color=COLORS[kind], alpha=.7, label=f'{kind} '+('S1' if stage == 'main' else 'S2'))
                ax.set_xlabel(label); ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0, 100); ax.set_xscale('log')
    for ax in axs:
        ax.set_title('State / parameter budgets; actual completed conditions')
        if ax.collections:
            ax.legend(fontsize=8)
    savefig(figures/'state_parameter_budget.png')
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for kind in COLORS:
        losses = defaultdict(list); accuracies = defaultdict(list)
        for r in groups[(kind, 512)]:
            logs = [json.loads(line) for line in (Path(r['_folder'])/'metrics.jsonl').read_text().splitlines()]
            for row in logs:
                if 'validation_loss' in row:
                    losses[row['step']].append(row['validation_loss']); accuracies[row['step']].append(row['validation_accuracy'])
        if losses:
            steps = sorted(losses)
            axs[0].plot([s*16 for s in steps], [np.mean(losses[s]) for s in steps], 'o-', color=COLORS[kind], label=kind)
            axs[1].plot([s*16 for s in steps], [100*np.mean(accuracies[s]) for s in steps], 'o-', color=COLORS[kind], label=kind)
    for ax, label in zip(axs, ('Validation cross entropy', 'Validation accuracy (%)')):
        ax.set_xlabel('Training episodes seen'); ax.set_ylabel(label); ax.set_title('N512 sample efficiency; actual validation checkpoints')
        if ax.lines:
            ax.legend()
    savefig(figures/'learning_curves.png')
    # Complete per-task/size/depth tables are machine readable; never silently drop poor slices.
    for r in runs:
        c = r['protocol']['condition']
        if 'test' not in r:
            continue
        values = {}
        for key in ('family', 'objects', 'depth', 'irrelevant', 'holdout'):
            values[key] = {str(v): {'accuracy': score(r, key=key, value=v), 'loss': score(r, key=key, value=v, metric='loss'),
                                   'episodes': sum(e[key] == v for e in r['test']['episode_results'])}
                           for v in sorted({e[key] for e in r['test']['episode_results']})}
        summary['stages'][c['id']] = {'condition': c, 'accuracy': score(r), 'ood_accuracy': score(r, True) if c['stage'] != 'public' else None, 'loss': r['test']['loss'],
                                    'parameters': r['ledger']['trainable_parameters'], 'state_floats': r['ledger']['current_state_floats_per_sample'],
                                    'optimization_seconds': r['optimization_seconds'], 'slices': values}
    summary['interventions'] = intervention_rows; summary['phase_samples'] = phase_count
    write(root/'summary.json', summary)
    deferred = [{'stage': stage, 'status': 'not_executed', 'detail': detail} for stage, detail in [
        ('P1_confirmation', '53/71两个追加种子；未作五种子确认。'),
        ('C0_C4', '训练匹配的独立无通信网络与共同clock控制。'),
        ('A_stream_CoGenT', '流式持久记忆与CoGenT组合验证。'),
        ('P4_P5', '固定稀疏专家、动态微专家与二阶/概率/多频率/新优化器。')]]
    remaining = [c for c in repeat_plan(json.loads(Path('configs/v03/base.json').read_text())) if c['id'] not in byid]
    pending = [{'condition': c, 'status': statuses.get(c['id'], {}).get('status', 'not_executed')} for c in remaining]
    if pending:
        deferred.append({'stage': 'P2_P3_repeated_seeds', 'status': 'pending', 'detail': f"种子23/37还有{len(pending)}/90个追加条件未完成；具体状态逐项保存在unexecuted_work.json。"})
    write(root/'unexecuted_work.json', {'deferred_groups': deferred, 'pending_repeat_conditions': pending})
    lines = ['# Bio-Matrix v0.3 网络规模实验结果', '', f"生成时间（UTC）：{summary['generated_utc']}。仅汇总有完整 result.json 的运行；部分运行、失败与中止不混入完成均值。", '',
             '分支：`bio-matrix-v03-network-scale`。首轮23份源文件与327份结果的SHA256复核无变化。首轮报告快照、预登记、实现说明分别见同目录文件。', '',
             '## 执行状态', '', f"下表计当前已登记的{sum(summary['status_counts'].values())}个网络训练条件；原计划81个，P2/P3追加90个，总目标171个。输入独立基线与后续工作项另列，不将一个工作组当作一次模型训练。", '', '| 状态 | 条件数 |', '|---|---:|']
    for status in ('completed', 'running', 'failed', 'aborted', 'not_executed'):
        lines.append(f"| {status} | {summary['status_counts'].get(status, 0)} |")
    if len(mains) == 27:
        coupled = summary['main']['C3_N512']; ordinary = summary['main']['C1_N512']
        delta = summary['paired']['C3-C1_N512_overall']['mean']
        delta_ood = summary['paired']['C3-C1_N512_ood']['mean']
        lines[2:2] = [f"主扫描27次已完成（三架构×三个节点规模×三个种子）。N512的C3为{100*coupled['overall']['mean']:.2f}%，C1为{100*ordinary['overall']['mean']:.2f}%，配对差{100*delta:+.2f}个百分点；主要OOD差{100*delta_ood:+.2f}个百分点。本轮保持初筛身份，结合基线、连续损失与成本判断，不把单个正差值当成成功。", '']
    histories = [(k, h) for k, v in statuses.items() for h in v.get('history', []) if h['status'] in ('failed', 'aborted')]
    lines += ['', f'历史失败/中止事件：{len(histories)}。当前完成状态不抹去历史事件；完整记录见 conditions.json。P0首次图切分实现检查失败已修复；旧验证程序最终硬件查询被沙箱阻止，已用独立只读哈希复核完成档案检查，未改写首轮结果。', '',
              'P0：7项主实现检查，另补4项对照/隔离/反事实/重复计划检查；与旧测试合计21项逻辑检查。初次沙箱回归有1项MPS检查跳过，最终复核日志见P0/final_tests.log；另在真实MPS执行正反向一致性和9组剖析。最大CPU/MPS梯度绝对误差7.45e-8。训练使用MPS float32，无隐式CPU回退。', '',
              '## 主规模扫描', '', '每格为准确率均值 ± 跨训练种子的样本SD。区间图为跨种子的t-95%区间，n=3时很不稳定；测试问题数不作为独立训练次数。OOD包括M64、深度4、保留关系组合。', '',
              '| N | 架构 | 完成种子 | 全切片准确率 | OOD准确率 | CE | 优化秒/次 |', '|---:|---|---|---:|---:|---:|---:|']
    for n in NODES:
        for kind in COLORS:
            rs = groups[(kind, n)]; s = summary['main'].get(f'{kind}_N{n}')
            if s:
                lines.append(f"| {n} | {kind} | {','.join(str(r['protocol']['condition']['seed']) for r in rs)} | {formatted(s['overall'])} | {formatted(s['ood'])} | {s['loss']['mean']:.4f} | {s['optimization_seconds']['mean']:.2f} |")
            else:
                lines.append(f'| {n} | {kind} | — | 未执行 | 未执行 | — | — |')
    lines += ['', f"![Scale]({(figures/'scale.png').resolve()})", '', '## 相位收益与规模交互', '',
              '差值以百分点表示，正数有利于C3。逐种子差值保存在summary.json；规模交互为[N512的差值]−[N32的差值]。', '',
              '| 对比 | N | 全切片差值 ± SD | OOD差值 ± SD |', '|---|---:|---:|---:|']
    for ref in ('C1', 'C2'):
        for n in NODES:
            a = summary['paired'][f'C3-{ref}_N{n}_overall']; b = summary['paired'][f'C3-{ref}_N{n}_ood']
            lines.append(f"| C3−{ref} | {n} | {formatted(a)} | {formatted(b)} |")
        lines.append(f"| C3−{ref} 规模交互 | 512−32 | {formatted(summary['scale_interactions'][f'C3-{ref}_overall'])} | {formatted(summary['scale_interactions'][f'C3-{ref}_ood'])} |")
    lines += ['', f"![Gains]({(figures/'paired_gains.png').resolve()})", '',
              '不能仅凭单个正差值、同步曲线或某个显著阈值宣称成功。未做五种子确认时，本轮保持初筛身份。若问题独立基线接近网络分数、关系任务未打通，当前结果不能用于否定相位机制。少量种子准确率差全为零时，样本SD和t区间也会退化为零，不代表已经精确证明总体差异为零。', '',
              '## 答案先验与输入独立基线', '']
    prior = root/'diagnostic_priors.json'
    if prior.exists():
        v = json.loads(prior.read_text()); ood_priors = [e for e in v['episode_results'] if e['objects'] == 64 or e['depth'] == 4 or e['holdout']]
        lines.append(f"开发集选择的多数类在测试上：{100*v['majority_accuracy']:.2f}%（OOD {100*np.mean([e['majority_correct'] for e in ood_priors]):.2f}%）；按开发分布随机答案的期望：{100*v['random_answer_expected_accuracy']:.2f}%。机会水平随任务与答案分布变化，不沿用首轮25%。")
    for mode in ('question_only', 'evidence_only'):
        values = [json.loads(path.read_text()) for seed in (11, 23, 37) if (path := root/baseline_identifier(mode, seed)/'result.json').exists()]
        if values:
            lines.append(f"- {mode}：{formatted(stats([score(v) for v in values]))}（OOD {formatted(stats([score(v, True) for v in values]))}），平均CE {np.mean([v['test']['loss'] for v in values]):.4f}；各2048步，单独训练的泄漏/先验诊断，非严格等参数核心对照。")
        else:
            lines.append(f'- {mode}：未执行。')
    lines += ['', '## 固定容量、迭代与拓扑', '', '先完成种子11配对，再按补充登记对所有条件补23/37。表中均值、样本SD与n来自实际完成的种子；n=1时不构造区间，三种子仍为初筛。S2精确匹配N(d+2)=8192，但不等参数或计算；使用32维分解自反馈，仍逐项报告参数和实际成本。', '',
              '| 阶段 | 架构 | N | d | J | 图 | 准确率 ± SD | 平均CE | 参数 | 当前状态浮点数 | 平均优化秒 |', '|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|']
    supplementary = defaultdict(list)
    for r in runs:
        c = r['protocol']['condition']
        if c['stage'] in ('capacity', 'iterations', 'topology'):
            supplementary[tuple(c[k] for k in ('stage', 'model', 'nodes', 'state_dim', 'iterations', 'topology'))].append(r)
    for key, values in sorted(supplementary.items()):
        stage, kind, n, d, j, topology = key; r = values[0]
        lines.append(f"| {stage} | {kind} | {n} | {d} | {j} | {topology} | {formatted(stats([score(v) for v in values]))} | {np.mean([v['test']['loss'] for v in values]):.4f} | {r['ledger']['trainable_parameters']} | {r['ledger']['current_state_floats_per_sample']} | {np.mean([v['optimization_seconds'] for v in values]):.2f} |")
    lines += ['', f"![Capacity]({(figures/'fixed_capacity.png').resolve()})", '', f"![J]({(figures/'trained_iterations.png').resolve()})", '',
              'J16训练后改变测试轮数的原始结果见各主运行diagnostics.json/test_time_iterations；与重训练J2/8/32严格区分。图切分保持8度但仍共享证据读取和最终读出，因此不等于隔绝所有信息通道。', '',
              '## 公开任务：CLEVR-SG 派生验证', '',
              '来源：[CLEVR官方页面](https://cs.stanford.edu/people/jcjohns/clevr/)，CC BY 4.0，官方无图像包SHA256 `d4435dceab30022b64f7d34b3ec217162bfc1c2950c79999784750ced22674c4`。固定50,000训练问题，按场景划分开发1500/测试3000问题；场景/问题ID、哈希和1000个程序离线标签校验保存在data/clevr/v03_manifest.json。神经输入不包含程序，保留原始问题文本。', '',
              '提供了真值对象和关系，故不是标准图像VQA，也不证明物体绑定涌现。2048×16=32,768次样本曝光，按固定子集有放回采样，未覆盖完整一轮训练集；不能将欠训练等同于架构能力上限。CoGenT未执行。', '',
              '| N | 架构 | 准确率 ± SD | 平均CE | 完成种子 |', '|---:|---|---:|---:|---|']
    for n in (128, 512):
        for kind in COLORS:
            values = [r for r in runs if r['protocol']['condition']['stage'] == 'public' and r['protocol']['condition']['nodes'] == n and r['protocol']['condition']['model'] == kind]
            if values:
                lines.append(f"| {n} | {kind} | {formatted(stats([score(v) for v in values]))} | {np.mean([v['test']['loss'] for v in values]):.4f} | {','.join(str(v['protocol']['condition']['seed']) for v in values)} |")
    lines += ['', '## 分任务与机制诊断', '', f"![Breakdowns]({(figures/'breakdowns.png').resolve()})", '',
              '切片来自预先固定的六组条件，M、深度、干扰不是完全正交；完整各条件逐任务/M/深度/干扰表见summary.json，不只展示有利切片。OOD指训练分布之外，开发集也包含同类型难度用于选择检查点，因此不称为开发阶段完全未见的分布。', '',
              f"![Phase]({(figures/'phase_behavior.png').resolve()})", '', f"![Interventions]({(figures/'interventions.png').resolve()})", '',
              'R1/R2/R4、图边余弦、最后8轮PLV与逐样本表现见diagnostics.json。PLV高可能只是稳定相位差；关联不证明功能分工。干预在同一检查点、同一输入上进行，立即与恢复4轮分开，carry对照也获得相同4轮。内容扰动匹配相位交换的总L2能量，不等于函数破坏完全相同。关κ需结合重训C2解释。', '',
              '| 架构 | N | 关κ准确率变化(pp) | 关辅助消息门变化(pp) | 同问题换证据后预测变化比例 | 平均预测分布TV |', '|---|---:|---:|---:|---:|---:|']
    for identifier, d in diagnostics.items():
        if 'same_question_evidence_counterfactual' not in d:
            continue
        r = byid[identifier]; c = r['protocol']['condition']; base = r['test']['accuracy']
        cf = d['same_question_evidence_counterfactual']
        lines.append(f"| {c['model']} | {c['nodes']} | {100*(d['disable_coupling_full_solve']['accuracy']-base):+.2f} | {100*(d['auxiliary_message_gate_off']['accuracy']-base):+.2f} | {100*cf['prediction_flip_fraction']:.2f}% | {cf['mean_prediction_total_variation']:.6f} |")
    lines += ['', '上表为种子11的事后机制诊断。保持问题/物体ID、重新抽取合法属性和关系直到答案改变，不用于选择模型或作新的主准确率。关辅助消息门在C1关闭非周期门、C2/C3关闭相位门；读出仍保留辅助统计。关闭内容消息不等于完全关闭所有辅助通信。所有测试时消融都有分布偏移限制。', '',
              '## 成本与证据边界', '', f"![Learning]({(figures/'learning_curves.png').resolve()})", '', f"![Cost]({(figures/'cost_performance.png').resolve()})", '',
              f"![Budgets]({(figures/'state_parameter_budget.png').resolve()})", '',
              '主运行等数据曝光、同图/角色/读取规则/当前状态维数，C3多一个活跃κ参数。所有MPS计时边界同步。完整推理包含编码器，固定16个相同输入；训练数据处理、评测、检查点时间与优化时间分开。内存为采样值，可能漏过瞬时峰值，tensor/driver不相加；进程RSS峰值为串行训练进程的累计峰值，不是可重置的单条件瞬时峰值。当前状态大小之外还有证据缓存、固定角色、初始相位、图索引和反向激活。', '',
              '当前没有等实际训练时间的独立重训练控制；成本图用于呈现代价，不代替等时间胜利的证据。固定容量组参数变化仍是限制。三种子初筛不代替五种子确认；流式记忆、CoGenT、固定/动态专家、二阶/概率/新优化器均未在本轮展开。', '',
              '明确未执行的后续工作（也保存为unexecuted_work.json）：', '', *[f"- {e['stage']}：{e['detail']}" for e in deferred], '',
              '## 文件与重现', '',
              f"- [原始条件状态]({(root/'conditions.json').resolve()})", f"- [汇总与全部切片]({(root/'summary.json').resolve()})", f"- [串行控制台]({(root/'experiment_console.log').resolve()})", '',
              '命令：`.venv/bin/python -m v03.profile`；`.venv/bin/python -m v03.run --stage all --dry-run`；`.venv/bin/python -m v03.run --stage all`；训练结束后执行`.venv/bin/python -m v03.baselines`、`.venv/bin/python -m v03.diagnose`、`.venv/bin/python -m v03.report`。不要同时启动多个训练进程。', '']
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_text('\n'.join(lines))
    print(json.dumps({'report': args.output, 'completed_main_runs': len(mains), 'status_counts': summary['status_counts']}))


if __name__ == '__main__':
    main()
