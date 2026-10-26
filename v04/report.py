"""Regenerate summaries from raw results, including incomplete conditions."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np


def report(root):
    root = Path(root)
    statuses = json.loads((root/'conditions.json').read_text()) if (root/'conditions.json').exists() else {}
    rows = []
    lines = ['# Bio-Matrix v0.4 实验结果', '',
             '本报告由原始结果自动生成。仅包含实际完成的训练；实现检查与短程试跑不能作为架构效果结论。', '',
             '## 执行状态', '', '| 条件 | 状态 | 安全检查点步数 |', '|---|---|---:|']
    for identifier, item in statuses.items():
        lines.append(f"| {identifier} | {item['status']} | {item.get('safe_checkpoint_step', item.get('step', '—'))} |")
        path = root/identifier/'result.json'
        if item['status'] != 'completed' or not path.exists():
            continue
        result = json.loads(path.read_text()); test = result['test']; ledger = result['ledger']
        row = dict(**result['condition'], accuracy=test['accuracy'], nll=test['nll'], brier=test['brier'], ece=test['ece'],
                   parameters=ledger['parameters'], state_floats=ledger['state_floats_per_stream'],
                   seconds=result['elapsed_seconds'], best_step=result['best_step'])
        if 'untrained_test' in result:
            row['untrained_accuracy'] = result['untrained_test']['accuracy']
            row['learning_gain_pp'] = 100*(row['accuracy']-row['untrained_accuracy'])
        row.update({f'{p}_final_acc': v['final_accuracy'] for p, v in test['by_profile'].items()})
        controls = result['interventions']
        row['history_gain_pp'] = 100*(test['by_profile']['history']['final_accuracy']-controls['reset_history']['by_profile']['history']['final_accuracy'])
        row['return_gain_pp'] = 100*(test['by_profile']['return']['final_accuracy']-controls['reset_history']['by_profile']['return']['final_accuracy'])
        base = json.loads((root/identifier/'test_predictions.json').read_text())
        swapped = json.loads((root/identifier/'swap_history_predictions.json').read_text())
        selected = [(a, b) for a, b in zip(base, swapped) if a['profile'] == 'independent' and a['final']]
        row['independent_history_TV'] = float(np.mean([.5*np.abs(np.array(a['probability'])-b['probability']).sum() for a, b in selected]))
        final_groups = defaultdict(list)
        for record in base:
            if record['final']:
                final_groups[(record['profile'], record['stream_hash'])] = record
        # Quartet streams are consecutive; their questions and cue trajectories are identical.
        blocks = [r for r in base if r['final']]
        row['counterfactual_prediction_diversity'] = float(np.mean([len({r['prediction'] for r in blocks[i:i+4]})/4 for i in range(0, len(blocks), 4)]))
        rows.append(row)
    lines += ['', '## 完成条件', '', '| 模型 | 路径 | N | seed | 参数 | 状态浮点数 | 准确率 | 零训练准确率 | NLL | 秒 |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        initial = f"{r['untrained_accuracy']:.2%}" if 'untrained_accuracy' in r else '未测'
        lines.append(f"| {r['variant']} | {r.get('read_path', 'shared')} | {r['nodes']} | {r['seed']} | {r['parameters']} | {r['state_floats']} | {r['accuracy']:.2%} | {initial} | {r['nll']:.4f} | {r['seconds']:.1f} |")
    lines += ['', '## 配对交互', '', '联合收益按同一规模、同一种子计算：full − probability − fourier + matrix。准确率差以百分点计。', '']
    paired = defaultdict(dict)
    for r in rows:
        paired[(r.get('read_path', 'shared'), r['nodes'], r['seed'])][r['variant']] = r
    interactions = []
    for (path, nodes, seed), group in paired.items():
        if {'full', 'probability', 'fourier', 'matrix'} <= group.keys():
            gain = 100*(group['full']['accuracy']-group['probability']['accuracy']-group['fourier']['accuracy']+group['matrix']['accuracy'])
            interactions.append(dict(read_path=path, nodes=nodes, seed=seed, interaction_pp=gain))
            lines.append(f'- {path}, N={nodes}, seed={seed}：{gain:+.3f} pp。')
    if not interactions:
        lines.append('完整的四组配对尚未完成。')
    lines += ['', '## 解释范围', '',
              '普通图网络与矩阵网络共享情景槽控制器、图、端口、读取算子和数据。向量节点使用有效瓶颈宽度尽量匹配参数；状态浮点数严格匹配，实际计算量不强行宣称相等。', '',
              'shared 读取路径用于检查记忆机制；跨节点协作需结合 separated 端口和断图控制。数据是中文可读的结构化合成流，算术组合读出预先给定；尚未完成自然中文语言建模或公开任务验证。', '',
              '四类答案平衡不保证任务充分困难。规模结论需要跨种子、固定总状态容量和计算预算控制，并检查证据干预、校准、节点多样性与实际耗时。']
    (root/'FINDINGS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    from .run import write
    write(root/'summary.json', dict(conditions=rows, interactions=interactions, statuses={k: v['status'] for k, v in statuses.items()}))
    if rows:
        columns = list(dict.fromkeys(k for row in rows for k in row))
        with (root/'summary.csv').open('w', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root', type=Path)
    report(parser.parse_args().root)
