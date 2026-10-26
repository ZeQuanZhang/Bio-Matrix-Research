"""Report raw paired outcomes, including retention, correction and interference."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from v04.run import write


def paired_tv(first, second, profile):
    selected = [(a, b) for a, b in zip(first, second) if a['profile'] == profile and a['final']]
    return float(np.mean([.5*np.abs(np.array(a['probability'])-b['probability']).sum() for a, b in selected])) if selected else None


def correction(records):
    groups = defaultdict(list)
    for row in records:
        if row['stage'] == 'correction':
            groups[(row['profile'], row['stream_hash'])].append(row)
    measured = []
    for (profile, stream_hash), rows in groups.items():
        correct = [r['prediction'] == r['target'] for r in rows]
        delay = next((i+1 for i in range(len(correct)-1) if correct[i] and correct[i+1]), None)
        measured.append(dict(profile=profile, stream_hash=stream_hash,
                             first_of_two_correct_queries=delay, censored=delay is None, queries=len(rows)))
    return measured


def report(root):
    root = Path(root); path = root/'conditions.json'
    statuses = json.loads(path.read_text()) if path.exists() else {}
    rows, delays = [], {}
    for identifier, item in statuses.items():
        file = root/identifier/'result.json'
        if item['status'] != 'completed' or not file.exists():
            continue
        result = json.loads(file.read_text()); test, controls, ledger = result['test'], result['interventions'], result['ledger']
        predictions = json.loads((root/identifier/'test_predictions.json').read_text())
        swapped = json.loads((root/identifier/'swap_history_predictions.json').read_text())
        row = dict(**result['condition'], accuracy=test['accuracy'], nll=test['nll'], brier=test['brier'], ece=test['ece'],
                   macro_final_accuracy=test.get('macro_final_accuracy', float(np.mean([v['final_accuracy'] for v in test['by_profile'].values()]))),
                   macro_final_nll=test.get('macro_final_nll', float(np.mean([v['final_nll'] for v in test['by_profile'].values()]))),
                   untrained_accuracy=result['untrained_test']['accuracy'],
                   learning_gain_pp=100*(test['accuracy']-result['untrained_test']['accuracy']),
                   parameters=ledger['parameters'], adaptive_weights=ledger['adaptive_local_weight_elements'],
                   state_floats=ledger['state_floats_per_stream'], seconds=result['elapsed_seconds'], best_step=result['best_step'])
        row.update({f'{p}_final_acc': v['final_accuracy'] for p, v in test['by_profile'].items()})
        for name, profile, intervention in (('history_gain_pp', 'history', 'reset_history'),
                                             ('slow_retention_gain_pp', 'consolidation', 'clear_slow'),
                                             ('return_gain_pp', 'return', 'clear_slow')):
            row[name] = 100*(test['by_profile'][profile]['final_accuracy']-controls[intervention]['by_profile'][profile]['final_accuracy'])
        row['independent_history_TV'] = paired_tv(predictions, swapped, 'independent')
        row['no_messages_accuracy'] = controls['no_messages']['accuracy']
        row['no_messages_nll'] = controls['no_messages']['nll']
        delays[identifier] = correction(predictions)
        entries = [r for r in delays[identifier] if r['profile'] == 'switch']
        row['correction_censored_fraction'] = float(np.mean([r['censored'] for r in entries])) if entries else None
        observed = [r['first_of_two_correct_queries'] for r in entries if not r['censored']]
        row['correction_delay_observed_mean'] = float(np.mean(observed)) if observed else None
        rows.append(row)
    lines = ['# Bio-Matrix v0.5 原始结果汇总', '',
             '仅汇总状态为 completed 且具有 result.json 的条件。单种子、零训练先验、短跑和工程检查分别解释。', '',
             '| 条件 | 状态 | 安全检查点 |', '|---|---|---:|']
    for name, item in statuses.items():
        lines.append(f"| {name} | {item['status']} | {item.get('safe_checkpoint_step', item.get('step', '—'))} |")
    lines += ['', '| 阶段 | 模型 | 路径 | N | seed | 末次查询宏平均 | 所有计分查询 | 零训练 | NLL | 慢记忆收益 pp | 无关历史 TV |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['stage']} | {r['variant']} | {r['read_path']} | {r['nodes']} | {r['seed']} | {r['macro_final_accuracy']:.2%} | {r['accuracy']:.2%} | {r['untrained_accuracy']:.2%} | {r['nll']:.4f} | {r['slow_retention_gain_pp']:+.2f} | {r['independent_history_TV']:.5f} |")
    lines += ['', '配对比较要求同阶段、规模、种子、端口、时间轮数、容量与信用协议。先完成同规模的全部配对，再扩大节点和补种子。', '',
              '慢记忆收益比较 consolidation 末次查询与 clear_slow；该任务已显式清除 M/P/X/资格迹。return 任务还在新情境学习后清除短期状态。', '',
              '纠错延迟记录首次连续两次答对中的第一个查询编号；未达到者标为 censored，单独报告比例，不能从均值中静默删除。', '',
              '主指标为各任务末次查询的宏平均；dev 用相同宏平均 NLL 选择检查点。重复声明在没有独立证据前的请求不计入主损失，避免要求模型猜不可知的真值。', '',
              'capacity 精确匹配全部运行时浮点状态，调整有实际用途的时间带、模式和资格迹数量；基础参数与计算仍逐项报告。compute 只固定 N×J，实际时间不是严格匹配。', '',
              '局部可塑权重计入状态/存储预算，逐轨迹独立。共享规则由 AdamW 训练；在线阶段只用已到达的可观察报告，不使用评分标签。', '',
              '数据是中文可读的结构化合成流。没有执行自然中文 LLM 对照、真实来源可靠性学习或跨来源相关性估计。', '',
              '共享端口可绕开跨节点传播；跨节点结论结合 separated、cut、no_messages 与任务梯度，规模结论结合重复种子、容量与成本。']
    (root/'FINDINGS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write(root/'summary.json', dict(conditions=rows, correction_delays=delays,
                                   statuses={k: v['status'] for k, v in statuses.items()}))
    if rows:
        columns = list(dict.fromkeys(k for row in rows for k in row))
        with (root/'summary.csv').open('w', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root', type=Path)
    report(parser.parse_args().root)
