"""Summarize saved paired results without running a model or opening the test split."""
import json
from pathlib import Path
import statistics

from v07.common import write


def report(root):
    root=Path(root)
    summary=json.loads((root/'summary.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    rows={}
    for identity,result in summary['conditions'].items():
        if result['status']!='completed':continue
        qa=result['validation']['qa'];lm=result['validation']['lm']
        rows[identity]=dict(condition=result['condition'],seed=result['seed'],
            parameters=result['total_parameters'],context_limit=result['context_limit'],
            qa_nll=qa['warm_nll'],first_answer_nll=qa['first_answer_nll'],
            first_answer_accuracy=qa['first_answer_accuracy'],lm_nll=lm['warm_nll'],lm_ppl=lm['warm_ppl'],
            remote_history_cost=qa['reencoded_history_cost'],read_off_cost=qa['total_read_off_cost'],
            train_seconds=result['train_seconds'],validation_seconds=result['validation_seconds'],
            input_tokens_per_second=result['input_tokens']/max(result['train_seconds'],1e-9),
            answer_targets=result['answer_targets'])
    aggregate={}
    for condition in ('trans_local','trans_full','v3_static','v3_dynamic'):
        selected=[row for row in rows.values() if row['condition']==condition]
        if not selected:continue
        aggregate[condition]=dict(completed_seeds=len(selected))
        for field in ('qa_nll','first_answer_nll','first_answer_accuracy','lm_nll','lm_ppl',
                      'remote_history_cost','read_off_cost','train_seconds','validation_seconds','input_tokens_per_second'):
            values=[r[field] for r in selected if r[field] is not None]
            aggregate[condition][field]=dict(mean=statistics.mean(values),seed_sd=statistics.stdev(values) if len(values)>1 else None) if values else None
    paired={}
    for other in ('trans_local','trans_full','v3_static'):
        values=[]
        for row in rows.values():
            identity=f'{other}_s{row["seed"]}'
            if row['condition']=='v3_dynamic' and identity in rows:
                values.append(dict(seed=row['seed'],qa_nll=row['qa_nll']-rows[identity]['qa_nll'],
                                   lm_nll=row['lm_nll']-rows[identity]['lm_nll'],
                                   training_time_ratio=row['train_seconds']/max(rows[identity]['train_seconds'],1e-9)))
        paired[f'v3_dynamic-minus-{other}']=values
    result=dict(status=summary['status'],per_seed=rows,aggregate=aggregate,paired_differences=paired,
                test_evaluated=False,protocol='protocol.json',
                note='Paired training seeds reuse the same fixed validation questions. Seed SD is unavailable for one seed. Full-context Transformer has a larger explicit KV budget.')
    write(root/'comparison.json',result)
    lines=['# Bio-Matrix v3 配对实验',f'\n状态：**{summary["status"]}**。最终测试集未评分。',
           '\n四组全部从零训练；Bio由向量节点层组成，无token注意力或KV。相同种子读取相同样本、mask与优化预算。',
           '\n| 条件 | 完成种子 | QA NLL均值↓ | 首答案NLL↓ | 首token准确率↑ | 文本PPL↓ | 训练秒数均值 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,row in aggregate.items():
        lines.append(f'| {name} | {row["completed_seeds"]} | {row["qa_nll"]["mean"]:.5f} | {row["first_answer_nll"]["mean"]:.5f} | {row["first_answer_accuracy"]["mean"]:.2%} | {row["lm_ppl"]["mean"]:.4f} | {row["train_seconds"]["mean"]:.2f} |')
    lines += ['\n`comparison.json`保留每个种子的原值、种子标准差与配对差。',
              '\nBio没有原文KV；局部Transformer窗口仅用于定义相同的远史替换范围。完整窗口Transformer保留整个采样窗口。参数接近，状态容量与FLOPs分别报告。',
              '\n两Bio组具有同样的向量状态容量；静态组固定运行态key和相位，动态组联合修正key、value和相位。两组都更新value，并通过因果前缀读取块内较早写入。',
              '\n主指标：答案和首答案NLL、首token准确率、替换远史后的损失变化，同时查看文本损失与计算成本。',
              '\n训练预算：'+json.dumps(protocol['training'],ensure_ascii=False),
              '\n状态表：'+json.dumps({k:v['status'] for k,v in summary['conditions'].items()},ensure_ascii=False)]
    (root/'COMPARISON.md').write_text('\n'.join(lines),encoding='utf-8')
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    args=parser.parse_args();report(args.root)
