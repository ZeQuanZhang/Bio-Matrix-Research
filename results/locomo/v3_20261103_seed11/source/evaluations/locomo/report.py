"""Aggregate per-question F1 and paired state interventions without rescoring models."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from v07.common import write


def aggregate(rows):
    done=[r for r in rows if r['status']=='completed']
    answerable=[r for r in done if r['category'] in (1,2,3,4)]
    adversarial=[r for r in done if r['category']==5]
    targets=sum(r['answer_tokens'] for r in answerable)
    first=[r for r in answerable if r['first_hit'] is not None]
    return dict(processed=len(rows),completed=len(done),not_applicable=len(rows)-len(done),
        main_questions=len(answerable),main_f1=float(np.mean([r['f1'] for r in answerable])) if answerable else None,
        per_category={str(c):dict(questions=len(v),f1=float(np.mean([r['f1'] for r in v]))) for c in range(1,6)
                      if (v:=[r for r in done if r['category']==c])},
        answer_nll=sum(r['answer_loss_sum'] for r in answerable)/targets if targets else None,
        answer_tokens=targets,first_token_accuracy=sum(r['first_hit'] for r in first)/len(first) if first else None,
        first_answer_nll=float(np.mean([r['first_answer_nll'] for r in first])) if first else None,
        adversarial_questions=len(adversarial),adversarial_refusal=float(np.mean([r['f1'] for r in adversarial])) if adversarial else None,
        mean_generated_tokens=float(np.mean([r['generated_tokens'] for r in done])) if done else None,
        token_limit_fraction=float(np.mean([r['hit_token_limit'] for r in done])) if done else None,
        distinct_generated_answers=len({r['prediction'] for r in done}))


def paired(a,b):
    left={r['id']:r for r in a if r['status']=='completed' and r['category']!=5}
    right={r['id']:r for r in b if r['status']=='completed' and r['category']!=5}
    ids=sorted(set(left)&set(right));groups=sorted({left[i]['conversation'] for i in ids})
    differences=np.array([left[i]['f1']-right[i]['f1'] for i in ids])
    sums=np.array([sum(left[i]['f1']-right[i]['f1'] for i in ids if left[i]['conversation']==c) for c in groups])
    counts=np.array([sum(left[i]['conversation']==c for i in ids) for c in groups])
    rng=np.random.default_rng(1103);indices=rng.integers(0,len(groups),size=(20000,len(groups)))
    draws=sums[indices].sum(1)/counts[indices].sum(1)
    return dict(questions=len(ids),conversations=len(groups),f1_difference=float(differences.mean()),
        conversation_bootstrap_95ci=np.quantile(draws,[.025,.975]).tolist(),
        note='One predeclared training seed; confidence interval covers evaluation conversation sampling only.')


def report(root):
    root=Path(root);registry=json.loads((root/'registry.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text());rows={};summary={}
    for identity,status in registry['conditions'].items():
        group=[]
        for p in sorted((root/identity).glob('c*_b*.json')):group+=json.loads(p.read_text())['rows']
        if not group:continue
        if len({r['id'] for r in group})!=len(group):raise ValueError('Duplicate question result')
        rows[identity]=group
        summary[identity]=dict(status=status['status'],**aggregate(group),
            oracle_eligible_subset=aggregate([r for r in group if r['oracle_eligible']]))
        if status['status']=='completed' and len(group)!=protocol['planned_questions']:raise ValueError('Incomplete condition')
    comparisons={}
    for other in ('S-main','D-clear','D-donor','D-oracle','L128','F512'):
        if 'D-main' in rows and other in rows:comparisons[f'D-main-minus-{other}']=paired(rows['D-main'],rows[other])
    result=dict(status=registry['status'],conditions=summary,paired_differences=comparisons,
        seed=protocol['seed'],metric='Official LoCoMo stemmed word F1, categories 1-4; category5 separately',
        judge_used=False,training_performed=False,not_comparable_to_memt_llm_judge_accuracy=True)
    write(root/'summary.json',result)
    lines=['# V3 LoCoMo 评估','',f'状态：{registry["status"]}。固定seed{protocol["seed"]}；未训练或调参。',
        '',f'官方固定版本：{protocol["dataset"]["revision"]}。10段对话、1986题；主分数为第1–4类1540题的词级F1。',
        '','| 条件 | 主评分题数 | F1 % | 答案NLL | 首token命中 % | 对抗拒答 % | 不适用题数 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def number(value,scale=1,digits=4):return '—' if value is None else f'{value*scale:.{digits}f}'
    for name,row in summary.items():
        lines.append(f'| {name} | {row["main_questions"]} | {number(row["main_f1"],100)} | {number(row["answer_nll"])} | {number(row["first_token_accuracy"],100,2)} | {number(row["adversarial_refusal"],100,2)} | {row["not_applicable"]} |')
    lines+=['','D/S为动态/静态Bio；L128/F512分别为原训练128/512窗口Transformer，均只保留各自近期KV。',
        'clear清空历史状态但保留时钟；donor使用其他对话并匹配token数；oracle只输入标注证据，使用证据标注，单独解释。',
        '','缺失或无效证据不会移除主评估题；oracle中标为not_applicable。比较oracle时使用summary.json中的共同证据有效子集。',
        '','Oracle Transformer窗口扩展到容纳完整证据，属于额外诊断；主评估窗口保持128/512。',
        '','每题从同一历史快照独立分支，生成上限50个本项目BPE token、贪心解码、遇EOS结束；原始答案仅用于F1与额外teacher-forced NLL评分。',
        '','本评估使用文本和数据中已有图像caption，未加载图像、生成的观察或摘要。无外部检索、LLM回答器或付费裁判。',
        '','与MemT的LLM裁判准确率采用不同评分协议，单独报告；一个训练种子的LoCoMo结果不足以评估训练种子稳定性。',
        '','对抗题采用原始问题直接生成及官方短语拒答规则，属于单列诊断；未使用原官方部分脚本的随机两选项提示。',
        '','全部逐题原始预测、参考答案、token IDs保存在本地cXX_bXXX.json；公开副本只提供净化数值与摘要。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result
