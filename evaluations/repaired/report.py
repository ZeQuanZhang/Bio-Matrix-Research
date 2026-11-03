"""Final holdout evaluation after all development choices, with paired article statistics."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from v07.common import write, digest, file_hash
from .model import build, NAMES
from .engine import score_lm, score_qa, generate


def generation_summary(generation, bins):
    groups = {}
    for label, bounds in (('128', (-1, 128)), ('256', (128, 256)), ('10000', (256, 10000))):
        rows = [r for r in generation['records'] if bounds[0] < r['distance'] <= bounds[1]]
        groups[label] = dict(population=bins[label]['population'], questions=len(rows),
            **{k: float(np.mean([r[k] for r in rows])) for k in ('em', 'f1', 'hit_cap')})
    population = sum(b['population'] for b in bins.values())
    return dict(strata=groups, questions=generation['questions'],
        **{k: sum(g[k]*g['population'] for g in groups.values())/population for k in ('em', 'f1', 'hit_cap')})


def paired_interval(a, b, task, rng):
    sums, counts = [], []
    for aa, bb in zip(a, b):
        aa, bb = aa[task], bb[task]
        if task == 'cpt_lm' or task == 'post_lm':
            records_a, records_b, key, loss, count = aa['documents'], bb['documents'], 'document', 'nll_sum', 'targets'
        else:
            records_a, records_b, key, loss, count = aa['records'], bb['records'], 'article_id', 'answer_nll_sum', 'answer_targets'
        left, right = defaultdict(lambda: [0., 0]), defaultdict(lambda: [0., 0])
        for records, dest in ((records_a, left), (records_b, right)):
            for r in records:
                dest[r[key]][0] += r[loss]; dest[r[key]][1] += r[count]
        keys = sorted(left)
        assert keys == sorted(right) and all(left[k][1] == right[k][1] for k in keys)
        sums.append([left[k][0]-right[k][0] for k in keys]); counts.append([left[k][1] for k in keys])
    sums, counts = np.asarray(sums), np.asarray(counts)
    seed_draws = rng.integers(len(a), size=(20000, len(a)))
    doc_draws = rng.integers(sums.shape[1], size=(20000, sums.shape[1]))
    values = sums[seed_draws[:, :, None], doc_draws[:, None, :]].sum((1, 2))/counts[seed_draws[:, :, None], doc_draws[:, None, :]].sum((1, 2))
    tail = .05/(2*(len(NAMES)-1))
    return dict(nll_difference=float(sums.sum()/counts.sum()), paired_seed_article_95ci=np.quantile(values, [.025, .975]).tolist(),
        simultaneous_six_contrasts_nominal95ci=np.quantile(values, [tail, 1-tail]).tolist(),
        seeds=len(a), articles=sums.shape[1], bootstrap_repetitions=20000,
        scope='Conditional bootstrap of observed seeds/articles; does not include hyperparameter-search variance')


def final_evaluation(root, protocol, corpus, qa, tokenizer, device, stage, stop):
    c = protocol['training']; folder = root/'heldout'; folder.mkdir(exist_ok=True)
    registry_file = folder/'registry.json'
    registry = json.loads(registry_file.read_text()) if registry_file.exists() else {
        f'{name}_s{s}': dict(status='not_executed') for s in c['seeds'] for name in NAMES}
    write(registry_file, registry)
    bins = protocol['test_generation']; indices = [i for b in bins.values() for i in b['indices']]
    for seed in c['seeds']:
        for name in NAMES:
            identity = f'{name}_s{seed}'; target = folder/f'{identity}.json'
            if registry[identity]['status'] == 'completed':
                continue
            registry[identity] = dict(status='running'); write(registry_file, registry)
            model = build(name, seed).to(device)
            result = dict(condition=name, seed=seed, status='running')
            for phase, key in (('cpt', 'cpt_lm'), ('qa', 'post_lm')):
                if stop.requested:
                    registry[identity]['status'] = 'aborted'; write(registry_file, registry); raise KeyboardInterrupt
                phase_file = folder/f'{identity}_{key}.json'
                checkpoint = root/f'final_{name}_s{seed}_{phase}'/'last.pt'
                saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                if saved['protocol_sha256'] != digest(protocol):
                    raise ValueError('Final checkpoint identity changed')
                model.load_state_dict(saved['model'])
                if phase_file.exists():
                    result[key] = json.loads(phase_file.read_text())
                else:
                    stage(f'保留测试 {identity} · {key} 全文覆盖')
                    result[key] = score_lm(model, corpus, 'test', device, c['evaluation_batch']); write(phase_file, result[key])
            tasks = [('qa', lambda: score_qa(model, qa['test'], device, batch=c['evaluation_batch']))]
            tasks += [(mode, lambda m=mode: score_qa(model, qa['test'], device, indices, c['evaluation_batch'], m))
                      for mode in ('warm', 'unrelated_context', 'remote_context')]
            independent = [i for i in indices if qa['test'].rows[i]['prefix_tokens'] <= 384][:64]
            tasks += [(key, lambda m=mode: score_qa(model, qa['test'], device, independent, c['evaluation_batch'], m))
                      for key, mode in (('independent_fresh', 'warm'), ('independent_prior', 'independent_prior'))]
            for key, evaluate in tasks:
                path = folder/f'{identity}_{key}.json'
                if path.exists():
                    result[key] = json.loads(path.read_text()); continue
                if stop.requested:
                    registry[identity]['status'] = 'aborted'; write(registry_file, registry); raise KeyboardInterrupt
                stage(f'保留测试 {identity} · {key} 问答评分')
                result[key] = evaluate(); write(path, result[key])
            for mode in ('warm', 'unrelated_context'):
                path = folder/f'{identity}_generation_{mode}.private.json'
                if path.exists():
                    generated = json.loads(path.read_text())
                else:
                    stage(f'保留测试 {identity} · {mode} 192题实际生成')
                    generated = generate(model, qa['test'], indices, tokenizer, device, c['evaluation_batch'], mode)
                    write(path, generated)
                result['generation_'+mode] = generation_summary(generated, bins)
                projection = folder/f'{identity}_generation_{mode}.metrics.jsonl'
                projection.write_text(''.join(json.dumps({k: v for k, v in r.items() if k not in ('prediction', 'reference_answers', 'generated_ids')}
                    | dict(prediction_sha256=hashlib.sha256(r['prediction'].encode()).hexdigest()))+'\n' for r in generated['records']))
            result['status'] = 'completed'; write(target, result)
            registry[identity] = dict(status='completed'); write(registry_file, registry)
            model.to('cpu'); del model
            if device.type == 'mps':
                torch.mps.empty_cache()
    summarize(root, protocol, corpus)


def summarize(root, protocol, corpus):
    c = protocol['training']; rows = {}; stats = {}; rng = np.random.default_rng(61209)
    for name in NAMES:
        rows[name] = [json.loads((root/f'heldout/{name}_s{s}.json').read_text()) for s in c['seeds']]
        values = [dict(seed=r['seed'], cpt_ppl=r['cpt_lm']['ppl'], post_ppl=r['post_lm']['ppl'],
            qa_answer_nll=r['qa']['answer_nll'], first_accuracy=r['qa']['first_accuracy'],
            em=r['generation_warm']['em'], f1=r['generation_warm']['f1'],
            unrelated_f1=r['generation_unrelated_context']['f1'],
            unrelated_answer_nll_effect=r['unrelated_context']['answer_nll']-r['warm']['answer_nll'],
            remote_answer_nll_effect=r['remote_context']['answer_nll']-r['warm']['answer_nll'],
            independent_prior_answer_nll_effect=r['independent_prior']['answer_nll']-r['independent_fresh']['answer_nll']) for r in rows[name]]
        for result, value in zip(rows[name], values):
            warm = [r for r in result['warm']['records'] if r['distance'] > 256]
            remote = [r for r in result['remote_context']['records'] if r['distance'] > 256]
            assert [r['question_id'] for r in warm] == [r['question_id'] for r in remote]
            value['remote_answer_nll_effect_far_bin'] = (sum(r['answer_nll_sum'] for r in remote)-sum(r['answer_nll_sum'] for r in warm))/sum(r['answer_targets'] for r in warm)
        seconds = []
        for s in c['seeds']:
            seconds.append(sum(json.loads((root/f'final_{name}_s{s}_{p}/result.json').read_text())['train_seconds'] for p in ('cpt', 'qa')))
        stats[name] = dict(parameters=protocol['models'][name]['total_parameters'], per_seed=values,
            means={key: float(np.mean([v[key] for v in values])) for key in values[0] if key != 'seed'},
            train_seconds_mean=float(np.mean(seconds)))
    comparisons = {name: {task: paired_interval(rows[name], rows['v3_dynamic'], task, rng)
        for task in ('cpt_lm', 'post_lm', 'qa')} for name in NAMES if name != 'v3_dynamic'}
    targets = np.ones(len(corpus.tokens['test']), dtype=bool); targets[corpus.offsets['test'][:-1]] = False
    unigram = {}
    for seed in c['seeds']:
        counts = np.ones(4096, dtype=np.int64); data_rng = np.random.default_rng(32000+seed)
        for _ in range(c['cpt_tokens']//(c['batch']*512)):
            values, _, _ = corpus.sample('train', data_rng, c['batch'], 512)
            counts += np.bincount(values[:, 1:].ravel(), minlength=4096)
        unigram[str(seed)] = float(np.exp(-np.log(counts/counts.sum())[corpus.tokens['test'][targets]].mean()))
    pairing = {}
    for seed in [101]+c['seeds']:
        for phase in ('cpt', 'qa'):
            paths = []
            if seed == 101:
                paths = [root/f'{phase}_search_{name}_r{i}/metrics.jsonl' for name in NAMES for i in range(3)]
            else:
                paths = [root/f'final_{name}_s{seed}_{phase}/metrics.jsonl' for name in NAMES]
            traces = [[json.loads(x) for x in p.read_text().splitlines()] for p in paths]
            assert all([x['sample_sha256'] for x in trace] == [x['sample_sha256'] for x in traces[0]] for trace in traces)
            pairing[f'{phase}_s{seed}'] = dict(conditions=len(paths), updates_per_condition=len(traces[0]), all_input_mask_hashes_matched=True)
    write(root/'summary.json', dict(status='completed', statistics=stats, comparisons_to_dynamic=comparisons,
        unigram_same_cpt_targets_test_ppl=unigram, pairing=pairing,
        selection=json.loads((root/'selection.json').read_text()),
        limitations='Micro-configurations; three final seeds; same samples and tokens, not equal FLOPs; finite search budgets; native initializers; BPE PPL; MPS reference kernels'))
    table = '\n'.join(f'| {n} | {s["parameters"]:,} | {s["means"]["cpt_ppl"]:.2f} | {s["means"]["post_ppl"]:.2f} | {100*s["means"]["first_accuracy"]:.2f}% | {100*s["means"]["em"]:.2f}% | {100*s["means"]["f1"]:.2f}% | {s["train_seconds_mean"]:.1f} |' for n, s in stats.items())
    controls = {n: json.loads((root/f'overfit_{n}/result.json').read_text())['diagnostic_passed'] for n in NAMES}
    report = f'''# 修正版本地微型架构对比

本轮冻结协议：7个预先指定配置；各阶段3个学习率候选；搜索种子101；最终种子11/23/37。所有最终配置训练完成后才评分保留测试。原实验独立保留。

## 保留测试：三种子均值

| 配置 | 参数 | 预训练PPL↓ | 问答后PPL↓ | 首答案命中↑ | 生成EM↑ | 生成F1↑ | 核心训练秒/种子 |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

文本覆盖WikiText保留测试的全部非BOS目标，每个位置评分一次；每段最多256个前置token，接着评分最多256个新目标。问答teacher forcing覆盖全部合格官方SQuAD dev题；生成按证据距离三桶各64题，表中EM/F1按各桶总体题数加权。原始逐题生成保存在本地private文件，公开数值投影不含题目、答案或预测原文。

## 学习链路和历史证据

8道真实短问答的固定预算过拟合检查通过情况：`{json.dumps(controls, ensure_ascii=False)}`。阈值为答案NLL<0.5、首答案准确率≥87.5%、训练题生成EM≥50%；此项仅检验学习链路。

summary.json给出每个种子的无关上下文替换、远处上下文替换及独立新任务前插入无关历史的答案NLL效应，并列出无关上下文生成F1。正确使用证据须同时检查生成表现与配对干预，不能仅依据训练损失或单个平均NLL。

## 范围与统计

主文本比较使用问答后训前的检查点；后训PPL单独衡量语言建模退化。Bio动态与静态对照用于检验动态地址的额外作用。DeepSeek原配置和均衡配置分别报告，不做家族内部择优后冒充相同搜索预算。

summary.json保存20000次配对种子/文章交叉bootstrap，含95%区间及六个对比的Bonferroni名义调整区间。区间受三个已观察种子和当前任务限制，未覆盖完整超参数搜索方差。

同目标词频控制的保留测试PPL：`{json.dumps(unigram)}`。各模型损失都使用同一新BPE；与其他分词口径的论文PPL不能直接比较。不同结构的FLOPs、运行态、原生初始化与实现效率不同；本轮时间仅代表Mac MPS参考实现。

学习率选择见selection.json；完整协议、逐更新日志、阶段检查点、开发评分及保留测试均在本目录。失败/中止/未执行条件分别记在registry.json和heldout/registry.json。原始实验结果不覆盖。
'''
    (root/'REPORT.md').write_text(report)
