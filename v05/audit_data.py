"""Read-only dataset audit: shortcuts, split structure and reported noise."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib
import json
from pathlib import Path
import numpy as np
from v04.data import digest
from v04.run import write

PROJECT = Path(__file__).resolve().parents[1]


def structure(row):
    signature = []
    for e in row['events']:
        role = (0 if e['key'] == e['query_a'] else 1 if e['key'] == e['query_b'] else 2) if e['observe'] else -1
        signature.append((role, bool(e.get('request', e['query'])), bool(e.get('restart', False)),
                          bool(e['combine']), int(e['noise'] > .3), int(e['world'])))
    return digest(signature)


def rule_accuracy(rows, profile, first=False):
    correct = total = 0
    for row in rows:
        if row['profile'] != profile:
            continue
        events = [e for e in row['events'] if e['observe'] and e['key'] == e['query_a']]
        event = events[0 if first else -1]
        predicted = (int(np.argmax(event['value']))-1) % 4
        target = row['events'][-1]['truth'][0]
        correct += predicted == target; total += 1
    return dict(correct=int(correct), streams=total, independent_base_groups=total//4,
                accuracy=correct/total if total else None)


def reference_check(rows):
    checked = errors = unresolved = 0
    for row in rows:
        ledger = {}
        for e in row['events']:
            if e['observe'] and e['noise'] < .3 and e.get('grounded', True):
                ledger[(e['world'], e['key'])] = int(np.argmax(e['value']))
            if not e['query']:
                continue
            addresses = [e['query_a']]+([e['query_b']] if e['combine'] else [])
            if any((e['world'], address) not in ledger for address in addresses):
                unresolved += 1; continue
            expected = sum(ledger[(e['world'], address)] for address in addresses) % 4
            checked += 1; errors += expected != e['target']
    return dict(checked=checked, errors=int(errors), unresolved=unresolved,
                method='Separate raw-report ledger with auditor-only world and grounded flags; no stored truth field used.')


def inspect(version, cfg, train_groups=128):
    module = importlib.import_module(f'{version}.data'); keys = cfg['matrix_size']
    rng = np.random.default_rng(cfg['data_seed']+cfg['seeds'][0])
    splits = dict(train=[r for _ in range(train_groups) for r in module.training_batch(rng, 4, keys)],
                  dev=module.bank('dev', cfg['development_quartets'], keys),
                  test=module.bank('test', cfg['test_quartets'], keys))
    label_errors, balance_errors = 0, 0
    for rows in splits.values():
        for start in range(0, len(rows), 4):
            group = rows[start:start+4]
            for t in range(len(group[0]['events'])):
                if group[0]['events'][t]['query']:
                    balance_errors += sorted(r['events'][t]['target'] for r in group) != [0, 1, 2, 3]
            for row in group:
                for e in row['events']:
                    label_errors += e['target'] != (e['truth'][0]+(e['truth'][1] if e['combine'] else 0)) % 4
    hashes = {name: {r['stream_hash'] for r in rows} for name, rows in splits.items()}
    templates = {name: {structure(r) for r in rows} for name, rows in splits.items()}
    noise = defaultdict(list)
    for row in splits['test']:
        seen = set()
        for e in row['events']:
            if not e['observe'] or e['evidence_id'] in seen:
                continue
            seen.add(e['evidence_id'])
            if e['key'] in (e['query_a'], e['query_b']) and e['noise'] < .3:
                target = e['truth'][0 if e['key'] == e['query_a'] else 1]
                mse = float(np.mean((np.array(e['value'])-np.eye(4)[target])**2))
                noise[row['profile']].append((mse, e['noise']**2))
    direct = []
    for row in splits['test']:
        if row['profile'] == 'independent':
            e = row['events'][-1]
            direct.append(int(np.argmax(e['value'])) == e['target'])
    return dict(source_sha256=hashlib.sha256((PROJECT/version/'data.py').read_bytes()).hexdigest(),
                config=cfg, label_errors_against_hidden_truth=label_errors, quartet_balance_errors=balance_errors,
                raw_report_reference={name: reference_check(rows) for name, rows in splits.items()},
                hidden_fields_passed_to_model=sorted({'world', 'target', 'truth', 'profile', 'shift', 'stage', 'feedback', 'grounded'} & set(module.INPUT_FIELDS)),
                stream_counts={name: len(rows) for name, rows in splits.items()},
                independent_groups={name: len(rows)//4 for name, rows in splits.items()},
                exact_train_dev_overlap=len(hashes['train'] & hashes['dev']),
                exact_train_test_overlap=len(hashes['train'] & hashes['test']),
                exact_dev_test_overlap=len(hashes['dev'] & hashes['test']),
                structural_template_test_seen_in_train_fraction=float(np.mean([structure(r) in templates['train'] for r in splits['test']])),
                structural_template_test_seen_in_dev_fraction=float(np.mean([structure(r) in templates['dev'] for r in splits['test']])),
                train_profiles=sorted({r['profile'] for r in splits['train']}),
                dev_profiles=sorted({r['profile'] for r in splits['dev']}),
                test_profiles=sorted({r['profile'] for r in splits['test']}),
                independent_current_observation_accuracy=float(np.mean(direct)),
                invert_first_claim=rule_accuracy(splits['test'], 'repetition', first=True),
                invert_last_report_outlier=rule_accuracy(splits['test'], 'outlier'),
                low_reported_noise_by_profile={p: dict(measured_mse=float(np.mean([v[0] for v in values])),
                    reported_variance=float(np.mean([v[1] for v in values])),
                    mse_over_reported=float(np.mean([v[0] for v in values])/np.mean([v[1] for v in values]))) for p, values in noise.items()},
                limits=['Exact hashes include random noise, report IDs and cues; no exact overlap does not establish unseen task structure.',
                        'Targets are checked against generator hidden truth; this is not an independent real-world annotation audit.',
                        'Four color counterfactuals share one base trajectory and are not four independent data sources.',
                        'A shared task grammar remains even when exact templates or delay lengths are held out.',
                        'Inputs are structured synthetic signals, not natural Chinese language.'])


def audit(root):
    results = {}
    for version in ('v04', 'v05'):
        cfg = json.loads((PROJECT/'configs'/version/'base.json').read_text())
        directory = 'verified_streams' if version == 'v05' else 'continuous'
        manifest = PROJECT/'results'/version/directory/'experiment.json'
        if manifest.exists():
            cfg = json.loads(manifest.read_text())['config']
        results[version] = inspect(version, cfg)
    write(Path(root)/'audit.json', results)
    lines = ['# 训练/开发/测试数据审计', '', '| 版本 | 标签不一致 | 平衡错误 | 隐藏字段泄漏 | 精确训练测试重复 | 首条报告减1捷径 |',
             '|---|---:|---:|---:|---:|---:|']
    for name, r in results.items():
        score = r['invert_first_claim']['accuracy']
        lines.append(f"| {name} | {r['label_errors_against_hidden_truth']} | {r['quartet_balance_errors']} | {len(r['hidden_fields_passed_to_model'])} | {r['exact_train_test_overlap']} | {'不适用' if score is None else format(score, '.2%')} |")
    lines += ['', '## 需要处理的问题', '',
              ('检测到首条报告减1的高准确率捷径，需要修改数据。' if results['v05']['invert_first_claim']['accuracy'] > .8
               else '首条报告减1规则没有超过25%；修订数据保持不可信声明不变、只改变真实历史和后续证据。'), '',
              'repetition 的前期请求没有独立真值支持；修订版保留请求但不计入主损失，后续有证据的纠正/保持探针计分。', '',
              '源噪声字段在人工错误报告中与实际误差不匹配，应明确模拟的是失准来源，并增加质量未知/错误标注的独立条件。', '',
              'dev/test 使用同一任务语法。修订版将 long 和16/32步延迟保留给测试；这仍未提供跨任务语法的盲结构外推。', '',
              '更多独立基础轨迹、盲难度切分、反事实仅改变历史而保持当前证据，以及公开中文任务仍需补充。', '',
              '本审计不修改 v0.4 运行中的数据、源码和结果。具体版本哈希与计数见 audit.json。']
    Path(root).mkdir(parents=True, exist_ok=True)
    (Path(root)/'FINDINGS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, default=PROJECT/'results/v05/data_audit')
    result = audit(parser.parse_args().root)
    for version, value in result.items():
        print(version, json.dumps({k: value[k] for k in ('label_errors_against_hidden_truth', 'quartet_balance_errors', 'invert_first_claim', 'structural_template_test_seen_in_dev_fraction')}, ensure_ascii=False))
