"""Fixed-budget corrected comparison with independent search seeds and final holdouts."""
import argparse
from datetime import datetime
from importlib.metadata import version
import inspect
import io
import json
import os
import platform
from pathlib import Path
import time
import traceback
import unittest

import numpy as np
import torch
from tokenizers import Tokenizer
from transformers.utils import logging

from .console import Progress
from v07.common import PROJECT, RunLock, device_for, digest, file_hash, save, write
from v07.data import Corpus
from v09.train import StopRequest, learning_rate_scale, capture_rng, restore_random
from v1.train import optimizer_for
from v3.train import source_hashes as historical_sources
from . import data
from .model import build, metadata, NAMES
from .engine import update, score_lm, score_qa, generate

CONFIG = dict(batch=8, evaluation_batch=16, cpt_tokens=1048576, qa_updates=512,
              search_cpt_tokens=262144, search_qa_updates=64, overfit_updates=512,
              learning_rates=[.0003, .001, .003], seeds=[11, 23, 37], search_seed=101,
              weight_decay=.01, gradient_clip=1., generation_per_distance_bin=64, threads=4)


def sources():
    paths = [*Path(__file__).parent.glob('*.py'), PROJECT/'run_biomatrix_repaired.py',
             PROJECT/'evaluations/tiny_arch/model.py', PROJECT/'scale/engine.py', PROJECT/'scale/data.py',
             PROJECT/'.run/Bio-Matrix 修正版架构对比.run.xml']
    return historical_sources() | {str(p.relative_to(PROJECT)): file_hash(p) for p in paths}


def append(path, row):
    with Path(path).open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')


def ordered_indices(qa, seed, count=None):
    indices = np.random.default_rng(seed).permutation(len(qa.rows)).tolist()
    return indices if count is None else indices[:count]


def follow(root):
    registry = json.loads((root/'registry.json').read_text())
    ui = Progress([dict(id=k) for k in registry['conditions']], 1, pid=registry.get('pid'))
    ui.label = '修正版 · 同预算架构对照'
    try:
        while True:
            state = json.loads((root/'registry.json').read_text())
            ui.total.n = sum(r['status'] == 'completed' for r in state['conditions'].values())
            for k, r in state['conditions'].items():
                ui.details[k].n = r.get('fraction', 0.)
                ui.details[k].postfix = f'CE={r.get("ce", 0.):.4f}'
            ui.stage(state.get('stage', state['status']))
            if state['status'] in ('completed', 'failed', 'aborted'):
                break
            if ui.stop.wait(1.):
                break
    except KeyboardInterrupt:
        pass
    finally:
        ui.close()


def run(args):
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    if args.mode == 'follow':
        return follow(root)
    if args.mode == 'prepare':
        print(json.dumps(data.prepare(), ensure_ascii=False)); return
    if (root/'registry.json').exists():
        old_state = json.loads((root/'registry.json').read_text())
        if old_state.get('status') == 'completed':
            return follow(root)
        if old_state.get('status') == 'running' and old_state.get('pid') != os.getpid():
            try:
                os.kill(old_state['pid'], 0)
            except (ProcessLookupError, KeyError):
                pass
            else:
                return follow(root)
    logging.set_verbosity_error(); torch.set_num_threads(CONFIG['threads']); device = device_for(args.device)
    c = CONFIG
    jobs = [f'overfit_{n}' for n in NAMES]
    jobs += [f'cpt_search_{n}_r{i}' for n in NAMES for i in range(3)]
    jobs += [f'qa_search_{n}_r{i}' for n in NAMES for i in range(3)]
    jobs += [f'final_{n}_s{s}_{p}' for s in c['seeds'] for n in NAMES for p in ('cpt', 'qa')]
    ui = Progress([dict(id=k) for k in jobs], 1, not args.quiet); ui.device = device
    ui.label = '修正版 · 同预算架构对照'
    state = dict(status='running', stage='准备', pid=os.getpid(), conditions={k: dict(status='not_executed', fraction=0.) for k in jobs})
    active = None

    def stage(message):
        state['stage'] = message; ui.stage(message)
        write(root/'registry.json', state)

    try:
        with RunLock(PROJECT/'results/repaired/.train.lock'), StopRequest() as stop:
            if (root/'registry.json').exists():
                state = json.loads((root/'registry.json').read_text())
                if state['status'] == 'completed':
                    return
                state['status'] = 'running'
                state['pid'] = os.getpid()
                ui.total.n = sum(r['status'] == 'completed' for r in state['conditions'].values())
            stage('核对冻结源码、官方语料和修正版数据')
            historical = json.loads((PROJECT/'results/tiny_arch/paired_20261103_local333k/protocol.json').read_text())
            if any(file_hash(PROJECT/p) != sha for p, sha in historical['sources'].items()):
                raise ValueError('Historical source changed; restore frozen sources')
            manifest = data.prepare(stage=stage)
            corpus = Corpus(data.ROOT); qa = {s: data.QA(data.ROOT, s) for s in ('train', 'validation', 'test')}
            tokenizer = Tokenizer.from_file(str(data.ROOT/'tokenizer.json'))
            stage('代码自检：因果性、完整段落、EOS、覆盖验证与干预隔离')
            out = io.StringIO()
            checks = unittest.TextTestRunner(stream=out, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('evaluations.repaired.test_repaired'))
            (root/'verification.txt').write_text(out.getvalue())
            write(root/'verification.json', dict(status='completed' if checks.wasSuccessful() else 'failed', tests=checks.testsRun, failures=len(checks.failures), errors=len(checks.errors)))
            if not checks.wasSuccessful():
                raise RuntimeError('Preflight failed')
            infos, externals = {}, {}
            for name in NAMES:
                model = build(name, c['search_seed'])
                infos[name] = metadata(name, model)
                native = getattr(model, 'model', model)
                externals[name] = dict(module=type(native).__module__, sha256=file_hash(inspect.getfile(type(native))))
            del model
            tests = {}
            for lower, upper in ((-1, 128), (128, 256), (256, 10000)):
                eligible = [i for i, r in enumerate(qa['test'].rows) if lower < r['answer_distance'] <= upper]
                if len(eligible) < c['generation_per_distance_bin']:
                    raise ValueError('Insufficient questions in a predeclared distance bin')
                ids = np.random.default_rng(44109+upper).choice(eligible, c['generation_per_distance_bin'], replace=False).tolist()
                tests[str(upper)] = dict(population=len(eligible), indices=ids)
            protocol = dict(format='repaired-local-architecture-comparison-1', created='2026-11-03', training=c,
                models=infos, sources=sources(), external=externals,
                packages={k: version(k) for k in ('torch', 'transformers', 'numpy', 'tokenizers')},
                dataset_manifest_sha256=digest(manifest), device=str(device), dtype='float32',
                selection='Search seed101 only; CPT LR chosen by full dev LM NLL; QA LR independently chosen by full dev answer+EOS NLL; no test model selection',
                search_budget='Three fixed rates per declared configuration and per stage; original and balanced DS reported separately, never pooled into best-of-family',
                final_test='All eligible WikiText targets once per checkpoint; all eligible official SQuAD dev questions teacher-forced; generation 64 per distance bin with population weighting',
                test_generation=tests, generation_limit=64, qa_objective='Answer and EOS; loss excludes padding and prompt',
                compute_scope='Same targets/samples and optimizer search opportunities; native initialization, different FLOPs/states; MPS reference timing only')
            if (root/'protocol.json').exists():
                if json.loads((root/'protocol.json').read_text()) != protocol:
                    raise ValueError('Resume identity mismatch')
            else:
                write(root/'protocol.json', protocol)
                for p in protocol['sources']:
                    target = root/'source'/p; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes((PROJECT/p).read_bytes())
                write(root/'hardware.json', dict(platform=platform.platform(), machine=platform.machine(),
                    torch=torch.__version__, device=str(device), torch_threads=torch.get_num_threads(),
                    metal_sample='Current allocation after updates; not an exact allocation peak',
                    rss_units='bytes on macOS, KiB on Linux'))
            preservation_file = root/'historical_preservation.json'
            if not preservation_file.exists():
                roots = [PROJECT/'results/v3/paired_20261103T094249_111075',
                         PROJECT/'results/tiny_arch/paired_20261103_local333k']
                write(preservation_file, {str(p.relative_to(PROJECT)): file_hash(p) for directory in roots
                    for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts})

            def train_job(identity, name, seed, phase, rate, budget, base=None, fixed=None):
                nonlocal active
                active = identity; folder = root/identity; folder.mkdir(exist_ok=True)
                if state['conditions'][identity]['status'] == 'completed':
                    ui.details[identity].n = 1.; return json.loads((folder/'result.json').read_text())
                model = build(name, seed).to(device)
                if base is not None:
                    model.load_state_dict(torch.load(base, map_location='cpu', weights_only=True)['model'])
                opt = optimizer_for(model, dict(learning_rate=rate, weight_decay=c['weight_decay']))
                rng = np.random.default_rng((31001 if phase == 'cpt' else 31002) if seed == c['search_seed'] else (32000 if phase == 'cpt' else 33000)+seed)
                updates = budget//(c['batch']*512) if phase == 'cpt' else budget
                counters = dict(updates=0, train_seconds=0., targets=0, inputs=0)
                checkpoint = folder/'last.pt'
                last_registry = time.monotonic()
                if checkpoint.exists():
                    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                    if saved['protocol_sha256'] != digest(protocol):
                        raise ValueError('Checkpoint protocol mismatch')
                    model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
                    counters = saved['counters']; rng.bit_generator.state = saved['data_rng']; restore_random(saved['rng'], device)
                    if (folder/'metrics.jsonl').exists():
                        lines = [json.loads(x) for x in (folder/'metrics.jsonl').read_text().splitlines()]
                        if any(r['update'] > counters['updates'] for r in lines):
                            (folder/'metrics.jsonl').rename(folder/f'uncommitted_{time.time_ns()}.jsonl')
                            (folder/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in lines if r['update'] <= counters['updates']))

                def persist():
                    if sources() != protocol['sources']:
                        raise ValueError('Frozen experiment source changed during run')
                    save(checkpoint, dict(protocol_sha256=digest(protocol), model=model.state_dict(), optimizer=opt.state_dict(),
                        counters=counters, data_rng=rng.bit_generator.state, rng=capture_rng(device)))

                state['conditions'][identity] = dict(status='running', fraction=counters['updates']/updates)
                stage(f'{identity} · {phase} · 开始/恢复')
                for step in range(counters['updates'], updates):
                    if stop.requested:
                        persist(); raise KeyboardInterrupt
                    if phase == 'cpt':
                        values, docs, starts = corpus.sample('train', rng, c['batch'], 512)
                        mask = np.ones((c['batch'], 512), dtype=bool); sample = dict(documents=docs, starts=starts)
                    else:
                        indices = fixed if fixed is not None else rng.integers(len(qa['train'].rows), size=c['batch']).tolist()
                        values, mask, _ = qa['train'].batch(indices); sample = dict(indices=indices)
                    lr = rate if fixed is not None else rate*learning_rate_scale(step+1, updates)
                    for group in opt.param_groups:
                        group['lr'] = lr
                    row = update(model, opt, values, mask, device, infos[name]['auxiliary_weight'])
                    counters['updates'] += 1; counters['train_seconds'] += row['seconds']
                    counters['targets'] += row['targets']; counters['inputs'] += row['inputs']
                    append(folder/'metrics.jsonl', dict(**row, update=step+1, learning_rate=lr,
                        sample_sha256=digest(dict(values=values.tolist(), mask=mask.tolist())), sample=sample))
                    record = dict(status='running', fraction=(step+1)/updates, ce=row['ce'], **counters)
                    state['conditions'][identity] = record; ui.details[identity].n = record['fraction']
                    ui.details[identity].postfix = f'CE={row["ce"]:.4f}'
                    ui.stage(f'{identity} · {step+1}/{updates} · CE={row["ce"]:.4f}')
                    if (step+1) % 16 == 0 or time.monotonic()-last_registry >= 1.:
                        write(root/'registry.json', state)
                        last_registry = time.monotonic()
                    if (step+1) % 64 == 0:
                        persist()
                persist(); stage(f'{identity} · 固定开发验证')
                result = dict(status='completed', condition=name, seed=seed, phase=phase, learning_rate=rate, **counters)
                if phase == 'cpt':
                    result['development'] = score_lm(model, corpus, 'validation', device, c['evaluation_batch'])
                else:
                    split = 'train' if fixed is not None else 'validation'
                    result['development'] = score_qa(model, qa[split], device, fixed, c['evaluation_batch'])
                    if fixed is not None:
                        generated = generate(model, qa['train'], fixed, tokenizer, device, c['evaluation_batch'])
                        write(folder/'overfit_generation.private.json', generated)
                        result['generation'] = {k: v for k, v in generated.items() if k != 'records'}
                        result['diagnostic_passed'] = result['development']['answer_nll'] < .5 and result['development']['first_accuracy'] >= .875 and generated['em'] >= .5
                write(folder/'result.json', result)
                state['conditions'][identity] = dict(status='completed', fraction=1., **counters)
                ui.total.n += 1; ui.details[identity].n = 1.; write(root/'registry.json', state)
                model.to('cpu'); del model, opt
                if device.type == 'mps':
                    torch.mps.empty_cache()
                return result

            short = [i for i in ordered_indices(qa['train'], 55001) if qa['train'].rows[i]['prefix_tokens'] <= 128
                     and qa['train'].rows[i]['answer_end']-qa['train'].rows[i]['answer_start'] <= 12][:8]
            if len(short) != 8:
                raise ValueError('Need eight real short QA controls')
            write(root/'positive_control.json', dict(indices=short, updates=c['overfit_updates'], learning_rate=.003,
                threshold='Answer NLL<0.5, first accuracy>=0.875, greedy EM>=0.5 on eight training questions; diagnostic only'))
            for name in NAMES:
                train_job(f'overfit_{name}', name, 101, 'qa', .003, c['overfit_updates'], fixed=short)
            selection = {}
            for name in NAMES:
                trials = [train_job(f'cpt_search_{name}_r{i}', name, 101, 'cpt', rate, c['search_cpt_tokens'])
                          for i, rate in enumerate(c['learning_rates'])]
                best = min(range(3), key=lambda i: (trials[i]['development']['nll'], c['learning_rates'][i]))
                selection[name] = dict(cpt_rate=c['learning_rates'][best], cpt_trial=f'cpt_search_{name}_r{best}')
            for name in NAMES:
                base = root/selection[name]['cpt_trial']/'last.pt'
                trials = [train_job(f'qa_search_{name}_r{i}', name, 101, 'qa', rate, c['search_qa_updates'], base)
                          for i, rate in enumerate(c['learning_rates'])]
                best = min(range(3), key=lambda i: (trials[i]['development']['nll'], c['learning_rates'][i]))
                selection[name]['qa_rate'] = c['learning_rates'][best]
            write(root/'selection.json', selection)
            for seed in c['seeds']:
                order = list(NAMES); np.random.default_rng(56000+seed).shuffle(order)
                for name in order:
                    identity = f'final_{name}_s{seed}'
                    train_job(identity+'_cpt', name, seed, 'cpt', selection[name]['cpt_rate'], c['cpt_tokens'])
                    train_job(identity+'_qa', name, seed, 'qa', selection[name]['qa_rate'], c['qa_updates'], root/(identity+'_cpt')/'last.pt')
            active = None
            stage('全部配置与学习率已冻结；开始一次性保留测试评分')
            from .report import final_evaluation
            final_evaluation(root, protocol, corpus, qa, tokenizer, device, stage, stop)
            preserved = json.loads(preservation_file.read_text())
            changed = [p for p, sha in preserved.items() if not (PROJECT/p).is_file() or file_hash(PROJECT/p) != sha]
            write(root/'historical_preservation_after.json', dict(status='completed' if not changed else 'failed', files=len(preserved), changed=changed))
            if changed:
                raise ValueError('Historical artifacts changed; inspect preservation report')
            state['status'] = 'completed'; stage('修正版对比完成；原始结果和独立保留测试已保存')
    except BaseException as error:
        state['status'] = 'aborted' if isinstance(error, KeyboardInterrupt) else 'failed'
        if active:
            state['conditions'][active]['status'] = state['status']
        write(root/'registry.json', state)
        (root/'traceback.private.txt').write_text(traceback.format_exc())
        raise
    finally:
        ui.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('run', 'follow', 'prepare'), default='run')
    parser.add_argument('--output', type=Path, default=PROJECT/'results/repaired/local333k_corrected_20261103')
    parser.add_argument('--device', choices=('auto', 'mps', 'cpu'), default='auto')
    parser.add_argument('--quiet', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
