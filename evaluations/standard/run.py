"""Frozen V3 checkpoints scored with the unmodified lm-eval causal backend."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from importlib.metadata import version
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import lm_eval
from lm_eval.api.task import ConfigurableTask
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from lm_eval.tasks._yaml_loader import load_yaml
import torch
from transformers import PretrainedConfig, PreTrainedTokenizerFast

from scripts.v2_console import Progress
from v07.common import PROJECT, device_for, digest, file_hash, write

LOCK = Path(__file__).with_name('protocol.json')
REFERENCE = PROJECT/'results/repaired/local333k_corrected_20261103'
CONDITIONS = ('trans_full', 'v3_static', 'v3_dynamic')


class LogitsModel(torch.nn.Module):
    """Translate the project's tuple output into the causal HFLM logits interface."""

    def __init__(self, network, tokenizer, observe=lambda n: None):
        super().__init__()
        self.network = network
        self.config = PretrainedConfig(is_encoder_decoder=False, vocab_size=tokenizer.vocab_size,
            bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id)
        self.observe = observe

    @property
    def device(self):
        return next(self.network.parameters()).device

    @property
    def dtype(self):
        return next(self.network.parameters()).dtype

    def tie_weights(self):
        # The underlying architecture already defines its output projection.
        pass

    def forward(self, input_ids):
        output = self.network(input_ids)
        self.observe(input_ids.numel())
        return SimpleNamespace(logits=output[0] if isinstance(output, tuple) else output.logits)


def backend(network, tokenizer, max_length, batch_size, observe=lambda n: None):
    return HFLM(pretrained=LogitsModel(network, tokenizer, observe), backend='causal',
        tokenizer=tokenizer, max_length=max_length, batch_size=batch_size,
        add_bos_token=False, softmax_dtype='float32')


def protocol():
    p = json.loads(LOCK.read_text())
    if version('lm_eval') != p['harness']:
        raise ValueError('lm-eval version differs from the locked protocol')
    upstream = Path(lm_eval.__file__).parent
    for name, expected in p['upstream_files'].items():
        if file_hash(upstream/name) != expected:
            raise ValueError(f'Upstream evaluation definition changed: {name}')
    return p


def reference_tokenizer(reference=REFERENCE):
    from evaluations.repaired.data import ROOT, validate
    manifest = validate(ROOT)
    p = json.loads((reference/'protocol.json').read_text())
    if digest(manifest) != p['dataset_manifest_sha256']:
        raise ValueError('Reference tokenizer data identity changed')
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(ROOT/'tokenizer.json'),
        bos_token='[BOS]', eos_token='[EOS]', pad_token='[PAD]', unk_token='[UNK]')
    assert (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) == (1, 2, 0)
    return tokenizer, p


def run(args):
    from evaluations.repaired.model import build

    p = protocol()
    tokenizer, frozen = reference_tokenizer(args.reference)
    if any(file_hash(PROJECT/name) != sha for name, sha in frozen['sources'].items()):
        raise ValueError('Reference model source identity changed')
    device = device_for(args.device)
    torch.set_num_threads(4)
    output = args.output or PROJECT/'results/standard_eval'/datetime.now().strftime('%Y%m%dT%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    conditions = args.condition or list(CONDITIONS)
    tasks = list(p['tasks'])
    identities = [f'{name}-{task}' for name in conditions for task in tasks]
    state = dict(status='running', mode='engineering_smoke' if args.smoke else 'full',
        conditions={k: dict(status='not_executed') for k in identities})
    write(output/'registry.json', state)
    write(output/'protocol.json', p | dict(seed=args.seed, conditions=conditions, device=str(device),
        reference_phase='cpt', reference_protocol_sha256=digest(frozen), batch_size=args.batch,
        limit=2 if args.smoke else None, packages={n: version(n) for n in
        ('lm_eval', 'datasets', 'transformers', 'torch', 'tokenizers')},
        adapter_sha256=file_hash(__file__)))
    (output/'adapter_source.py').write_bytes(Path(__file__).read_bytes())
    model_labels = {'trans_full': 'TF', 'v3_static': 'BioS', 'v3_dynamic': 'BioD'}
    task_labels = dict(zip(tasks, ('WT', 'LAMB', 'PIQA', 'ARC-E', 'HS')))
    labels = {f'{n}-{t}': f'{model_labels[n]}-{task_labels[t]}' for n in conditions for t in tasks}
    ui = Progress([dict(id=labels[k]) for k in identities], 1, not args.quiet)
    ui.label = 'V3 · 公开标准评测'; ui.device = device
    active = None
    try:
        with (output/'harness.private.log').open('w', encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
            logging.basicConfig(stream=log, level=logging.INFO, force=True)
            manager = TaskManager(include_defaults=False)
            for name in conditions:
                checkpoint = args.reference/f'final_{name}_s{args.seed}_cpt/last.pt'
                saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                if saved['protocol_sha256'] != digest(frozen):
                    raise ValueError('Reference checkpoint identity changed')
                network = build(name, args.seed)
                network.load_state_dict(saved['model'], strict=True)
                network.to(device).eval()
                weight_sha = file_hash(checkpoint)
                slots = 0

                def observe(n):
                    nonlocal slots
                    slots += n
                    ui.stage(f'{active} · 已计算 {slots:,} 个输入 token 槽')

                lm = backend(network, tokenizer, p['max_length'], args.batch, observe)
                for task, spec in p['tasks'].items():
                    active = f'{name}-{task}'; slots = 0
                    state['conditions'][active] = dict(status='running')
                    write(output/'registry.json', state)
                    ui.stage(f'{name} · {task} · 读取公开数据与评分')
                    upstream = Path(lm_eval.__file__).parent/spec['yaml']
                    config = load_yaml(upstream, resolve_func=True)
                    config['dataset_kwargs'] = (config.get('dataset_kwargs') or {}) | dict(revision=spec['revision'])
                    loaded = ConfigurableTask(config=config)
                    split = 'test' if loaded.has_test_docs() else 'validation'
                    if loaded.get_config('dataset_path') != spec['dataset'] or split != spec['split']:
                        raise ValueError(f'Unexpected dataset or split for {task}')
                    cache = loaded.dataset[split].cache_files
                    if not cache or any(spec['revision'] not in Path(f['filename']).parts for f in cache):
                        raise ValueError(f'Loaded cache revision differs from the lock for {task}')
                    count = len(loaded.eval_docs)
                    result = lm_eval.simple_evaluate(model=lm, tasks=[loaded], task_manager=manager,
                        num_fewshot=0, apply_chat_template=False, log_samples=False,
                        limit=2 if args.smoke else None, bootstrap_iters=0 if args.smoke else 100000)
                    samples = result['n-samples'][task]
                    assert samples['original'] == count and samples['effective'] == (min(count, 2) if args.smoke else count)
                    assert not result.get('samples') and all(f'{m},none' in result['results'][task] for m in spec['metrics'])
                    # Keep upstream scores and configuration without storing raw prompts/answers.
                    record = json.loads(json.dumps(result, default=str, allow_nan=False))
                    write(output/f'{active}.json', record | dict(checkpoint_sha256=weight_sha,
                        dataset_revision=spec['revision'], split=split, inference_input_slots=slots,
                        status='completed', engineering_smoke=args.smoke))
                    state['conditions'][active] = dict(status='completed', **samples)
                    ui.details[labels[active]].n = 1; ui.total.update(1)
                    write(output/'registry.json', state)
                del lm, network, saved
                if device.type == 'mps':
                    torch.mps.empty_cache()
        state['status'] = 'completed'
        ui.stage('标准评测已保存')
    except BaseException as error:
        state['status'] = 'aborted' if isinstance(error, KeyboardInterrupt) else 'failed'
        if active and state['conditions'][active]['status'] == 'running':
            state['conditions'][active] = dict(status=state['status'], error=repr(error))
        raise
    finally:
        write(output/'registry.json', state)
        ui.close()
    print(f'标准评测结果：{output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='每任务2条；仅验收')
    parser.add_argument('--condition', choices=CONDITIONS, action='append')
    parser.add_argument('--seed', type=int, choices=(11, 23, 37), default=11)
    parser.add_argument('--device', choices=('auto', 'mps', 'cpu'), default='auto')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reference', type=Path, default=REFERENCE, help='已完成的修正版运行目录')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    if args.batch < 1 or args.condition and len(set(args.condition)) != len(args.condition):
        parser.error('batch须为正数；条件不可重复')
    run(args)
