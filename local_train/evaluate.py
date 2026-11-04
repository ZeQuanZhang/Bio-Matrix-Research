"""Use locked upstream tasks and scores; only redirect their progress notifications."""
from contextlib import contextmanager
from functools import lru_cache
import json
import math
from pathlib import Path

import lm_eval
from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks import TaskManager
from lm_eval.tasks._yaml_loader import load_yaml
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from evaluations.standard.run import backend, protocol
from v07.common import write


@lru_cache(maxsize=5)
def load_task(name):
    spec = protocol()['tasks'][name]
    config = load_yaml(Path(lm_eval.__file__).parent/spec['yaml'], resolve_func=True)
    config['dataset_kwargs'] = (config.get('dataset_kwargs') or {}) | dict(revision=spec['revision'])
    task = ConfigurableTask(config=config)
    split = 'test' if task.has_test_docs() else 'validation'
    cache = task.dataset[split].cache_files
    if split != spec['split'] or task.get_config('dataset_path') != spec['dataset']:
        raise ValueError(f'Unexpected benchmark split: {name}')
    if not cache or any(spec['revision'] not in Path(f['filename']).parts for f in cache):
        raise ValueError(f'Unexpected cached benchmark revision: {name}')
    return task


def tasks(ui, stop):
    for name in protocol()['tasks']:
        if stop.requested:
            raise KeyboardInterrupt
        ui.stage(f'载入标准评测集 · {name} · 校验固定版本')
        yield name, load_task(name)


@contextmanager
def progress_notifications(callback):
    import lm_eval.models.huggingface as hf
    original = hf.tqdm

    class SilentProgress(tqdm):
        def __init__(self, *args, **kwargs):
            self.scoring = kwargs.get('desc') == 'Running loglikelihood requests'
            kwargs['disable'] = True
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            if self.scoring:
                callback(n)
            return super().update(n)

    hf.tqdm = SilentProgress
    try:
        yield
    finally:
        hf.tqdm = original


def run(model, data_root, folder, c, ui, stop, smoke=False):
    folder.mkdir(parents=True, exist_ok=True)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(data_root/'tokenizer.json'),
        bos_token='<|endoftext|>', eos_token='<|endoftext|>', pad_token='<|endoftext|>')
    slots = 0

    def observe(n):
        nonlocal slots
        if stop.requested:
            raise KeyboardInterrupt
        slots += n
        ui.stage(f'{folder.parent.name} · {active} · {bar.n:,.0f}/{bar.total:,} 评分请求 · 输入 {slots:,} token槽')

    lm = backend(model.eval(), tokenizer, c['sequence'], c['eval_batch'], observe)
    manager = TaskManager(include_defaults=False)
    for active, task in tasks(ui, stop):
        bar = ui.details[active]; slots = 0
        path = folder/f'{active}.json'
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['engineering_smoke'] != smoke:
                raise ValueError('Full evaluation and smoke outputs cannot be mixed')
            bar.n = bar.total
            ui.total.update()
            continue
        docs = task.eval_docs.select(range(min(2, len(task.eval_docs)))) if smoke else task.eval_docs
        output_type = task.get_config('output_type')
        if output_type == 'loglikelihood_rolling':
            count = sum(math.ceil(len(lm.tok_encode(task.doc_to_target(d)))/c['sequence']) for d in docs)
        elif output_type == 'multiple_choice':
            count = sum(len(task.doc_to_choice(d)) for d in docs)
        else:
            count = len(docs)
        bar.reset(max(1, count)); bar.postfix = '评分中'

        def advance(n):
            if stop.requested:
                raise KeyboardInterrupt
            bar.update(n)

        write(folder/f'{active}.status.json', dict(status='running'))
        try:
            with progress_notifications(advance):
                result = lm_eval.simple_evaluate(model=lm, tasks=[task], task_manager=manager,
                    num_fewshot=0, apply_chat_template=False, log_samples=False,
                    limit=2 if smoke else None, bootstrap_iters=0 if smoke else 100000)
            if bar.n != count:
                raise ValueError(f'Scoring progress mismatch: {active}, {bar.n} != {count}')
        except BaseException as error:
            write(folder/f'{active}.status.json', dict(status='aborted' if isinstance(error, KeyboardInterrupt) else 'failed'))
            raise
        assert result['n-samples'][active]['effective'] == len(docs) and not result.get('samples')
        write(path, json.loads(json.dumps(result, default=str, allow_nan=False)) |
              dict(status='completed', engineering_smoke=smoke, inference_input_slots=slots))
        bar.n = bar.total; bar.postfix = '已完成'
        write(folder/f'{active}.status.json', dict(status='completed', scoring_requests=count))
        ui.total.update()
