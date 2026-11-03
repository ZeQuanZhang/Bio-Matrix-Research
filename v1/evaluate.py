"""History interventions with identical scored text and no parameter updates."""
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from v07.common import sync


def clear_memory(model, state):
    """Keep positions and current attention KV; remove only the Bio experience."""
    fresh = model.initial_state(len(state['position']))
    return state | {name: fresh[name] for name in ('memory', 'pending', 'field', 'field_summary')
                    if name in state and name in fresh}


@torch.no_grad()
def evaluate_history(model, windows, device, prefix_tokens, batch=2, seed=7191, progress=None):
    values = np.asarray(windows)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError('Evaluation windows must be integer [windows, inputs+1]')
    if len(values) < 1 or not 0 < prefix_tokens < values.shape[1]-1 or batch < 1:
        raise ValueError('Need nonempty history, scored suffix, and positive batch')
    if int(values.min()) < 0 or int(values.max()) >= model.cfg.vocab_size:
        raise ValueError('Evaluation token outside the vocabulary')
    limit = model.cfg.context_limit
    old_tokens = max(0, prefix_tokens-limit)
    names = ['warm', 'empty_memory', 'memory_read_off']
    if len(values) > 1 and old_tokens > 0:
        names += ['unrelated_history', 'shuffled_history']
    rng = np.random.default_rng(seed)
    shuffled = values[:, :prefix_tokens].copy()
    for row in shuffled:
        row[:old_tokens] = row[rng.permutation(old_tokens)]
    mismatched = np.roll(values, 1, axis=0)[:, :prefix_tokens].copy()
    mismatched[:, old_tokens:] = values[:, old_tokens:prefix_tokens]
    conditions, warm_states = {}, {}
    mode = model.training
    parameters = {name: value._version for name, value in model.named_parameters()}
    model.eval()
    sync(device)
    started = time.perf_counter()
    try:
        for name in names:
            rows = []
            for start in range(0, len(values), batch):
                end = min(start+batch, len(values))
                source = (mismatched if name == 'unrelated_history' else
                          shuffled if name == 'shuffled_history' else values[:, :prefix_tokens])
                inputs = torch.as_tensor(values[start:end, prefix_tokens:-1].copy(), dtype=torch.long, device=device)
                targets = torch.as_tensor(values[start:end, prefix_tokens+1:].copy(), dtype=torch.long, device=device)
                read = name != 'memory_read_off'
                if name == 'warm':
                    prefix = torch.as_tensor(values[start:end, :prefix_tokens].copy(), dtype=torch.long, device=device)
                    _, state, _ = model(prefix)
                    warm_states[start] = state
                else:
                    state = warm_states[start]
                    if name in ('unrelated_history', 'shuffled_history'):
                        prefix = torch.as_tensor(source[start:end].copy(), dtype=torch.long, device=device)
                        _, donor, _ = model(prefix)
                        # Transplant Bio experience only; current KV and clocks stay identical.
                        state = state | {key: donor[key] for key in ('memory', 'pending', 'field', 'field_summary')
                                         if key in state and key in donor}
                if name == 'empty_memory':
                    state = clear_memory(model, state)
                logits, _, _ = model(inputs, state, memory_read=read)
                losses = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), reduction='none')
                per_window = losses.reshape(len(inputs), -1).mean(1).cpu().tolist()
                if not all(math.isfinite(value) for value in per_window):
                    raise FloatingPointError('Nonfinite history evaluation loss')
                rows.extend(per_window)
                if progress:
                    progress(name, end, len(values))
            average = float(np.mean(rows))
            conditions[name] = dict(status='completed', nll=average, ppl=math.exp(min(average, 80)),
                                    per_window_nll=rows, windows=len(rows),
                                    targets=len(rows)*(values.shape[1]-1-prefix_tokens))
        for name in ('unrelated_history', 'shuffled_history'):
            if name not in conditions:
                conditions[name] = dict(status='not_applicable', reason='Need two windows and history beyond local KV')
    finally:
        model.train(mode)
    if parameters != {name: value._version for name, value in model.named_parameters()}:
        raise RuntimeError('History evaluation unexpectedly modified model parameters')
    sync(device)
    warm = conditions['warm']['nll']
    return dict(conditions=conditions, warm_nll=warm, warm_ppl=conditions['warm']['ppl'],
                memory_clear_cost=conditions['empty_memory']['nll']-warm,
                read_off_cost=conditions['memory_read_off']['nll']-warm,
                unrelated_cost=(conditions['unrelated_history']['nll']-warm
                                if conditions['unrelated_history']['status'] == 'completed' else None),
                prefix_tokens=prefix_tokens, local_context_tokens=limit,
                scored_tokens_per_window=values.shape[1]-1-prefix_tokens,
                seconds=time.perf_counter()-started,
                intervention_contract='All conditions enter the scored suffix with identical attention KV and positions; only Bio experience/read access changes.',
                note='Paired windows, not independent training seeds. Mismatched windows are a content proxy for unrelated history.')
