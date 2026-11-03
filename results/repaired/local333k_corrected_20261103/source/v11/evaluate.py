"""Paired history interventions with identical KV and masked suffix scoring."""
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from v07.common import sync


def clear_memory(model, state):
    """Reset Bio experience while retaining the incoming text KV and clocks."""
    fresh = model.initial_state(len(state['position']))
    return state | {name: fresh[name] for name in ('memory', 'pending', 'field', 'field_summary')
                    if name in state and name in fresh}


def _scoring_masks(values, prefix, target_mask, first_answer_mask):
    shape = (len(values), values.shape[1]-1)
    targets = np.ones(shape, dtype=bool) if target_mask is None else np.asarray(target_mask)
    first = np.zeros(shape, dtype=bool) if first_answer_mask is None else np.asarray(first_answer_mask)
    if targets.shape != shape or targets.dtype != np.bool_ or first.shape != shape or first.dtype != np.bool_:
        raise ValueError('target_mask and first_answer_mask must be boolean [windows,input_tokens]')
    if bool((first & ~targets).any()) or bool(first[:, :prefix].any()):
        raise ValueError('First answers must be supervised suffix targets')
    if not bool(targets[:, prefix:].any()):
        raise ValueError('No supervised targets in the scored suffix')
    return targets[:, prefix:], first[:, prefix:]


def _aggregate(loss_sums, counts, first_sums, first_counts, correct):
    total = sum(counts)
    per_window = [loss/count if count else None for loss, count in zip(loss_sums, counts)]
    scored = [value for value in per_window if value is not None]
    first_total = sum(first_counts)
    nll = sum(loss_sums)/total
    result = dict(status='completed', nll=nll, ppl=math.exp(min(nll, 80)),
                  windows=len(counts), scored_windows=len(scored), targets=total,
                  per_window_nll=per_window, per_window_targets=counts,
                  window_mean_nll=float(np.mean(scored)),
                  first_answer_nll=sum(first_sums)/first_total if first_total else None,
                  first_answer_accuracy=sum(correct)/first_total if first_total else None,
                  first_answer_targets=first_total,
                  first_answer_nll_per_window=[loss/count if count else None for loss, count in zip(first_sums, first_counts)],
                  first_answer_accuracy_per_window=[hits/count if count else None for hits, count in zip(correct, first_counts)],
                  first_answer_targets_per_window=first_counts)
    return result


@torch.no_grad()
def evaluate_history(model, windows, device, prefix_tokens, batch=2, seed=7191, progress=None,
                     target_mask=None, first_answer_mask=None):
    """Compare direct snapshot edits and whole-path interventions on paired text.

    Masks select measurements only and are never given to the model. Right
    padding is allowed after scored positions because the model remains causal.
    """
    values = np.asarray(windows)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError('Evaluation windows must be integer [windows,input_tokens+1]')
    if len(values) < 1 or not 0 < prefix_tokens < values.shape[1]-1 or batch < 1:
        raise ValueError('Need nonempty history, scored suffix, and positive batch')
    if int(values.min()) < 0 or int(values.max()) >= model.cfg.vocab_size:
        raise ValueError('Evaluation token outside the vocabulary')
    masks, first_masks = _scoring_masks(values, prefix_tokens, target_mask, first_answer_mask)
    limit = model.cfg.context_limit
    old_tokens = max(0, prefix_tokens-limit)
    baseline_receptive_tokens = getattr(model.cfg, 'layers', 1)*limit
    remote_tokens = max(0, prefix_tokens-baseline_receptive_tokens)
    names = ['warm', 'empty_memory', 'memory_read_off']
    if len(values) > 1 and old_tokens > 0:
        names += ['unrelated_history', 'shuffled_history']
    names += ['memory_read_off_all']
    if len(values) > 1 and remote_tokens > 0:
        names += ['remote_history_reencoded']
    rng = np.random.default_rng(seed)
    shuffled = values[:, :prefix_tokens].copy()
    for row in shuffled:
        row[:old_tokens] = row[rng.permutation(old_tokens)]
    mismatched = np.roll(values, 1, axis=0)[:, :prefix_tokens].copy()
    mismatched[:, old_tokens:] = values[:, old_tokens:prefix_tokens]
    reencoded = np.roll(values, 1, axis=0)[:, :prefix_tokens].copy()
    reencoded[:, remote_tokens:] = values[:, remote_tokens:prefix_tokens]
    conditions, warm_states = {}, {}
    mode = model.training
    parameters = {name: value._version for name, value in model.named_parameters()}
    model.eval()
    sync(device)
    started = time.perf_counter()
    try:
        for name in names:
            loss_sums, counts, first_sums, first_counts, correct = [], [], [], [], []
            for start in range(0, len(values), batch):
                end = min(start+batch, len(values))
                inputs = torch.as_tensor(values[start:end, prefix_tokens:-1].copy(), dtype=torch.long, device=device)
                targets = torch.as_tensor(values[start:end, prefix_tokens+1:].copy(), dtype=torch.long, device=device)
                if name == 'warm':
                    prefix = torch.as_tensor(values[start:end, :prefix_tokens].copy(), dtype=torch.long, device=device)
                    _, state, _ = model(prefix)
                    warm_states[start] = state
                elif name in ('memory_read_off_all', 'remote_history_reencoded'):
                    source = values[:, :prefix_tokens] if name == 'memory_read_off_all' else reencoded
                    prefix = torch.as_tensor(source[start:end].copy(), dtype=torch.long, device=device)
                    _, state, _ = model(prefix, memory_read=name != 'memory_read_off_all')
                else:
                    state = warm_states[start]
                    if name in ('unrelated_history', 'shuffled_history'):
                        source = mismatched if name == 'unrelated_history' else shuffled
                        prefix = torch.as_tensor(source[start:end].copy(), dtype=torch.long, device=device)
                        _, donor, _ = model(prefix)
                        state = state | {key: donor[key] for key in ('memory', 'pending', 'field', 'field_summary')
                                         if key in state and key in donor}
                if name == 'empty_memory':
                    state = clear_memory(model, state)
                logits, _, _ = model(inputs, state, memory_read=name not in ('memory_read_off', 'memory_read_off_all'))
                losses = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), reduction='none')
                losses = losses.reshape(len(inputs), -1).cpu().numpy().astype(np.float64)
                hits = (logits.argmax(-1) == targets).cpu().numpy()
                for at in range(end-start):
                    selected, first = masks[start+at], first_masks[start+at]
                    scored = losses[at, selected]
                    if not bool(np.isfinite(scored).all()):
                        raise FloatingPointError('Nonfinite scored history evaluation loss')
                    loss_sums.append(float(scored.sum()))
                    counts.append(int(selected.sum()))
                    first_sums.append(float(losses[at, first].sum()))
                    first_counts.append(int(first.sum()))
                    correct.append(int(hits[at, first].sum()))
                if progress:
                    progress(name, end, len(values))
            conditions[name] = _aggregate(loss_sums, counts, first_sums, first_counts, correct)
        for name in ('unrelated_history', 'shuffled_history', 'remote_history_reencoded'):
            if name not in conditions:
                bound = 'the baseline multi-layer receptive bound' if name == 'remote_history_reencoded' else 'local KV'
                conditions[name] = dict(status='not_applicable', reason=f'Need two windows and history beyond {bound}')
    finally:
        model.train(mode)
    if parameters != {name: value._version for name, value in model.named_parameters()}:
        raise RuntimeError('History evaluation unexpectedly modified model parameters')
    sync(device)
    warm = conditions['warm']['nll']
    return dict(conditions=conditions, warm_nll=warm, warm_ppl=conditions['warm']['ppl'],
                memory_clear_cost=conditions['empty_memory']['nll']-warm,
                read_off_cost=conditions['memory_read_off']['nll']-warm,
                total_read_off_cost=conditions['memory_read_off_all']['nll']-warm,
                unrelated_cost=(conditions['unrelated_history']['nll']-warm
                                if conditions['unrelated_history']['status'] == 'completed' else None),
                reencoded_history_cost=(conditions['remote_history_reencoded']['nll']-warm
                                        if conditions['remote_history_reencoded']['status'] == 'completed' else None),
                first_answer_nll=conditions['warm']['first_answer_nll'],
                first_answer_accuracy=conditions['warm']['first_answer_accuracy'],
                prefix_tokens=prefix_tokens, local_context_tokens=limit,
                baseline_receptive_tokens=baseline_receptive_tokens, remote_tokens=remote_tokens,
                scored_tokens_per_window=conditions['warm']['per_window_targets'],
                scored_targets=conditions['warm']['targets'], seconds=time.perf_counter()-started,
                intervention_contract='Direct snapshot conditions retain identical incoming warm KV and clocks. memory_read_off_all reencodes prefix with all Bio reads disabled; remote_history_reencoded replaces only tokens older than the baseline multi-layer receptive bound and reencodes KV plus Bio state. Whole-path conditions may change KV. Suffix writes remain enabled.',
                direct_conditions=['warm', 'empty_memory', 'memory_read_off', 'unrelated_history', 'shuffled_history'],
                whole_path_conditions=['memory_read_off_all', 'remote_history_reencoded'],
                aggregation='Token-weighted NLL; window means and first-answer measurements are reported separately.',
                note='Paired windows, not independent training seeds. Direct clear costs retain Bio information already encoded in warm KV. Total read-off also removes static Bio contributions; reencoded-history cost targets remote input dependence.')
