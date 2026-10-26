"""Paired public-text likelihood windows and frozen-weight causal interventions."""
from contextlib import contextmanager
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .common import digest, write
from .model import invalidate_feedback


def windows(corpus, split, cfg):
    rng = np.random.default_rng(cfg['evaluation_seed']+(0 if split == 'validation' else 1))
    length = cfg['prefix_tokens']+cfg['target_tokens']
    count = cfg['validation_windows'] if split == 'validation' else cfg['test_windows']
    a, docs, starts = corpus.sample(split, rng, count, length)
    b, other, _ = corpus.sample(split, rng, count, length)
    for i in range(count):
        for _ in range(256):
            if other[i] != docs[i]:
                break
            row, ids, _ = corpus.sample(split, rng, 1, length)
            b[i], other[i] = row[0], ids[0]
        else:
            raise ValueError('Evaluation needs two distinct documents with sufficiently long windows')
    return dict(a=a, b=b, docs=docs, starts=starts, other=other,
                sha256=digest(dict(a=a.tolist(), b=b.tolist(), docs=docs, starts=starts, other=other)))


@contextmanager
def runtime_intervention(model, condition):
    """Inference switches never modify learned parameters or the saved model config."""
    name = {'no_local': 'runtime_local_enabled', 'no_slow': 'runtime_slow_enabled'}.get(condition)
    previous = getattr(model, name) if name is not None else None
    try:
        if name is not None:
            setattr(model, name, False)
        yield
    finally:
        if name is not None:
            setattr(model, name, previous)


@torch.no_grad()
def feed(model, tokens, device, chunk, state=None, *, diagnostics=None, reset_each_token=False,
         head_only=False, stop=None):
    outputs = []
    # Target diagnostics are sampled at every visible token, including Transformer cache steps.
    size = 1 if diagnostics is not None or reset_each_token else chunk
    for start in range(0, len(tokens), size):
        if stop and stop['requested']:
            raise KeyboardInterrupt
        x = torch.as_tensor(np.asarray(tokens[start:start+size]).copy(), dtype=torch.long, device=device)[None]
        if head_only:
            logits = model.head_only(x); diag = {}; state = None
        else:
            logits, state, diag = model(x, None if reset_each_token else state)
        outputs.append(logits[0])
        if diagnostics is not None:
            diagnostics.append({name: value.detach().mean() if isinstance(value, torch.Tensor) else float(value)
                                for name, value in diag.items()})
    return (torch.cat(outputs) if outputs else None), state


def materialize_diagnostics(rows, device):
    entries = [(i, name, value) for i, row in enumerate(rows) for name, value in row.items()]
    if not entries:
        return rows
    # One device transfer per scored window avoids a synchronization for every scalar and token.
    values = torch.stack([torch.as_tensor(value, device=device) for _, _, value in entries]).cpu().tolist()
    result = [{} for _ in rows]
    for (i, name, _), value in zip(entries, values):
        result[i][name] = float(value)
    return result


def diagnostic_summary(rows):
    names = sorted({name for row in rows for name in row})
    return {name: dict(mean=float(np.mean(values)), minimum=float(np.min(values)), maximum=float(np.max(values)),
                       samples=len(values))
            for name in names if (values := [row[name] for row in rows if name in row])}


def calibration_error(pairs):
    pairs = np.asarray(pairs)
    # Integer assignment makes boundary values exclusive; probability 1 belongs to the final bin.
    bins = np.minimum((pairs[:, 0]*10).astype(np.int64), 9)
    error = 0.
    for index in range(10):
        mask = bins == index
        if mask.any():
            error += float(mask.mean()*abs(pairs[mask, 0].mean()-pairs[mask, 1].mean()))
    return error


def history_subsets(corpus, selected, prefix, target):
    """Predeclared lexical evidence masks, independent of any model or predictions."""
    frequency = None; cutoff = None
    if hasattr(corpus, 'tokens') and 'train' in corpus.tokens:
        frequency = np.bincount(corpus.tokens['train'])
        positive = frequency[4:][frequency[4:] > 0]
        cutoff = float(np.quantile(positive, .75)) if len(positive) else 0.
    masks = {name: np.zeros((len(selected['a']), target), dtype=bool)
             for name in ('remote_repeat', 'no_remote_token_match')}
    if frequency is not None:
        masks['remote_rare_repeat'] = np.zeros_like(masks['remote_repeat'])
    for i, (a, b) in enumerate(zip(selected['a'], selected['b'])):
        history = set(a[:prefix].tolist()); unrelated = set(b[:prefix].tolist())
        for t in range(target):
            position = prefix+t+1; token = int(a[position])
            if token < 4:
                continue
            remote = a[:min(prefix, max(0, position-32+1))]
            recent = a[max(0, position-16):position]
            repeat = token in remote and token not in recent and token not in unrelated
            masks['remote_repeat'][i, t] = repeat
            masks['no_remote_token_match'][i, t] = token not in history and token not in unrelated
            if frequency is not None:
                masks['remote_rare_repeat'][i, t] = (repeat and token < len(frequency)
                                                    and 0 < frequency[token] <= cutoff)
    protocol = dict(minimum_repeat_distance=32, excluded_recent_tokens=16, excluded_special_ids=[0, 1, 2, 3],
                    rare_train_frequency_quantile=.75, rare_train_frequency_cutoff=cutoff,
                    train_frequency_sha256=digest(frequency.tolist()) if frequency is not None else None,
                    interpretation='Model-independent public-text lexical evidence proxies; not ground-truth necessary long dependencies',
                    unavailable=[] if frequency is not None else ['remote_rare_repeat: train tokens unavailable'])
    return masks, protocol


def subset_scores(losses, masks):
    result = {}
    for name, mask in masks.items():
        count = int(mask.sum()); early = mask[:, :16]; early_count = int(early.sum())
        summary = {condition: dict(nll=float(np.asarray(values)[mask].mean()) if count else None,
                                    first_16_nll=float(np.asarray(values)[:, :16][early].mean()) if early_count else None)
                   for condition, values in losses.items()}
        contrasts = {}
        if count and 'A_B_A' in summary:
            contrasts = dict(remote_relevant_retention_gain=summary['B_B_A']['nll']-summary['A_B_A']['nll'],
                             ordered_remote_history_gain=summary['shuffled_A_B_A']['nll']-summary['A_B_A']['nll'],
                             age_matched_interference_cost=summary['A_B_A']['nll']-summary['B_A_A']['nll'],
                             irrelevant_history_cost=summary['unrelated']['nll']-summary['reset']['nll'])
            if early_count:
                contrasts.update(first_16_remote_relevant_retention_gain=summary['B_B_A']['first_16_nll']-summary['A_B_A']['first_16_nll'],
                                 first_16_ordered_remote_history_gain=summary['shuffled_A_B_A']['first_16_nll']-summary['A_B_A']['first_16_nll'],
                                 first_16_age_matched_interference_cost=summary['A_B_A']['first_16_nll']-summary['B_A_A']['first_16_nll'])
        if count and 'credit_only_reset_after_prefix' in summary:
            for name in ('reset_fast_after_prefix', 'clear_slow_after_prefix'):
                if name in summary:
                    contrasts[name+'_vs_credit_only_cost'] = summary[name]['nll']-summary['credit_only_reset_after_prefix']['nll']
        result[name] = dict(status='scored' if count else 'no_eligible_tokens', tokens=count,
                            first_16_tokens=early_count, windows=int(mask.any(1).sum()),
                            conditions=summary, contrasts=contrasts)
    return result


@torch.no_grad()
def evaluate(model, corpus, split, cfg, device, ui, root=None, interventions=False, stop=None):
    was_training = model.training; model.eval()
    try:
        return _evaluate(model, corpus, split, cfg, device, ui, root, interventions, stop)
    finally:
        model.train(was_training); ui.close()


def _evaluate(model, corpus, split, cfg, device, ui, root, interventions, stop):
    selected = windows(corpus, split, cfg)
    conditions = ['carry']
    applicability = {}
    if interventions:
        conditions += ['reset', 'shuffled', 'unrelated', 'A_B_A', 'B_B_A', 'shuffled_A_B_A', 'B_A_A',
                       'unrelated_boundary', 'reset_each_token']
        for name, supported in [('no_local', hasattr(model, 'runtime_local_enabled')),
                                ('no_slow', hasattr(model, 'runtime_slow_enabled')),
                                ('reset_fast_after_prefix', callable(getattr(model, 'reset_fast', None))),
                                ('clear_slow_after_prefix', callable(getattr(model, 'clear_slow', None))),
                                ('credit_only_reset_after_prefix', callable(getattr(model, 'apply_feedback', None))),
                                ('head_only', getattr(model.cfg, 'variant', None) == 'bypass' and hasattr(model, 'head_only'))]:
            applicability[name] = dict(status='applied' if supported else 'not_applicable',
                                       reason='' if supported else 'This architecture has no corresponding runtime path')
            if supported:
                conditions.append(name)
    prefix, target, chunk = cfg['prefix_tokens'], cfg['target_tokens'], cfg['bptt_tokens']
    masks, subset_protocol = history_subsets(corpus, selected, prefix, target)
    target_sha = digest(dict(inputs=selected['a'][:, prefix:prefix+target].tolist(),
                             targets=selected['a'][:, prefix+1:prefix+target+1].tolist(),
                             documents=selected['docs'], starts=selected['starts']))
    evaluation_sha = digest(dict(version='v07-paired-target-step-3', windows_sha256=selected['sha256'],
                                 scored_tokens_sha256=target_sha, prefix_tokens=prefix, target_tokens=target,
                                 shuffle_seed=cfg['evaluation_seed'], boundary_tokens=[2, 1],
                                 history_controls=['A_B_A', 'B_B_A', 'shuffled_A_B_A', 'B_A_A',
                                                   'reset_fast_after_prefix', 'clear_slow_after_prefix'],
                                 credit_control='invalidate feedback metadata only after prefix',
                                 subset_protocol=subset_protocol))
    ui.stage('验证' if split == 'validation' else '测试与历史干预',
             f'{split}；{len(selected["a"])} 窗口 × {len(conditions)} 条件',
             total=len(selected['a'])*len(conditions), unit='窗口')
    rows, losses, divergence = [], {c: [] for c in conditions}, {c: [] for c in conditions}
    calibration = {c: [] for c in conditions}; diagnostics = {c: [] for c in conditions}
    for i, (a, b) in enumerate(zip(selected['a'], selected['b'])):
        reference = None
        for condition in conditions:
            state = None
            history = np.empty(0, dtype=np.int64)
            with runtime_intervention(model, condition):
                if condition not in ('reset', 'reset_each_token', 'head_only'):
                    history = a[:prefix].copy()
                    if condition in ('shuffled', 'shuffled_A_B_A'):
                        np.random.default_rng(cfg['evaluation_seed']+i).shuffle(history)
                    elif condition in ('unrelated', 'unrelated_boundary', 'B_B_A', 'B_A_A'):
                        history = b[:prefix].copy()
                    if condition in ('A_B_A', 'B_B_A', 'shuffled_A_B_A'):
                        history = np.concatenate((history, b[:prefix]))
                    elif condition == 'B_A_A':
                        history = np.concatenate((history, a[:prefix]))
                    elif condition == 'unrelated_boundary':
                        history = np.concatenate((history, [2, 1]))
                    _, state = feed(model, history, device, chunk, stop=stop)
                    if condition == 'reset_fast_after_prefix':
                        state = model.reset_fast(state)
                    elif condition == 'clear_slow_after_prefix':
                        state = model.clear_slow(state)
                    elif condition == 'credit_only_reset_after_prefix':
                        state = invalidate_feedback(state)
                token_diagnostics = []
                logits, _ = feed(model, a[prefix:prefix+target], device, chunk, state,
                                 diagnostics=token_diagnostics, reset_each_token=condition == 'reset_each_token',
                                 head_only=condition == 'head_only', stop=stop)
                token_diagnostics = materialize_diagnostics(token_diagnostics, device)
            truth = torch.as_tensor(a[prefix+1:prefix+target+1].copy(), dtype=torch.long, device=device)
            logp = logits.log_softmax(-1)
            nll = F.nll_loss(logp, truth, reduction='none')
            probability, prediction = logp.exp().max(-1)
            if reference is None:
                reference = logp
            kl = (reference.exp()*(reference-logp)).sum(-1).clamp_min(0)
            values = nll.cpu().tolist(); drifts = kl.cpu().tolist()
            correct = (prediction == truth).float().cpu().tolist(); confidence = probability.cpu().tolist()
            losses[condition].append(values); divergence[condition].append(drifts)
            calibration[condition].extend(zip(confidence, correct)); diagnostics[condition].extend(token_diagnostics)
            rows.append(dict(window=i, condition=condition, document=selected['docs'][i],
                             unrelated_document=selected['other'][i], start=selected['starts'][i],
                             input_tokens=a[prefix:prefix+target].tolist(), target_tokens=truth.cpu().tolist(),
                             history_tokens=history.tolist(), initial_history_length=len(history),
                             scored_tokens_sha256=target_sha, nll=values, kl_from_carry=drifts,
                             top1=prediction.cpu().tolist(), confidence=confidence,
                             per_token_diagnostics=token_diagnostics,
                             subset_masks={name: mask[i].tolist() for name, mask in masks.items()}))
            ui.update()
    summary = {}
    for condition in conditions:
        values = np.asarray(losses[condition]); drift = np.asarray(divergence[condition])
        pairs = np.asarray(calibration[condition]); ece = calibration_error(pairs)
        mean = float(values.mean())
        summary[condition] = dict(nll=mean, bpe_perplexity=math.exp(min(mean, 50)), tokens=int(values.size),
                                  top1_accuracy=float(pairs[:, 1].mean()), ece_10=ece,
                                  mean_kl_from_carry=float(drift.mean()), first_16_nll=float(values[:, :16].mean()),
                                  per_window_nll=values.mean(1).tolist(), per_position_nll=values.mean(0).tolist(),
                                  diagnostics=diagnostic_summary(diagnostics[condition]))
    result = dict(split=split, windows_sha256=selected['sha256'], scored_tokens_sha256=target_sha,
                  evaluation_sha256=evaluation_sha, conditions=summary, intervention_applicability=applicability,
                  metric='Windowed, train-only byte-BPE token NLL/perplexity; not official word-level perplexity',
                  evaluation='Frozen weights; causal teacher forcing; scored inputs/targets identical in every condition',
                  history_definition='Relevant prefix is contiguous public text; individual targets need not require history',
                  subset_protocol=subset_protocol, history_subsets=subset_scores(losses, masks))
    if interventions:
        result['contrasts'] = dict(relevant_history_gain=summary['reset']['nll']-summary['carry']['nll'],
                                  irrelevant_history_cost=summary['unrelated']['nll']-summary['reset']['nll'],
                                  boundary_irrelevant_cost=summary['unrelated_boundary']['nll']-summary['reset']['nll'],
                                  return_cost=summary['A_B_A']['nll']-summary['carry']['nll'],
                                  first_16_relevant_history_gain=summary['reset']['first_16_nll']-summary['carry']['first_16_nll'],
                                  first_16_irrelevant_history_cost=summary['unrelated']['first_16_nll']-summary['reset']['first_16_nll'],
                                  reset_each_token_cost=summary['reset_each_token']['nll']-summary['carry']['nll'],
                                  remote_relevant_retention_gain=summary['B_B_A']['nll']-summary['A_B_A']['nll'],
                                  ordered_remote_history_gain=summary['shuffled_A_B_A']['nll']-summary['A_B_A']['nll'],
                                  age_matched_interference_cost=summary['A_B_A']['nll']-summary['B_A_A']['nll'],
                                  first_16_remote_relevant_retention_gain=summary['B_B_A']['first_16_nll']-summary['A_B_A']['first_16_nll'],
                                  first_16_ordered_remote_history_gain=summary['shuffled_A_B_A']['first_16_nll']-summary['A_B_A']['first_16_nll'],
                                  first_16_age_matched_interference_cost=summary['A_B_A']['first_16_nll']-summary['B_A_A']['first_16_nll'])
        result['contrast_interpretation'] = dict(
            return_cost='Legacy A_B_A minus carry mixes extra state age and interference',
            remote_relevant_retention_gain='A_B_A vs B_B_A: same age and recent B; remote relevant A vs repeated B',
            ordered_remote_history_gain='A_B_A vs shuffled_A_B_A: same age, recent B and remote-token frequencies',
            age_matched_interference_cost='A_B_A minus B_A_A: same age; remote A with recent B vs recent A reference',
            reset_fast_after_prefix='After learning A, reset fast state and feedback, preserve slow contents and slow exposure support',
            clear_slow_after_prefix='After learning A, erase slow contents and invalidate feedback while preserving fast state',
            credit_only_reset_after_prefix='After learning A, invalidate only credit metadata; retain every content and dynamic state',
            content_probe_costs='Legacy probe_cost compares to carry, including credit invalidation; probe_vs_credit_only_cost subtracts the matched invalidation baseline',
            no_remote_token_match='No exact target-token match in either prefix; this does not guarantee semantic independence',
            seams='No hidden state or feedback reset at artificial seams; context-switch schedule labels are never model inputs')
        for name in ('no_local', 'no_slow', 'head_only', 'reset_fast_after_prefix', 'clear_slow_after_prefix',
                     'credit_only_reset_after_prefix'):
            if name in summary:
                result['contrasts'][name+'_cost'] = summary[name]['nll']-summary['carry']['nll']
        if 'credit_only_reset_after_prefix' in summary:
            for name in ('reset_fast_after_prefix', 'clear_slow_after_prefix'):
                if name in summary:
                    result['contrasts'][name+'_vs_credit_only_cost'] = summary[name]['nll']-summary['credit_only_reset_after_prefix']['nll']
                    result['contrasts']['first_16_'+name+'_vs_credit_only_cost'] = summary[name]['first_16_nll']-summary['credit_only_reset_after_prefix']['first_16_nll']
    if root is not None:
        root = Path(root); write(root/f'{split}_predictions.json', rows); write(root/f'{split}_metrics.json', result)
    return result
