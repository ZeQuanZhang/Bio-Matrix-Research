"""Fixed public-text likelihood windows and paired history interventions."""
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .common import detach, digest, write


def windows(corpus, split, cfg):
    rng = np.random.default_rng(cfg['evaluation_seed']+(0 if split == 'validation' else 1))
    length = cfg['prefix_tokens']+cfg['target_tokens']
    count = cfg['validation_windows'] if split == 'validation' else cfg['test_windows']
    a, docs, starts = corpus.sample(split, rng, count, length)
    b, other, _ = corpus.sample(split, rng, count, length)
    for i in range(count):
        while other[i] == docs[i]:
            row, ids, _ = corpus.sample(split, rng, 1, length)
            b[i], other[i] = row[0], ids[0]
    return dict(a=a, b=b, docs=docs, starts=starts, other=other,
                sha256=digest(dict(a=a.tolist(), b=b.tolist())))


@torch.no_grad()
def feed(model, tokens, device, chunk, state=None):
    outputs = []
    for start in range(0, len(tokens), chunk):
        x = torch.as_tensor(np.asarray(tokens[start:start+chunk]).copy(), dtype=torch.long, device=device)[None]
        logits, state, _ = model(x, state)
        outputs.append(logits[0])
    return (torch.cat(outputs) if outputs else None), state


@torch.no_grad()
def evaluate(model, corpus, split, cfg, device, ui, root=None, interventions=False, stop=None):
    was_training = model.training; model.eval()
    selected = windows(corpus, split, cfg)
    conditions = ('carry', 'reset', 'shuffled', 'unrelated', 'A_B_A', 'unrelated_boundary') if interventions else ('carry',)
    prefix, target, chunk = cfg['prefix_tokens'], cfg['target_tokens'], cfg['bptt_tokens']
    ui.stage('验证' if split == 'validation' else '测试与历史干预',
             f'{split}；{len(selected["a"])} 窗口 × {len(conditions)} 条件',
             total=len(selected['a'])*len(conditions), unit='窗口')
    rows, losses, divergence = [], {c: [] for c in conditions}, {c: [] for c in conditions}
    calibration = {c: [] for c in conditions}
    for i, (a, b) in enumerate(zip(selected['a'], selected['b'])):
        reference = None
        for condition in conditions:
            if stop and stop['requested']:
                model.train(was_training); ui.close(); raise KeyboardInterrupt
            state = None
            if condition != 'reset':
                history = a[:prefix].copy()
                if condition == 'shuffled':
                    np.random.default_rng(cfg['evaluation_seed']+i).shuffle(history)
                elif condition in ('unrelated', 'unrelated_boundary'):
                    history = b[:prefix]
                _, state = feed(model, history, device, chunk)
                if condition == 'A_B_A':
                    _, state = feed(model, b[:prefix], device, chunk, state)
                elif condition == 'unrelated_boundary':
                    _, state = feed(model, np.array([2, 1]), device, chunk, state)
            logits, state = feed(model, a[prefix:prefix+target], device, chunk, state)
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
            calibration[condition].extend(zip(confidence, correct))
            rows.append(dict(window=i, condition=condition, document=selected['docs'][i],
                             unrelated_document=selected['other'][i], start=selected['starts'][i],
                             target_tokens=truth.cpu().tolist(), nll=values, kl_from_carry=drifts,
                             top1=prediction.cpu().tolist(), confidence=confidence))
            ui.update()
    summary = {}
    for condition in conditions:
        values = np.asarray(losses[condition]); drift = np.asarray(divergence[condition])
        pairs = np.asarray(calibration[condition]); ece = 0.
        for lo in np.arange(0, 1, .1):
            mask = (pairs[:, 0] >= lo) & (pairs[:, 0] < lo+.1+1e-10)
            if mask.any():
                ece += float(mask.mean()*abs(pairs[mask, 0].mean()-pairs[mask, 1].mean()))
        mean = float(values.mean())
        summary[condition] = dict(nll=mean, bpe_perplexity=math.exp(min(mean, 50)), tokens=int(values.size),
                                  top1_accuracy=float(pairs[:, 1].mean()), ece_10=ece,
                                  mean_kl_from_carry=float(drift.mean()),
                                  first_16_nll=float(values[:, :16].mean()),
                                  per_window_nll=values.mean(1).tolist(), per_position_nll=values.mean(0).tolist())
    result = dict(split=split, windows_sha256=selected['sha256'], conditions=summary,
                  metric='Windowed, train-only byte-BPE token NLL/perplexity; not official word-level perplexity',
                  evaluation='Teacher-forced newly visible tokens; no optimizer updates; runtime local plasticity remains active')
    if interventions:
        carry = summary['carry']['nll']
        result['contrasts'] = dict(relevant_history_gain=summary['reset']['nll']-carry,
                                    irrelevant_history_cost=summary['unrelated']['nll']-summary['reset']['nll'],
                                    boundary_irrelevant_cost=summary['unrelated_boundary']['nll']-summary['reset']['nll'],
                                    return_cost=summary['A_B_A']['nll']-carry)
    if root is not None:
        root = Path(root); write(root/f'{split}_predictions.json', rows); write(root/f'{split}_metrics.json', result)
    model.train(was_training); ui.close()
    return result
