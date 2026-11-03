"""Shared objectives, coverage validation and batched answer generation."""
from collections import defaultdict
import resource
import time
import numpy as np
import torch
from torch.nn import functional as F

from scale.engine import qa_score
from v07.common import sync, digest
from .data import lm_windows


def update(model, opt, values, mask, device, auxiliary_weight=0.):
    x = torch.as_tensor(values[:, :-1].copy(), device=device)
    y = torch.as_tensor(values[:, 1:].copy(), device=device)
    selected = torch.as_tensor(mask.copy(), device=device)
    if selected.shape != y.shape or not bool(selected.any()):
        raise ValueError('Empty or misaligned objective')
    model.train(); opt.zero_grad(set_to_none=True); sync(device); start = time.perf_counter()
    logits, _, diag = model(x)
    ce = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction='none').reshape_as(y)[selected].mean()
    aux = diag.get('native_auxiliary', ce.new_zeros(()))
    loss = ce+auxiliary_weight*aux
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite objective')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    opt.step(); sync(device)
    seconds = time.perf_counter()-start
    return dict(ce=float(ce.detach().cpu()), objective=float(loss.detach().cpu()),
                native_auxiliary=float(aux.detach().cpu()), gradient_norm=float(norm.detach().cpu()),
                seconds=seconds, targets=int(mask.sum()), inputs=int(x.numel()), input_tokens_per_second=x.numel()/seconds,
                metal_allocated_bytes=torch.mps.current_allocated_memory() if device.type == 'mps' else None,
                process_peak_rss_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


@torch.inference_mode()
def score_lm(model, corpus, split, device, batch=16):
    model.eval(); windows = list(lm_windows(corpus, split)); records = []
    for begin in range(0, len(windows), batch):
        group = windows[begin:begin+batch]
        values = np.stack([r[0] for r in group]); masks = np.stack([r[1] for r in group])
        x = torch.as_tensor(values, device=device)
        logits, _, _ = model(x[:, :-1])
        loss = F.cross_entropy(logits.float().flatten(0, 1), x[:, 1:].flatten(), reduction='none').reshape_as(x[:, 1:]).cpu().numpy()
        for row, mask, losses in zip(group, masks, loss):
            records.append(dict(document=row[2], nll_sum=float(losses[mask].astype(np.float64).sum()), targets=int(mask.sum())))
    total = sum(r['targets'] for r in records)
    nll = sum(r['nll_sum'] for r in records)/total
    if total != len(corpus.tokens[split])-len(corpus.offsets[split])+1:
        raise ValueError('Coverage validation missed or duplicated a target')
    grouped = defaultdict(lambda: [0., 0])
    for r in records:
        grouped[r['document']][0] += r['nll_sum']; grouped[r['document']][1] += r['targets']
    return dict(nll=nll, ppl=float(np.exp(nll)), targets=total, windows=len(windows),
                documents=[dict(document=k, nll_sum=v[0], targets=v[1]) for k, v in grouped.items()])


def intervention(qa, indices, values, rows, mode):
    if mode == 'warm':
        return values, rows
    changed = values.copy(); metadata = [dict(r) for r in rows]
    for i, (index, row) in enumerate(zip(indices, rows)):
        rng = np.random.default_rng(int(digest([row['question_id'], 'unrelated-donor'])[:16], 16))
        donor_index = int(rng.integers(len(qa.rows)))
        while qa.rows[donor_index]['article_id'] == row['article_id']:
            if len({r['article_id'] for r in qa.rows}) < 2:
                raise ValueError('No unrelated article for control')
            donor_index = int(rng.integers(len(qa.rows)))
        donor = qa.values[donor_index, 1:1+qa.rows[donor_index]['context_tokens']]
        if mode == 'independent_prior':
            if row['prefix_tokens'] > 384:
                raise ValueError('Independent-prior probe exceeds fixed length budget')
            extra = np.resize(donor, 64)
            v = np.concatenate((values[i, :1], extra, values[i, 1:row['supervised_end']+1]))
            if changed.shape[1] < len(v):
                changed = np.pad(changed, ((0, 0), (0, len(v)-changed.shape[1])))
            changed[i] = 0; changed[i, :len(v)] = v
            for key in ('prefix_tokens', 'answer_start', 'answer_end', 'supervised_end', 'evidence_end'):
                metadata[i][key] += 64
        else:
            length = row['context_tokens'] if mode == 'unrelated_context' else min(row['context_tokens'], max(0, row['prefix_tokens']-256)-1)
            if mode not in ('unrelated_context', 'remote_context'):
                raise ValueError(mode)
            if length > 0:
                changed[i, 1:1+length] = np.resize(donor, length)
    return changed, metadata


@torch.inference_mode()
def score_qa(model, qa, device, indices=None, batch=16, mode='warm'):
    model.eval(); indices = list(range(len(qa.rows))) if indices is None else list(indices)
    records = []
    for begin in range(0, len(indices), batch):
        selected = indices[begin:begin+batch]
        values, _, rows = qa.batch(selected)
        values, rows = intervention(qa, selected, values, rows, mode)
        x = torch.as_tensor(values, device=device)
        logits, _, _ = model(x[:, :-1])
        losses = F.cross_entropy(logits.float().flatten(0, 1), x[:, 1:].flatten(), reduction='none').reshape_as(x[:, 1:]).cpu().numpy()
        hits = (logits.argmax(-1) == x[:, 1:]).cpu().numpy()
        for r, loss, hit in zip(rows, losses, hits):
            a, b, e = r['answer_start'], r['answer_end'], r['supervised_end']
            records.append(dict(question_id=r['question_id'], article_id=r['article_id'], distance=r['answer_distance'],
                nll_sum=float(loss[a:e].astype(np.float64).sum()), targets=e-a,
                answer_nll_sum=float(loss[a:b].astype(np.float64).sum()), answer_targets=b-a,
                first_nll=float(loss[a]), first_correct=int(hit[a]), eos_nll=float(loss[b])))
    return dict(nll=sum(r['nll_sum'] for r in records)/sum(r['targets'] for r in records),
        answer_nll=sum(r['answer_nll_sum'] for r in records)/sum(r['answer_targets'] for r in records),
        first_nll=float(np.mean([r['first_nll'] for r in records])),
        first_accuracy=float(np.mean([r['first_correct'] for r in records])), questions=len(records), records=records)


@torch.inference_mode()
def generate(model, qa, indices, tokenizer, device, batch=16, mode='warm', limit=64):
    model.eval(); records = []
    for begin in range(0, len(indices), batch):
        selected = list(indices[begin:begin+batch]); original, _, rows = qa.batch(selected)
        original, rows = intervention(qa, selected, original, rows, mode)
        lengths = np.asarray([r['prefix_tokens'] for r in rows])
        values = np.zeros((len(rows), 513), dtype=np.int64)
        for i, n in enumerate(lengths):
            values[i, :n] = original[i, :n]
        output = [[] for _ in rows]; active = np.ones(len(rows), dtype=bool)
        for step in range(limit):
            end = int(lengths.max())+step
            if end > 512:
                raise ValueError('Generation exceeded declared context budget')
            x = torch.as_tensor(values[:, :end].copy(), device=device)
            logits, _, _ = model(x)
            pos = torch.as_tensor(lengths+step-1, device=device)
            tokens = logits[torch.arange(len(rows), device=device), pos].argmax(-1).cpu().numpy()
            for i in np.flatnonzero(active):
                token = int(tokens[i]); output[i].append(token); values[i, lengths[i]+step] = token
                if token == 2:
                    active[i] = False
            if not active.any():
                break
        for r, ids in zip(rows, output):
            prediction = tokenizer.decode(ids, skip_special_tokens=True).strip()
            em, f1 = qa_score(prediction, r['reference_answers'])
            records.append(dict(question_id=r['question_id'], article_id=r['article_id'], distance=r['answer_distance'],
                prediction=prediction, reference_answers=r['reference_answers'], generated_ids=ids,
                em=em, f1=f1, hit_cap=not ids or ids[-1] != 2))
    return dict(em=float(np.mean([r['em'] for r in records])), f1=float(np.mean([r['f1'] for r in records])),
                cap_rate=float(np.mean([r['hit_cap'] for r in records])), questions=len(records), records=records)
