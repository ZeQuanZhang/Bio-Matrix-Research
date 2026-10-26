"""Frozen-weight, document-paired validation evaluation with explicit anchors.

Only validation arrays are opened. Prefixes and scoring inputs use complete
C16 blocks; no diagnostic option changes the model's execution schedule.
Observed tokens may update runtime memory, but learned parameters never update.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from v07.common import PROJECT, device_for, digest, file_hash, write
from v07.model import invalidate_feedback


CONDITIONS = ('A0R', 'U0R', 'shuffle_block_A0R', 'A0BR', 'U0BR',
              'credit_only', 'clear_slow', 'base_only')
CONTRASTS = dict(history_gain=('U0R', 'A0R'), ordered_history_gain=('shuffle_block_A0R', 'A0R'),
                 ABA_retention_gain=('U0BR', 'A0BR'),
                 insertion_and_extra_age_cost=('A0BR', 'A0R'),
                 old_credit_reset_cost=('credit_only', 'A0R'),
                 slow_content_contribution=('clear_slow', 'credit_only'),
                 bio_read_contribution=('base_only', 'credit_only'))


def validation_corpus(root):
    """Unlike Corpus, this loader never opens the test split, even its header."""
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    verified = {}
    for name in ('validation_tokens.npy', 'validation_offsets.npy'):
        actual = file_hash(root/name)
        expected = manifest.get('prepared_hashes', {}).get(name)
        if expected is not None and actual != expected:
            raise ValueError(f'Validation prepared hash mismatch: {name}')
        verified[name] = actual
    return SimpleNamespace(root=root, manifest=manifest, verified_validation_hashes=verified,
             tokens={'validation': np.load(root/'validation_tokens.npy', mmap_mode='r')},
             offsets={'validation': np.load(root/'validation_offsets.npy', mmap_mode='r')})


def evaluation_sources():
    paths = [*sorted((PROJECT/'v09').glob('*.py')), PROJECT/'v08/model.py',
             PROJECT/'v08/recurrence.py', PROJECT/'v07/model.py', PROJECT/'v07/graph.py',
             PROJECT/'v04/model.py', PROJECT/'v05/model.py']
    return {str(path.relative_to(PROJECT)): file_hash(path) for path in paths if path.is_file()}


def _span(tokens, offsets, rng, length, excluded):
    eligible = [(doc, int(offsets[doc]), int(offsets[doc+1]-offsets[doc])-length+1)
                for doc in range(len(offsets)-1)
                if doc not in excluded and offsets[doc+1]-offsets[doc] >= length]
    if not eligible:
        return None
    counts = np.asarray([row[2] for row in eligible], dtype=np.int64)
    draw = int(rng.integers(int(counts.sum())))
    cumulative = counts.cumsum(); index = int(np.searchsorted(cumulative, draw, side='right'))
    doc, start, _ = eligible[index]
    start += draw-(int(cumulative[index-1]) if index else 0)
    return dict(document=doc, start=start, tokens=np.asarray(tokens[start:start+length]).tolist())


def select_windows(corpus, count=64, data_seed=7191, history=1024, recent=256,
                   target=256, intrusion=64, block=16):
    lengths = (history, recent, target, intrusion)
    if count < 1 or block < 1 or any(not isinstance(n, int) or n < 1 or n % block for n in lengths):
        raise ValueError('Positive H/R/T/B lengths must be complete execution blocks')
    tokens, offsets = corpus.tokens['validation'], corpus.offsets['validation']
    if len(offsets) < 2 or int(offsets[0]) != 0 or int(offsets[-1]) != len(tokens) or np.any(np.diff(offsets) < 0):
        raise ValueError('Invalid validation document offsets')
    total = history+recent+target+1
    candidates = [(doc, int(offsets[doc])+offset) for doc in range(len(offsets)-1)
                  for offset in range(0, max(0, int(offsets[doc+1]-offsets[doc])-total+1), target)]
    rng = np.random.default_rng(data_seed)
    order = rng.permutation(len(candidates)); selected = []; unavailable = 0
    for index in order:
        if len(selected) == count:
            break
        doc, start = candidates[int(index)]
        u = _span(tokens, offsets, rng, history, {doc})
        if u is None:
            unavailable += 1; continue
        insertion = _span(tokens, offsets, rng, intrusion, {doc, u['document']})
        if insertion is None:
            insertion = _span(tokens, offsets, rng, intrusion, {doc})
        if insertion is None:
            unavailable += 1; continue
        a = np.asarray(tokens[start:start+total]).tolist()
        block_order = rng.permutation(history//block).tolist()
        shuffled = np.asarray(a[:history]).reshape(-1, block)[block_order].reshape(-1).tolist()
        anchor = start+history+recent
        selected.append(dict(window=len(selected), document=doc, start=start,
                    A0=a[:history], U0=u, R=a[history:history+recent], B=insertion,
                    segment=a[history+recent:], shuffled_A0=shuffled, shuffle_block_order=block_order,
                    anchor=dict(position=anchor, token=int(tokens[anchor]), scored=False),
                    target_start=anchor+1, target_end=anchor+1+target))
    positions = {(row['document'], p) for row in selected for p in range(row['target_start'], row['target_end'])}
    coverage = dict(status='sufficient' if len(selected) == count else 'insufficient_windows',
                    requested_windows=count, candidate_nonoverlap_windows=len(candidates),
                    rejected_missing_other_document=unavailable, actual_windows=len(selected),
                    actual_documents=len({row['document'] for row in selected}),
                    scored_positions=len(selected)*target, unique_scored_positions=len(positions),
                    duplicate_scored_positions=len(selected)*target-len(positions),
                    eligible_candidate_target_positions=len(candidates)*target,
                    total_validation_tokens=len(tokens), fraction_of_validation_tokens=len(positions)/max(1, len(tokens)))
    plan = dict(version='v09-frozen-paired-anchor-1', split='validation', data_seed=data_seed,
                lengths=dict(history=history, recent=recent, target=target, intrusion=intrusion, block=block),
                anchor_protocol='Prefix then T known inputs segment[:T] predict segment[1:T+1]; anchor itself is unscored.',
                natural_interference_note='U0/B are different-document natural interference; statistical independence is not assumed.',
                coverage=coverage, windows=selected)
    plan['sample_sha256'] = digest(plan)
    return plan


def clone_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {name: clone_state(item) for name, item in value.items()}
    if isinstance(value, list):
        return [clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_state(item) for item in value)
    return value


@torch.no_grad()
def feed_aligned(model, values, device, state=None, block=16, collect=False):
    if len(values) % block:
        raise ValueError('Evaluation feed must contain complete execution blocks')
    limit = int(getattr(model.cfg, 'context_limit', 256))
    chunk = max(block, min(64, limit)//block*block)
    if limit < block:
        raise ValueError('Saved context limit is shorter than an evaluation block')
    outputs = []
    for start in range(0, len(values), chunk):
        x = torch.tensor(np.asarray(values[start:start+chunk]).copy(), dtype=torch.long, device=device)[None]
        logits, state, _ = model(x, state)
        if collect:
            outputs.append(logits[0])
    return (torch.cat(outputs) if outputs else None), state


@contextmanager
def read_intervention(model, base_only=False):
    previous = getattr(model, 'runtime_bio_read_enabled', None)
    try:
        if base_only:
            if previous is None or not getattr(model.cfg, 'content_base', False):
                raise ValueError('This checkpoint has no content-base read ablation')
            model.runtime_bio_read_enabled = False
        yield
    finally:
        if previous is not None:
            model.runtime_bio_read_enabled = previous


@torch.no_grad()
def score_segment(model, segment, state, device, block=16):
    output, _ = feed_aligned(model, segment[:-1], device, clone_state(state), block, collect=True)
    target = torch.tensor(segment[1:], dtype=torch.long, device=device)
    losses = F.cross_entropy(output, target, reduction='none').float().cpu().numpy()
    if not np.isfinite(losses).all():
        raise FloatingPointError('Nonfinite validation token likelihood')
    return losses


def loss_metrics(losses):
    values = np.asarray(losses, dtype=np.float64)
    result = dict(nll=float(values.mean()), scored_tokens=len(values))
    for length in (16, 64, 256):
        result[f'first_{length}_nll'] = float(values[:length].mean())
        result[f'first_{length}_tokens'] = min(length, len(values))
    return result


def document_summary(rows, repetitions=2000, seed=7282):
    if repetitions < 1:
        raise ValueError('bootstrap repetitions must be positive')
    grouped = {}
    for row in rows:
        grouped.setdefault(row['document'], {}).setdefault(row['condition'], []).append(row['token_nll'])

    def pooled(chunks):
        result = loss_metrics([value for chunk in chunks for value in chunk])
        for length in (16, 64, 256):
            values = [value for chunk in chunks for value in chunk[:length]]
            result[f'first_{length}_nll'] = float(np.mean(values))
            result[f'first_{length}_tokens'] = len(values)
        return result

    docs = []
    for doc, conditions in sorted(grouped.items()):
        docs.append(dict(document=doc, conditions={name:pooled(values) for name, values in conditions.items()}))
    available = sorted({row['condition'] for row in rows})
    means = {condition:pooled([row['token_nll'] for row in rows if row['condition'] == condition])
             for condition in available}
    contrasts = {}
    rng = np.random.default_rng(seed)
    identifiers = list(sorted(grouped)); draws = rng.integers(0, len(identifiers), size=(repetitions, len(identifiers))) if identifiers else None
    for name, (left, right) in CONTRASTS.items():
        paired = [doc for doc in identifiers if left in grouped[doc] and right in grouped[doc]]
        if not paired:
            contrasts[name] = dict(status='not_applicable'); continue
        if len(paired) != len(identifiers):
            contrasts[name] = dict(status='incomplete_paired_documents'); continue
        sums = np.asarray([sum(map(sum, grouped[doc][left]))-sum(map(sum, grouped[doc][right])) for doc in identifiers])
        counts = np.asarray([sum(map(len, grouped[doc][right])) for doc in identifiers])
        point = float(sums.sum()/counts.sum())
        interval = None
        if len(identifiers) > 1:
            bootstrap = sums[draws].sum(1)/counts[draws].sum(1)
            interval = np.quantile(bootstrap, [.025, .975]).tolist()
        contrasts[name] = dict(status='computed' if interval is not None else 'insufficient_documents_for_ci',
                   nll_difference=point, ci95=interval, paired_documents=len(identifiers),
                   bootstrap_repetitions=repetitions, interpretation='Document-resampled evaluation-sample CI; not training-seed uncertainty.')
    for name, (left, right) in CONTRASTS.items():
        if left in means and right in means:
            for length in (16, 64, 256):
                values = []
                for window in sorted({row['window'] for row in rows}):
                    pair = {row['condition']:row['token_nll'][:length] for row in rows if row['window'] == window}
                    if left in pair and right in pair:
                        values.extend(np.asarray(pair[left])-np.asarray(pair[right]))
                contrasts[name][f'first_{length}_nll_difference'] = float(np.mean(values)) if values else None
    return dict(conditions=means, contrasts=contrasts, documents=docs,
                contrast_notes='Positive history/ABA/slow/Bio contributions favor the retained component; insertion cost also includes B extra aging.')


def reconstruct(payload, corpus):
    from .benchmark import build, initialize_model
    if 'protocol' not in payload or 'training' not in payload['protocol']:
        raise ValueError('Evaluation requires a training last.pt or best.pt with protocol.training')
    protocol = payload['protocol']; training = protocol['training']; saved = protocol.get('model', {})
    context = int(training.get('context_limit', saved.get('context_limit', 256)))
    model, cfg = build(dict(condition=training['condition'], nodes=training.get('nodes', 128)),
            corpus.manifest['vocab_size'], training.get('group_size', 16), training.get('route_mode', 'summary'),
            context_limit=context)
    initialize_model(model, training['condition'], training.get('seed', 11))
    model.load_state_dict(payload['model'], strict=True)
    for name, value in payload.get('runtime_attributes', {}).items():
        if name in ('gate_blend', 'active_fraction') and hasattr(model, name):
            setattr(model, name, value)
    model.eval()
    return model, cfg


def run_evaluation(checkpoint, output, device='mps', windows=64, data_seed=7191, data_root=None,
                   history_tokens=1024, recent_tokens=256, target_tokens=256, intrusion_tokens=64,
                   bootstrap_repetitions=2000, corpus=None):
    device = device_for(str(device))
    if device.type not in ('cpu', 'mps'):
        raise ValueError('Evaluation is restricted to this Mac: cpu or mps')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    live = dict(status='running', phase='loading', completed_windows=0, requested_windows=windows,
                last_nll=None, progress=0., completed=0, total=0, nll=None)
    write(root/'live.json', live)
    rows = []
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        checkpoint = Path(checkpoint).resolve()
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        protocol = payload.get('protocol', {})
        corpus = corpus or validation_corpus(data_root or protocol['data_root'])
        expected = protocol.get('data_manifest_sha256')
        if expected is not None and digest(corpus.manifest) != expected:
            raise ValueError('Validation corpus manifest differs from the training checkpoint')
        model, cfg = reconstruct(payload, corpus); model.to(device)
        plan = select_windows(corpus, windows, data_seed, history_tokens, recent_tokens, target_tokens, intrusion_tokens)
        write(root/'sample_plan.json', plan)
        manifest = dict(status='running', checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint),
               checkpoint_format=payload.get('format'), checkpoint_counters=payload.get('counters'),
               checkpoint_source_hashes=protocol.get('source_hashes', {}), evaluation_source_hashes=evaluation_sources(),
               checkpoint_training=protocol.get('training'), rebuilt_model_config=asdict(cfg),
               runtime_attributes={name:getattr(model, name) for name in ('gate_blend', 'active_fraction') if hasattr(model, name)},
               device=str(device), split='validation', sample_sha256=plan['sample_sha256'], coverage=plan['coverage'],
               data_manifest_sha256=digest(corpus.manifest), data_seed=data_seed, model_seed=protocol['training'].get('seed'),
               verified_validation_hashes=getattr(corpus, 'verified_validation_hashes', {}),
               anchor_protocol=plan['anchor_protocol'], first_target_pre_intervention=False,
               frozen_parameters=True, runtime_memory_updates=True, training_resumed=False,
               uncertainty='Bootstrap covers validation documents only; no training-seed CI.',
               checkpoint_policy='Scores exactly the supplied last/best file; no automatic best selection.')
        write(root/'config.json', manifest)
        applicability = {name:True for name in CONDITIONS}
        initial = getattr(model, 'initial_state', None)
        applicability['credit_only'] = 'feedback_valid' in (initial(1) if callable(initial) else {})
        applicability['clear_slow'] = hasattr(model, 'clear_slow')
        applicability['base_only'] = hasattr(model, 'runtime_bio_read_enabled') and getattr(cfg, 'content_base', False)
        cases = [name for name in CONDITIONS if applicability[name]]
        live['total_condition_rows'] = len(cases)*len(plan['windows'])
        live['total'] = live['total_condition_rows']
        for window in plan['windows']:
            live.update(window=window['window'], actual_windows=len(plan['windows']))
            prefixes = dict(A0R=window['A0']+window['R'], U0R=window['U0']['tokens']+window['R'],
                   shuffle_block_A0R=window['shuffled_A0']+window['R'],
                   A0BR=window['A0']+window['B']['tokens']+window['R'],
                   U0BR=window['U0']['tokens']+window['B']['tokens']+window['R'])
            same_age = None
            for case_index, condition in enumerate(cases):
                live.update(phase='prefix' if condition in prefixes else 'same_age_ablation', condition=condition,
                            condition_index=case_index, condition_count=len(cases))
                write(root/'live.json', live)
                if condition in prefixes:
                    _, state = feed_aligned(model, prefixes[condition], device)
                    if condition == 'A0R':
                        same_age = clone_state(state)
                else:
                    state = invalidate_feedback(clone_state(same_age))
                    if condition == 'clear_slow':
                        state = model.clear_slow(state)
                live['phase'] = 'scoring'; write(root/'live.json', live)
                with read_intervention(model, condition == 'base_only'):
                    losses = score_segment(model, window['segment'], state, device)
                row = dict(window=window['window'], document=window['document'], condition=condition,
                       anchor=window['anchor'], target_start=window['target_start'], target_end=window['target_end'],
                       first_target_pre_intervention=False, metrics=loss_metrics(losses), token_nll=losses.tolist())
                with (root/'raw.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                rows.append(row)
                live.update(last_nll=row['metrics']['nll'], phase='condition_completed',
                            completed_condition_rows=len(rows), progress=len(rows)/max(1, live['total_condition_rows']),
                            completed=len(rows), nll=row['metrics']['nll'])
                write(root/'live.json', live)
            live.update(completed_windows=window['window']+1, phase='window_completed')
            write(root/'live.json', live)
        aggregate = document_summary(rows, bootstrap_repetitions, data_seed+91)
        with (root/'documents.jsonl').open('w', encoding='utf-8') as stream:
            for row in aggregate.pop('documents'):
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        status = 'completed' if plan['windows'] else 'insufficient_data'
        summary = dict(status=status, coverage=plan['coverage'], sample_sha256=plan['sample_sha256'],
                       applicability={name:'scored' if enabled else 'not_applicable' for name,enabled in applicability.items()},
                       **aggregate)
        write(root/'summary.json', summary)
        manifest['status'] = status; write(root/'config.json', manifest)
        live.update(status=status, phase='done'); write(root/'live.json', live)
        return summary
    except BaseException as error:
        status = 'aborted' if isinstance(error, KeyboardInterrupt) else 'failed'
        live.update(status=status, phase=status, error=repr(error)); write(root/'live.json', live)
        write(root/'status.json', dict(status=status, error=repr(error), scored_condition_rows=len(rows)))
        (root/'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        torch.set_num_threads(previous_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    parser.add_argument('--windows', type=int, default=64)
    parser.add_argument('--data-seed', type=int, default=7191)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--history-tokens', type=int, default=1024)
    parser.add_argument('--recent-tokens', type=int, default=256)
    parser.add_argument('--target-tokens', type=int, default=256)
    parser.add_argument('--intrusion-tokens', type=int, default=64)
    args = parser.parse_args()
    result = run_evaluation(**vars(args))
    print(f'v0.9 evaluation {result["status"]}: {args.output}')


if __name__ == '__main__':
    main()
