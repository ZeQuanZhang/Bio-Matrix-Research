"""Short matched-data diagnostics; these are neither benchmarks nor validation runs.

Every condition uses the same three contiguous optimizer updates. Interventions
use cloned, invalidated runtime state and fixed trained weights. Coefficient
hooks observe only the final internal round because the core does not expose
intermediate rounds. Full matrices remain intact during every intervention.
"""
from dataclasses import asdict
from pathlib import Path
import gc
import json
import math
import time
import traceback
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from v07.common import PROJECT, detach, digest, file_hash, save, sync, write
from v07.model import invalidate_feedback, parameter_count


def clone_state(state):
    return {name: value.detach().clone() for name, value in state.items()}


def gradient_summary(model):
    rows = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        # CPU FP64 prevents the norm reduction itself overflowing on MPS/FP32.
        gradient = parameter.grad.detach().cpu()
        rows.append(dict(name=name, finite=bool(gradient.isfinite().all()),
                         norm64=float(gradient.double().norm()), max_abs=float(gradient.abs().max())))
    finite = all(row['finite'] for row in rows)
    total = math.sqrt(sum(row['norm64']**2 for row in rows)) if finite else None
    selected = {row['name']: row['norm64'] for row in rows if row['name'] in
                ('content_projection.weight', 'address_projection.weight', 'route_key.weight',
                 'route_query.weight', 'raw_slow_rate', 'raw_content_gain')}
    rows.sort(key=lambda row: row['norm64'] if math.isfinite(row['norm64']) else float('inf'), reverse=True)
    # Failed elementwise gradients are represented explicitly in strict JSON.
    for row in rows:
        for name in ('norm64', 'max_abs'):
            if not math.isfinite(row[name]):
                row[name] = None
    selected = {name: value if math.isfinite(value) else None for name, value in selected.items()}
    return dict(all_elements_finite=finite, global_norm64=total, selected_norms64=selected,
                largest_parameters=rows[:8])


def _capture(model, tokens, state):
    captured = []
    original = model.block_coefficients

    def coefficients(block, compact):
        result = original(block, compact)
        buffer = getattr(model, 'graph_buffer', lambda name: getattr(model, name))
        ids = getattr(model, '_active_indices', None)
        ids = torch.arange(compact['m'].shape[1], device=block.device) if ids is None else ids
        row = dict(batch=len(block), time=block.shape[1], ids=ids.detach().cpu(),
                   shell=buffer('shell').detach().cpu(), ports=buffer('read_ports').detach().cpu(),
                   tokens=block.detach().cpu())
        for name in ('transport', 'content_base_transport', 'content_base_mass', 'arrived'):
            if name in result:
                row[name] = result[name].detach().cpu()
        captured.append(row)
        return result

    with patch.object(model, 'block_coefficients', side_effect=coefficients):
        logits, end, diag = model(tokens, clone_state(state))
    routes = [[dict(groups=list(row['groups']), active_ids=row['active_ids'].detach().cpu().tolist())
               for row in block] for block in getattr(model, 'last_routes', [])]
    return logits, end, {name: float(value.detach().cpu()) for name, value in diag.items()}, captured, routes


def _comparison(reference, other, targets):
    difference = (reference-other).float()
    a = F.cross_entropy(reference.flatten(0, 1), targets.flatten())
    b = F.cross_entropy(other.flatten(0, 1), targets.flatten())
    log_a, log_b = reference.float().log_softmax(-1), other.float().log_softmax(-1)
    divergence = (log_a.exp()*(log_a-log_b)).sum(-1).mean().clamp_min(0)
    return dict(logit_max_abs=float(difference.abs().max().cpu()),
                logit_rms=float(difference.square().mean().sqrt().cpu()),
                kl_reference_to_other=float(divergence.cpu()),
                reference_nll=float(a.cpu()), other_nll=float(b.cpu()), nll_difference=float((b-a).cpu()))


def _shell_trace(before, after, position):
    rows = []
    if len(before) != len(after):
        return dict(status='unmatched_routes', before_calls=len(before), after_calls=len(after))
    for number, (left, right) in enumerate(zip(before, after)):
        if not torch.equal(left['ids'], right['ids']):
            return dict(status='unmatched_routes', call=number)
        b, t, n = left['batch'], left['time'], len(left['ids'])
        at = min(position, t-1)
        record = dict(call=number, active_ids=left['ids'].tolist(), time_position=at, shells=[])
        for level in sorted(set(left['shell'].tolist())):
            mask = left['shell'] == level
            item = dict(shell=level, active_nodes=int(mask.sum()))
            for field in ('transport', 'content_base_transport'):
                if field not in left:
                    continue
                delta = (right[field]-left[field]).reshape(b, t, n, *left[field].shape[-2:])[:, at, mask]
                norms = delta.double().norm(dim=(-1, -2))
                item[field+'_delta_mean_fro'] = float(norms.mean())
                item[field+'_delta_max_fro'] = float(norms.max())
            if 'content_base_mass' in left:
                mass = left['content_base_mass'].reshape(b, t, n)[:, at, mask]
                item['base_mass_mean'] = float(mass.double().mean())
            if 'arrived' in left:
                item['fresh_drive_arrival_fraction'] = float(left['arrived'].reshape(b, t, n)[:, at, mask].double().mean())
            record['shells'].append(item)
        delta = (right['transport']-left['transport']).reshape(b, t, n, *left['transport'].shape[-2:])[:, at]
        record['raw_transport_read_port_delta_fro'] = float((delta*left['ports'][None, :, None, None]).sum(1).double().norm())
        if 'content_base_transport' in left:
            base_pools = []
            for entry in (left, right):
                mass = entry['content_base_mass'].reshape(b, t, n)[:, at]*entry['ports'][None]
                weights = mass/mass.sum(-1, keepdim=True).clamp_min(1e-6)
                content = entry['content_base_transport'].reshape(b, t, n, *entry['content_base_transport'].shape[-2:])[:, at]
                base_pools.append((weights[..., None, None]*content).sum(1))
            record['base_read_port_delta_fro'] = float((base_pools[1]-base_pools[0]).double().norm())
        rows.append(record)
    return dict(status='completed', resolution='final_internal_round_only',
                note='No intermediate-round tensors are exposed by core; this is not a per-round trace.', calls=rows)


@torch.no_grad()
def input_interventions(model, tokens, targets, warm, history, seed):
    """Read interventions at fixed parameters; no optimizer update occurs here."""
    training = model.training; model.eval()
    bio_flag = getattr(model, 'runtime_bio_read_enabled', None)
    try:
        warm = invalidate_feedback(clone_state(warm))
        position = tokens.shape[1]//2
        changed = tokens.clone(); changed[:, position] = (changed[:, position]+139)%model.cfg.vocab_size
        reference, _, diag, first, routes = _capture(model, tokens, warm)
        altered, _, _, second, _ = _capture(model, changed, warm)
        token = _comparison(reference, altered, targets)
        token['position'] = position
        token['prefix_logit_max_abs'] = float((reference[:, :position]-altered[:, :position]).abs().max().cpu()) if position else 0.
        with patch.object(model.content_projection, 'forward', side_effect=lambda embedding: embedding.new_zeros(*embedding.shape[:-1], model.cfg.matrix_size**2)):
            erased, _, _, _, _ = _capture(model, tokens, warm)
        cleared, _, _, _, _ = _capture(model, tokens, model.clear_slow(clone_state(warm)))
        cold, _, _, _, _ = _capture(model, tokens, model.initial_state(len(tokens)))
        # Same trained weights, same context multiset; order is the only history change.
        rng = np.random.default_rng(seed)
        permuted = history.clone()
        for lane in range(len(history)):
            order = torch.tensor(rng.permutation(history.shape[1]), dtype=torch.long, device=history.device)
            permuted[lane] = history[lane].index_select(0, order)
        _, ordered_state, _ = model(history)
        _, shuffled_state, _ = model(permuted)
        ordered, _, _, _, _ = _capture(model, tokens, invalidate_feedback(ordered_state))
        shuffled, _, _, _, _ = _capture(model, tokens, invalidate_feedback(shuffled_state))
        result = dict(status='completed', token_replacement=token, routes=routes, diagnostics=diag,
                      erase_fresh_matrix_content=_comparison(reference, erased, targets),
                      erase_note='Erases only width→d² fresh content; original 2d value/key packet remains.',
                      clear_slow=_comparison(reference, cleared, targets), warm_vs_cold=_comparison(reference, cold, targets),
                      replay_ordered_vs_shuffled_history=_comparison(ordered, shuffled, targets),
                      history_token_count=history.shape[1], history_sha256=digest(history.cpu().tolist()),
                      shuffled_history_sha256=digest(permuted.cpu().tolist()),
                      input_difference_by_shell=_shell_trace(first, second, position))
        if getattr(model.cfg, 'content_base', False):
            # This is an attribution control, not a cheaper architecture: all state
            # updates and graph work still execute with the dynamic read disabled.
            model.runtime_bio_read_enabled = False
            base, _, _, _, _ = _capture(model, tokens, warm)
            base_changed, _, _, _, _ = _capture(model, changed, warm)
            base_clear, _, _, _, _ = _capture(model, tokens, model.clear_slow(clone_state(warm)))
            base_cold, _, _, _, _ = _capture(model, tokens, model.initial_state(len(tokens)))
            base_ordered, _, _, _, _ = _capture(model, tokens, invalidate_feedback(ordered_state))
            base_shuffled, _, _, _, _ = _capture(model, tokens, invalidate_feedback(shuffled_state))
            result['content_base_only'] = dict(status='completed', scope='read-only ablation; all state computation retained',
                       full_vs_base_only=_comparison(reference, base, targets),
                       token_replacement=_comparison(base, base_changed, targets),
                       clear_slow=_comparison(base, base_clear, targets),
                       warm_vs_cold=_comparison(base, base_cold, targets),
                       ordered_vs_shuffled_history=_comparison(base_ordered, base_shuffled, targets))
        else:
            result['content_base_only'] = dict(status='not_applicable', reason='This control has no content_base channel.')
        return result
    finally:
        if bio_flag is not None:
            model.runtime_bio_read_enabled = bio_flag
        model.train(training)


@torch.no_grad()
def cut_feedback_check(device, seed=11):
    """Warm unreachable outer credit must not receive global observed tokens."""
    from .model import Config, Network, initialize
    cfg = Config(vocab_size=32, nodes=32, matrix_size=4, width=16, bands=2, modes=2,
                 block_size=4, group_size=2, active_fraction=.5, topology='cut')
    model = Network(cfg).to(device).eval(); initialize(model, seed)
    state = model.initial_state(1); outer = model.shell >= cfg.shells//2
    state['trace'][:, :, :, 0] = .5; state['support'].fill_(1.)
    state['read_query'][:, :, 0] = 1.; state['read_weight'].fill_(1/cfg.nodes)
    state['feedback_gain'].fill_(1.); state['read_scale'].fill_(1.)
    state['feedback_h'].fill_(.02); state['feedback_valid'].fill_(True)
    state['group_summary'] = model._summaries(state)
    first = torch.tensor([1], device=device); other = torch.tensor([2], device=device)
    a, _ = model._apply_feedback(clone_state(state), first)
    b, _ = model._apply_feedback(clone_state(state), other)
    x = torch.tensor([[1, 3, 5, 7]], device=device)
    y = x.clone(); y[:, 0] = 2
    original, _, _ = model(x, clone_state(state))
    changed, _, _ = model(y, clone_state(state))
    outer_difference = float((a['slow'][:, outer]-b['slow'][:, outer]).norm().cpu())
    inner_difference = float((a['slow'][:, ~outer]-b['slow'][:, ~outer]).norm().cpu())
    output_difference = float((original-changed).abs().max().cpu())
    passed = outer_difference == 0. and output_difference == 0. and inner_difference > 0.
    return dict(status='completed' if passed else 'failed', config=asdict(cfg),
                runtime_automatic_feedback=model.runtime_automatic_feedback,
                outer_slow_difference=outer_difference, reachable_inner_positive_control=inner_difference,
                logits_max_difference=output_difference, meaning='Cut blocks both legal propagation and warm pending-supervision writes.')


def run_diagnostics(output_root, corpus, device, seed=11):
    """Run four matched N128 three-update probes; never launches another device."""
    from .benchmark import build, initialize_model, route_snapshot, sources
    device = torch.device(device)
    folder = Path(output_root)/'diagnostics'
    folder.mkdir(parents=True, exist_ok=False)
    conditions = ('v08_seq', 'v09_E', 'v09_dense', 'v09_sparse50')
    values, docs, starts = corpus.sample('train', np.random.default_rng(seed), 2, 8*64)
    tensor = torch.tensor(values, dtype=torch.long, device=device)
    hashes = sources()
    manifest = dict(status='running', seed=seed, nodes=128, batch=2, sequence_tokens=64, steps=3,
                    device=str(device), dtype='float32', source_hashes=hashes,
                    sample=dict(split='train', documents=docs, starts=starts, sha256=digest(values.tolist())),
                    corpus_manifest_sha256=file_hash(corpus.root/'manifest.json'),
                    conditions={name:'not_executed' for name in conditions}, formal_training=False,
                    note='Diagnostic gradient copies/hooks/replays make these timings unsuitable for throughput comparisons.')
    write(folder/'manifest.json', manifest)
    for condition in conditions:
        target = folder/condition; target.mkdir()
        model = optimizer = state = None
        try:
            torch.manual_seed(seed)
            model, cfg = build(dict(condition=condition, nodes=128), corpus.manifest['vocab_size'])
            initialize_model(model, condition, seed); model.to(device).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
            write(target/'config.json', dict(config=asdict(cfg), parameters=parameter_count(model),
                       source_hashes=hashes, sample=manifest['sample']))
            for step in range(3):
                sync(device); began = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                batch = tensor[:, step*64:step*64+65]
                logits, state, diagnostics = model(batch[:, :-1], state)
                loss = F.cross_entropy(logits.flatten(0, 1), batch[:, 1:].flatten())
                loss.backward(); gradient = gradient_summary(model)
                record = dict(step=step, loss=float(loss.detach().cpu()), gradients=gradient,
                     gate_blend=model.gate_blend, active_fraction=getattr(model, 'active_fraction', 1.),
                     routes=route_snapshot(model), diagnostics={name:float(value.detach().cpu()) for name,value in diagnostics.items()})
                try:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                except RuntimeError as error:
                    record.update(status='failed', error=str(error), optimizer_applied=False)
                    with (target/'raw.jsonl').open('a') as stream:
                        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
                    raise
                optimizer.step(); state = invalidate_feedback(detach(state)); sync(device)
                record.update(status='completed', optimizer_applied=True, clip_norm32=float(norm.detach().cpu()),
                              seconds_including_diagnostic_copies=time.perf_counter()-began)
                with (target/'raw.jsonl').open('a') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
            model.eval()
            probe = tensor[:, 3*64:3*64+16]; labels = tensor[:, 3*64+1:3*64+17]
            interventions = input_interventions(model, probe, labels, state, tensor[:, :3*64], seed)
            write(target/'interventions.json', interventions)
            save(target/'last.pt', dict(model=model.state_dict(), runtime_state=state, optimizer=optimizer.state_dict(),
                 config=asdict(cfg), step=3, formal_training=False, exact_resume=False))
            manifest['conditions'][condition] = 'completed'
        except Exception as error:
            write(target/'failure.json', dict(status='failed', error=repr(error)))
            (target/'traceback.txt').write_text(traceback.format_exc())
            manifest['conditions'][condition] = 'failed'
        finally:
            del model, optimizer, state
            gc.collect()
            write(folder/'manifest.json', manifest)
    write(folder/'cut_feedback_check.json', cut_feedback_check(device, seed))
    manifest['status'] = 'completed' if all(value == 'completed' for value in manifest['conditions'].values()) else 'completed_with_failures'
    write(folder/'manifest.json', manifest)
    return manifest
