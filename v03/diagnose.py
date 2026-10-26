"""Post-training interventions, test-time iteration changes and sparse phase diagnostics."""
import argparse
import copy
from dataclasses import fields
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from .data import bank, collate, solve, digest, ANSWERS
from .graph import graph
from .model import Config, Network
from .run import ROOT, write, inputs, evaluate


def load(folder, device):
    checkpoint = torch.load(folder/'best.pt', map_location=device, weights_only=True)
    model = Network(Config(**checkpoint['protocol']['model'])).to(device)
    model.load_state_dict(checkpoint['model']); model.eval()
    return model


def phase_metrics(z_history, neighbors):
    z = z_history[-1]
    def doubled(v):
        return torch.stack((v[..., 0].square()-v[..., 1].square(), 2*v[..., 0]*v[..., 1]), -1)
    r1 = z.mean(1).norm(dim=-1); z2 = doubled(z); r2 = z2.mean(1).norm(dim=-1); r4 = doubled(z2).mean(1).norm(dim=-1)
    edges = z[:, neighbors]
    cosine = (z[:, :, None]*edges).sum(-1).mean((1, 2))
    edge_differences = []
    for current in z_history[-8:]:
        neighbor = current[:, neighbors]
        c = (current[:, :, None]*neighbor).sum(-1)
        s = current[:, :, None, 0]*neighbor[..., 1]-current[:, :, None, 1]*neighbor[..., 0]
        edge_differences.append(torch.stack((c, s), -1))
    plv = torch.stack(edge_differences).mean(0).norm(dim=-1).mean((1, 2))
    return {'R1': r1.tolist(), 'R2': r2.tolist(), 'R4': r4.tolist(), 'edge_cosine': cosine.tolist(), 'edge_PLV_last8': plv.tolist()}


def counterfactual_pairs(examples):
    rng = np.random.default_rng(19403); pairs = []
    for original in examples[::12]:
        if original['family'] == 'compare' and original['spec']['start'] == original['spec']['other']:
            continue  # This fixed query is always true, so its answer cannot be changed.
        changed = copy.deepcopy(original); world = changed['world']; count = len(world['ids'])
        for _ in range(100):
            world['attrs'] = np.stack((rng.integers(8, size=count), rng.integers(3, size=count), rng.integers(2, size=count)), -1).tolist()
            world['relations'] = np.stack((rng.permutation(count), rng.permutation(count)), -1).tolist()
            target = solve(world, changed['spec'])
            if target != original['target']:
                break
        else:
            continue
        by_id = {r[0]: r for r in changed['records']}
        for i, oid in enumerate(world['ids']):
            by_id[oid+1][1:6] = [513+world['attrs'][i][0], 521+world['attrs'][i][1], 524+world['attrs'][i][2],
                                world['ids'][world['relations'][i][0]]+1, world['ids'][world['relations'][i][1]]+1]
        changed['target'] = target; changed['answer_zh'] = ANSWERS[target]; changed['world_hash'] = digest(world)
        assert changed['question'] == original['question'] and target != original['target']
        pairs.append((original, changed))
    return pairs


@torch.no_grad()
def evidence_sensitivity(model, examples, device):
    pairs = counterfactual_pairs(examples); records = []
    for start in range(0, len(pairs), 8):
        group = pairs[start:start+8]; a = collate([p[0] for p in group]); b = collate([p[1] for p in group])
        left = model(**inputs(a, device))[0].cpu(); right = model(**inputs(b, device))[0].cpu()
        for i, (original, changed) in enumerate(group):
            records.append({'family': original['family'], 'world_hash': original['world_hash'], 'changed_world_hash': changed['world_hash'],
                            'old_target': original['target'], 'changed_target': changed['target'],
                            'old_prediction': int(left[i].argmax()), 'changed_prediction': int(right[i].argmax()),
                            'logit_absolute_change_max': float((left[i]-right[i]).abs().max()),
                            'prediction_total_variation': float((left[i].softmax(-1)-right[i].softmax(-1)).abs().sum()/2)})
    return {'protocol': 'Post-hoc diagnostic: same valid question, same object IDs, resampled attributes/relations until solver answer differs; not a primary accuracy benchmark.',
            'pairs': records, 'prediction_flip_fraction': float(np.mean([r['old_prediction'] != r['changed_prediction'] for r in records])),
            'mean_logit_absolute_change_max': float(np.mean([r['logit_absolute_change_max'] for r in records])),
            'mean_prediction_total_variation': float(np.mean([r['prediction_total_variation'] for r in records]))}


@torch.no_grad()
def interventions(model, examples, device):
    kinds = ('carry', 'rotate', 'phase_swap', 'phase_scramble', 'content_matched', 'cut_graph')
    result = {f'{kind}_{recovery}': [] for kind in kinds for recovery in (0, 4)}
    measured_energy = {k: [] for k in kinds}
    for start in range(0, len(examples), 16):
        group = examples[start:start+16]; batch = collate(group); x = inputs(batch, device); encoded = model.encode(**x)
        n = model.cfg.nodes; s = encoded[0].new_zeros(len(group), n, model.cfg.state_dim)
        z = model.initial_z[None].expand(len(group), -1, -1)
        for _ in range(model.cfg.iterations):
            s, z = model.update(s, z, encoded)
        original_graph = model.neighbors.clone()
        for kind in kinds:
            ss, zz = model.intervene(s, z, kind)
            measured_energy[kind].extend(((ss-s).square().sum((1, 2))+(zz-z).square().sum((1, 2))).cpu().tolist())
            if kind == 'cut_graph':
                model.neighbors = graph(n, model.cfg.graph_seed, 'cut').to(device)
            for recovery in (0, 4):
                if recovery:
                    for _ in range(recovery):
                        ss, zz = model.update(ss, zz, encoded)
                logits = model.read(encoded[0], ss, zz); target = batch['target'].to(device)
                losses = F.cross_entropy(logits, target, reduction='none').cpu().tolist()
                predictions = logits.argmax(-1).cpu().tolist()
                result[f'{kind}_{recovery}'].extend([{'world_hash': e['world_hash'], 'family': e['family'], 'correct': int(p == e['target']), 'loss': loss}
                                                   for e, p, loss in zip(group, predictions, losses)])
            model.neighbors = original_graph
    return {'intervention_at': 'after trained J iterations', 'recovery': '0 or 4 extra updates; carry also receives same extra updates',
            'content_matched_reference': 'total L2 energy of phase_swap, not coordinate-wise functional equivalence',
            'mean_squared_perturbation_energy': {k: float(np.mean(v)) for k, v in measured_energy.items()},
            'conditions': {k: {'accuracy': float(np.mean([r['correct'] for r in rows])), 'loss': float(np.mean([r['loss'] for r in rows])),
                               'episode_results': rows} for k, rows in result.items()}}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--stage', default='all', choices=('all', 'main')); args = p.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS required for post-training inference')
    torch.set_num_threads(4); device = torch.device('mps'); test = bank('test', 24)
    for folder in sorted(ROOT.glob('*')):
        if not folder.is_dir() or not (folder/'result.json').exists() or not (folder/'best.pt').exists():
            continue
        result = json.loads((folder/'result.json').read_text())
        if 'protocol' not in result or result['protocol']['condition']['stage'] == 'pilot':
            continue
        c = result['protocol']['condition']; path = folder/'diagnostics.json'
        if path.exists():
            continue
        if args.stage == 'main' and c['stage'] != 'main':
            continue
        model = load(folder, device)
        examples = test
        if c['stage'] == 'public':
            from .public import PublicData
            examples = PublicData().test
        sample = collate(examples[:16]); x = inputs(sample, device)
        with torch.no_grad():
            for _ in range(3):
                model(**x)
            elapsed = []
            for _ in range(7):
                torch.mps.synchronize(); started = time.perf_counter(); model(**x); torch.mps.synchronize()
                elapsed.append(time.perf_counter()-started)
            logits, (s, z, _) = model(**x)
            h = model.encode(**x)[0]
            feature = torch.cat((s, model.roles[None].expand(len(s), -1, -1)), -1)
            attention = (feature*model.read_query(h)[:, None]).sum(-1).softmax(-1)
            mean = (attention[..., None]*s).sum(1)
            coherence = (attention[..., None]*z).sum(1).square().sum(-1, keepdim=True).add(1e-8).sqrt()
            gate = model.read_gate(torch.cat((mean, coherence), -1)).sigmoid()
            zeroed = model.read(h, torch.zeros_like(s), torch.zeros_like(z))
            swapped = model.read(h, s.roll(1, 0), z.roll(1, 0))
        diagnostic = {'status': 'completed', 'inference_batch': 16, 'inference_records': int(sample['records'].shape[1]),
                      'inference_question_tokens': int(sample['question'].shape[1]), 'inference_seconds_samples': elapsed,
                      'median_inference_batch_seconds': float(np.median(elapsed)),
                      'timing_note': 'Synchronized full-model GPU latency on first fixed 16 test inputs; encoding included, CPU collation excluded.',
                      'state_node_std_mean': float(s.std(1).mean()), 'content_saturation_abs_gt_095': float((s.abs()>.95).float().mean()),
                      'read_gate_min': float(gate.min()), 'read_gate_max': float(gate.max()),
                      'logit_zero_state_abs_max': float((logits-zeroed).abs().max()),
                      'logit_cross_sample_state_swap_abs_max': float((logits-swapped).abs().max()),
                      'state_probe_subset': 'First fixed 16 test inputs; A is retrieval/M8, not a population estimate across task families.',
                      'learned_internal_dt': float((.5*model.raw_dt.sigmoid()).detach().cpu()),
                      'learned_phase_coupling': float((2*model.raw_coupling.sigmoid()).detach().cpu()) if hasattr(model, 'raw_coupling') else 0.,
                      'constant_float_buffers_roles_and_initial_phase': model.cfg.nodes*10}
        if c['stage'] == 'main':
            # Evaluate all training seeds for test-time J sweeps; never mix with retrained J cases.
            diagnostic['test_time_iterations'] = {str(j): evaluate(model, examples, device, iterations=j) for j in (2, 8, 16, 32)}
            if c['seed'] == 11:
                diagnostic['same_question_evidence_counterfactual'] = evidence_sensitivity(model, examples, device)
                diagnostic['interventions'] = interventions(model, examples, device)
                diagnostic['disable_coupling_full_solve'] = evaluate(model, examples, device, intervention='no_coupling')
                diagnostic['disable_messages_full_solve'] = evaluate(model, examples, device, intervention='no_messages')
                diagnostic['disable_messages_note'] = 'Only content messages are disabled; auxiliary-state communication/phase coupling remains. This is not a trained C0 no-communication baseline.'
                model.cfg.phase_to_messages = False
                diagnostic['auxiliary_message_gate_off'] = evaluate(model, examples, device)
                diagnostic['auxiliary_message_gate_off_note'] = 'C1: nonperiodic auxiliary gate; C2/C3: relative-phase gate. Test-only ablation; readout auxiliary statistic remains active.'
                model.cfg.phase_to_messages = True
            if c['seed'] == 11:
                phase_rows = []; attention_by_family = {}
                # Fixed, evenly spread sample selection made without correctness labels.
                subset = examples[::12]
                for start in range(0, len(subset), 16):
                    group = subset[start:start+16]; batch = collate(group); x = inputs(batch, device)
                    with torch.no_grad():
                        logits, (s, z, history) = model(**x, trace=True)
                        h = model.encode(**x)[0]
                        feature = torch.cat((s, model.roles[None].expand(len(s), -1, -1)), -1)
                        attention = (feature*model.read_query(h)[:, None]).sum(-1).softmax(-1).cpu()
                    metrics = phase_metrics([v.cpu() for v in history], model.neighbors.cpu()) if c['model'] != 'C1' else {}
                    predictions = logits.argmax(-1).cpu().tolist()
                    for i, e in enumerate(group):
                        phase_rows.append({'world_hash': e['world_hash'], 'family': e['family'], 'correct': int(predictions[i] == e['target']),
                                           **{k: v[i] for k, v in metrics.items()}})
                        attention_by_family.setdefault(e['family'], []).append(attention[i].tolist())
                if c['model'] != 'C1':
                    diagnostic['phase_behavior'] = phase_rows
                diagnostic['mean_node_read_attention_by_task'] = {k: np.mean(v, axis=0).tolist() for k, v in attention_by_family.items()}
                diagnostic['phase_interpretation'] = 'Associations from fixed samples; high PLV/global coherence alone is not functional specialization.'
        write(path, diagnostic); print(json.dumps({'diagnosed': folder.name}), flush=True)
        del model; torch.mps.empty_cache()


if __name__ == '__main__':
    main()
