"""Frozen-weight CPU mechanism, resource and numerical conformance records."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
from v07.common import state_bytes
from v07.data import Corpus
from v07.model import Config, Network, bounded, initialize, joseph, unit_rotate


def main():
    torch.set_num_threads(1)
    root = Path(__file__).parent
    sources = [PROJECT/'v07/model.py', PROJECT/'v07/graph.py', PROJECT/'v07/test_model.py',
               Path('<LOCAL_HOME>/Downloads/BioMatrix_完整构想描述.md'),
               PROJECT/'docs/V06_DESIGN.md', PROJECT/'docs/V07_DESIGN_AND_EXPERIMENT.md']
    payload = dict(scope='CPU one thread; fixed learned weights; no optimizer, no training claims',
                   sources={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
    sizes = []
    for nodes in (32, 128, 512):
        model = Network(Config(nodes=nodes)).eval()
        state = model.initial_state(1)
        fields = {name: dict(shape=list(value.shape), dtype=str(value.dtype), bytes=state_bytes(value))
                  for name, value in state.items()}
        expected = 2244*nodes+4136
        assert state_bytes(state) == expected
        sizes.append(dict(nodes=nodes, bytes_per_stream=state_bytes(state), fields=fields,
                          formula_default_fp32='2244*N + 4136 bytes; d=8,bands=3,modes=3,trace_bands=2,width=64,H=8'))
    payload['runtime_storage'] = sizes
    payload['complexity'] = dict(main='O(B*T*J*N*k*d^2)', covariance='O(B*T*N*d^2)',
                                 input_filter='O(B*T*J*N*d)', credit_metadata='O(B*H*(N*d+2*N+2*width+1))',
                                 history_dependence='H=8 is explicit fixed credit metadata, never node-state/context copies')
    normalization = {}
    for mode in ('none', 'participation'):
        rows = []
        for nodes in (32, 128, 512):
            model = Network(Config(vocab_size=48, matrix_size=4, width=16, modes=2,
                                   nodes=nodes, local_write_normalization=mode)).double()
            initialize(model, 71)
            state = model.initial_state(1)
            distribution = torch.ones(1, nodes, dtype=torch.float64)/nodes
            scale, effective = model.write_scale(distribution)
            state['read_scale'] = scale; state['feedback_gain'] = torch.ones_like(distribution)
            state['read_query'][..., 0] = 1
            direction = torch.zeros_like(state['slow']); direction[..., 0, 0] = 1/nodes
            query = torch.zeros(1, nodes, 4, dtype=torch.float64); query[..., 0] = 1
            with torch.no_grad(), patch.object(model, 'local_direction', return_value=direction):
                updated = model.local_update(state, model.embedding(torch.tensor([7])))
            rows.append(dict(nodes=nodes, effective_nodes=float(effective.item()), scale=float(scale.item()),
                             slow_read_change=float(model.slow_read(updated, query, distribution).norm())))
        normalization[mode] = rows
    assert max(row['slow_read_change'] for row in normalization['participation'])-min(row['slow_read_change'] for row in normalization['participation']) < 1e-12
    payload['isolated_uniform_local_write'] = normalization
    model = Network(Config(vocab_size=48, matrix_size=4, width=16, modes=2)).eval()
    initialize(model, 71); state = model.initial_state(1)
    key = torch.zeros(1, 32, 4); key[..., 0] = 1
    value = torch.ones_like(key)*.2; variance = torch.ones(1, 32)*.1
    with torch.no_grad():
        for _ in range(4096):
            mean, covariance = model.stationary_prior(state)
            m, p = joseph(mean, covariance, key, value, variance, torch.ones_like(variance)*.5,
                           torch.ones_like(variance)*model.cfg.covariance_process)
            state = state | dict(m=bounded(m), p=p)
    eigenvalues = torch.linalg.eigvalsh(state['p'])
    bound = model.covariance_upper_bound()
    assert float(eigenvalues.min()) > 0 and float(eigenvalues.max()) <= bound+1e-3
    phase = torch.tensor([[1., 0.], [0., 1.]])
    for _ in range(10000):
        phase = unit_rotate(phase, torch.tensor([.031, .027]))
    payload['numerical_bounds'] = dict(kernel_steps=4096, covariance_float32_min_eigenvalue=float(eigenvalues.min()),
                                      covariance_float32_max_eigenvalue=float(eigenvalues.max()), covariance_theoretical_upper_bound=bound,
                                      old_unobserved_variance_at_25651=1+.002001*25651,
                                      phase_steps=10000, max_phase_norm_error=float((phase.norm(dim=-1)-1).abs().max()),
                                      note='Covariance bound is for rho prior + one fresh Joseph event; mean is projected to norm 4')
    corpus = Corpus(); tokens, docs, starts = corpus.sample('train', np.random.default_rng(7101), 4, 32)
    calibration = []
    for nodes in (32, 128, 512):
        model = Network(Config(vocab_size=corpus.manifest['vocab_size'], nodes=nodes)).eval(); initialize(model, 11)
        reached, opened = [], []
        original = model.weights
        def record(*args):
            weight, gate, sine = original(*args)
            active = model.outward & model.edge_mask.bool()
            opened.append([float(gate[:, active & (model.shell[:, None] == shell)].mean()) for shell in (1, 2, 3)])
            return weight, gate, sine
        model.weights = record
        with torch.no_grad():
            state = None
            for position in range(32):
                previous = torch.zeros(4, nodes, dtype=torch.long) if state is None else state['evidence_count']
                _, state, diag = model(torch.tensor(tokens[:, position:position+1]), state)
                arrived = state['evidence_count'] > previous
                reached.append([float(arrived[:, model.shell == shell].float().mean()) for shell in range(4)])
        calibration.append(dict(nodes=nodes, reached_by_shell=np.mean(reached, axis=0).tolist(),
                                 outward_open_target_shell=np.mean(opened, axis=0).tolist(),
                                 slow_rms=float(state['slow'].square().mean().sqrt()),
                                 final_effective_read_nodes=float(diag['effective_read_nodes']),
                                 final_local_write_scale=float(diag['local_write_scale'])))
    payload['train_only_routing_calibration'] = dict(seed=11, docs=docs, starts=starts, windows=4, tokens_per_window=32,
                                                     conditions=calibration, online_state_updates=True)
    snapshot = root/'model_source_snapshot'; snapshot.mkdir(exist_ok=True)
    for path in sources[:3]:
        shutil.copy2(path, snapshot/path.name)
    (root/'model_conformance.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({name: payload[name] for name in ('numerical_bounds', 'isolated_uniform_local_write', 'train_only_routing_calibration')}, indent=2))


if __name__ == '__main__':
    main()
