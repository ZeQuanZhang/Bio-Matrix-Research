"""Post-hoc CPU state diagnostics; no training or model selection."""
import json
from pathlib import Path
import statistics

import numpy as np
import torch

from .data import generate
from .models import Config, MemoryModel


def run():
    torch.set_num_threads(1)
    events, _, queries = generate(np.random.default_rng(190001), batch=64, pairs=2, delay=24)
    rows = []
    for path in sorted(Path('results/matrix').glob('*/best.pt')):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        cfg = Config(**checkpoint['protocol']['model'])
        model = MemoryModel(cfg); model.load_state_dict(checkpoint['model']); model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            x = model.encoder(model.operation(events[..., 0])+model.key(events[..., 1])+model.value(events[..., 2]))
            encoded = model.input_state(x)
        state = torch.zeros(64, cfg.state)
        saturated, updates, readout_saturated, ratios, local_norms = [], [], [], [], []
        for t in range(events.shape[1]):
            with torch.no_grad():
                if cfg.cell == 'gru':
                    projection = model.state_state(state)
                    retain = (encoded[:, t, cfg.state:2*cfg.state]+projection[:, cfg.state:2*cfg.state]).sigmoid()
                    update = 1-retain
                else:
                    gate = encoded[:, t, cfg.state:]
                    feedback = model.state_state(state) if cfg.cell != 'linear' else torch.zeros_like(encoded[:, t])
                    update = (gate+feedback[:, cfg.state:]).sigmoid()
                    if cfg.cell == 'reaction':
                        p, q = model.pq(state).chunk(2, -1)
                        reaction = model.reaction(p*q)
                        base = encoded[:, t, :cfg.state]+feedback[:, :cfg.state]
                        ratios.append(float(reaction.square().mean().sqrt()/(base.square().mean().sqrt()+1e-8)))
                new = model.step(encoded[:, t], state)
                if t >= 2:
                    saturated.append(float((new.abs() > .95).float().mean()))
                    updates.append(float(update.mean()))
                    gate = model.gate(new).sigmoid()
                    readout_saturated.append(float(((gate < .01) | (gate > .99)).float().mean()))
            if t == queries[0]-1:
                for i in range(4):
                    jac = torch.autograd.functional.jacobian(lambda s: model.step(encoded[i:i+1, t], s[None])[0], state[i])
                    local_norms.append(float(torch.linalg.svdvals(jac).max()))
            state = new
        row = {'variant': checkpoint['protocol']['variant'], 'seed': checkpoint['protocol']['seed'],
               'state_abs_gt_095_fraction': statistics.mean(saturated), 'update_gate_mean': statistics.mean(updates),
               'readout_gate_saturated_fraction': statistics.mean(readout_saturated),
               'local_jacobian_max_singular_mean': statistics.mean(local_norms),
               'quadratic_to_linear_rms_ratio': statistics.mean(ratios) if ratios else None}
        assert all(np.isfinite(v) for v in row.values() if isinstance(v, float))
        rows.append(row)
    assert len(rows) == 21
    Path('results/matrix/state_diagnostics.json').write_text(json.dumps({'device': 'cpu', 'posthoc': True,
        'condition': '64 fresh episodes, 2 keys,24 distractors; Jacobian at final distractor,4 examples',
        'limitation': 'local sensitivity and saturation are descriptive; not a causal diagnosis of training failure', 'runs': rows}, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    run()
