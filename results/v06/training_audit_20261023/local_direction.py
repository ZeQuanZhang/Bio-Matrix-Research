"""Frozen-checkpoint CPU diagnostic; never edits training sources or updates weights."""
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import torch
from torch.nn import functional as F

from v06.common import PROJECT, detach, source_hashes, write
from v06.data import Corpus
from v06.model import Config
from v06.run import build

torch.set_num_threads(1)
folder = PROJECT/'results/v06/public_lm/core_n32_full_s11'
out = Path(__file__).resolve().parent
protocol = json.loads((folder/'protocol.json').read_text())
before = source_hashes()
assert before == protocol['source_hashes'], 'Training sources differ from checkpoint protocol'
model = build(Config(**protocol['model']))
checkpoint = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
model.load_state_dict(checkpoint['model'])
model.double().eval()
for parameter in model.parameters():
    parameter.requires_grad_(False)
corpus = Corpus()
values, documents, starts = corpus.sample('train', np.random.default_rng(236401), 16, 80)
original_update = model.local_update
rows = []
for window, sequence in enumerate(values):
    state = None
    with torch.no_grad():
        _, state, _ = model(torch.as_tensor(sequence[:32])[None], state)
    for t in range(32, 40):
        prior = detach(state)
        # Only the evolving slow matrix needs gradients. Other state is fixed history.
        prior['slow'] = prior['slow'].clone().requires_grad_(True)
        token = torch.tensor([int(sequence[t])])
        truth = torch.tensor([int(sequence[t+1])])
        logits, state, _ = model.step(token, prior)
        loss = F.cross_entropy(logits, truth)
        gradient, = torch.autograd.grad(loss, state['slow'])
        with torch.no_grad():
            delta = original_update(state, model.embedding(truth))-state['slow']
            dot = (gradient*delta).sum()
            norm_product = gradient.norm()*delta.norm()
            assert norm_product > 0 and torch.isfinite(dot)
            cosine = -dot/norm_product
            fixed_slow = state['slow'].detach()
            baseline = float(loss)
            # Replay the SAME prediction with a substituted slow matrix; verify
            # the derivative by symmetric finite differences, not future loss.
            replay = {}
            epsilon = 1e-4
            for scale in (0., -1., 1., -epsilon, epsilon):
                replacement = fixed_slow+scale*delta
                model.local_update = lambda _state, _embedding, x=replacement: x
                alternative, _, _ = model.step(token, prior)
                replay[scale] = float(F.cross_entropy(alternative, truth))
            model.local_update = original_update
            assert abs(replay[0.]-baseline) < 1e-10
            finite_difference = (replay[epsilon]-replay[-epsilon])/(2*epsilon)
            assert abs(finite_difference-float(dot)) < max(1e-10, abs(float(dot))*1e-3), (window, t, float(dot), finite_difference, float(delta.norm()), replay)
            rows.append(dict(window=window, position=t, loss=baseline,
                             slow_norm=float(fixed_slow.norm()), delta_norm=float(delta.norm()),
                             gradient_norm=float(gradient.norm()), gradient_dot_update=float(dot),
                             cosine_with_descent=float(cosine), finite_difference=finite_difference,
                             replay_loss_change=replay[1.]-baseline))
        state = detach(state)
assert source_hashes() == before
summary = dict(samples=len(rows), windows=len(documents),
               descent_fraction=sum(r['gradient_dot_update'] < 0 for r in rows)/len(rows),
               mean_cosine_with_descent=float(np.mean([r['cosine_with_descent'] for r in rows])),
               mean_gradient_dot_update=float(np.mean([r['gradient_dot_update'] for r in rows])),
               mean_replay_loss_change=float(np.mean([r['replay_loss_change'] for r in rows])),
               replay_descent_fraction=sum(r['replay_loss_change'] < 0 for r in rows)/len(rows),
               mean_update_norm=float(np.mean([r['delta_norm'] for r in rows])),
               max_derivative_check_error=max(abs(r['finite_difference']-r['gradient_dot_update']) for r in rows))
write(out/'local_direction.json', dict(
    checkpoint=str(folder/'best.pt'), source_hashes=before, precision='CPU float64',
    split='train', documents=documents, starts=starts, summary=summary, rows=rows,
    interpretation='A newly visible token provides feedback about the previous prediction. '
    'This tests whether its slow-state update is a first-order descent direction for '
    'that previous prediction. It does not test future-token utility or retrain the model.'))
print(json.dumps(summary, indent=2))
