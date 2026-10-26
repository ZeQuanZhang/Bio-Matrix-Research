"""Separate periodic forecasting test; not evidence for general language memory."""
import json
import hashlib
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch import nn

from .extensions import Oscillator
from .run import sync

ROOT = Path('results/matrix_signals')
CONFIG = {'steps': 1000, 'batch': 64, 'context': 16, 'train_horizon': 16, 'state': 32,
          'width': 16, 'lr': .003, 'seeds': [42, 137, 2026], 'device': 'mps',
          'variants': ['spectral_zero', 'spectral_fixed', 'spectral_learned'],
          'train_frequencies': [.08, .17, .31], 'test_frequencies': [.12, .25]}


def examples(rng, batch, horizon, frequencies, noise=0.):
    omega = rng.choice(frequencies, batch)
    phase = rng.uniform(0, 2*np.pi, batch)
    amplitude = rng.uniform(.6, 1.2, batch)
    signal = amplitude[:, None]*np.sin(np.arange(16+horizon)[None]*omega[:, None]+phase[:, None])
    observed = np.zeros((batch, 16+horizon, 2), dtype=np.float32)
    observed[:, :16, 0] = signal[:, :16]+rng.normal(0, noise, (batch, 16))
    observed[:, :16, 1] = 1.
    return torch.from_numpy(observed), torch.tensor(signal[:, 16:], dtype=torch.float32)


class SignalModel(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2, 16), nn.Tanh())
        self.oscillator = Oscillator(mode, 16, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, events, reset=False):
        x = self.encoder(events); state = x.new_zeros(len(x), 32); predictions = []
        for t in range(x.shape[1]):
            if reset and t == 16:
                state = torch.zeros_like(state)
            state = self.oscillator(state, x[:, t])
            if t >= 16:
                predictions.append(self.output(state).squeeze(-1))
        return torch.stack(predictions, 1)


def run():
    torch.set_num_threads(4); device = torch.device('mps')
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS required')
    CONFIG['source_hashes'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (Path(__file__), Path('matrix_lab/extensions.py'))}
    ROOT.mkdir(parents=True, exist_ok=True); (ROOT/'config.json').write_text(json.dumps(CONFIG, indent=2))
    for seed in CONFIG['seeds']:
        for mode in CONFIG['variants']:
            path = ROOT/f'{mode}_seed{seed}.json'
            if path.exists():
                if json.loads(path.read_text())['config'] != CONFIG:
                    raise ValueError('Existing signal result uses another protocol; choose a new results directory')
                continue
            torch.manual_seed(seed); rng = np.random.default_rng(seed)
            model = SignalModel(mode).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.0001)
            best, seconds, trace, peak = float('inf'), 0., [], 0
            vx, vy = examples(np.random.default_rng(10001), 128, 16, CONFIG['train_frequencies'])
            vx, vy = vx.to(device), vy.to(device)
            for step in range(CONFIG['steps']):
                x, y = examples(rng, 64, 16, CONFIG['train_frequencies'], noise=.05)
                x, y = x.to(device), y.to(device)
                sync(device); tick = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss = (model(x)-y).square().mean(); loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not bool(loss.isfinite() and norm.isfinite()):
                    raise FloatingPointError('Nonfinite signal training')
                optimizer.step(); sync(device); seconds += time.perf_counter()-tick
                peak = max(peak, torch.mps.driver_allocated_memory())
                if (step+1) % 100 == 0:
                    trace.append({'step': step+1, 'train_mse': float(loss.detach())})
                if (step+1) % 250 == 0:
                    with torch.no_grad():
                        mse = float((model(vx)-vy).square().mean())
                    trace.append({'step': step+1, 'validation_mse': mse})
                    if mse < best:
                        best = mse
                        torch.save({'model': model.state_dict(), 'step': step+1, 'mode': mode, 'seed': seed, 'config': CONFIG}, ROOT/f'{mode}_seed{seed}.pt')
            checkpoint = torch.load(ROOT/f'{mode}_seed{seed}.pt', map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model']); scores = {}
            with torch.no_grad():
                for name, horizon, frequencies, noise in [('seen16', 16, CONFIG['train_frequencies'], 0.),
                    ('seen64', 64, CONFIG['train_frequencies'], 0.), ('unseen16', 16, CONFIG['test_frequencies'], 0.),
                    ('noise20', 16, CONFIG['train_frequencies'], .2)]:
                    x, y = examples(np.random.default_rng(11001), 512, horizon, frequencies, noise)
                    y = y.to(device); scores[name] = {'carry': [], 'reset': []}
                    for reset, key in [(False, 'carry'), (True, 'reset')]:
                        losses = []
                        for start in range(0, 512, 128):
                            pred = model(x[start:start+128].to(device), reset)
                            losses.extend((pred-y[start:start+128]).square().mean(1).cpu().tolist())
                        scores[name][key] = {'mse': float(np.mean(losses)), 'episode_mse': losses}
            result = {'mode': mode, 'seed': seed, 'config': CONFIG, 'parameters': sum(p.numel() for p in model.parameters()),
                      'state_floats': 32, 'seconds': seconds, 'events_per_second': 1000*64*32/seconds,
                      'metal_driver_bytes': peak, 'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      'rss_scope': 'cumulative process peak, not an independent per-run peak', 'trace': trace, 'test': scores}
            path.write_text(json.dumps(result, indent=2))
            print(mode, seed, {k: round(v['carry']['mse'], 5) for k, v in scores.items()}, flush=True)


if __name__ == '__main__':
    run()
