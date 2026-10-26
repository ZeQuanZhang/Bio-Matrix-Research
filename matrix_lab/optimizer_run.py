"""Independent CPU tests of gradient filtering and periodic parameter exploration."""
import json
import hashlib
import math
from pathlib import Path
import resource
import time

import numpy as np
import torch

ROOT = Path('results/matrix_optimizer')
VARIANTS = ('adam', 'adam_cosine', 'gaussian_cosine', 'gaussian_periodic_cosine')


class GradientFilter:
    def __init__(self, parameters, seed):
        self.parameters = list(parameters)
        generator = torch.Generator().manual_seed(seed)
        self.states = []
        for p in self.parameters:
            direction = torch.randn(p.shape, generator=generator)
            direction /= direction.norm().clamp_min(1e-8)
            self.states.append([torch.zeros_like(p), torch.ones_like(p), torch.ones_like(p), direction])

    @torch.no_grad()
    def apply(self):
        for p, state in zip(self.parameters, self.states):
            mean, variance, noise, _ = state
            residual = p.grad - mean
            noise.mul_(.95).add_(residual.square(), alpha=.05).clamp_(min=1e-6)
            prior = variance + .01*noise + 1e-5
            gain = prior/(prior+noise)
            mean.add_(gain*residual)
            variance.copy_((1-gain)*prior)
            p.grad.copy_(mean)

    @torch.no_grad()
    def explore(self, step, lr):
        for p, (mean, variance, noise, direction) in zip(self.parameters, self.states):
            uncertainty = variance.mean().sqrt()/(1+noise.mean().sqrt())
            amplitude = lr*.2*uncertainty*math.sin(2*math.pi*step/50)/math.sqrt(1+step/50)
            p.add_(amplitude*direction)


def objective(name, x):
    if name == 'rosenbrock':
        return 100*(x[1]-x[0].square()).square()+(1-x[0]).square()
    if name == 'rastrigin':
        return 20+x.square().sum()-10*(2*math.pi*x).cos().sum()
    if name == 'ackley':
        return -20*torch.exp(-.2*torch.sqrt(x.square().mean()+1e-12))-torch.exp((2*math.pi*x).cos().mean())+20+math.e
    raise ValueError(name)


def update(optimizer, filtering, variant, step, steps, base_lr):
    lr = base_lr if variant == 'adam' else base_lr*(.1+.45*(1+math.cos(math.pi*step/steps)))
    optimizer.param_groups[0]['lr'] = lr
    if variant.startswith('gaussian'):
        filtering.apply()
    optimizer.step()
    if variant == 'gaussian_periodic_cosine':
        filtering.explore(step, lr)


def self_check():
    assert float(objective('rosenbrock', torch.ones(2))) == 0.
    assert abs(float(objective('rastrigin', torch.zeros(2)))) < 1e-6
    assert abs(float(objective('ackley', torch.zeros(2)))) < 1e-4
    p = torch.nn.Parameter(torch.zeros(2)); p.grad = torch.full((2,), 2.)
    filtering = GradientFilter([p], 1); filtering.apply()
    assert bool(((p.grad > 0) & (p.grad < 2)).all())
    assert bool((filtering.states[0][1] > 0).all())


def run():
    torch.set_num_threads(1); self_check(); ROOT.mkdir(parents=True, exist_ok=True)
    config = {'device': 'cpu', 'reason': '2D analytic and tiny MLP workloads; independent of MPS architecture experiments',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'steps': 2000, 'seeds': list(range(10)), 'variants': VARIANTS, 'function_lr': .03,
              'thresholds': {'rosenbrock': .001, 'rastrigin': .001, 'ackley': .01},
              'gradient_assumption': 'diagonal Gaussian random-walk gradient + noisy observations; approximate, uncalibrated',
              'periodic_exploration': 'fixed direction, period50; amplitude depends on posterior variance and decays with time'}
    if (ROOT/'config.json').exists():
        if json.loads((ROOT/'config.json').read_text()) != json.loads(json.dumps(config)):
            raise ValueError('Existing optimizer experiment uses another protocol')
        if all((ROOT/name).exists() for name in ('functions.json', 'neural.json', 'runtime.json')):
            print('Independent optimizer tests already complete.', flush=True)
            return
    (ROOT/'config.json').write_text(json.dumps(config, indent=2))
    rows = []
    for name in config['thresholds']:
        for seed in config['seeds']:
            initial = torch.tensor(np.random.default_rng(seed).uniform(-3, 3, 2), dtype=torch.float32)
            for variant in VARIANTS:
                parameter = torch.nn.Parameter(initial.clone())
                optimizer = torch.optim.AdamW([parameter], lr=.03, weight_decay=0.)
                filtering = GradientFilter([parameter], seed) if variant.startswith('gaussian') else None
                best, first, trace = float('inf'), None, []
                tick = time.perf_counter()
                for step in range(config['steps']):
                    optimizer.zero_grad(set_to_none=True); loss = objective(name, parameter)
                    if not bool(loss.isfinite()):
                        raise FloatingPointError('Nonfinite optimizer objective')
                    best = min(best, float(loss.detach()))
                    if first is None and best < config['thresholds'][name]:
                        first = step
                    if step % 20 == 0:
                        trace.append({'step': step, 'loss': float(loss.detach()), 'best': best, 'x': parameter.detach().tolist()})
                    loss.backward(); update(optimizer, filtering, variant, step, config['steps'], .03)
                final = float(objective(name, parameter).detach()); best = min(best, final)
                if first is None and final < config['thresholds'][name]:
                    first = config['steps']
                rows.append({'function': name, 'seed': seed, 'variant': variant, 'initial': initial.tolist(),
                             'loss': final, 'best_loss': best, 'first_success_step': first,
                             'success': best < config['thresholds'][name], 'gradient_calls': config['steps'],
                             'function_calls': config['steps']+1, 'seconds': time.perf_counter()-tick, 'trace': trace})
        print('Optimizer function complete:', name, flush=True)
    (ROOT/'functions.json').write_text(json.dumps(rows, indent=2))
    neural = []
    data_rng = np.random.default_rng(901)
    train = torch.tensor(data_rng.normal(size=(2048, 2)), dtype=torch.float32)
    test = torch.tensor(data_rng.normal(size=(2048, 2)), dtype=torch.float32)
    labels = (train[:, 0]*train[:, 1] > 0).long(); truth = (test[:, 0]*test[:, 1] > 0).long()
    for seed in (42, 137, 2026):
        for variant in VARIANTS:
            torch.manual_seed(seed)
            model = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.Tanh(), torch.nn.Linear(32, 2))
            optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=0.)
            filtering = GradientFilter(model.parameters(), seed) if variant.startswith('gaussian') else None
            rng = np.random.default_rng(seed); trace = []; tick = time.perf_counter()
            for step in range(1000):
                ids = rng.integers(2048, size=64)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(train[ids]), labels[ids])
                loss.backward(); update(optimizer, filtering, variant, step, 1000, .003)
                if step % 100 == 0:
                    trace.append({'step': step, 'loss': float(loss.detach())})
            with torch.no_grad():
                pred = model(test); accuracy = float((pred.argmax(-1) == truth).float().mean())
                ce = float(torch.nn.functional.cross_entropy(pred, truth))
            neural.append({'variant': variant, 'seed': seed, 'accuracy': accuracy, 'loss': ce,
                           'seconds': time.perf_counter()-tick, 'gradient_calls': 1000, 'trace': trace})
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'seed': seed, 'variant': variant,
                        'filter_states': filtering.states if filtering is not None else None,
                        'data_rng': rng.bit_generator.state, 'step': 1000}, ROOT/f'xor_{variant}_seed{seed}.pt')
    (ROOT/'neural.json').write_text(json.dumps(neural, indent=2))
    (ROOT/'runtime.json').write_text(json.dumps({'device': 'cpu',
        'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'memory_scope': 'cumulative peak of the process executing all function and MLP trials; not per-trial peak',
        'function_trials': len(rows), 'neural_trials': len(neural)}, indent=2))
    print('Independent optimizer tests complete.', flush=True)


if __name__ == '__main__':
    run()
