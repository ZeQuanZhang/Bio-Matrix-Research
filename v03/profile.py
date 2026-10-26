"""P0 audit, numerical checks, data validation and synchronized MPS resource profiling."""
import hashlib
import io
import json
import platform
import resource
import subprocess
import time
import unittest
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import validate, collate, example
from .graph import statistics
from .model import Config, Network
from . import test_v03


def sync():
    torch.mps.synchronize()


def main():
    root = Path('results/v03/P0'); root.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO()
    tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_v03))
    (root/'tests.log').write_text(stream.getvalue())
    if not tests.wasSuccessful():
        raise RuntimeError('CPU implementation checks failed; see tests.log')
    validated = validate(root)
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS required, no CPU training fallback')
    torch.set_num_threads(4)
    environment = {'python': platform.python_version(), 'torch': str(torch.__version__), 'platform': platform.platform(),
                   'git_branch': subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
                   'mps_available': True, 'mps_cpu_fallback_environment': __import__('os').environ.get('PYTORCH_ENABLE_MPS_FALLBACK', 'unset'),
                   'hardware': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string', 'hw.memsize'], text=True).splitlines(),
                   'first_round_findings_sha256': hashlib.sha256(Path('matrix_lab/docs/FINDINGS.md').read_bytes()).hexdigest()}
    (root/'environment.json').write_text(json.dumps(environment, indent=2))
    # Check CPU/MPS forward and gradients on the exact recurrent path.
    torch.manual_seed(31)
    cpu = Network(Config(nodes=32, iterations=3)); gpu = Network(cpu.cfg).to('mps'); gpu.load_state_dict(cpu.state_dict())
    batch = collate([example(np.random.default_rng(i), 8, 2) for i in range(2)])
    args = {k: v for k, v in batch.items() if k != 'target'}
    a, _ = cpu(**args); b, _ = gpu(**{k: v.to('mps') for k, v in args.items()})
    torch.testing.assert_close(a, b.cpu(), atol=2e-4, rtol=2e-4)
    F.cross_entropy(a, batch['target']).backward(); F.cross_entropy(b, batch['target'].to('mps')).backward()
    max_gradient_error = 0.
    for p, q in zip(cpu.parameters(), gpu.parameters()):
        torch.testing.assert_close(p.grad, q.grad.cpu(), atol=3e-4, rtol=3e-3)
        max_gradient_error = max(max_gradient_error, float((p.grad-q.grad.cpu()).abs().max()))
    del cpu, gpu, a, b; torch.mps.empty_cache()
    rows = []
    for n in (32, 128, 512):
        for kind in ('C1', 'C2', 'C3'):
            torch.manual_seed(11); model = Network(Config(model=kind, nodes=n)).to('mps')
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            x = collate([example(np.random.default_rng(i), 32, 3, .8) for i in range(16)])
            x = {k: v.to('mps') for k, v in x.items()}; times = []; current = []; driver = []
            for step in range(8):
                sync(); tick = time.perf_counter(); optimizer.zero_grad(set_to_none=True)
                logits, state = model(**{k: v for k, v in x.items() if k != 'target'})
                loss = F.cross_entropy(logits, x['target']); loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step(); sync()
                assert bool(loss.isfinite()) and bool(norm.isfinite())
                if step >= 3:
                    times.append(time.perf_counter()-tick)
                    current.append(torch.mps.current_allocated_memory()); driver.append(torch.mps.driver_allocated_memory())
            model.eval()
            with torch.no_grad():
                sync(); tick = time.perf_counter()
                for _ in range(5):
                    model(**{k: v for k, v in x.items() if k != 'target'})
                sync(); latency = (time.perf_counter()-tick)/5
            row = {'model': kind, 'nodes': n, 'batch': 16, 'records': x['records'].shape[1],
                   'warmup_steps': 3, 'measured_steps': 5, 'train_step_seconds': times,
                   'median_train_step_seconds': float(np.median(times)), 'inference_batch_seconds': latency,
                   'sampled_tensor_bytes_max': max(current), 'sampled_driver_bytes_max': max(driver),
                   'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   'ledger': model.ledger(x['records'].shape[1]), 'graph': statistics(model.neighbors.cpu())}
            rows.append(row)
            (root/'profile.json').write_text(json.dumps(rows, indent=2))
            print(json.dumps({k: row[k] for k in ('model', 'nodes', 'median_train_step_seconds', 'sampled_driver_bytes_max')}), flush=True)
            del model, optimizer, x, logits, state, loss, norm; torch.mps.empty_cache()
    result = {'status': 'completed', 'cpu_tests': tests.testsRun, 'mps_forward_backward_checked': True,
              'max_cpu_mps_gradient_absolute_error': max_gradient_error, 'dataset_worlds_disjoint': validated['worlds_disjoint'],
              'profile_conditions': len(rows), 'micro_batch_size_selected': 16,
              'memory_note': 'Sampled current tensor and driver values are distinct; not additive and not instantaneous peaks.'}
    (root/'result.json').write_text(json.dumps(result, indent=2)); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
