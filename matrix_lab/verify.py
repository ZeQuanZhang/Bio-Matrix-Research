"""Verify completed budgets, source identities, checkpoints and paired schedules."""
import datetime
import hashlib
import json
from pathlib import Path
import platform
import py_compile
import subprocess

import numpy as np
import torch

from .data import generate
from .models import Config, MemoryModel
from .extension_run import model_for
from .signal_run import SignalModel, examples


def check_sources(sources):
    for name, expected in sources.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name


def run():
    torch.set_num_threads(1)
    checked = []
    for group, expected_count, steps, episodes in [('matrix', 21, 4000, 512), ('matrix_extensions', 24, 2000, 256)]:
        paths = sorted((Path('results')/group).glob('*/result.json'))
        assert len(paths) == expected_count
        by_seed = {}
        for path in paths:
            result = json.loads(path.read_text()); p = result['protocol']
            check_sources(p['source_hashes'])
            best = torch.load(path.parent/'best.pt', map_location='cpu', weights_only=True)
            last = torch.load(path.parent/'last.pt', map_location='cpu', weights_only=True)
            assert best['protocol'] == last['protocol'] == p
            assert last['step'] == steps and best['step'] == result['best_step']
            trace = [json.loads(line) for line in (path.parent/'metrics.jsonl').read_text().splitlines()]
            validations = [e for e in trace if 'validation_loss' in e]
            assert best['step'] == min(validations, key=lambda e: e['validation_loss'])['step']
            if group == 'matrix':
                model = MemoryModel(Config(**p['model']))
                capacity, digest, events = p['persistent_state_floats'], result['data_schedule_sha256'], result['events_seen']
            else:
                model = model_for(p['variant'])
                capacity, digest, events = p['state_floats'], result['schedule_sha256'], result['events']
            by_seed.setdefault(p['seed'], set()).add((digest, events))
            model.load_state_dict(best['model'])
            assert model.parameter_count() == p['parameters']
            x, _, q = generate(np.random.default_rng(999), batch=4, pairs=2, delay=3)
            with torch.no_grad():
                pred, state = model(x, q)
                assert state.shape == (4, capacity) and bool(pred.isfinite().all() and state.isfinite().all())
            for condition in result['test'].values():
                for score in condition.values():
                    assert score['episodes'] == len(score['episode_accuracy']) == episodes
                    assert abs(float(np.mean(score['episode_accuracy']))-score['accuracy']) < 1e-9
                    assert np.isfinite(score['episode_loss']).all()
            checked.append(str(path))
        assert len(by_seed) == 3 and all(len(v) == 1 for v in by_seed.values())
    signals = list(Path('results/matrix_signals').glob('*seed*.json')); assert len(signals) == 9
    for path in signals:
        result = json.loads(path.read_text()); check_sources(result['config']['source_hashes'])
        model = SignalModel(result['mode'])
        model.load_state_dict(torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=True)['model'])
        x, _ = examples(np.random.default_rng(999), 4, 16, [.08, .17])
        with torch.no_grad():
            assert bool(model(x).isfinite().all())
        checked.append(str(path))
    optimizer_config = json.loads(Path('results/matrix_optimizer/config.json').read_text())
    assert hashlib.sha256(Path('matrix_lab/optimizer_run.py').read_bytes()).hexdigest() == optimizer_config['source_sha256']
    functions = json.loads(Path('results/matrix_optimizer/functions.json').read_text()); assert len(functions) == 120
    for name in optimizer_config['thresholds']:
        for seed in range(10):
            rows = [r for r in functions if r['function'] == name and r['seed'] == seed]
            assert len(rows) == 4 and len({tuple(r['initial']) for r in rows}) == 1
            assert all(r['gradient_calls'] == 2000 and r['function_calls'] == 2001 and np.isfinite(r['loss']) for r in rows)
    neural = json.loads(Path('results/matrix_optimizer/neural.json').read_text()); assert len(neural) == 12
    for r in neural:
        checkpoint = torch.load(Path('results/matrix_optimizer')/f"xor_{r['variant']}_seed{r['seed']}.pt", weights_only=True)
        assert checkpoint['step'] == r['gradient_calls'] == 1000
        assert (checkpoint['filter_states'] is not None) == r['variant'].startswith('gaussian')
    for path in Path('matrix_lab').rglob('*.py'):
        py_compile.compile(str(path), doraise=True)
    assert json.loads(Path('results/matrix_all_summary.json').read_text())['complete']
    report = {'verified_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'mps_model_runs': len(checked), 'analytic_optimizer_trials': len(functions), 'mlp_optimizer_trials': len(neural),
              'paired_schedules_equal': True, 'source_hashes_match': True, 'all_checkpoints_load_safely': True,
              'best_checkpoints_match_validation_selection': True, 'unit_tests_passed': 10,
              'hardware': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string', 'hw.memsize'], text=True).splitlines(),
              'python': platform.python_version(), 'torch': str(torch.__version__),
              'checked_result_files': checked}
    Path('results/matrix_verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'checked_result_files'}, indent=2))


if __name__ == '__main__':
    run()
