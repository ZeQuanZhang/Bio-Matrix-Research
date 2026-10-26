"""Complete P2/P3 screening seeds after the initial serial queue and diagnostics."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .run import ROOT, cases, write, train, select_rates, now


def plan(cfg):
    result = []
    for seed in (23, 37):
        for stage in ('capacity', 'iterations', 'topology', 'public'):
            for condition in cases(stage, cfg):
                c = condition.copy(); c['seed'] = seed
                c['id'] = c['id'].rsplit('_s', 1)[0]+f'_s{seed}'
                result.append(c)
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--dry-run', action='store_true'); args = p.parse_args()
    cfg = json.loads(Path('configs/v03/base.json').read_text()); conditions = plan(cfg)
    budget = {'registered_utc': now(), 'conditions': len(conditions), 'updates': sum(c['steps'] for c in conditions),
              'episodes': sum(c['steps']*cfg['batch_size'] for c in conditions), 'cases': conditions,
              'rule': 'All initial P2/P3 conditions, same frozen protocol; add seeds 23/37, no test-score selection.'}
    write(ROOT/'repeat_plan.json', budget)
    if args.dry_run:
        print(json.dumps(budget, indent=2)); return
    write(ROOT/'repeat_state.json', {'status': 'waiting_for_initial_postprocess', 'conditions': len(conditions)})
    try:
        while True:
            status = json.loads((ROOT/'postprocess_state.json').read_text())
            if status['status'] == 'completed':
                break
            if status['status'] in ('failed', 'blocked_by_training_condition'):
                raise RuntimeError(f'Initial queue needs repair: {status}')
            time.sleep(45)
        marker = ROOT/'initial_screen_snapshot.json'
        if not marker.exists():
            source_figures = ROOT/'figures'; archived_figures = ROOT/'initial_screen_figures'
            shutil.copytree(source_figures, archived_figures, dirs_exist_ok=True)
            report = Path('docs/V03_FINDINGS.md').read_text().replace(str(source_figures.resolve()), str(archived_figures.resolve()))
            Path('docs/V03_INITIAL_SCREEN_FINDINGS.md').write_text(report)
            shutil.copy2(ROOT/'summary.json', ROOT/'initial_screen_summary.json')
            write(marker, {'status': 'completed', 'utc': now(), 'formal_training_conditions': 81,
                           'report': 'docs/V03_INITIAL_SCREEN_FINDINGS.md', 'figures': str(archived_figures)})
        statuses = json.loads((ROOT/'conditions.json').read_text())
        for c in conditions:
            statuses.setdefault(c['id'], {'condition': c, 'status': 'not_executed', 'registration': 'P2/P3 repeat amendment'})
        write(ROOT/'conditions.json', statuses)
        rates = select_rates(ROOT, cfg)
        for i, c in enumerate(conditions):
            c['lr'] = rates[c['model']]
            write(ROOT/'repeat_state.json', {'status': 'running', 'index': i+1, 'conditions': len(conditions), 'condition': c['id']})
            train(c, cfg, ROOT)
        for module in ('v03.diagnose', 'v03.verify', 'v03.report'):
            write(ROOT/'repeat_state.json', {'status': 'postprocessing', 'module': module})
            subprocess.run([sys.executable, '-u', '-m', module], check=True)
    except KeyboardInterrupt:
        write(ROOT/'repeat_state.json', {'status': 'aborted', 'reason': 'Interrupted; safe checkpoints retained'}); raise
    except Exception as error:
        write(ROOT/'repeat_state.json', {'status': 'failed', 'error': repr(error)}); raise
    write(ROOT/'repeat_state.json', {'status': 'completed', 'conditions': len(conditions), 'utc': now()})


if __name__ == '__main__':
    main()
