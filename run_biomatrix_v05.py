"""PyCharm: right-click -> Run, using this project's .venv/bin/python."""
import json
from pathlib import Path
import sys

MODE = 'paired'                        # paired / single / smoke / plan / preflight
STAGE = 'core'                         # core / capacity / iterations / topology / credit / compute / all
SCALES = [32, 128, 512]
SEEDS = [11, 23, 37]
VARIANTS = ['message', 'full', 'no_fourier', 'single_timescale', 'no_traces',
            'no_consolidation', 'frequency_only', 'always_read']
READ_PATHS = ['shared', 'separated']
TRAIN_STEPS = 1200
DEVICE = 'auto'
OUTPUT_DIRECTORY = 'results/v05/verified_streams'


def main():
    from v05.run import execute, main as command_line
    if len(sys.argv) > 1:
        command_line(); return
    project = Path(__file__).resolve().parent
    cfg = json.loads((project/'configs/v05/base.json').read_text())
    cfg.update(stage=STAGE, nodes=SCALES, seeds=SEEDS, variants=VARIANTS,
               read_paths=READ_PATHS, train_steps=TRAIN_STEPS, device=DEVICE)
    root = project/OUTPUT_DIRECTORY
    if MODE == 'preflight':
        from v05.profile import preflight
        preflight(cfg, project/'results/v05/preflight'); return
    if MODE not in ('paired', 'single', 'smoke', 'plan'):
        raise ValueError('Unknown MODE')
    if MODE == 'single':
        cfg.update(stage='core', nodes=[SCALES[0]], seeds=[SEEDS[0]], variants=['full'], read_paths=[READ_PATHS[0]])
    if MODE == 'smoke':
        cfg.update(stage='core', nodes=[32], seeds=[11], variants=['message', 'full'], read_paths=['shared'],
                   train_steps=2, internal_steps=2, checkpoint_events=False, development_quartets=1,
                   test_quartets=1, eval_every=2, checkpoint_every=1)
        root = project/'results/v05/pycharm_smoke'
    execute(cfg, root, dry_run=MODE == 'plan')


if __name__ == '__main__':
    main()
