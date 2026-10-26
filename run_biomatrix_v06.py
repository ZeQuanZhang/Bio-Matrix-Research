"""PyCharm: right-click Run; enable Emulate terminal in output console."""
import os
from pathlib import Path
import subprocess
import sys

# Set before importing torch/tokenizers; keep Apple's MPS memory protection enabled.
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '4')

MODE = 'paired'                 # paired / single / smoke / preflight / plan / prepare
STAGE = 'core'                  # core / mechanism / capacity / iterations / topology / credit / all
SCALES = [32, 128, 512]
SEEDS = [11, 23, 37]
DEVICE = 'auto'                 # Apple Silicon -> mps; unavailable MPS raises an error
TRAIN_STEPS = 4000
OUTPUT_DIRECTORY = 'results/v06/public_lm'
DATA_DIRECTORY = 'data/v06/wikitext2'
KEEP_AWAKE = True              # Mac: prevent idle sleep while this process is alive


def main():
    from v06.common import PROJECT
    from v06.run import execute, load_config, main as command_line, smoke_config
    if len(sys.argv) > 1:
        command_line(); return
    cfg = load_config() | dict(stage=STAGE, nodes=SCALES, seeds=SEEDS, device=DEVICE, train_steps=TRAIN_STEPS)
    data_root = PROJECT/DATA_DIRECTORY
    if MODE == 'prepare':
        from v06.data import prepare
        prepare(data_root, cfg['vocab_size']); return
    if MODE == 'preflight':
        from v06.profile import preflight
        preflight(cfg, PROJECT/'results/v06/preflight', data_root); return
    if MODE == 'smoke':
        cfg = smoke_config(cfg); root = PROJECT/'results/v06/pycharm_smoke'
    elif MODE == 'single':
        cfg.update(stage='core', nodes=[SCALES[0]], seeds=[SEEDS[0]], variants=['full'])
        root = PROJECT/OUTPUT_DIRECTORY
    elif MODE in ('paired', 'plan'):
        root = PROJECT/OUTPUT_DIRECTORY
    else:
        raise ValueError(f'Unknown MODE: {MODE}')
    execute(cfg, root, data_root, dry_run=MODE == 'plan')


if __name__ == '__main__':
    awake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) if KEEP_AWAKE and sys.platform == 'darwin' else None
    try:
        main()
    except Exception as error:
        print(f'\n[v0.6 停止] {error}\n详细记录见输出目录的 error.txt / events.jsonl。', file=sys.stderr)
        raise
    finally:
        if awake is not None:
            awake.terminate(); awake.wait()
