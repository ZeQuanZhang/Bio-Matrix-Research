"""PyCharm local Mac entry: v1.1 staged memory training, in-place 1 Hz progress."""
import os
import sys

os.environ['TQDM_NROWS'] = '24'
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '4')

MODE = 'train'  # train / verify
DEVICE = 'auto'  # Apple Silicon selects native MPS; expired A800 is never used.
CPT_TARGETS = 65_536
PT_UPDATES = 128
RESUME = None  # Existing results/v11/... directory or condition/last.pt

if __name__ == '__main__':
    from v11.train import main
    arguments = ['--mode', MODE, '--device', DEVICE]
    if RESUME is not None:
        arguments += ['--resume', str(RESUME)]
    elif '--resume' not in sys.argv[1:]:
        arguments += ['--cpt-tokens', str(CPT_TARGETS), '--pt-updates', str(PT_UPDATES)]
    main(arguments+sys.argv[1:])
