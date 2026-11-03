"""PyCharm: verify first, then adapt local Transformer memory with live progress."""
import os
import sys

os.environ['TQDM_NROWS'] = '24'
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '4')

MODE = 'train'  # train / verify; train starts the finite local pilot after checks.
DEVICE = 'auto'
TOKENS_PER_CONDITION = 65_536
RESUME = None  # Existing results/v1/... directory or its condition/last.pt.


if __name__ == '__main__':
    from v1.train import main
    arguments = ['--mode', MODE, '--device', DEVICE]
    if RESUME is not None:
        arguments += ['--resume', str(RESUME)]
    elif '--resume' not in sys.argv[1:]:
        arguments += ['--tokens', str(TOKENS_PER_CONDITION)]
    main(arguments+sys.argv[1:])
