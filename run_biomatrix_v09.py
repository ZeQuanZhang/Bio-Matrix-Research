"""PyCharm entry: native Mac GPU verification and all paired engineering cases."""
import os
import sys

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '4')

DEVICE = 'auto'
NODES = [32, 128, 512]
SEQUENCE_TOKENS = 64
BATCH_SIZE = 2
WARMUP = 6
REPEATS = 3


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv += ['--device', DEVICE, '--nodes', *map(str, NODES),
                     '--sequence-tokens', str(SEQUENCE_TOKENS), '--batch', str(BATCH_SIZE),
                     '--warmup', str(WARMUP), '--repeats', str(REPEATS)]
    from v09.benchmark import main
    main()
