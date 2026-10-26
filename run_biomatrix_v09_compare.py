"""PyCharm: run this file to start/resume the local MPS comparison queue."""
import os
import sys

# PyCharm can start a 0×0 PTY; tqdm otherwise caches a negative row limit.
os.environ['TQDM_NROWS'] = '24'

from v07.common import PROJECT
from v09.experiment import main


RUN_DIRECTORY = PROJECT/'results/v09/comparison_no_v08'
TOKENS_PER_JOB = 12_000_000
NODES = (32, 128, 512)
SEEDS = (11, 23, 37)
CONDITIONS = ('content_only', 'v09_dense', 'v09_sparse50', 'transformer')
EVALUATE_ON_COMPLETE = True
EVALUATION_WINDOWS = 64
RETRY_FAILED = False


if __name__ == '__main__':
    arguments = ['--output', str(RUN_DIRECTORY), '--tokens', str(TOKENS_PER_JOB),
                 '--nodes', *map(str, NODES), '--seeds', *map(str, SEEDS),
                 '--conditions', *CONDITIONS, '--evaluation-windows', str(EVALUATION_WINDOWS)]
    if not EVALUATE_ON_COMPLETE: arguments.append('--skip-evaluation')
    if RETRY_FAILED: arguments.append('--retry-failed')
    main(arguments+sys.argv[1:])
