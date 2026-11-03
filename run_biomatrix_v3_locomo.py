"""PyCharm: one frozen-checkpoint LoCoMo evaluation, with in-place progress."""
import os
import sys

os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('OMP_NUM_THREADS','4')

DEVICE='auto'
SEED=11
BATCH=16
MAX_NEW_TOKENS=50
RESUME=None  # results/locomo/v3_...; completed batches are reused.

if __name__=='__main__':
    from evaluations.locomo.run import main
    args=['--device',DEVICE,'--seed',str(SEED),'--batch',str(BATCH),'--max-new-tokens',str(MAX_NEW_TOKENS)]
    if RESUME is not None:args+=['--resume',str(RESUME)]
    main(args+sys.argv[1:])
