"""PyCharm: standalone V3 vector nodes, paired against plain Transformers."""
import os
import sys

os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('OMP_NUM_THREADS','4')

MODE='train'  # train / verify
DEVICE='auto'  # Native Mac MPS when available; no remote connection.
CPT_TARGETS=1_048_576
PT_UPDATES=512
SEEDS='11,23,37'
RESUME=None  # results/v3/paired_...; incompatible with older checkpoints.

if __name__=='__main__':
    from v3.train import main
    args=['--mode',MODE,'--device',DEVICE]
    if RESUME is not None: args+=['--resume',str(RESUME)]
    elif '--resume' not in sys.argv[1:]:
        args+=['--cpt-tokens',str(CPT_TARGETS),'--pt-updates',str(PT_UPDATES),'--seeds',SEEDS]
    main(args+sys.argv[1:])
