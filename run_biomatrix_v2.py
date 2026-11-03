"""PyCharm entry: phase-addressed block memory vs fully trained Transformers."""
import os
import sys

os.environ['TQDM_NROWS']='24'
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('OMP_NUM_THREADS','4')

MODE='train'  # train / verify
DEVICE='auto'  # native Apple Silicon MPS
CPT_TARGETS=131_072
PT_UPDATES=512
SEEDS='11,23,37'
RESUME=None  # results/v2/paired_... or condition/last.pt

if __name__=='__main__':
    from v2.train import main
    args=['--mode',MODE,'--device',DEVICE]
    if RESUME is not None:args+=['--resume',str(RESUME)]
    elif '--resume' not in sys.argv[1:]:
        args+=['--cpt-tokens',str(CPT_TARGETS),'--pt-updates',str(PT_UPDATES),'--seeds',SEEDS]
    main(args+sys.argv[1:])
