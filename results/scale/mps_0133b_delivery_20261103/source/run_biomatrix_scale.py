"""PyCharm entry: inspect/verify locally; prepare/profile/train on the new GPU."""
import sys
from pathlib import Path
from scale.run import main

PROJECT=Path(__file__).resolve().parent
MODE='inspect'             # inspect / verify / prepare / profile / train
DEVICE='auto'              # New GPU: cuda. Mac acceptance: mps.
PRECISION='auto'           # CUDA BF16; Mac FP32.
DATA_ROOT=PROJECT/'data/scale/fineweb_edu_100bt_gpt2'
CONFIG=PROJECT/'configs/scale/v3_0133b.json'
RESUME=None

if __name__=='__main__':
    args=sys.argv[1:]
    if not args:
        args=['--mode',MODE,'--device',DEVICE,'--precision',PRECISION,'--data-root',str(DATA_ROOT),'--config',str(CONFIG)]
        if RESUME is not None:args+=['--resume',str(RESUME)]
    main(args)
