"""PyCharm: right-click this file -> Run. Settings below apply without CLI arguments."""
import json
from pathlib import Path
import sys

# PyCharm 使用本项目 .venv/bin/python。建议勾选 Emulate terminal in output console。
# 首次可将 MODE 改成 "smoke" 检查环境；正式配对使用 "paired"。
MODE = 'paired'                       # paired / single / smoke / plan
SCALES = [32, 128, 512]
SEEDS = [11, 23, 37]
VARIANTS = ['graph', 'full', 'matrix', 'probability', 'fourier']
TRAIN_STEPS = 1200
DEVICE = 'auto'                        # 本机优先 MPS；也支持 cuda / cpu
READ_PATHS = ['shared', 'separated']    # 先检查记忆，再检验跨节点传播；两者均完整执行
OUTPUT_DIRECTORY = 'results/v04/continuous'


def main():
    from v04.run import execute, main as command_line
    if len(sys.argv) > 1:
        command_line(); return
    if MODE not in ('paired', 'single', 'smoke', 'plan'):
        raise ValueError('MODE must be paired, single, smoke or plan')
    project = Path(__file__).resolve().parent
    cfg = json.loads((project/'configs/v04/base.json').read_text())
    cfg.update(nodes=SCALES, seeds=SEEDS, variants=VARIANTS, train_steps=TRAIN_STEPS,
               device=DEVICE, read_paths=READ_PATHS)
    root = project/OUTPUT_DIRECTORY
    if MODE == 'single':
        cfg.update(nodes=[SCALES[0]], seeds=[SEEDS[0]], variants=['full'], read_paths=[READ_PATHS[0]])
    if MODE == 'smoke':
        cfg.update(train_steps=2, internal_steps=2, nodes=[32], seeds=[11], variants=['graph', 'full'],
                   read_paths=['shared'], development_quartets=1, test_quartets=1, eval_every=2, checkpoint_every=1)
        root = project/'results/v04/pycharm_smoke'
    execute(cfg, root, dry_run=MODE == 'plan')


if __name__ == '__main__':
    main()
