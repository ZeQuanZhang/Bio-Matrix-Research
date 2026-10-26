"""Saved learning curves; raw JSONL remains the source of truth."""
import argparse
import json
from pathlib import Path

import numpy as np


def plot_training(folder, gate_warmup=400):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    folder = Path(folder)
    rows = [json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
    validation = [json.loads(line) for line in (folder/'validation.jsonl').read_text().splitlines()]
    x = [r['step'] for r in rows]; y = [r['loss'] for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, layout='constrained')
    axes[0].plot(x, y, alpha=.25, label='Train NLL')
    window = min(40, len(rows))
    axes[0].plot(x[window-1:], np.convolve(y, np.ones(window)/window, mode='valid'), label=f'Train MA({window})')
    axes[0].plot([r['step'] for r in validation], [r['conditions']['carry']['nll'] for r in validation], 'o-', label='Validation NLL (hard gates)')
    if 0 < gate_warmup <= max(x):
        axes[0].axvline(gate_warmup, color='gray', linestyle=':', label='End of soft-gate warmup')
    axes[0].set_ylabel('BPE token NLL'); axes[0].legend(fontsize=8)
    axes[1].plot(x, [r['tokens_per_second'] for r in rows], label='Tokens / second')
    axes[1].set_ylabel('Training throughput'); axes[1].set_xlabel('Optimizer update'); axes[1].legend()
    fig.suptitle(folder.name); fig.savefig(folder/'learning_curve.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root', type=Path)
    args = parser.parse_args()
    from .run import report
    value = json.loads((args.root/'conditions.json').read_text()); report(args.root, value['conditions'])
    for record in value['conditions']:
        folder = args.root/record['id']
        if (folder/'validation.jsonl').exists():
            plot_training(folder, value['config']['gate_warmup_steps'])
