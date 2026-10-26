"""Reproducible frozen-weight public-text evaluator check; no optimizer or GPU."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from v07.common import Console, digest, file_hash, write
from v07.data import Corpus
from v07.evaluate import evaluate, history_subsets, windows
from v07.model import Config, Network, initialize
from v07.run import load_config


def main():
    torch.set_num_threads(1)
    project = Path(__file__).resolve().parents[3]
    tracked = [project/'v07/model.py', project/'v07/evaluate.py']
    before = {str(p.relative_to(project)): file_hash(p) for p in tracked}
    root = Path(__file__).parent/f'public_evaluation_{digest(before)[:12]}'
    if root.exists():
        raise FileExistsError(f'Preserving previous check: {root}')
    root.mkdir()
    cfg = load_config(); corpus = Corpus()
    selected = windows(corpus, 'test', cfg)
    masks, protocol = history_subsets(corpus, selected, cfg['prefix_tokens'], cfg['target_tokens'])
    write(root/'public_subset_manifest.json', dict(definitions=protocol, windows_sha256=selected['sha256'],
          subsets={name: dict(tokens=int(mask.sum()), windows=int(mask.any(1).sum()),
                             first_16_tokens=int(mask[:, :16].sum())) for name, mask in masks.items()},
          capability_result=False))
    model = Network(Config(vocab_size=corpus.manifest['vocab_size'], width=16, matrix_size=4, modes=2,
                           internal_steps=2)).eval()
    initialize(model, 71)
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    started = time.perf_counter()
    result = evaluate(model, corpus, 'test', cfg | {'test_windows': 1}, torch.device('cpu'),
                      Console(enabled=False), root, interventions=True)
    after = {str(p.relative_to(project)): file_hash(p) for p in tracked}
    assert before == after, 'Source changed during the probe; do not accept this check'
    assert all(torch.equal(weights[name], value) for name, value in model.state_dict().items())
    rows = json.loads((root/'test_predictions.json').read_text())
    assert len({tuple(row['target_tokens']) for row in rows}) == 1
    assert len({tuple(row['input_tokens']) for row in rows}) == 1
    check = dict(status='completed', device='cpu', threads=1, optimizer_updates=0,
                 weights='Initialized frozen seed71; engineering validation only', seconds=time.perf_counter()-started,
                 source_hashes=before, model_config=asdict(model.cfg), conditions=list(result['conditions']),
                 evaluation_sha256=result['evaluation_sha256'], scored_targets_per_condition=cfg['target_tokens'])
    write(root/'check.json', check)
    print(json.dumps(dict(output=str(root), **check), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
