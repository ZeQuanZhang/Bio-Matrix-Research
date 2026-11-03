"""Compare evaluation batches on development data only, using independent frozen models."""
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from evaluations.repaired.model import build, NAMES
from evaluations.repaired.data import ROOT, QA
from evaluations.repaired.engine import score_qa, score_lm, generate
from v07.data import Corpus
from v07.common import digest, write
from tokenizers import Tokenizer
from transformers.utils import logging


def main():
    root = PROJECT/'results/repaired/local333k_corrected_20261103'
    # Keep the auxiliary profile inside a validation interval, outside timed production updates.
    while True:
        state = json.loads((root/'registry.json').read_text())
        active = [k for k, v in state['conditions'].items() if v['status'] == 'running']
        if ('固定开发验证' in state['stage'] and active and active[0].endswith('_qa')
                and any(n in active[0] for n in ('v3_', 'deepseek', 'qwen'))
                and time.time()-(root/'registry.json').stat().st_mtime < 5):
            break
        if state['status'] != 'running':
            raise RuntimeError('Training ended before calibration window')
        time.sleep(1.)
    start = time.time(); parent_phase = active[0]
    logging.set_verbosity_error(); torch.set_num_threads(4); device = torch.device('mps')
    protocol = json.loads((root/'protocol.json').read_text()); qa = QA(ROOT, 'validation')
    tokenizer = Tokenizer.from_file(str(ROOT/'tokenizer.json'))
    indices = np.random.default_rng(63001).choice(len(qa.rows), 128, replace=False).tolist()
    corpus = Corpus(ROOT)
    fixture = SimpleNamespace(tokens={'validation': corpus.tokens['validation'][:8193]}, offsets={'validation': np.array([0, 8193])})
    results = {}
    for name in NAMES:
        model = build(name, 101).to(device)
        saved = torch.load(root/f'qa_search_{name}_r2/last.pt', map_location='cpu', weights_only=True)
        assert saved['protocol_sha256'] == digest(protocol); model.load_state_dict(saved['model'])
        q, l, elapsed = {}, {}, {}
        for batch in (16, 64):
            torch.mps.synchronize(); before = time.perf_counter()
            q[batch] = score_qa(model, qa, device, indices, batch)
            l[batch] = score_lm(model, fixture, 'validation', device, batch)
            torch.mps.synchronize(); elapsed[batch] = time.perf_counter()-before
        differences = [abs(a['nll_sum']/a['targets']-b['nll_sum']/b['targets']) for a, b in zip(q[16]['records'], q[64]['records'])]
        assert max(differences) < 2e-5 and abs(l[16]['nll']-l[64]['nll']) < 2e-5
        assert [r['first_correct'] for r in q[16]['records']] == [r['first_correct'] for r in q[64]['records']]
        generated = {}
        for batch in (16, 64):
            g = generate(model, qa, indices[:64], tokenizer, device, batch, limit=4)
            generated[batch] = [r['generated_ids'] for r in g['records']]
        assert generated[16] == generated[64]
        results[name] = dict(status='completed', qa_max_per_question_nll_difference=max(differences),
            lm_nll_difference=abs(l[16]['nll']-l[64]['nll']), first_answer_predictions_agree=True,
            first4_greedy_tokens_identical_on64_questions=True, elapsed_seconds=elapsed,
            approximate_speedup=elapsed[16]/elapsed[64])
        print(name, results[name], flush=True)
        model.to('cpu'); del model; torch.mps.empty_cache()
    final = json.loads((root/'registry.json').read_text())
    write(root/'eval_batch_calibration.json', dict(status='completed', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        split='validation only', production_phase_at_start=parent_phase, production_phase_at_end=final['stage'],
        started_unix=start, ended_unix=time.time(), conditions=results,
        note='Timing is a development calibration under concurrent validation, not a standalone architecture speed benchmark'))


if __name__ == '__main__':
    main()
