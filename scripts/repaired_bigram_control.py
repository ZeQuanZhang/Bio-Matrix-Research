"""Same-target count bigram diagnostic, without neural training or test-based tuning."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from evaluations.repaired.data import ROOT, validate
from v07.data import Corpus
from v07.common import write


def main():
    root = PROJECT/'results/repaired/local333k_corrected_20261103'
    assert json.loads((root/'registry.json').read_text())['status'] == 'completed'
    declaration = json.loads((root/'bigram_control_protocol.json').read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == declaration['source_sha256']
    validate(ROOT); corpus = Corpus(ROOT)
    protocol = json.loads((root/'protocol.json').read_text()); c = protocol['training']; results = {}
    test = corpus.tokens['test']; mask = np.ones(len(test), dtype=bool); mask[corpus.offsets['test'][:-1]] = False
    positions = np.flatnonzero(mask); before, targets = test[positions-1], test[positions]
    summary = json.loads((root/'summary.json').read_text())
    for seed in c['seeds']:
        counts = np.zeros((4096, 4096), dtype=np.int32); rng = np.random.default_rng(32000+seed)
        for _ in range(c['cpt_tokens']//(c['batch']*512)):
            values, _, _ = corpus.sample('train', rng, c['batch'], 512)
            np.add.at(counts, (values[:, :-1].ravel(), values[:, 1:].ravel()), 1)
        assert counts.sum() == c['cpt_tokens']
        unigram = (counts.sum(0)+1)/(counts.sum()+4096)
        row_count = counts.sum(1); types = np.maximum(1, np.count_nonzero(counts, axis=1))
        probabilities = (counts[before, targets]+types[before]*unigram[targets])/(row_count[before]+types[before])
        assert bool(np.isfinite(probabilities).all()) and bool((probabilities > 0).all())
        nll = float(-np.log(probabilities).mean()); uni_ppl = float(np.exp(-np.log(unigram[targets]).mean()))
        assert abs(uni_ppl-summary['unigram_same_cpt_targets_test_ppl'][str(seed)]) < 1e-9
        results[str(seed)] = dict(status='completed', nll=nll, ppl=float(np.exp(nll)),
            test_targets=len(targets), training_target_draws=int(counts.sum()),
            observed_transition_types=int(np.count_nonzero(counts)), dense_count_table_bytes=counts.nbytes,
            same_target_unigram_ppl=uni_ppl)
    write(root/'bigram_control.json', dict(status='completed', seeds=results,
        ppl_mean=float(np.mean([r['ppl'] for r in results.values()])),
        formula='p(v|u)=[count(u,v)+max(1,distinct_followers(u))*p_unigram(v)]/[count(u)+max(1,distinct_followers(u))]',
        fitting='Exactly the same CPT token-transition draws as each final neural seed; all smoothing quantities derived from train counts; no test or dev tuning',
        scope='One-token-context diagnostic; dense symbolic count state is not matched to neural parameter or compute budgets. Not a flagship comparison.'))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
