"""Post-hoc descriptive output audit; no training, ranking, or model selection."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2]
sys.path.insert(0, str(PROJECT))
from scale.engine import normalize
from v07.common import write


def distribution(texts, weights):
    counts = Counter(texts)
    mode = min(counts, key=lambda s: (-counts[s], s))
    return dict(distinct_outputs=len(counts), modal_count=counts[mode],
                modal_share=sum(w for t, w in zip(texts, weights) if t == mode),
                modal_sha256=hashlib.sha256(mode.encode()).hexdigest())


def main():
    assert normalize('The U.S.') == 'us'
    check = distribution(['x', 'y', 'x'], [.2, .3, .5])
    assert check['modal_count'] == 2 and abs(check['modal_share']-.7) < 1e-12
    assert json.loads((ROOT/'registry.json').read_text())['status'] == 'completed'
    protocol = json.loads((ROOT/'protocol.json').read_text())
    populations = {k: b['population'] for k, b in protocol['test_generation'].items()}
    total = sum(populations.values())
    results = {}
    for name in protocol['models']:
        rows = []
        for seed in protocol['training']['seeds']:
            files = {m: ROOT/f'heldout/{name}_s{seed}_generation_{m}.private.json'
                     for m in ('warm', 'unrelated_context')}
            generations = {m: json.loads(p.read_text())['records'] for m, p in files.items()}
            warm, unrelated = generations.values()
            assert len(warm) == 192 and [r['question_id'] for r in warm] == [r['question_id'] for r in unrelated]
            labels = [str(128 if r['distance'] <= 128 else 256 if r['distance'] <= 256 else 10000) for r in warm]
            counts = Counter(labels)
            assert set(counts.values()) == {64}
            weights = [populations[k]/(total*counts[k]) for k in labels]
            assert abs(sum(weights)-1.) < 1e-12
            raw = {m: [r['prediction'] for r in records] for m, records in generations.items()}
            normalized = {m: list(map(normalize, texts)) for m, texts in raw.items()}
            row = dict(seed=seed, questions=192,
                input_sha256={m: hashlib.sha256(p.read_bytes()).hexdigest() for m, p in files.items()},
                distributions={m: dict(raw=distribution(raw[m], weights), normalized=distribution(normalized[m], weights)) for m in files})
            for label, texts in [('raw', raw), ('normalized', normalized)]:
                row[label+'_changed_fraction'] = sum(w for a, b, w in zip(texts['warm'], texts['unrelated_context'], weights) if a != b)
            rows.append(row)
        results[name] = dict(per_seed=rows,
            mean_normalized_changed_fraction=sum(r['normalized_changed_fraction'] for r in rows)/len(rows),
            mean_warm_normalized_modal_share=sum(r['distributions']['warm']['normalized']['modal_share'] for r in rows)/len(rows))
    write(ROOT/'analysis/generation_dependence.json', dict(status='completed', conditions=results,
        scope='Post-hoc descriptive audit of every completed model and seed. Uses fixed heldout outputs only; no training, checkpoint changes, or selection. Population weights match primary generation metrics.',
        mode_selection='Most frequent output by unweighted sample count, lexicographic tie break; modal_share uses population weights.',
        privacy='Prediction and answer strings remain local; only hashes, counts, and fractions are exported.',
        normalization='Existing scale.engine.normalize used by the primary QA metric'))


if __name__ == '__main__':
    main()
