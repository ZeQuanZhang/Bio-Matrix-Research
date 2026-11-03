"""Training-only constant-answer controls; run after the frozen comparison finishes."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from evaluations.repaired.data import ROOT, QA
from scale.engine import normalize, qa_score
from v07.common import write


def main():
    root = PROJECT/'results/repaired/local333k_corrected_20261103'
    protocol = json.loads((root/'protocol.json').read_text())
    assert json.loads((root/'registry.json').read_text())['status'] == 'completed', 'Wait for the frozen comparison to finish'
    declaration = json.loads((root/'prior_control_protocol.json').read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == declaration['source_sha256']
    train, test = QA(ROOT, 'train'), QA(ROOT, 'test')
    answers = Counter(normalize(r['answer_text']) for r in train.rows)
    constant = min(answers, key=lambda a: (-answers[a], a))
    first_tokens = Counter(int(v[r['prefix_tokens']]) for v, r in zip(train.values, train.rows))
    first = min(first_tokens, key=lambda t: (-first_tokens[t], t))
    ids = [i for b in protocol['test_generation'].values() for i in b['indices']]
    results = {}
    for name, text in [('empty_answer', ''), ('most_frequent_training_answer', constant)]:
        values = [qa_score(text, r['reference_answers']) for r in test.rows]
        results[name] = dict(full_test_questions=len(values),
            full_test_em=sum(x[0] for x in values)/len(values), full_test_f1=sum(x[1] for x in values)/len(values),
            generation_sample_questions=len(ids), prediction_sha256=hashlib.sha256(text.encode()).hexdigest())
        total = sum(b['population'] for b in protocol['test_generation'].values())
        for k, index in [('em', 0), ('f1', 1)]:
            results[name]['same_generation_strata_weighted_'+k] = sum(
                b['population']*sum(values[i][index] for i in b['indices'])/len(b['indices'])
                for b in protocol['test_generation'].values())/total
    results['most_frequent_training_first_token'] = dict(
        first_accuracy=sum(int(v[r['prefix_tokens']]) == first for v, r in zip(test.values, test.rows))/len(test.rows),
        training_frequency=first_tokens[first]/len(train.rows))
    write(root/'qa_prior_controls.json', dict(status='completed', controls=results,
        selection='Constants derived only from training annotations; no model or test-based tuning',
        note='Supplement declared before final holdout evaluation; separate from the frozen main protocol'))
    write(root/'qa_prior_controls.private.json', dict(constant_answer=constant, first_token_id=first))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
