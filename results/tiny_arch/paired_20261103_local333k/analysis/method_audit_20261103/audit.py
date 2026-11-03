"""Read-only audit of frozen data and results; no model training or checkpoint writes."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import zlib

PROJECT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from v07.common import digest, file_hash
from v07.data import documents

OUT = Path(__file__).resolve().parent
NEW = OUT.parents[1]
OLD = PROJECT / 'results/v3/paired_20261103T094249_111075'
WT = PROJECT / 'data/v06/wikitext2'
QA = PROJECT / 'data/v11/squad_remote256'
SPLITS = ('train', 'validation', 'test')
NAMES = ('trans_local', 'trans_full', 'v3_static', 'v3_dynamic', 'qwen35_tiny', 'deepseek4_tiny')
read = lambda p: json.loads(p.read_text())
protocol = read(NEW / 'protocol.json')
assert all(file_hash(PROJECT / p) == sha for p, sha in protocol['sources'].items())
wm, qm = read(WT / 'manifest.json'), read(QA / 'manifest.json')
assert digest(wm) == protocol['data_manifest_sha256']
assert digest(qm) == protocol['qa_manifest_sha256']
for root, manifest in ((WT, wm), (QA, qm)):
    assert all(file_hash(root / p) == sha for p, sha in manifest['prepared_hashes'].items())
assert all(file_hash(WT / f'{s}-00000-of-00001.parquet') == v['sha256'] for s, v in wm['raw'].items())
assert all(file_hash(QA / p) == v['sha256'] for p, v in qm['sources'].items())
tokenizer = Tokenizer.from_file(str(WT / 'tokenizer.json'))
wt = {s: documents(pq.read_table(WT / f'{s}-00000-of-00001.parquet', columns=['text'])['text'].to_pylist()) for s in SPLITS}
rows = {s: [json.loads(x) for x in (QA / f'{s}_records.jsonl').read_text().splitlines()] for s in SPLITS}
seq = {s: np.load(QA / f'{s}_sequences.npy', mmap_mode='r') for s in SPLITS}
result = dict(status='completed', source_files_unchanged=len(protocol['sources']), prepared_file_hashes='all verified',
              method='Read-only; exact normalized document/context checks and verified exact 32-word overlap; not semantic deduplication.')
result['split_sizes'] = dict(wikitext={s: {k: wm['splits'][s][k] for k in ('documents', 'tokens')} for s in SPLITS},
                            squad=qm['splits'], filtering=qm['filtering'])
result['qa_integrity'] = {}
for s in SPLITS:
    records, values = rows[s], seq[s]
    assert len(records) == len(values)
    bad_decode = bad_span = wrong_eos = 0
    for row, v in zip(records, values):
        a, b = row['answer_input_start'], row['answer_input_end']
        bad_decode += tokenizer.decode(v[a:b].tolist()).strip() != row['answer_text'].strip()
        x, y = row['context_answer_start'], row['context_answer_end']
        bad_span += row['answer_text'].casefold() not in tokenizer.decode(v[x:y].tolist()).casefold()
        wrong_eos += v[b] != 2
        assert (row['answer_target_start'], row['answer_target_end']) == (a-1, b-1)
        assert 0 <= x < y < 128 and row['answer_distance_tokens'] > 256
        assert v[a] not in v[a-256:a]
    assert not (bad_decode or bad_span or wrong_eos)
    result['qa_integrity'][s] = dict(records=len(records), bad_answer_decode=int(bad_decode),
                                   missing_visible_span=int(bad_span), wrong_eos=int(wrong_eos),
                                   answer_distance_min=min(r['answer_distance_tokens'] for r in records),
                                   answer_distance_max=max(r['answer_distance_tokens'] for r in records))
result['split_overlap'] = {}
for i, a in enumerate(SPLITS):
    for b in SPLITS[i+1:]:
        overlap = {k: len({r[k] for r in rows[a]} & {r[k] for r in rows[b]})
                   for k in ('article_id', 'context_id', 'question_id', 'question_sha256', 'history_sha256')}
        result['split_overlap'][f'{a}/{b}'] = overlap
        assert not any(v for k, v in overlap.items() if k != 'question_sha256')
        for normalize in (lambda t: ' '.join(t.split()), lambda t: ' '.join(t.casefold().split())):
            assert not ({normalize(t) for t in wt[a]} & {normalize(t) for t in wt[b]})
meta = protocol['evaluation']
starts = meta['lm']['starts']
val_tokens = np.load(WT / 'validation_tokens.npy')
positions = [p for s in starts for p in range(s+385, s+513)]
counts = Counter(positions)
valqa = [rows['validation'][i] for i in meta['qa']['indices']]
result['validation_sample'] = dict(lm_windows=len(starts), lm_unique_windows=len(set(starts)),
    lm_documents=len(set(meta['lm']['documents'])), lm_scored_targets=len(positions),
    lm_unique_target_positions=len(counts), lm_repeated_target_occurrences=len(positions)-len(counts),
    lm_unique_target_fraction_of_split=len(counts)/len(val_tokens),
    qa_draws=len(valqa), qa_unique_questions=len({r['question_id'] for r in valqa}),
    qa_unique_contexts=len({r['context_id'] for r in valqa}), qa_unique_histories=len({r['history_sha256'] for r in valqa}),
    qa_articles=len({r['article_id'] for r in valqa}),
    qa_fraction_of_prepared_validation=len(valqa)/len(rows['validation']))
# Gold presence is checked above; preserve a separate check of the full annotated passage.
byqid = {r['question_id']: r for r in rows['validation']}
selected = {r['question_id'] for r in valqa}
coverage = {'all_validation': [], 'sampled64': []}
for article in read(QA / 'train-v1.1.json')['data']:
    if not any(q['id'] in byqid for p in article['paragraphs'] for q in p['qas']):
        continue
    offsets = tokenizer.encode('\n\n'.join(p['context'] for p in article['paragraphs'])).offsets
    cursor = 0
    for paragraph in article['paragraphs']:
        end = cursor + len(paragraph['context'])
        for q in paragraph['qas']:
            if q['id'] not in byqid:
                continue
            start = byqid[q['id']]['history_start']
            left, right = offsets[start][0], offsets[start+383][1]
            item = (left <= cursor and right >= end,
                    max(0, min(right, end)-max(left, cursor))/len(paragraph['context']))
            coverage['all_validation'].append(item)
            if q['id'] in selected:
                coverage['sampled64'].append(item)
        cursor = end+2
result['annotated_paragraph_coverage'] = {k: dict(questions=len(v), full_annotated_paragraph=sum(x[0] for x in v),
    mean_character_coverage=float(np.mean([x[1] for x in v])), min_character_coverage=float(min(x[1] for x in v)))
    for k, v in coverage.items()}
assert result['annotated_paragraph_coverage']['sampled64']['questions'] == 64
train_tokens = np.load(WT / 'train_tokens.npy')
freq = np.bincount(train_tokens, minlength=wm['vocab_size']) + 1
logp = np.log(freq / freq.sum())
unigram_nll = float(-logp[val_tokens[positions]].mean())
result['posthoc_unigram_control'] = dict(training='Full frozen WikiText train counts plus Laplace alpha=1; no QA or held-out fitting',
    scope='Post-hoc diagnostic, sees more train targets than neural models; not a matched-budget competitor',
    validation_targets=len(positions), nll=unigram_nll, ppl=float(np.exp(unigram_nll)))
result['training_coverage'] = {}
result['learning_stages'] = {}
result['bio_history_interventions'] = {}
for name in NAMES:
    root = OLD if name in NAMES[:4] else NEW
    stages = {}
    for filename in ('initial_validation.json', 'cpt_validation.json', 'result.json'):
        values = []
        for seed in (11, 23, 37):
            item = read(root / f'{name}_s{seed}' / filename)
            item = item.get('validation', item)
            values.append(dict(seed=seed, lm_ppl=item['lm']['warm_ppl'], qa_nll=item['qa']['warm_nll'],
                               first_accuracy=item['qa']['first_answer_accuracy']))
        stages[filename] = dict(per_seed=values, lm_ppl_mean=float(np.mean([v['lm_ppl'] for v in values])),
                                qa_nll_mean=float(np.mean([v['qa_nll'] for v in values])))
    result['learning_stages'][name] = stages
    if name in ('v3_static', 'v3_dynamic'):
        effects = {}
        for task in ('lm', 'qa'):
            per_condition = {}
            for seed in (11, 23, 37):
                conditions = read(root / f'{name}_s{seed}/result.json')['validation'][task]['conditions']
                for key, item in conditions.items():
                    per_condition.setdefault(key, []).append(item['nll']-conditions['warm']['nll'])
            effects[task] = {key: float(np.mean(values)) for key, values in per_condition.items()}
        result['bio_history_interventions'][name] = effects
for seed in (11, 23, 37):
    metrics = [json.loads(x) for x in (OLD / f'trans_full_s{seed}/metrics.jsonl').read_text().splitlines()]
    covered = np.zeros(len(train_tokens), dtype=bool)
    sampled_frequencies = np.zeros(wm['vocab_size'], dtype=np.int64)
    qa_indices, lm_starts = [], []
    rng = np.random.default_rng(protocol['training']['data_seed'] + seed)
    for m in metrics:
        if m['stage'] == 'cpt':
            for start in m['sample']['starts']:
                covered[start+1:start+513] = True
                sampled_frequencies += np.bincount(train_tokens[start+1:start+513], minlength=wm['vocab_size'])
                lm_starts.append(start)
            offsets = np.load(WT / 'train_offsets.npy')
            cumulative = np.maximum(np.diff(offsets)-512, 0).cumsum()
            draws = rng.integers(int(cumulative[-1]), size=4)
        else:
            indices = rng.integers(0, len(rows['train']), size=4)
            assert indices.tolist() == m['sample']['indices']
            qa_indices.extend(indices.tolist())
    matched = 0
    for name in NAMES[4:]:
        other = [json.loads(x) for x in (NEW / f'{name}_s{seed}/metrics.jsonl').read_text().splitlines()]
        assert [m['sample_sha256'] for m in other] == [m['sample_sha256'] for m in metrics]
        matched += len(other)
    sampled = Counter(int(seq['train'][i][rows['train'][i]['answer_input_start']]) for i in qa_indices)
    assert sampled_frequencies.sum() == 1048576
    sampled_logp = np.log((sampled_frequencies+1)/(sampled_frequencies.sum()+wm['vocab_size']))
    unigram_ppl = float(np.exp(-sampled_logp[val_tokens[positions]].mean()))
    result['training_coverage'][str(seed)] = dict(lm_windows=len(lm_starts), lm_unique_windows=len(set(lm_starts)),
        lm_drawn_targets=1048576, lm_unique_target_positions=int(covered.sum()),
        lm_unique_fraction_of_train=float(covered.mean()), qa_draws=len(qa_indices), qa_unique_records=len(set(qa_indices)),
        qa_fraction_of_prepared_train=len(set(qa_indices))/len(rows['train']),
        train_first_answer_token_types=len(sampled), validation_first_answer_seen_in_training=sum(int(seq['validation'][i][rows['validation'][i]['answer_input_start']]) in sampled for i in meta['qa']['indices']),
        unigram_ppl_same_cpt_target_draws_alpha1=unigram_ppl,
        paired_new_update_hashes=matched)

def words(text):
    return re.findall(r'\w+', text.casefold())

def grams(tokens, n=32):
    if len(tokens) < n:
        return
    base, mask = 1000003, (1 << 64)-1
    power = pow(base, n-1, 1 << 64)
    ids = [zlib.crc32(w.encode()) for w in tokens]
    h = 0
    for value in ids[:n]:
        h = (h*base+value)&mask
    yield h, 0
    for i in range(n, len(ids)):
        h = ((h-ids[i-n]*power)*base+ids[i])&mask
        yield h, i-n+1

def overlaps(sources, targets):
    target_words = [words(t) for t in targets]
    wanted = {h for w in target_words for h, _ in grams(w)}
    found = {}
    for text in sources:
        w = words(text)
        for h, i in grams(w):
            if h in wanted:
                found.setdefault(h, set()).add(tuple(w[i:i+32]))
    hits = [i for i, w in enumerate(target_words) if any(tuple(w[j:j+32]) in found.get(h, ()) for h, j in grams(w))]
    return dict(targets=len(targets), targets_with_verified_exact_32word_overlap=len(hits), matching_target_indices=hits)

# Check the actual cropped histories, including the chosen 64, across task pipelines.
qa_val_history = [tokenizer.decode(v[:384].tolist()) for v in seq['validation']]
qa_train_history = [tokenizer.decode(v[:384].tolist()) for v in seq['train']]
result['cross_task_overlap'] = dict(
    wiki_train_to_qa_validation_histories=overlaps(wt['train'], qa_val_history),
    qa_train_to_wiki_validation_documents=overlaps(qa_train_history, wt['validation']),
    qa_train_to_qa_validation_histories=overlaps(qa_train_history, qa_val_history),
    wiki_train_to_wiki_validation_documents=overlaps(wt['train'], wt['validation']))
hits = set(result['cross_task_overlap']['wiki_train_to_qa_validation_histories']['matching_target_indices'])
result['cross_task_overlap']['sampled_qa_histories_with_32word_match_in_wiki_train'] = len(hits & set(meta['qa']['indices']))
hits_qa = set(result['cross_task_overlap']['qa_train_to_qa_validation_histories']['matching_target_indices'])
result['cross_task_overlap']['sampled_qa_histories_with_32word_match_in_qa_train'] = len(hits_qa & set(meta['qa']['indices']))
hits_lm = set(result['cross_task_overlap']['wiki_train_to_wiki_validation_documents']['matching_target_indices'])
result['cross_task_overlap']['sampled_lm_documents_with_32word_match_in_wiki_train'] = len(hits_lm & set(meta['lm']['documents']))
assert all(file_hash(PROJECT / p) == sha for p, sha in protocol['sources'].items())
(OUT / 'evidence.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k: v for k, v in result.items() if k not in ('learning_stages', 'split_sizes')}, ensure_ascii=False, indent=2))
