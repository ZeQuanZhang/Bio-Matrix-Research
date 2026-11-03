"""Full-passage SQuAD, EOS supervision, deduplicated WikiText and coverage scoring."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

from v07.common import PROJECT, digest, file_hash, write
from v07.data import ROOT as OLD_WT, documents, prepare as verify_wiki, Corpus
from v11.data import SOURCES, REVISION, article_id, article_split, _validated_articles, _query_answer

ROOT = PROJECT/'data/repaired/full_passage_4096'
FORMAT = 'repaired-full-passage-eos-1'
LENGTH, PREFIX_LIMIT, ANSWER_LIMIT = 512, 448, 64


def fingerprints(text):
    words = re.findall(r'\w+', text.casefold())
    return {hashlib.blake2b(' '.join(words[i:i+32]).encode(), digest_size=16).digest()
            for i in range(len(words)-31)}


def qa_example(tokenizer, paragraph, question):
    """Keep every character of the passage; only length limits select examples."""
    context = tokenizer.encode(paragraph['context'])
    answer = question['answers'][0]
    if paragraph['context'][answer['answer_start']:answer['answer_start']+len(answer['text'])] != answer['text']:
        raise ValueError('Official answer span mismatch')
    prompt, labels = _query_answer(tokenizer, question['question'], answer['text'])
    prefix = [1]+context.ids+prompt
    if len(prefix) > PREFIX_LIMIT:
        return None, 'full_passage_prompt_too_long'
    if not 0 < len(labels) <= ANSWER_LIMIT:
        return None, 'answer_too_long'
    positions = [i for i, (a, b) in enumerate(context.offsets)
                 if b > answer['answer_start'] and a < answer['answer_start']+len(answer['text'])]
    if not positions:
        raise ValueError('Answer evidence token alignment failed')
    values = prefix+labels+[2]
    row = dict(question_id=question['id'], context_id=digest(' '.join(paragraph['context'].casefold().split())),
               context_tokens=len(context.ids), prefix_tokens=len(prefix), answer_start=len(prefix)-1,
               answer_end=len(prefix)+len(labels)-1, supervised_end=len(prefix)+len(labels),
               evidence_end=positions[-1]+2, answer_distance=len(prefix)-(positions[-1]+2),
               answer_text=answer['text'], reference_answers=list(dict.fromkeys(a['text'] for a in question['answers'])),
               full_passage=True, values=values)
    return row, None


def validate(root=ROOT):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest.get('format') != FORMAT or manifest.get('status') != 'completed':
        raise ValueError('Incomplete or incompatible corrected data')
    if manifest.get('preparation_source_sha256') != file_hash(__file__):
        raise ValueError('Data preparation source changed; preserve this cache and choose a fresh directory')
    for path, sha in manifest['files'].items():
        if Path(path).name != path or file_hash(root/path) != sha:
            raise ValueError(f'Frozen corrected data changed: {path}')
    return manifest


def prepare(root=ROOT, stage=lambda text: None):
    root = Path(root)
    if (root/'manifest.json').exists():
        return validate(root)
    root.mkdir(parents=True, exist_ok=True)
    verify_wiki(OLD_WT)
    wt = {s: documents(pq.read_table(OLD_WT/f'{s}-00000-of-00001.parquet', columns=['text'])['text'].to_pylist())
          for s in ('train', 'validation', 'test')}
    qa = {s: [] for s in wt}
    for filename, expected in SOURCES.items():
        path = PROJECT/'data/v11/squad_remote256'/filename
        if file_hash(path) != expected['sha256']:
            raise ValueError('Pinned official SQuAD changed')
        for a in _validated_articles(json.loads(path.read_text()), filename):
            split = 'test' if filename.startswith('dev-') else article_split(a['article_id'])
            for p in a['paragraphs']:
                qa[split].append((a['article_id'], p))
    stage('构建开发/保留测试指纹；过滤训练来源的32词交叉重合')
    held = set()
    for s in ('validation', 'test'):
        for text in wt[s]+[p['context'] for _, p in qa[s]]:
            held.update(fingerprints(text))
    old_sizes = dict(wiki=len(wt['train']), qa_paragraphs=len(qa['train']))
    wt['train'] = [t for t in wt['train'] if not fingerprints(t) & held]
    qa['train'] = [(a, p) for a, p in qa['train'] if not fingerprints(p['context']) & held]
    stage('仅在清理后的WikiText训练文章拟合4096词BPE')
    tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(wt['train'], trainers.BpeTrainer(vocab_size=4096, min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=['[PAD]', '[BOS]', '[EOS]', '[UNK]'], show_progress=False))
    tokenizer.save(str(root/'tokenizer.json'))
    files = ['tokenizer.json']
    splits, counts, filtering = {}, {}, {}
    for s in wt:
        stage(f'编码 {s}：WikiText完整覆盖与SQuAD完整段落')
        tokens, offsets = [], [0]
        for text in wt[s]:
            tokens.extend([1]+tokenizer.encode(text).ids+[2]); offsets.append(len(tokens))
        for suffix, values in (('tokens', np.asarray(tokens, dtype=np.int32)), ('offsets', np.asarray(offsets, dtype=np.int64))):
            name = f'{s}_{suffix}.npy'; np.save(root/name, values); files.append(name)
        splits[s] = dict(documents=len(wt[s]), tokens=len(tokens))
        records, sequences, excluded = [], [], Counter()
        for a, paragraph in qa[s]:
            for question in paragraph['qas']:
                row, reason = qa_example(tokenizer, paragraph, question)
                if row is None:
                    excluded[reason] += 1; continue
                values = row.pop('values'); sequences.append(values+[0]*(513-len(values)))
                records.append(row | dict(article_id=a))
        if not records:
            raise ValueError('Empty full-passage QA split')
        name = f'{s}_qa.npy'; np.save(root/name, np.asarray(sequences, dtype=np.uint16)); files.append(name)
        name = f'{s}_qa.private.jsonl'
        (root/name).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records)); files.append(name)
        counts[s] = dict(records=len(records), articles=len({r['article_id'] for r in records}))
        filtering[s] = dict(excluded)
    manifest = dict(format=FORMAT, status='completed', preparation_source_sha256=file_hash(__file__), vocab_size=tokenizer.get_vocab_size(), splits=splits,
        qa=counts, filtering=filtering, bos=1, eos=2, source_revision=REVISION,
        source_hashes={name: item['sha256'] for name, item in SOURCES.items()},
        tokenizer_fit='Deduplicated WikiText train only', sequence=512, full_passage_prompt_limit=448,
        maximum_answer_tokens=64, removed_train=dict(wiki_documents=old_sizes['wiki']-len(wt['train']),
            qa_paragraphs=old_sizes['qa_paragraphs']-len(qa['train'])),
        deduplication='Remove train source documents/paragraphs sharing normalized 32-word BLAKE2b-128 fingerprints with either task held-out splits',
        qa_policy='Full original paragraph; no answer-aware crop or local-answer exclusion; answer+EOS targets; official train article hash split; official dev reserved test',
        files={name: file_hash(root/name) for name in files})
    write(root/'manifest.json', manifest)
    return validate(root)


class QA:
    def __init__(self, root, split):
        root = Path(root)
        self.values = np.load(root/f'{split}_qa.npy', mmap_mode='r')
        self.rows = [json.loads(x) for x in (root/f'{split}_qa.private.jsonl').read_text().splitlines()]
        if len(self.values) != len(self.rows):
            raise ValueError('QA metadata mismatch')

    def batch(self, indices):
        rows = [self.rows[int(i)] for i in indices]
        length = max(r['supervised_end']+1 for r in rows)
        values = np.asarray(self.values[indices, :length], dtype=np.int64)
        mask = np.zeros((len(rows), length-1), dtype=bool)
        for i, r in enumerate(rows):
            mask[i, r['answer_start']:r['supervised_end']] = True
            if values[i, r['supervised_end']] != 2:
                raise ValueError('EOS objective lost alignment')
        return values, mask, rows


def lm_windows(corpus, split):
    """Every target after document BOS is scored once, with up to 256 preceding tokens."""
    offsets = corpus.offsets[split]
    for doc, (left, right) in enumerate(zip(offsets[:-1], offsets[1:])):
        for first in range(int(left)+1, int(right), 256):
            end = min(int(right), first+256)
            start = max(int(left), first-256)
            values = np.zeros(513, dtype=np.int64)
            values[:end-start] = corpus.tokens[split][start:end]
            mask = np.zeros(512, dtype=bool)
            mask[first-start-1:end-start-1] = True
            yield values, mask, int(doc)
