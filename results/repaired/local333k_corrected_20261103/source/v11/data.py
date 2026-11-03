"""Pinned human-annotated SQuAD history QA with a reused WikiText tokenizer."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

from v07.common import digest, file_hash, write


VERSION = 'v11-squad-article-history-qa-2'
REVISION = 'eee5fdbf62f8613a7812b03419e6b29617b74fd1'
REPOSITORY = 'rajpurkar/SQuAD-explorer'
SPLITS = ('train', 'validation', 'test')
SOURCES = {
    'train-v1.1.json': dict(git_blob_sha1='77f1d9a7daee11f7bc293832020f7709f63410ad', bytes=30288272,
                            sha256='3527663986b8295af4f7fcdff1ba1ff3f72d07d61a20f487cb238a6ef92fd955'),
    'dev-v1.1.json': dict(git_blob_sha1='e9a3f913ad1468ebe105b891334ca7b0bc0e2510', bytes=4854279,
                          sha256='95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9'),
}
PAD, BOS, EOS = 0, 1, 2


def normalized(text):
    return ' '.join(text.casefold().split())


def article_id(title):
    return digest(normalized(title.replace('_', ' ')))


def article_split(identifier):
    return 'validation' if int(identifier[:16], 16) % 20 == 0 else 'train'


def _blob_sha(path):
    result = hashlib.sha1(b'blob '+str(path.stat().st_size).encode()+b'\0')
    with path.open('rb') as handle:
        while block := handle.read(1024*1024):
            result.update(block)
    return result.hexdigest()


def _source_file(root, name, allow_download):
    path = root/name
    expected = SOURCES[name]
    if not path.is_file():
        if not allow_download:
            raise FileNotFoundError(f'Official SQuAD file missing: {name}; enable download to prepare it')
        url = f'https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/dataset/{name}'
        temporary = path.with_suffix(path.suffix+'.part')
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as output:
            with tqdm(total=expected['bytes'], desc=f'SQuAD {name}', unit='B', unit_scale=True,
                      mininterval=1., dynamic_ncols=True) as bar:
                while block := response.read(256*1024):
                    output.write(block)
                    bar.update(len(block))
        if (temporary.stat().st_size != expected['bytes'] or _blob_sha(temporary) != expected['git_blob_sha1']
                or file_hash(temporary) != expected['sha256']):
            raise ValueError(f'Official SQuAD download failed pinned hash verification: {name}; .part retained')
        temporary.replace(path)
    if (path.stat().st_size != expected['bytes'] or _blob_sha(path) != expected['git_blob_sha1']
            or file_hash(path) != expected['sha256']):
        raise ValueError(f'Pinned official SQuAD artifact changed: {name}; restore it or choose a fresh directory')
    return path


def _validated_articles(raw, source):
    if not isinstance(raw, dict) or raw.get('version') != '1.1' or not isinstance(raw.get('data'), list):
        raise ValueError(f'{source}: expected official SQuAD v1.1 schema')
    seen_articles, seen_questions = set(), set()
    articles = []
    for article in raw['data']:
        if (not isinstance(article, dict) or not isinstance(article.get('title'), str)
                or not article['title'].strip() or not isinstance(article.get('paragraphs'), list)):
            raise ValueError(f'{source}: malformed article')
        identifier = article_id(article['title'])
        if identifier in seen_articles:
            raise ValueError(f'{source}: duplicate article title')
        seen_articles.add(identifier)
        paragraphs = []
        for paragraph in article['paragraphs']:
            if (not isinstance(paragraph, dict) or not isinstance(paragraph.get('context'), str)
                    or not paragraph['context'].strip() or not isinstance(paragraph.get('qas'), list)):
                raise ValueError(f'{source}: malformed paragraph')
            context, questions = paragraph['context'], []
            for qa in paragraph['qas']:
                if (not isinstance(qa, dict) or not isinstance(qa.get('id'), str) or not qa['id']
                        or not isinstance(qa.get('question'), str) or not qa['question'].strip()
                        or not isinstance(qa.get('answers'), list) or not qa['answers']):
                    raise ValueError(f'{source}: malformed answerable question')
                if qa['id'] in seen_questions:
                    raise ValueError(f'{source}: duplicate question id')
                seen_questions.add(qa['id'])
                for answer in qa['answers']:
                    if (not isinstance(answer, dict) or not isinstance(answer.get('text'), str)
                            or not answer['text'].strip() or isinstance(answer.get('answer_start'), bool)
                            or not isinstance(answer.get('answer_start'), int) or answer['answer_start'] < 0):
                        raise ValueError(f'{source}: malformed answer span')
                    start, text = answer['answer_start'], answer['text']
                    if context[start:start+len(text)] != text:
                        raise ValueError(f'{source}: answer_start/text mismatch for {qa["id"]}')
                questions.append(qa)
            paragraphs.append(dict(context=context, context_id=digest(normalized(context)), qas=questions))
        articles.append(dict(title=article['title'], article_id=identifier, paragraphs=paragraphs))
    return articles


def _no_split_overlap(groups, key):
    sets = {split: {row[key] for row in groups[split]} for split in SPLITS}
    for index, first in enumerate(SPLITS):
        for second in SPLITS[index+1:]:
            if sets[first] & sets[second]:
                raise ValueError(f'QA split leakage: {key} intersects {first}/{second}')


def _local_answer_leak(tokenizer, history, prompt, answer_ids, answer_text, limit):
    local = (history+prompt)[-limit:]
    if answer_ids[0] in local:
        return 'first_answer_token_in_local_context'
    for begin in range(len(local)-len(answer_ids)+1):
        if local[begin:begin+len(answer_ids)] == answer_ids:
            return 'answer_tokens_in_local_context'
    decoded = tokenizer.decode(local, skip_special_tokens=False)
    if normalized(answer_text) in normalized(decoded):
        return 'answer_text_in_decoded_local_context'
    return None


def _query_answer(tokenizer, question, answer):
    prompt = '\nQuestion: '+question+'\nAnswer:'
    encoded = tokenizer.encode(prompt+' '+answer)
    char_start = len(prompt)+1
    positions = [index for index, (_, end) in enumerate(encoded.offsets) if end > char_start]
    if not positions:
        raise ValueError('Answer has no canonical token overlapping its text')
    first = positions[0]
    query, answer_ids = encoded.ids[:first], encoded.ids[first:]
    if tokenizer.decode(answer_ids, skip_special_tokens=False).strip() != answer.strip():
        raise ValueError('Canonical answer tokens do not decode to the annotated answer')
    return query, answer_ids


def _records(articles, tokenizer, length, prefix, context_limit, history_exclusion, source, statistics):
    groups = {split: [] for split in SPLITS}
    sequences = {split: [] for split in SPLITS}
    for article in tqdm(articles, desc=f'编码 SQuAD {source}', unit='文章', mininterval=1., dynamic_ncols=True):
        split = 'test' if source == 'dev-v1.1.json' else article_split(article['article_id'])
        pieces, starts, cursor = [], [], 0
        for paragraph in article['paragraphs']:
            starts.append(cursor)
            pieces.append(paragraph['context'])
            cursor += len(paragraph['context'])+2
        article_text = '\n\n'.join(pieces)
        encoded = tokenizer.encode(article_text)
        token_ids = encoded.ids
        offsets = np.asarray(encoded.offsets, dtype=np.int64)
        for paragraph, paragraph_start in zip(article['paragraphs'], starts):
            for qa in paragraph['qas']:
                statistics[split]['questions_seen'] += 1
                if len(token_ids) < prefix:
                    statistics[split]['article_too_short'] += 1
                    continue
                candidates, failures = [], []
                for answer in qa['answers']:
                    text = answer['text']
                    if normalized(text) in normalized(qa['question']):
                        failures.append('answer_in_question')
                        continue
                    try:
                        prompt, answer_ids = _query_answer(tokenizer, qa['question'], text)
                    except ValueError:
                        failures.append('canonical_answer_boundary_rejected')
                        continue
                    if not answer_ids or prefix+len(prompt)+len(answer_ids)+1 > length+1:
                        failures.append('question_answer_too_long')
                        continue
                    char_start = paragraph_start+answer['answer_start']
                    char_end = char_start+len(text)
                    positions = np.flatnonzero((offsets[:, 1] > char_start) & (offsets[:, 0] < char_end))
                    if (not len(positions) or offsets[positions[0], 0] > char_start
                            or offsets[positions[-1], 1] < char_end):
                        failures.append('unaligned_answer_tokens')
                        continue
                    first, end = int(positions[0]), int(positions[-1])+1
                    lower = max(0, end-(prefix-history_exclusion)+1)
                    upper = min(first, len(token_ids)-prefix)
                    if lower > upper:
                        failures.append('answer_not_remote_in_complete_history')
                        continue
                    # Label-aware crop locations are deterministic and disclosed; no gold labels enter the model.
                    start = lower+int(digest([qa['id'], char_start])[:16], 16) % (upper-lower+1)
                    recent_start = int(offsets[start+prefix-history_exclusion, 0])
                    recent_end = int(offsets[start+prefix-1, 1])
                    if normalized(text) in normalized(article_text[recent_start:recent_end]):
                        failures.append('answer_repeated_in_recent_context')
                        continue
                    leak = _local_answer_leak(tokenizer, token_ids[start:start+prefix], prompt,
                                              answer_ids, text, history_exclusion)
                    if leak:
                        failures.append(leak)
                        continue
                    candidates.append((text, prompt, answer_ids, char_start, char_end, first, end, start))
                if not candidates:
                    statistics[split][failures[0] if failures else 'no_usable_answer'] += 1
                    continue
                text, prompt, answer_ids, char_start, char_end, first, end, start = candidates[0]
                history = token_ids[start:start+prefix]
                answer_at = prefix+len(prompt)
                answer_end = answer_at+len(answer_ids)
                values = [*history, *prompt, *answer_ids, EOS]
                values += [PAD]*(length+1-len(values))
                assert len(values) == length+1 and end-start < prefix-history_exclusion
                assert answer_at-(end-start) > history_exclusion
                record = dict(record_id=digest([source, qa['id'], start]), source=source,
                              article_id=article['article_id'], article_title=article['title'],
                              context_id=paragraph['context_id'], question_id=qa['id'],
                              history_start=start, history_sha256=digest(history),
                              context_answer_start=first-start, context_answer_end=end-start,
                              answer_input_start=answer_at, answer_input_end=answer_end,
                              answer_target_start=answer_at-1, answer_target_end=answer_end-1,
                              answer_start_character=char_start, answer_end_character=char_end,
                              answer_text=text, reference_answers=list(dict.fromkeys(a['text'] for a in qa['answers'])),
                              answer_distance_tokens=answer_at-(end-start),
                              local_answer_text_present=False, local_first_answer_token_present=False,
                              canonical_answer_tokens_sha256=digest(answer_ids),
                              history_exclusion=history_exclusion,
                              question_sha256=digest(qa['question']))
                groups[split].append(record)
                sequences[split].append(np.asarray(values, dtype=np.int32))
                statistics[split]['accepted'] += 1
    return groups, sequences


def _prepared_hashes(root):
    names = [f'{split}_{kind}.{extension}' for split in SPLITS
             for kind, extension in (('sequences', 'npy'), ('records', 'jsonl'))]
    return {name: file_hash(root/name) for name in names}


def prepare_qa(root, corpus, *, length=512, prefix=384, context_limit=128, backbone_layers=2, allow_download=True):
    """Prepare fixed human-QA windows; official dev is a reserved final test split."""
    if (any(isinstance(value, bool) or not isinstance(value, int) for value in (length, prefix, context_limit, backbone_layers))
            or backbone_layers < 1 or not 0 < backbone_layers*context_limit < prefix < length):
        raise ValueError('QA geometry requires positive integers and backbone_layers*context_limit < prefix < length')
    history_exclusion = backbone_layers*context_limit
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tokenizer_file = Path(corpus.root)/'tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    if (tokenizer.get_vocab_size() != corpus.manifest['vocab_size']
            or [tokenizer.token_to_id(name) for name in ('[PAD]', '[BOS]', '[EOS]')] != [PAD, BOS, EOS]):
        raise ValueError('QA requires the unchanged backbone tokenizer and special token IDs')
    protocol = dict(version=VERSION, length=length, prefix=prefix, context_limit=context_limit,
                    backbone_layers=backbone_layers, history_exclusion=history_exclusion,
                    maximum_evidence_end_exclusive=prefix-history_exclusion,
                    tokenizer_sha256=file_hash(tokenizer_file), vocab_size=tokenizer.get_vocab_size(),
                    revision=REVISION, source_pins=SOURCES, preparation_source_sha256=file_hash(__file__))
    sources = {}
    for name in SOURCES:
        path = _source_file(root, name, allow_download)
        sources[name] = dict(url=f'https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/dataset/{name}',
                             bytes=path.stat().st_size, git_blob_sha1=_blob_sha(path), sha256=file_hash(path))
    manifest_file = root/'manifest.json'
    if manifest_file.is_file():
        old = json.loads(manifest_file.read_text())
        if old.get('protocol') != protocol or old.get('sources') != sources:
            raise ValueError('QA preparation protocol changed; preserve it and choose another directory')
        if old.get('prepared_hashes') != _prepared_hashes(root):
            raise ValueError('Prepared QA artifacts changed; restore them or choose another directory')
        return QACorpus(root, old)
    raw = {name: _validated_articles(json.loads((root/name).read_text()), name) for name in SOURCES}
    train_articles = {a['article_id'] for a in raw['train-v1.1.json']}
    test_articles = {a['article_id'] for a in raw['dev-v1.1.json']}
    duplicated_articles = train_articles & test_articles
    contexts = {name: {p['context_id'] for a in rows for p in a['paragraphs']} for name, rows in raw.items()}
    statistics = {split: Counter() for split in SPLITS}
    memberships = defaultdict(set)
    for name, articles in raw.items():
        for article in articles:
            split = 'test' if name == 'dev-v1.1.json' else article_split(article['article_id'])
            for paragraph in article['paragraphs']:
                memberships[paragraph['context_id']].add(split)
    duplicated_contexts = {key for key, splits in memberships.items() if len(splits) > 1}
    overlap_audit = dict(official_train_dev_article_overlap=len(duplicated_articles),
                         official_train_dev_context_overlap=len(contexts['train-v1.1.json'] & contexts['dev-v1.1.json']),
                         cross_split_context_ids_excluded=len(duplicated_contexts),
                         policy='Exclude every cross-split duplicate article/context from all participating splits')
    for name, articles in raw.items():
        kept = []
        for article in articles:
            split = 'test' if name == 'dev-v1.1.json' else article_split(article['article_id'])
            if article['article_id'] in duplicated_articles:
                statistics[split]['duplicate_article_questions_excluded'] += sum(len(p['qas']) for p in article['paragraphs'])
                continue
            paragraphs = []
            for paragraph in article['paragraphs']:
                if paragraph['context_id'] in duplicated_contexts:
                    statistics[split]['duplicate_context_questions_excluded'] += len(paragraph['qas'])
                else:
                    paragraphs.append(paragraph)
            if paragraphs:
                kept.append(article | dict(paragraphs=paragraphs))
        raw[name] = kept
    groups, sequences = {split: [] for split in SPLITS}, {split: [] for split in SPLITS}
    for name, articles in raw.items():
        records, values = _records(articles, tokenizer, length, prefix, context_limit, history_exclusion, name, statistics)
        for split in SPLITS:
            groups[split].extend(records[split]); sequences[split].extend(values[split])
    histories = defaultdict(set)
    for split, records in groups.items():
        for row in records:
            histories[row['history_sha256']].add(split)
    duplicated_histories = {key for key, splits in histories.items() if len(splits) > 1}
    overlap_audit['cross_split_history_ids_excluded'] = len(duplicated_histories)
    for split in SPLITS:
        keep = [index for index, row in enumerate(groups[split]) if row['history_sha256'] not in duplicated_histories]
        statistics[split]['duplicate_history_questions_excluded'] = len(groups[split])-len(keep)
        groups[split] = [groups[split][index] for index in keep]
        sequences[split] = [sequences[split][index] for index in keep]
    for key in ('article_id', 'context_id', 'question_id', 'history_sha256'):
        _no_split_overlap(groups, key)
    if any(not groups[split] for split in SPLITS):
        raise ValueError('QA filtering left an empty split; no cache was marked complete')
    for split in SPLITS:
        values = np.stack(sequences[split])
        temporary = root/f'{split}_sequences.npy.tmp'
        with temporary.open('wb') as handle:
            np.save(handle, values)
        temporary.replace(root/f'{split}_sequences.npy')
        temporary = root/f'{split}_records.jsonl.tmp'
        with temporary.open('w', encoding='utf-8') as handle:
            for row in groups[split]:
                handle.write(json.dumps(row, ensure_ascii=False)+'\n')
        temporary.replace(root/f'{split}_records.jsonl')
    manifest = dict(protocol=protocol, sources=sources,
                    dataset='SQuAD v1.1, human-annotated answer spans on Wikipedia',
                    source_page='https://rajpurkar.github.io/SQuAD-explorer/',
                    license='CC-BY-SA-4.0', tokenizer_policy='Reuse frozen WikiText train-only tokenizer; no QA fitting',
                    splits={split: dict(records=len(groups[split]), articles=len({r['article_id'] for r in groups[split]}),
                                        context_ids_sha256=digest(sorted({r['context_id'] for r in groups[split]}))) for split in SPLITS},
                    filtering={split: dict(counts) for split, counts in statistics.items()},
                    split_overlap_audit=overlap_audit,
                    split_policy='Official train: article-title SHA256 modulo20=0 validation, others train. Official dev reserved as final test.',
                    overlap_checks='No shared article, normalized annotated context, question ID, or exact token history across prepared splits',
                    construction='Contiguous same-article text, concatenating original paragraphs with two newlines. Gold-span-aware deterministic crop; no invented questions/answers.',
                    answer_contract='Answer evidence ends strictly before prefix-layers*context_limit; exact answer and first answer token absent from the effective pre-answer window of layers*context_limit. This conservative window also covers stacked local-attention receptive fields. Labels/masks never enter model input.',
                    tokenization='Question/Answer labels and leading-space answer are jointly BPE-encoded; character offsets select canonical answer tokens. No separately encoded unspaced answer concatenation.',
                    supervision='Answer token targets only; first-answer mask separately returned. Query, context, EOS and PAD carry no loss.',
                    prepared_hashes=_prepared_hashes(root))
    write(manifest_file, manifest)
    return QACorpus(root, manifest)


class QACorpus:
    def __init__(self, root, manifest=None):
        self.root = Path(root)
        self.manifest = manifest or json.loads((self.root/'manifest.json').read_text())
        self.length = self.manifest['protocol']['length']
        self.prefix = self.manifest['protocol']['prefix']
        self.context_limit = self.manifest['protocol']['context_limit']
        self.backbone_layers = self.manifest['protocol']['backbone_layers']
        self.history_exclusion = self.manifest['protocol']['history_exclusion']
        self.records = {split: [json.loads(line) for line in (self.root/f'{split}_records.jsonl').read_text().splitlines()]
                        for split in SPLITS}
        self._sequences = {}

    def sample(self, split, rng, batch):
        if split not in SPLITS or isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
            raise ValueError('Need a named QA split and positive batch size')
        if split not in self._sequences:
            self._sequences[split] = np.load(self.root/f'{split}_sequences.npy', mmap_mode='r')
        indices = rng.integers(0, len(self.records[split]), size=batch)
        rows = [self.records[split][int(index)] for index in indices]
        values = np.asarray(self._sequences[split][indices], dtype=np.int64)
        mask = np.zeros((batch, self.length), dtype=bool)
        first = np.zeros_like(mask)
        for lane, row in enumerate(rows):
            start, end = row['answer_target_start'], row['answer_target_end']
            if not 0 <= start < end <= self.length:
                raise ValueError('Prepared QA answer target interval is invalid')
            mask[lane, start:end] = True
            first[lane, start] = True
        metadata = dict(indices=indices.tolist(), record_ids=[row['record_id'] for row in rows],
                        documents=[row['article_id'] for row in rows],
                        starts=[row['history_start'] for row in rows], question_ids=[row['question_id'] for row in rows],
                        answer_target_starts=[row['answer_target_start'] for row in rows],
                        answer_target_ends=[row['answer_target_end'] for row in rows],
                        first_answer_mask=first, rows=rows,
                        sha256=digest(dict(values=values.tolist(), mask=mask.tolist())))
        return values, mask, metadata
