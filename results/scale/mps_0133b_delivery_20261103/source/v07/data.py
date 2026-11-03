"""Pinned official WikiText-2, train-only BPE, document-disjoint sampling."""
import argparse
import json
from pathlib import Path
import re
import urllib.request

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .common import Console, PROJECT, digest, file_hash, write

DATASET = 'Salesforce/wikitext'
REVISION = '6231e49f19a707241d6b84d9cff60a3a86b85a85'
SUBSET = 'wikitext-2-raw-v1'
HASHES = {
    'train': '49280f3dd993cbdfd322186194133d5d1935f81e1c91064c239f2a70ebb02701',
    'validation': 'a10eb5a42c4264c45dd38bfef4da6a74ddef5fc22cb419e1a89d54c4cc6d748f',
    'test': '3ee89cd6a2ab912afd5d01e98867b46a67d9ec7a9eca0910e5e4c3cdd4cc1925',
}
# The frozen v0.6 tokenizer/splits are the comparison data, not a new v0.7 fit.
VERSION = 'v06-wikitext2-doc-bpe-1'
ROOT = PROJECT/'data/v06/wikitext2'
BOS, EOS = 1, 2
CHANGE_CUES = (' was renamed', ' was replaced', ' became', ' no longer', ' previously', ' formerly')


def documents(lines):
    result, current = [], []
    for line in lines:
        if re.fullmatch(r'= [^=]+ =', line.strip()) and current:
            result.append(''.join(current)); current = []
        current.append(line)
    if current:
        result.append(''.join(current))
    return [x for x in result if x.strip()]


def prepare(root=ROOT, vocab_size=4096, console=None):
    root = Path(root)
    manifest_path = root/'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if (old['version'] != VERSION or old['requested_vocab'] != vocab_size or
                old['revision'] != REVISION or old['tokenizer_fit_split'] != 'train' or
                {k: v['sha256'] for k, v in old['raw'].items()} != HASHES):
            raise ValueError('Prepared data protocol changed; choose another DATA_DIRECTORY')
        files = old['prepared_hashes'] | {f'{split}-00000-of-00001.parquet': sha for split, sha in HASHES.items()}
        for name, expected in files.items():
            if not (root/name).is_file() or file_hash(root/name) != expected:
                raise ValueError(f'Prepared artifact damaged: {name}; restore it or use a fresh data directory')
        if console:
            console.stage('数据校验', '复用并校验 v0.6 冻结切分与 train-only BPE；不重新下载或拟合')
        return old
    if root.resolve() == ROOT.resolve():
        raise FileNotFoundError('Frozen v0.6 manifest missing; restore data/v06/wikitext2 or explicitly choose a new --data-root')
    root.mkdir(parents=True, exist_ok=True)
    ui = console or Console()
    texts, raw = {}, {}
    for split, expected in HASHES.items():
        name = f'{split}-00000-of-00001.parquet'; target = root/name
        url = f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{SUBSET}/{name}'
        if not target.exists() or file_hash(target) != expected:
            temporary = target.with_suffix('.part')
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as output:
                ui.stage('下载公开数据', split, total=int(response.headers.get('Content-Length', 0)) or None, unit='B')
                while block := response.read(256*1024):
                    output.write(block); ui.update(len(block))
            if file_hash(temporary) != expected:
                raise ValueError(f'{split}: SHA-256 mismatch; partial file retained')
            temporary.replace(target)
        raw[split] = dict(url=url, sha256=expected, bytes=target.stat().st_size)
        texts[split] = documents(pq.read_table(target, columns=['text'])['text'].to_pylist())
    ui.stage('数据校验', '官方三份切分；检查文档交集，分词器仅学习 train')
    hashes = {k: [digest(' '.join(x.split())) for x in values] for k, values in texts.items()}
    for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        if set(hashes[a]) & set(hashes[b]):
            raise ValueError(f'Exact normalized document overlap: {a}/{b}')
    tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    ui.stage('训练分词器', f'仅 train；词表上限 {vocab_size}')
    tokenizer.train_from_iterator(texts['train'], trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2, show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=['[PAD]', '[BOS]', '[EOS]', '[UNK]']))
    tokenizer.save(str(root/'tokenizer.json'))
    stats = {}; prepared = ['tokenizer.json']
    for split, values in texts.items():
        offsets, tokens = [0], []
        ui.stage('编码数据', split, total=len(values), unit='文档')
        for text in values:
            tokens.extend([BOS, *tokenizer.encode(text).ids, EOS]); offsets.append(len(tokens)); ui.update()
        for suffix, array in (('tokens', np.asarray(tokens, dtype=np.int32)), ('offsets', np.asarray(offsets, dtype=np.int64))):
            name = f'{split}_{suffix}.npy'
            with (root/(name+'.tmp')).open('wb') as handle:
                np.save(handle, array)
            (root/(name+'.tmp')).replace(root/name); prepared.append(name)
        stats[split] = dict(documents=len(values), tokens=len(tokens), document_hashes=hashes[split])
    # Repeated paragraphs are disclosed separately; they are not independent documents.
    paragraphs = {k: {digest(' '.join(p.split())) for x in values for p in x.split('\n') if len(p.strip()) >= 100}
                  for k, values in texts.items()}
    overlaps = {f'{a}/{b}': len(paragraphs[a] & paragraphs[b]) for a, b in
                (('train', 'validation'), ('train', 'test'), ('validation', 'test'))}
    manifest = dict(version=VERSION, dataset=DATASET, subset=SUBSET, revision=REVISION,
                    license=['CC-BY-SA-3.0', 'GFDL'], requested_vocab=vocab_size,
                    vocab_size=tokenizer.get_vocab_size(), raw=raw, splits=stats,
                    exact_document_overlap=0, repeated_long_paragraphs=overlaps,
                    tokenizer_fit_split='train', bos=BOS, eos=EOS,
                    prepared_hashes={name: file_hash(root/name) for name in prepared})
    write(manifest_path, manifest); ui.close()
    return manifest


class Corpus:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        self.tokens = {s: np.load(self.root/f'{s}_tokens.npy', mmap_mode='r') for s in HASHES}
        self.offsets = {s: np.load(self.root/f'{s}_offsets.npy', mmap_mode='r') for s in HASHES}
        self._change_proxies = {}

    def change_proxy_spans(self, split='train'):
        """Uncurated lexical change cues for exposure auditing, never rule labels."""
        if split in self._change_proxies:
            return self._change_proxies[split]
        tokenizer = Tokenizer.from_file(str(self.root/'tokenizer.json'))
        tokens, offsets, found = self.tokens[split], self.offsets[split], {}
        for cue in CHANGE_CUES:
            pattern = np.asarray(tokenizer.encode(cue).ids)
            candidates = np.flatnonzero(tokens == pattern[0])
            candidates = candidates[candidates+len(pattern) <= len(tokens)]
            matched = candidates[np.all(tokens[candidates[:, None]+np.arange(len(pattern))] == pattern, axis=1)]
            for position in matched:
                doc = int(np.searchsorted(offsets, position, side='right')-1)
                if position+len(pattern) > offsets[doc+1]:
                    continue
                begin, end = max(int(offsets[doc]), int(position)-24), min(int(offsets[doc+1]), int(position)+len(pattern)+24)
                found[int(position)] = dict(document=doc, offset=int(position-offsets[doc]), cue=cue.strip(),
                                            token_ids=pattern.tolist(), excerpt=tokenizer.decode(tokens[begin:end].tolist()))
        rows = [found[key] for key in sorted(found)]
        self._change_proxies[split] = rows
        return rows

    def sample(self, split, rng, batch, length):
        """Uniform valid start positions; each sample stays inside one official document."""
        offsets = self.offsets[split]
        counts = np.maximum(np.diff(offsets)-length, 0)
        cumulative = counts.cumsum()
        if not len(cumulative) or cumulative[-1] < 1:
            raise ValueError('No document long enough for requested window')
        draws = rng.integers(int(cumulative[-1]), size=batch)
        docs = np.searchsorted(cumulative, draws, side='right')
        previous = np.where(docs > 0, cumulative[np.maximum(docs-1, 0)], 0)
        starts = offsets[docs]+draws-previous
        values = np.stack([self.tokens[split][start:start+length+1] for start in starts])
        return values.astype(np.int64), docs.tolist(), starts.tolist()


class ContinuousStream:
    """Continuous sessions: full A documents, B intrusions, then unseen A continuation.

    Source records define only supervision/diagnostics. They are never given to the
    network or used to suppress local feedback at an unmarked context switch.
    """
    def __init__(self, corpus, batch, length, data_seed, seed, session_tokens=4096,
                 intrusion_tokens=64, intrusion_interval=512, marked_probability=.5,
                 reset_policy='session'):
        self.corpus = corpus
        self.batch, self.length = int(batch), int(length)
        if min(self.batch, self.length) < 1 or session_tokens < self.length or session_tokens % self.length:
            raise ValueError('Positive stream chunks must divide the finite session length')
        if min(intrusion_tokens, intrusion_interval) < 0 or not 0 <= marked_probability <= 1:
            raise ValueError('Invalid intrusion schedule or marked-boundary probability')
        if reset_policy not in ('session', 'document'):
            raise ValueError('Unknown reset policy')
        self.protocol = dict(version='v07-content-context-stream-3', batch=self.batch, length=self.length,
                             data_seed=int(data_seed), seed=int(seed), split='train',
                             session_tokens=int(session_tokens), intrusion_tokens=int(intrusion_tokens),
                             intrusion_interval=int(intrusion_interval), marked_probability=float(marked_probability),
                             reset_policy=reset_policy)
        proxies = corpus.change_proxy_spans('train') if hasattr(corpus, 'change_proxy_spans') else []
        self.proxy_positions = {}
        for row in proxies:
            self.proxy_positions.setdefault(row['document'], set()).add(row['offset'])
        self.protocol['natural_change_proxy_sha256'] = digest(proxies)
        offsets, tokens = corpus.offsets['train'], corpus.tokens['train']
        self.content_bounds = {}
        for doc in range(len(offsets)-1):
            size = int(offsets[doc+1]-offsets[doc]); lower, upper = 0, size
            if size and int(tokens[offsets[doc]]) == BOS:
                lower += 1
            if upper > lower and int(tokens[offsets[doc+1]-1]) == EOS:
                upper -= 1
            if upper > lower:
                self.content_bounds[doc] = (lower, upper)
        self.protocol['content_bounds_sha256'] = digest(self.content_bounds)
        self.eligible = np.asarray(list(self.content_bounds), dtype=np.int64)
        if not len(self.eligible):
            raise ValueError('No train document contains real text content')
        self.rng = np.random.default_rng(np.random.SeedSequence([data_seed, seed]))
        self.order = self.rng.permutation(self.eligible).tolist()
        self.order_cursor, self.epoch, self.resets, self.targets, self.inputs, self.change_proxy_hits = 0, 0, 0, 0, 0, 0
        self.switches = dict(independent=0, intrusion=0, return_to_A=0, marked=0, unmarked=0)
        self.lanes = [dict(base=-1, position=0, intrusion=-1, intrusion_position=0,
                           intrusion_left=0, since_intrusion=0, next_intrusion=self._interval(),
                           returning=False, markers=[], first=True, pending=None, age=0)
                      for _ in range(self.batch)]

    def _interval(self):
        interval = self.protocol['intrusion_interval']
        return int(self.rng.integers(max(1, interval//2), max(2, 3*interval//2+1))) if interval else 0

    def state_dict(self):
        from copy import deepcopy
        return deepcopy(dict(protocol=self.protocol, order=self.order, order_cursor=self.order_cursor,
                             epoch=self.epoch, resets=self.resets, targets=self.targets, inputs=self.inputs,
                             change_proxy_hits=self.change_proxy_hits,
                             switches=self.switches, lanes=self.lanes, rng=self.rng.bit_generator.state))

    def load_state_dict(self, value):
        from copy import deepcopy
        if value['protocol'] != self.protocol or sorted(value['order']) != sorted(self.eligible.tolist()):
            raise ValueError('Stream protocol/document order mismatch')
        if (len(value['lanes']) != self.batch or not 0 <= value['order_cursor'] <= len(self.eligible) or
                min(value['epoch'], value['resets'], value['targets'], value['inputs']) < 0):
            raise ValueError('Invalid stream cursor')
        for lane in value['lanes']:
            if not 0 <= lane['age'] < self.protocol['session_tokens'] or lane['age'] % self.length:
                raise ValueError('Invalid session age')
            for doc_name, pos_name in (('base', 'position'), ('intrusion', 'intrusion_position')):
                doc, position = lane[doc_name], lane[pos_name]
                if doc != -1 and (doc not in self.content_bounds or not self.content_bounds[doc][0] <= position <= self.content_bounds[doc][1]):
                    raise ValueError('Invalid stream lane/document position')
        for name in ('order', 'order_cursor', 'epoch', 'resets', 'targets', 'inputs', 'change_proxy_hits', 'switches', 'lanes'):
            setattr(self, name, deepcopy(value[name]))
        self.rng.bit_generator.state = deepcopy(value['rng'])

    def _new_document(self):
        if self.order_cursor == len(self.order):
            self.order = self.rng.permutation(self.eligible).tolist()
            self.order_cursor = 0; self.epoch += 1
        doc = int(self.order[self.order_cursor]); self.order_cursor += 1
        return doc

    def _boundary(self, lane, kind, old_doc=None, new_doc=None):
        self.switches[kind] += 1
        marked = bool(self.rng.random() < self.protocol['marked_probability'])
        self.switches['marked' if marked else 'unmarked'] += 1
        if marked:
            lane['markers'] = [dict(token=EOS, document=-1, offset=-1, kind=kind+'_EOS', boundary=True, change_proxy=False),
                               dict(token=BOS, document=-1, offset=-1, kind=kind+'_BOS', boundary=False, change_proxy=False)]
            if kind == 'independent':
                offsets, tokens = self.corpus.offsets['train'], self.corpus.tokens['train']
                end = self.content_bounds[old_doc][1]
                begin = self.content_bounds[new_doc][0]-1
                if end < offsets[old_doc+1]-offsets[old_doc] and tokens[offsets[old_doc]+end] == EOS:
                    lane['markers'][0].update(document=old_doc, offset=end, boundary=False)
                if begin >= 0 and tokens[offsets[new_doc]+begin] == BOS:
                    lane['markers'][1].update(document=new_doc, offset=begin)
                lane['markers'][1]['boundary'] = True
        return not marked

    def _record(self, lane):
        offsets = self.corpus.offsets['train']
        if lane['markers']:
            return lane['markers'].pop(0)
        boundary = False
        if lane['returning']:
            lane['intrusion'] = -1; lane['intrusion_position'] = 0; lane['returning'] = False
            boundary = self._boundary(lane, 'return_to_A')
            if lane['markers']:
                return lane['markers'].pop(0)
        if lane['base'] == -1 or lane['position'] >= self.content_bounds[lane['base']][1]:
            old_doc = lane['base']
            lane['base'] = self._new_document()
            lane['position'] = self.content_bounds[lane['base']][0]
            if not lane['first']:
                boundary = self._boundary(lane, 'independent', old_doc=old_doc, new_doc=lane['base'])
                if lane['markers']:
                    return lane['markers'].pop(0)
            else:
                boundary = True; lane['first'] = False
        if lane['intrusion_left'] == 0 and lane['intrusion'] == -1:
            enabled = self.protocol['intrusion_tokens'] > 0 and self.protocol['intrusion_interval'] > 0
            # Observe some A before an intrusion; then return to its still-unseen next token.
            if enabled and len(self.eligible) > 1 and lane['position']-self.content_bounds[lane['base']][0] >= 32 and lane['since_intrusion'] >= lane['next_intrusion']:
                candidates = self.eligible[self.eligible != lane['base']]
                doc = int(self.rng.choice(candidates))
                lower, upper = self.content_bounds[doc]
                count = min(self.protocol['intrusion_tokens'], upper-lower)
                lane['intrusion'], lane['intrusion_position'] = doc, int(self.rng.integers(lower, upper-count+1))
                lane['intrusion_left'] = count; lane['since_intrusion'] = 0
                lane['next_intrusion'] = self._interval()
                boundary = self._boundary(lane, 'intrusion')
                if lane['markers']:
                    return lane['markers'].pop(0)
        if lane['intrusion_left']:
            doc, position, kind = lane['intrusion'], lane['intrusion_position'], 'intrusion'
            lane['intrusion_position'] += 1; lane['intrusion_left'] -= 1
            if lane['intrusion_left'] == 0:
                lane['returning'] = True
        else:
            doc, position, kind = lane['base'], lane['position'], 'base'
            lane['position'] += 1; lane['since_intrusion'] += 1
        absolute = int(offsets[doc]+position)
        return dict(token=int(self.corpus.tokens['train'][absolute]), document=int(doc), offset=int(position),
                    kind=kind, boundary=boundary, change_proxy=position in self.proxy_positions.get(int(doc), ()))

    @staticmethod
    def _spans(records):
        spans = []
        for index, record in enumerate(records):
            if (spans and spans[-1]['document'] == record['document'] and spans[-1]['kind'] == record['kind'] and
                    record['offset'] == spans[-1]['offset']+index-spans[-1]['begin'] and not record['boundary'] and
                    record['change_proxy'] == spans[-1]['change_proxy']):
                spans[-1]['end'] = index+1
            else:
                spans.append(dict(begin=index, end=index+1, document=record['document'], offset=record['offset'],
                                  kind=record['kind'], boundary=record['boundary'], change_proxy=record['change_proxy']))
        return spans

    def next(self):
        values = np.empty((self.batch, self.length+1), dtype=np.int64)
        mask = np.zeros((self.batch, self.length), dtype=bool)
        reset = np.zeros_like(mask)
        ages, sources, documents, starts = [], [], [], []
        for index, lane in enumerate(self.lanes):
            records = [lane['pending']] if lane['pending'] is not None else []
            while len(records) < self.length+1:
                records.append(self._record(lane))
            lane['pending'] = records[-1]
            self.change_proxy_hits += sum(record['change_proxy'] for record in records[:-1])
            values[index] = [record['token'] for record in records]
            for t, (current, following) in enumerate(zip(records, records[1:])):
                mask[index, t] = (current['document'] >= 0 and current['document'] == following['document'] and
                                  following['offset'] == current['offset']+1)
                reset[index, t] = self.protocol['reset_policy'] == 'document' and current['boundary']
            reset[index, 0] |= lane['age'] == 0
            ages.append(lane['age']); lane['age'] = (lane['age']+self.length) % self.protocol['session_tokens']
            sources.append(self._spans(records)); documents.append(records[0]['document'])
            starts.append(int(self.corpus.offsets['train'][records[0]['document']]+records[0]['offset']) if records[0]['document'] >= 0 else -1)
        self.targets += int(mask.sum()); self.inputs += self.batch*self.length; self.resets += int(reset.sum())
        counts = mask.sum(1).tolist()
        metadata = dict(documents=documents, starts=starts, sources=sources, age_start=ages,
                        age_end=[age+self.length for age in ages], valid_tokens=counts,
                        visible_tokens=[self.length]*self.batch, reset=reset.any(1).tolist(),
                        reset_events=int(reset.sum()), cumulative_resets=self.resets,
                        cumulative_targets=self.targets, cumulative_inputs=self.inputs, epoch=self.epoch,
                        cumulative_natural_change_proxy_hits=self.change_proxy_hits,
                        switches=self.switches.copy(),
                        content_sha256=digest(dict(values=values.tolist(), mask=mask.tolist())),
                        sha256=digest(dict(values=values.tolist(), mask=mask.tolist(), reset=reset.tolist())))
        return values, mask, reset, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--vocab-size', type=int, default=4096)
    args = parser.parse_args(); prepare(args.root, args.vocab_size)
