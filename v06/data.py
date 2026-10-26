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
VERSION = 'v06-wikitext2-doc-bpe-1'
ROOT = PROJECT/'data/v06/wikitext2'
BOS, EOS = 1, 2


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
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
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
    manifest_path = root/'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old['version'] != VERSION or old['requested_vocab'] != vocab_size or old['raw'] != raw:
            raise ValueError('Prepared data protocol changed; choose another DATA_DIRECTORY')
        for name, expected in old['prepared_hashes'].items():
            if not (root/name).exists() or file_hash(root/name) != expected:
                raise ValueError(f'Prepared artifact damaged: {name}; restore it or use a fresh data directory')
        return old
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--vocab-size', type=int, default=4096)
    args = parser.parse_args(); prepare(args.root, args.vocab_size)
