"""Bounded FineWeb-Edu text-column reads, benchmark filtering, immutable token packs."""
import hashlib
import json
from pathlib import Path
import random
import re
import shutil

from huggingface_hub import HfFileSystem
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from scale import data as source
from v07.common import PROJECT, digest, file_hash, write
from .evaluate import tasks, protocol


def words(text):
    return re.findall(r'\w+', text.casefold())


def grams(tokens):
    for start in range(len(tokens)-12):
        yield hashlib.blake2b(' '.join(tokens[start:start+13]).encode(), digest_size=16).digest()


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from strings(child)


def benchmark_index(ui, stop):
    index = set(); counts = {}
    for name, task in tasks(ui, stop):
        ui.stage(f'建立评测重合过滤索引 · {name}')
        counts[name] = len(task.eval_docs)
        for row in task.eval_docs:
            if stop.requested:
                raise KeyboardInterrupt
            for text in strings(row):
                index.update(grams(words(text)))
    return index, counts


def validate(root, settings):
    m = json.loads((root/'manifest.json').read_text())
    if m['status'] != 'completed' or m['settings'] != settings:
        raise ValueError('数据缓存协议不同或未完成，请使用对应的数据目录')
    src = json.loads((root/'sources.json').read_text())
    if digest(src) != m['sources_sha256'] or src['dataset_revision'] != source.REVISION:
        raise ValueError('FineWeb source identity changed')
    if file_hash(root/'tokenizer.json') != source.TOKENIZER_SHA256:
        raise ValueError('Tokenizer changed')
    for row in m['files']:
        if Path(row['path']).name != row['path']:
            raise ValueError('Invalid cache path')
        path = root/row['path']
        if path.stat().st_size != 2*row['tokens'] or file_hash(path) != row['sha256']:
            raise ValueError(f'Token pack changed: {path.name}')
    return m


def prepare(root, c, ui, stop):
    root.mkdir(parents=True, exist_ok=True)
    required = c['pretrain_tokens']+c['sequence']*len(c['milestones'])+1
    limits = dict(train=required, validation=c['sequence']*c['validation_sequences']+1)
    settings = dict(format='local-fineweb-1', limits=limits, order_seed=c['data_seed'],
                    benchmark_protocol_sha256=digest(protocol()), overlap_words=13)
    bar = ui.details['data']; bar.total = sum(limits.values())
    if (root/'manifest.json').exists():
        result = validate(root, settings); bar.n = bar.total; bar.postfix = '已校验'
        return result
    if shutil.disk_usage(root).free < 10*2**30+2*sum(limits.values()):
        raise OSError('本地训练至少需要 10 GiB 检查点余量，加上 token 缓存空间')
    ui.stage('校验固定 FineWeb-Edu 来源与 GPT-2 分词器')
    old = PROJECT/'data/scale/fineweb_edu_100bt_gpt2'
    if not (root/'sources.json').exists() and (old/'sources.json').exists():
        saved = json.loads((old/'sources.json').read_text())
        if saved['dataset_revision'] == source.REVISION and file_hash(old/'tokenizer.json') == source.TOKENIZER_SHA256:
            for name in ('sources.json', 'tokenizer.json'):
                shutil.copyfile(old/name, root/name)
    src = source.sources(root)
    if file_hash(root/'tokenizer.json') != source.TOKENIZER_SHA256:
        raise ValueError('Tokenizer hash mismatch')
    index, counts = benchmark_index(ui, stop)
    tokenizer = Tokenizer.from_file(str(root/'tokenizer.json'))
    shards = list(src['shards']); random.Random(c['data_seed']).shuffle(shards)
    fs = HfFileSystem()
    seen = set(); totals = dict.fromkeys(limits, 0)
    audit = dict(scanned=0, duplicate_documents=0, benchmark_overlap_documents=0,
                 benchmark_documents=counts, benchmark_13grams=len(index), source_shards=[])
    handles = {k: (root/f'{k}.bin.part').open('wb') for k in limits}
    try:
        for shard in shards:
            if stop.requested:
                raise KeyboardInterrupt
            ui.stage(f'读取官方文本列 · {Path(shard["path"]).name} · 已编码 {totals["train"]:,}/{required:,}')
            remote = f'datasets/{source.REPO}@{source.REVISION}/{shard["path"]}'
            with fs.open(remote, 'rb', block_size=1024*1024) as reader:
                if reader.size != shard['bytes']:
                    raise ValueError('Pinned shard size mismatch')
                audit['source_shards'].append(shard)
                for batch in pq.ParquetFile(reader, pre_buffer=False).iter_batches(batch_size=32, columns=['text']):
                    if stop.requested:
                        raise KeyboardInterrupt
                    accepted = []
                    for text in batch.column(0).to_pylist():
                        audit['scanned'] += 1
                        if not isinstance(text, str) or not text.strip():
                            continue
                        normalized = words(text)
                        identity = hashlib.sha256(' '.join(normalized).encode()).digest()
                        if identity in seen:
                            audit['duplicate_documents'] += 1
                            continue
                        seen.add(identity)
                        if any(g in index for g in grams(normalized)):
                            audit['benchmark_overlap_documents'] += 1
                            continue
                        split = 'validation' if int.from_bytes(identity[:8], 'big') % 100 == 0 else 'train'
                        if totals[split] < limits[split]:
                            accepted.append((text, split))
                    encoded = tokenizer.encode_batch([text for text, _ in accepted], add_special_tokens=False)
                    for (_, split), tokenized in zip(accepted, encoded):
                        ids = (tokenized.ids+[source.EOS])[:limits[split]-totals[split]]
                        if ids:
                            if min(ids) < 0 or max(ids) >= 50257:
                                raise ValueError('Unexpected tokenizer IDs')
                            handles[split].write(np.asarray(ids, dtype='<u2').tobytes())
                            totals[split] += len(ids)
                    bar.n = sum(totals.values()); bar.postfix = f'扫描 {audit["scanned"]:,} 文档'
                    ui.stage(f'编码/去重/过滤 · train {totals["train"]:,}/{required:,} · val {totals["validation"]:,}/{limits["validation"]:,}')
                    if totals == limits:
                        break
            if totals == limits:
                break
        if totals != limits:
            raise ValueError('Pinned corpus exhausted before the requested budgets')
    finally:
        for handle in handles.values():
            handle.close()
    files = []
    for split, count in totals.items():
        path = root/f'{split}.bin'; path.with_suffix('.bin.part').replace(path)
        files.append(dict(path=path.name, split=split, tokens=count, sha256=file_hash(path)))
    result = dict(status='completed', settings=settings, sources_sha256=digest(src), files=files, totals=totals,
                  audit=audit, source_integrity='HTTPS range reads at immutable HF revision; source byte sizes checked; complete local token packs SHA256 checked. Partial remote shards do not have whole-shard checksum verification.',
                  filtering='Casefolded word 13-gram overlap against all five official evaluation splits; normalized full-document deduplication; hash-based 1% validation. Short/semantic paraphrase contamination is not covered.')
    write(root/'manifest.json', result)
    return validate(root, settings)
