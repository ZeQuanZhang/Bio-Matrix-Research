"""Prepare a reproducible, deduplicated Chinese corpus and a train-only BPE."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers


def prepare(root, documents, vocab_size):
    root = Path(root)
    source = json.loads((root / "source.json").read_text())
    splits = {"train": [], "valid": [], "test": []}
    seen = set()
    parquet = root / "train-00000-of-00003.parquet"
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=4096, columns=["text"]):
        for text in batch.column(0).to_pylist():
            text = " ".join((text or "").split())
            han = sum("\u4e00" <= c <= "\u9fff" for c in text)
            if len(text) < 80 or han < .4 * len(text):
                continue
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            bucket = int(digest[:8], 16) % 100
            split = "test" if bucket == 0 else "valid" if bucket == 1 else "train"
            splits[split].append((digest, text))
            if len(seen) >= documents:
                break
        if len(seen) >= documents:
            break
    if min(map(len, splits.values())) < 100:
        raise ValueError("Not enough independent validation/test documents.")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    tokenizer.train_from_iterator((text for _, text in splits["train"]), trainer=trainer)
    tokenizer.save(str(root / "tokenizer.json"))
    manifest = {"dataset": "RobinChen2001/TinyStories-Zh-2M", "revision": source["sha"],
                "shard": parquet.name, "shard_sha256": hashlib.file_digest(parquet.open("rb"), "sha256").hexdigest(),
                "selection": "first qualifying unique documents of pinned first shard",
                "split": "sha256(normalized_document) modulo 100: 0=test, 1=valid, remainder=train",
                "vocab_size": tokenizer.get_vocab_size(), "splits": {},
                "license_note": "Dataset card: MIT; upstream English TinyStories: CDLA-Sharing-1.0. Translation quality limits apply."}
    for name, rows in splits.items():
        tokens, offsets = [], [0]
        for start in range(0, len(rows), 1000):
            encoded = tokenizer.encode_batch([text for _, text in rows[start:start + 1000]])
            for encoding in encoded:
                tokens.extend([1, *encoding.ids, 2])
                offsets.append(len(tokens))
        np.save(root / f"{name}_tokens.npy", np.asarray(tokens, dtype=np.int32))
        np.save(root / f"{name}_offsets.npy", np.asarray(offsets, dtype=np.int64))
        (root / f"{name}_hashes.txt").write_text("\n".join(digest for digest, _ in rows) + "\n")
        manifest["splits"][name] = {"documents": len(rows), "tokens": len(tokens),
                                    "token_sha256": hashlib.sha256(np.asarray(tokens, dtype=np.int32).tobytes()).hexdigest()}
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--documents", type=int, default=80000)
    parser.add_argument("--vocab-size", type=int, default=8192)
    args = parser.parse_args()
    prepare(args.data, args.documents, args.vocab_size)
