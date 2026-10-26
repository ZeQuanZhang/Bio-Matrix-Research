"""Fresh Chinese-symbol key/value associations; targets never enter queries."""
import numpy as np
import torch

KEYS = tuple("甲乙丙丁戊己庚辛")
VALUES = ("红", "蓝", "黄", "绿")
OPERATIONS = ("写入", "干扰", "查询")


def generate(rng, batch=64, pairs=2, delay=8, overwrite=False, queries=1):
    if not 1 <= pairs <= len(KEYS) or min(batch, queries) < 1 or delay < 0:
        raise ValueError("Invalid associative-memory dimensions")
    keys = rng.random((batch, len(KEYS))).argsort(1)[:, :pairs]
    values = rng.integers(len(VALUES), size=(batch, pairs))
    observed = values.copy()
    writes = [(keys[:, i], values[:, i]) for i in range(pairs)]
    if overwrite:
        index = rng.integers(pairs, size=batch)
        replacement = rng.integers(len(VALUES), size=batch)
        observed[np.arange(batch), index] = replacement
        writes.append((keys[np.arange(batch), index], replacement))
    query_indices = rng.integers(pairs, size=(batch, queries))
    events, positions, labels = [], [], []
    for key, value in writes:
        events.append(np.stack((np.zeros(batch, dtype=int), key, value), 1))
    for _ in range(delay):
        events.append(np.stack((np.ones(batch, dtype=int), rng.integers(len(KEYS), size=batch),
                                rng.integers(len(VALUES), size=batch)), 1))
    for q in range(queries):
        if q:
            for _ in range(3):
                events.append(np.stack((np.ones(batch, dtype=int), rng.integers(len(KEYS), size=batch),
                                        rng.integers(len(VALUES), size=batch)), 1))
        index = query_indices[:, q]
        positions.append(len(events))
        events.append(np.stack((np.full(batch, 2), keys[np.arange(batch), index],
                                np.full(batch, len(VALUES))), 1))
        labels.append(observed[np.arange(batch), index])
    return (torch.from_numpy(np.stack(events, 1)).long(),
            torch.from_numpy(np.stack(labels, 1)).long(), tuple(positions))


CONDITIONS = {
    "one_short": dict(pairs=1, delay=0),
    "two_short": dict(pairs=2, delay=0),
    "four_short": dict(pairs=4, delay=0),
    "two_delay24": dict(pairs=2, delay=24),
    "four_delay24": dict(pairs=4, delay=24),
    "two_delay64": dict(pairs=2, delay=64),
    "four_delay128": dict(pairs=4, delay=128),
    "six_pairs": dict(pairs=6, delay=24),
    "eight_pairs": dict(pairs=8, delay=24),
    "overwrite": dict(pairs=3, delay=16, overwrite=True),
    "multi_query": dict(pairs=3, delay=16, queries=4),
}


def training_batch(rng, step, batch):
    # A fixed curriculum establishes basic readout before demanding delayed multi-key retrieval.
    max_pairs = 1 if step < 100 else 2 if step < 300 else 4
    max_delay = 4 if step < 100 else 12 if step < 300 else 24
    return generate(rng, batch, int(rng.integers(1, max_pairs + 1)),
                    int(rng.integers(max_delay + 1)), bool(rng.random() < .25),
                    2 if rng.random() < .25 else 1)
