"""Seeded, undirected degree-eight sparse graphs, independent of evidence relations."""
from collections import deque
from functools import lru_cache
import numpy as np
import torch


@lru_cache(maxsize=32)
def graph(nodes, seed=901, topology='connected', degree=8):
    if degree % 2 or nodes <= degree:
        raise ValueError('An even degree smaller than N is required')
    rng = np.random.default_rng(seed)
    # Ring plus degree-preserving edge swaps: no dense N*N*d allocation.
    groups = [np.arange(nodes)] if topology != 'cut' else np.array_split(np.arange(nodes), 2)
    edges = set()
    for group in groups:
        if len(group) <= degree:
            raise ValueError('Each isolated component must exceed its degree')
        local = {tuple(sorted((int(group[i]), int(group[(i + j) % len(group)]))))
                 for i in range(len(group)) for j in range(1, degree // 2 + 1)}
        for _ in range(len(group) * degree * 10):
            ordered = sorted(local)
            a, b = ordered[int(rng.integers(len(ordered)))]; c, d = ordered[int(rng.integers(len(ordered)))]
            if rng.random() < .5:
                c, d = d, c
            e, f = tuple(sorted((a, c))), tuple(sorted((b, d)))
            if len({a, b, c, d}) == 4 and e not in local and f not in local:
                local.remove(tuple(sorted((a, b)))); local.remove(tuple(sorted((c, d))))
                local.update((e, f))
        edges.update(local)
    adjacency = [set() for _ in range(nodes)]
    for a, b in edges:
        adjacency[a].add(b); adjacency[b].add(a)
    index = torch.tensor([sorted(row) for row in adjacency], dtype=torch.long)
    if index.shape != (nodes, degree):
        raise AssertionError('Degree mismatch')
    return index


def statistics(index, rounds=(2, 8, 16, 32)):
    adjacency = index.tolist(); distances = []
    for source in range(len(adjacency)):
        dist = {source: 0}; queue = deque([source])
        while queue:
            i = queue.popleft()
            for j in adjacency[i]:
                if j not in dist:
                    dist[j] = dist[i] + 1; queue.append(j)
        distances.append(dist)
    components = len({tuple(sorted(d)) for d in distances})
    lengths = [v for row in distances for k, v in row.items() if v]
    return {'nodes': len(adjacency), 'degree': len(adjacency[0]), 'components': components,
            'mean_reachable_shortest_path': float(np.mean(lengths)),
            'max_reachable_shortest_path': max(lengths),
            'reachable_fraction': {str(j): float(np.mean([sum(v <= j for v in d.values()) / len(adjacency)
                                                        for d in distances])) for j in rounds}}
