"""Fixed-degree incoming candidates in a solid ball of concentric shells."""
import numpy as np
import torch


def solid_ball(nodes, degree=8, shells=4, seed=6101, topology='connected'):
    if nodes < shells*2 or not 3 <= degree < nodes or shells < 2:
        raise ValueError('Insufficient nodes/degree for ball topology')
    if topology not in ('connected', 'cut'):
        raise ValueError('Unknown topology')
    rng = np.random.default_rng(seed)
    weights = np.diff(np.arange(1, shells+1, dtype=float)**3)
    counts = np.floor((nodes-1-(shells-1))*weights/weights.sum()).astype(int)+1
    counts[-1] += nodes-1-int(counts.sum())
    positions = [np.zeros((1, 3))]; levels = [0]
    golden = np.pi*(3-np.sqrt(5))
    for shell, count in enumerate(counts, 1):
        i = np.arange(count); z = 1-2*(i+.5)/count; radius = np.sqrt(1-z*z)
        phi = golden*i+rng.uniform(-np.pi, np.pi)
        positions.append(np.stack((radius*np.cos(phi), radius*np.sin(phi), z), -1)*shell/(shells-1))
        levels.extend([shell]*count)
    xyz = np.concatenate(positions); levels = np.asarray(levels)
    distance = ((xyz[:, None]-xyz[None])**2).sum(-1)
    neighbor = []
    for target in range(nodes):
        row = []
        # A radial parent guarantees that a new central input can reach every shell.
        for shell in (levels[target]-1, levels[target]+1):
            options = np.flatnonzero(levels == shell)
            if len(options):
                row.append(int(options[np.argmin(distance[target, options])]))
        same = np.flatnonzero((levels == levels[target]) & (np.arange(nodes) != target))
        # The two angular ring directions ensure within-shell connectivity.
        if len(same):
            members = np.flatnonzero(levels == levels[target])
            angle = np.arctan2(xyz[members, 1], xyz[members, 0])
            order = members[np.argsort(angle)]; at = int(np.flatnonzero(order == target)[0])
            row.extend([int(order[(at-1) % len(order)]), int(order[(at+1) % len(order)])])
        candidates = np.flatnonzero((abs(levels-levels[target]) <= 1) & (np.arange(nodes) != target))
        row = list(dict.fromkeys(row))
        for source in candidates[np.argsort(distance[target, candidates])]:
            if int(source) not in row:
                row.append(int(source))
            if len(row) >= degree:
                break
        # Small central shells may have fewer candidates; self padding is masked.
        mask = [1.]*len(row)
        while len(row) < degree:
            row.append(target); mask.append(0.)
        neighbor.append((row[:degree], mask[:degree]))
    indices = torch.tensor([x[0] for x in neighbor], dtype=torch.long)
    valid = torch.tensor([x[1] for x in neighbor], dtype=torch.float32)
    level = torch.tensor(levels, dtype=torch.long)
    if topology == 'cut':
        boundary = shells//2
        valid *= ((level[:, None] < boundary) == (level[indices] < boundary))
    outward = level[:, None] > level[indices]
    write = (level == 0).float(); read = (level == shells-1).float()
    read /= read.sum()
    return dict(positions=torch.tensor(xyz, dtype=torch.float32), shell=level,
                neighbors=indices, edge_mask=valid, outward=outward,
                write_ports=write, read_ports=read)
