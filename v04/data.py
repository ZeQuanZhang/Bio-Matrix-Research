"""Balanced Chinese-renderable streams. Hidden worlds never enter model inputs."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

VERSION = 'latent-context-device-streams-v2'
PROFILES = ('history', 'independent', 'outlier', 'switch', 'return', 'partial', 'duplicate', 'long')
TRAIN_PROFILES = PROFILES[:-1]
INPUT_FIELDS = ('key', 'value', 'noise', 'cue', 'source', 'evidence_id', 'observe',
                'query_a', 'query_b', 'combine', 'valid')
COLORS = ('红', '蓝', '绿', '黄')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def stream(rng, profile, keys=8):
    if profile not in PROFILES or keys < 4:
        raise ValueError('Unknown profile or fewer than four address keys')
    a, b = map(int, rng.choice(keys, 2, replace=False))
    active = [a, b] + [int(k) for k in rng.choice([k for k in range(keys) if k not in (a, b)], 2, replace=False)]
    worlds = [rng.integers(4, size=keys).tolist() for _ in range(2)]
    cues = rng.normal(0, 1.2, (2, 4))
    while np.linalg.norm(cues[0]-cues[1]) < 1.8:
        cues[1] = rng.normal(0, 1.2, 4)
    events = []; counter = int(rng.integers(1000, 1000000))
    combine = bool(rng.integers(2)) if profile != 'independent' else False
    duplicate = None

    def emit(world, key=None, noise=.10, value=None, query=False, copy=None):
        nonlocal counter
        counter += 1
        if copy is not None:
            event = dict(copy)
            event['cue'] = (cues[world]+rng.normal(0, .16, 4)).tolist()
        else:
            observed = key is not None
            if observed and value is None:
                value = (np.eye(4)[worlds[world][key]]+rng.normal(0, noise, 4)).tolist()
            event = dict(key=key if observed else 0, value=value if observed else [0.]*4,
                         noise=noise, cue=(cues[world]+rng.normal(0, .16, 4)).tolist(),
                         source=0 if noise < .3 else 1, evidence_id=counter if observed else -1,
                         observe=observed)
        truth = [worlds[world][a], worlds[world][b]]
        event.update(query_a=a, query_b=b, combine=combine, query=query, truth=truth,
                     target=(truth[0]+(truth[1] if combine else 0)) % 4, world=world)
        events.append(event)
        return event

    for key in rng.permutation(active):
        event = emit(0, int(key))
        if int(key) == a:
            duplicate = event
    for _ in range(3):
        emit(0, active[2])
    emit(0, query=True)
    if profile in ('independent', 'switch', 'return', 'long'):
        for key in rng.permutation(active):
            emit(1, int(key))
        for _ in range(3):
            emit(1, active[3])
        emit(1, a if profile == 'independent' else None, query=True)
        if profile in ('return', 'long'):
            for _ in range(16 if profile == 'long' else 4):
                emit(0, active[2])
            emit(0, query=True)
    elif profile == 'outlier':
        wrong = (worlds[0][a]+int(rng.integers(1, 4))) % 4
        emit(0, a, noise=.9, value=np.eye(4)[wrong].tolist())
        emit(0, query=True)
    elif profile == 'partial':
        worlds[0][a] = (worlds[0][a]+int(rng.integers(1, 4))) % 4
        for _ in range(4):
            emit(0, a)
        emit(0, query=True)
    elif profile == 'duplicate':
        for _ in range(4):
            emit(0, copy=duplicate)
        emit(0, query=True)
    else:
        for _ in range(5):
            emit(0)
        emit(0, query=True)
    return dict(profile=profile, events=events, rotate_key=a)


def quartet(rng, profile, keys=8):
    original = stream(rng, profile, keys); rows = []
    for shift in range(4):
        events = []
        for event in original['events']:
            e = dict(event)
            if e['observe'] and e['key'] == original['rotate_key']:
                e['value'] = np.roll(e['value'], shift).tolist()
            e['truth'] = [(e['truth'][0]+shift) % 4, e['truth'][1]]
            e['target'] = (e['target']+shift) % 4
            events.append(e)
        row = dict(profile=profile, events=events, shift=shift)
        row['stream_hash'] = digest(row); rows.append(row)
    return rows


def training_batch(rng, size=4, keys=8, profile=None):
    if size < 4 or size % 4:
        raise ValueError('batch_size must be a positive multiple of four')
    selected = profile or str(rng.choice(TRAIN_PROFILES))
    return [r for _ in range(size//4) for r in quartet(rng, selected, keys)]


def bank(split, quartets=8, keys=8, profiles=PROFILES):
    rng = np.random.default_rng({'dev': 810204, 'test': 910204, 'fit': 710204}[split])
    return [r for p in profiles for _ in range(quartets) for r in quartet(rng, p, keys)]


def collate(rows):
    b, t = len(rows), max(len(r['events']) for r in rows)
    batch = {k: torch.zeros(b, t, dtype=torch.long) for k in ('key', 'source', 'evidence_id', 'query_a', 'query_b', 'target', 'world')}
    batch.update({k: torch.zeros(b, t, dtype=torch.bool) for k in ('observe', 'combine', 'valid', 'query')})
    batch.update(value=torch.zeros(b, t, 4), cue=torch.zeros(b, t, 4), noise=torch.ones(b, t))
    batch['evidence_id'].fill_(-1)
    for i, row in enumerate(rows):
        for j, e in enumerate(row['events']):
            for k in batch:
                if k != 'valid':
                    batch[k][i, j] = torch.as_tensor(e[k])
            batch['valid'][i, j] = True
    return batch


def inputs(batch, device):
    return {k: batch[k].to(device) for k in INPUT_FIELDS}


def chinese(row):
    lines = []
    for e in row['events']:
        cue = np.round(e['cue'], 2).tolist()
        if e['observe']:
            lines.append(f"线索{cue}；来源{e['source']}，报告编号{e['evidence_id']}：设备{e['key']}读数{np.round(e['value'], 3).tolist()}，标准差{e['noise']}。")
        else:
            lines.append(f'线索{cue}；本步没有新的设备观测。')
        if e['query']:
            question = f"设备{e['query_a']}与设备{e['query_b']}颜色编号之和模4" if e['combine'] else f"设备{e['query_a']}的颜色"
            lines.append(f"问题：{question}？ 答案：{COLORS[e['target']]}。")
    return '\n'.join(lines)


def validate(root, cfg):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    splits = {s: bank(s, cfg['development_quartets'] if s == 'dev' else cfg['test_quartets'], cfg['matrix_size']) for s in ('dev', 'test')}
    for rows in splits.values():
        for start in range(0, len(rows), 4):
            group = rows[start:start+4]
            for t in range(len(group[0]['events'])):
                if group[0]['events'][t]['query']:
                    assert sorted(r['events'][t]['target'] for r in group) == [0, 1, 2, 3]
                for name in ('cue', 'key', 'source', 'noise', 'evidence_id', 'observe', 'query_a', 'query_b', 'combine'):
                    assert all(r['events'][t][name] == group[0]['events'][t][name] for r in group)
            for row in group:
                for e in row['events']:
                    expected = e['truth'][0]
                    if e['combine']:
                        expected = (expected+e['truth'][1]) % 4
                    assert e['target'] == expected
                batch = collate([row])
                assert not {'world', 'target', 'query', 'truth', 'profile', 'shift'} & set(INPUT_FIELDS)
                assert bool(torch.isfinite(batch['value']).all())
    sets = [{r['stream_hash'] for r in rows} for rows in splits.values()]
    assert not sets[0] & sets[1]
    result = dict(version=VERSION, question_only_accuracy_exact=.25, boundaries_visible=False,
                  model_inputs=list(INPUT_FIELDS), hidden=['world', 'target', 'query_loss_mask', 'truth', 'profile', 'rotation'],
                  labels_independently_recomputed=True,
                  splits={s: dict(streams=len(rows), sha256=digest(rows)) for s, rows in splits.items()},
                  input_format='Chinese-renderable structured events, not Chinese language-model training',
                  noise='Reported source standard deviation is observable; unknown source reliability remains untested')
    (root/'data_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (root/'chinese_examples.txt').write_text('\n\n'.join(chinese(r) for r in splits['dev'][:4]), encoding='utf-8')
    return result
