"""Chinese lifelong device streams; visible feedback is separate from scored truth."""
from pathlib import Path
import numpy as np
import torch
from v04.data import digest, COLORS

VERSION = 'lifelong-device-streams-v2'
PROFILES = ('history', 'independent', 'outlier', 'switch', 'return', 'consolidation',
            'repetition', 'delayed', 'long')
TRAIN_PROFILES = PROFILES[:-1]
INPUT_FIELDS = ('key', 'value', 'noise', 'cue', 'source', 'evidence_id', 'observe',
                'query_a', 'query_b', 'combine', 'request', 'restart', 'valid')


def stream(rng, profile, keys=8, split='train'):
    if profile not in PROFILES or keys < 4:
        raise ValueError('Unknown profile or fewer than four keys')
    a, b, distractor = map(int, rng.choice(keys, 3, replace=False))
    worlds = rng.integers(4, size=(2, keys))
    cues = rng.normal(0, 1.2, (2, 4))
    while np.linalg.norm(cues[0]-cues[1]) < 2.2:
        cues[1] = rng.normal(0, 1.2, 4)
    events = []; counter = int(rng.integers(10000, 10000000))
    combine = bool(rng.integers(2)) if profile in ('history', 'return', 'long') else False

    def emit(world=0, key=None, request=False, noise=.08, wrong=None, copy=None,
             restart=False, stage='learn', feedback=False):
        nonlocal counter
        counter += 1
        if copy is None:
            observed = key is not None
            color = int(worlds[world, key]) if observed else 0
            if wrong is not None:
                color = wrong
            value = (np.eye(4)[color]+rng.normal(0, noise, 4)).tolist() if observed else [0.]*4
            e = dict(key=key if observed else 0, value=value, noise=noise, source=0 if noise < .3 else 1,
                     evidence_id=counter if observed else -1, observe=observed, grounded=wrong is None)
        else:
            e = dict(copy)
        truth = [int(worlds[world, a]), int(worlds[world, b])]
        e.update(cue=(cues[world]+rng.normal(0, .10, 4)).tolist(), query_a=a, query_b=b,
                 combine=combine, request=request, restart=restart, query=request and stage != 'repeated_claim',
                 target=(truth[0]+(truth[1] if combine else 0)) % 4, truth=truth,
                 world=world, stage=stage, feedback=feedback)
        events.append(e)
        return e

    def practice(world, repeats=None):
        repeats = int(rng.integers(2, 5)) if repeats is None else repeats
        for _ in range(repeats):
            for key in rng.permutation([a, b]):
                emit(world, int(key))
            emit(world, request=True)
            for _ in range(int(rng.integers(4))):
                emit(world, distractor)
            for key in rng.permutation([a, b]):
                emit(world, int(key), feedback=True)

    if profile == 'repetition':
        wrong = int(rng.integers(4))
        report = emit(0, a, wrong=wrong)
        for _ in range(8):
            emit(0, copy=report, request=True, stage='repeated_claim')
        for _ in range(3):
            emit(0, a, feedback=True, stage='correction')
            emit(0, request=True, stage='correction')
        emit(0, restart=True, stage='restart')
        emit(0, request=True, stage='slow_probe')
    elif profile == 'delayed':
        lag = int(rng.choice([2, 4, 8] if split in ('train', 'dev', 'fit') else [16, 32]))
        for _ in range(2):
            emit(0, a); emit(0, request=True)
            for _ in range(lag):
                emit(0, distractor)
            emit(0, a, feedback=True, stage='delayed_feedback')
        emit(0, restart=True, stage='restart')
        emit(0, request=True, stage='slow_probe')
    else:
        practice(0)
        if profile == 'independent':
            emit(1, a, request=True, stage='independent_probe')
        elif profile == 'outlier':
            emit(0, a, noise=.9, wrong=int(rng.integers(4)))
            emit(0, request=True, stage='outlier_probe')
        elif profile == 'switch':
            worlds[0, a] = (int(worlds[0, a])+int(rng.integers(1, 4))) % 4
            for _ in range(int(rng.integers(2, 9))):
                emit(0, a, feedback=True, stage='correction')
                emit(0, request=True, stage='correction')
            emit(0, restart=True, stage='restart')
            emit(0, request=True, stage='slow_probe')
        elif profile in ('return', 'long'):
            practice(1, 2)
            for _ in range(24 if profile == 'long' else 2):
                emit(1, distractor)
            if profile == 'return':
                emit(0, restart=True, stage='restart')
            emit(0, request=True, stage='return_probe')
        elif profile == 'consolidation':
            emit(0, restart=True, stage='restart')
            emit(0, request=True, stage='slow_probe')
        else:
            for _ in range(int(rng.integers(2, 9))):
                emit(0)
            emit(0, request=True, stage='history_probe')
    return dict(profile=profile, events=events, rotate_key=a)


def quartet(rng, profile, keys=8, split='train'):
    original = stream(rng, profile, keys, split); rows = []
    for shift in range(4):
        events = []
        for event in original['events']:
            e = dict(event)
            if e['observe'] and e['key'] == original['rotate_key'] and e['grounded']:
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


def bank(split, quartets=4, keys=8, profiles=PROFILES):
    seed = {'dev': 810205, 'test': 910205, 'fit': 710205}[split]
    rng = np.random.default_rng(seed)
    profiles = tuple(p for p in profiles if split != 'dev' or p != 'long')
    return [r for profile in profiles for _ in range(quartets) for r in quartet(rng, profile, keys, split)]


def collate(rows):
    b, t = len(rows), max(len(r['events']) for r in rows)
    batch = {k: torch.zeros(b, t, dtype=torch.long) for k in
             ('key', 'source', 'evidence_id', 'query_a', 'query_b', 'target', 'world')}
    batch.update({k: torch.zeros(b, t, dtype=torch.bool) for k in
                  ('observe', 'combine', 'valid', 'query', 'request', 'restart')})
    batch.update(value=torch.zeros(b, t, 4), cue=torch.zeros(b, t, 4), noise=torch.ones(b, t))
    batch['evidence_id'].fill_(-1)
    for i, row in enumerate(rows):
        for j, event in enumerate(row['events']):
            for k in batch:
                if k != 'valid':
                    batch[k][i, j] = torch.as_tensor(event[k])
            batch['valid'][i, j] = True
    return batch


def inputs(batch, device):
    return {k: batch[k].to(device) for k in INPUT_FIELDS}


def chinese(row):
    lines = []
    for e in row['events']:
        text = f"情境线索{np.round(e['cue'], 2).tolist()}；"
        if e['restart']:
            text += '实验性清除短期状态，保留后天可塑参数与情境锚点。'
        elif e['observe']:
            text += f"报告{e['evidence_id']}：设备{e['key']}，读数{np.round(e['value'], 3).tolist()}，来源标准差{e['noise']}。"
        else:
            text += '没有新观测。'
        if e['request']:
            text += f" 查询设备{e['query_a']}"+(f"与{e['query_b']}颜色之和模4" if e['combine'] else '颜色')+'。'
            text += f" [仅评分器可见答案：{COLORS[e['target']]}；阶段：{e['stage']}；计入主评分：{e['query']}]"
        lines.append(text)
    return '\n'.join(lines)


def validate(root, cfg):
    from v04.run import write
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    splits = {s: bank(s, cfg['development_quartets'] if s == 'dev' else cfg['test_quartets'], cfg['matrix_size'])
              for s in ('dev', 'test')}
    for rows in splits.values():
        for start in range(0, len(rows), 4):
            group = rows[start:start+4]
            for t, event in enumerate(group[0]['events']):
                if event['query']:
                    assert sorted(r['events'][t]['target'] for r in group) == [0, 1, 2, 3]
                for name in ('cue', 'key', 'noise', 'source', 'evidence_id', 'observe', 'request', 'restart', 'query_a', 'query_b', 'combine'):
                    assert all(r['events'][t][name] == event[name] for r in group)
            for row in group:
                for e in row['events']:
                    assert e['target'] == (e['truth'][0]+(e['truth'][1] if e['combine'] else 0)) % 4
    assert not {r['stream_hash'] for r in splits['dev']} & {r['stream_hash'] for r in splits['test']}
    hidden = {'world', 'target', 'query', 'truth', 'profile', 'shift', 'stage', 'feedback', 'grounded'}
    assert not hidden & set(INPUT_FIELDS)
    result = dict(version=VERSION, question_only_accuracy_exact=.25, inputs=list(INPUT_FIELDS), hidden=sorted(hidden),
                  splits={s: dict(streams=len(rows), sha256=digest(rows), max_events=max(len(r['events']) for r in rows)) for s, rows in splits.items()},
                  boundaries='No hidden world/change labels. restart is an explicit experimental short-state erasure, not a world switch.',
                  feedback='Only noisy observable reports arrive after requests; scored targets never enter state updates.',
                  scoring='Requests before any grounded evidence in repetition are unscored; targets remain hidden. Unverified claims stay identical across all four counterfactual histories.',
                  generalization='Dev delayed lags 2/4/8; test lags 16/32. Long task excluded from dev. Shared task grammar remains.',
                  limitation='Chinese structured synthetic streams; reported noise is a source claim and may be false in the repetition stress test; cross-ID source correlation remains unresolved.')
    write(root/'data_validation.json', result)
    example_rng = np.random.default_rng(710205)
    (root/'chinese_examples.txt').write_text('\n\n'.join(
        chinese(quartet(example_rng, p, cfg['matrix_size'], split='fit')[0]) for p in PROFILES), encoding='utf-8')
    return result
