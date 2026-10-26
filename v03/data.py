"""Chinese relational worlds. The solver is outside the neural input path."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

VERSION = 'relational-world-zh-v03-1'
FAMILIES = ('retrieve', 'trace', 'compare', 'count', 'rule')
WORDS = ['对象', '的', '颜色', '形状', '大小', '是什么', '沿甲关系', '沿乙关系',
         '与', '相同吗', '有多少', '个', '若', '为大', '则', '否则', '之后', '忽略无关记录']
W = {v: 528+i for i, v in enumerate(WORDS)}
COLOR = tuple('红蓝绿黄紫青白黑'); SHAPE = ('圆', '方', '三角'); SIZE = ('小', '大')
VOCAB = {0: 'PAD', **{i+1: f'物{i:03d}' for i in range(512)},
         **{513+i: c for i, c in enumerate(COLOR)}, **{521+i: c for i, c in enumerate(SHAPE)},
         **{524+i: c for i, c in enumerate(SIZE)}, **{i: w for w, i in W.items()}}
ANSWERS = list(COLOR) + list(SHAPE) + list(SIZE) + ['否', '是'] + [str(i) for i in range(65)]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def follow(world, start, path):
    for relation in path:
        start = int(world['relations'][start][relation])
    return start


def solve(world, spec):
    target = follow(world, spec['start'], spec['path'])
    attrs = world['attrs']; family = spec['family']
    if family in ('retrieve', 'trace'):
        a = spec['attribute']; return (0, 8, 11)[a] + int(attrs[target][a])
    if family == 'compare':
        other = follow(world, spec['other'], spec['path'])
        return 13 + int(attrs[target][0] == attrs[other][0])
    if family == 'count':
        return 15 + sum(row[0] == spec['color'] for row in attrs)
    if family == 'rule':
        if attrs[target][2] == 0:
            target = follow(world, target, [0])
        return int(attrs[target][0])
    raise ValueError('Unknown task')


def example(rng, objects=16, depth=2, irrelevant=.0, family=None, holdout=False):
    if objects not in (8, 16, 32, 64) or depth not in (1, 2, 3, 4) or irrelevant not in (0., .5, .8):
        raise ValueError('Unregistered task dimensions')
    family = family or str(rng.choice(FAMILIES))
    ids = rng.permutation(64)[:objects]
    attrs = np.stack((rng.integers(8, size=objects), rng.integers(3, size=objects), rng.integers(2, size=objects)), -1)
    relations = np.stack((rng.permutation(objects), rng.permutation(objects)), -1)
    world = {'ids': ids.tolist(), 'attrs': attrs.tolist(), 'relations': relations.tolist()}
    path = rng.integers(2, size=depth).tolist()
    if depth >= 3:
        # The ABA composition is held out; both primitive relations remain in training.
        if holdout:
            path[:3] = [1, 0, 1]
        elif path[:3] == [1, 0, 1]:
            path[2] = 0
    spec = {'family': family, 'start': int(rng.integers(objects)), 'path': path,
            'other': int(rng.integers(objects)), 'attribute': int(rng.integers(3)), 'color': int(rng.integers(8))}
    if family in ('retrieve', 'count'):
        spec['path'] = []
    path_words = [W['沿甲关系'] if r == 0 else W['沿乙关系'] for r in spec['path']]
    question = [W['对象'], int(ids[spec['start']])+1] + path_words
    attr_word = W[('颜色', '形状', '大小')[spec['attribute']]]
    if family in ('retrieve', 'trace'):
        question += [W['的'], attr_word, W['是什么']]
    elif family == 'compare':
        question += [W['与'], W['对象'], int(ids[spec['other']])+1] + path_words + [W['的'], W['颜色'], W['相同吗']]
    elif family == 'count':
        question = [W['有多少'], W['个'], 513+spec['color'], W['颜色'], W['对象'], W['忽略无关记录']]
    else:
        question += [W['若'], W['大小'], W['为大'], W['则'], W['颜色'], W['否则'], W['沿甲关系'], W['颜色']]
    records = [[int(ids[i])+1, 513+int(attrs[i, 0]), 521+int(attrs[i, 1]), 524+int(attrs[i, 2]),
                int(ids[relations[i, 0]])+1, int(ids[relations[i, 1]])+1, 0] for i in range(objects)]
    # Irrelevant records are explicitly marked; necessary evidence is never hidden.
    # Their IDs and relations are disjoint from the queried world, not neural-node assignments.
    extra = int(round(objects*irrelevant/(1-irrelevant)))
    for i in range(extra):
        records.append([65+i, 513+int(rng.integers(8)), 521+int(rng.integers(3)), 524+int(rng.integers(2)),
                        65+int(rng.integers(max(1, extra))), 65+int(rng.integers(max(1, extra))), W['忽略无关记录']])
    rng.shuffle(records)
    y = solve(world, spec)
    return {'records': records, 'question': question, 'target': y, 'world_hash': digest(world),
            'family': family, 'objects': objects, 'depth': len(spec['path']), 'irrelevant': irrelevant,
            'holdout': holdout and family not in ('retrieve', 'count'), 'world': world, 'spec': spec,
            'question_zh': ' '.join(VOCAB[v] for v in question), 'answer_zh': ANSWERS[y]}


def collate(examples):
    count, rows, columns = len(examples), max(len(e['records']) for e in examples), max(len(e['records'][0]) for e in examples)
    qlen = max(len(e['question']) for e in examples)
    records = torch.zeros(count, rows, columns, dtype=torch.long)
    question = torch.zeros(count, qlen, dtype=torch.long); mask = torch.zeros(count, rows, dtype=torch.bool)
    for i, e in enumerate(examples):
        records[i, :len(e['records']), :len(e['records'][0])] = torch.tensor(e['records'])
        question[i, :len(e['question'])] = torch.tensor(e['question']); mask[i, :len(e['records'])] = True
    return {'records': records, 'question': question, 'mask': mask,
            'target': torch.tensor([e['target'] for e in examples])}


def training_batch(rng, batch=16, with_hashes=False):
    # Shared difficulty per batch avoids padding to the worst case in every update.
    objects = int(rng.choice([8, 16, 32])); depth = int(rng.choice([1, 2, 3])); irrelevant = float(rng.choice([0., .5, .8]))
    examples = [example(rng, objects, depth, irrelevant) for _ in range(batch)]
    value = collate(examples)
    return (value, [e['world_hash'] for e in examples]) if with_hashes else value


def bank(split='test', per_slice=24):
    rng = np.random.default_rng({'dev': 81003, 'test': 91003, 'pilot': 71003}[split]); result = []
    # Predefined axes, not a full Cartesian product; all reported slices retained.
    for family in FAMILIES:
        for objects, depth, noise, held in [(8, 1, 0., False), (16, 2, .5, False), (32, 3, .8, False),
                                           (64, 3, .5, False), (32, 4, .5, False), (32, 3, .5, True)]:
            result += [example(rng, objects, depth, noise, family, held) for _ in range(per_slice)]
    return result


def manifest(examples):
    return {'version': VERSION, 'episodes': len(examples), 'world_hashes': [e['world_hash'] for e in examples],
            'content_sha256': digest([{k: e[k] for k in ('records', 'question', 'target', 'world_hash')} for e in examples]),
            'family_counts': {f: sum(e['family'] == f for e in examples) for f in FAMILIES}}


def validate(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    splits = {s: bank(s) for s in ('dev', 'test', 'pilot')}
    hashes = {s: {e['world_hash'] for e in rows} for s, rows in splits.items()}
    for a, b in (('dev', 'test'), ('dev', 'pilot'), ('pilot', 'test')):
        assert not hashes[a] & hashes[b], 'World leakage across splits'
    for rows in splits.values():
        for e in rows:
            assert e['target'] == solve(e['world'], e['spec'])
            id_to_record = {r[0]: r for r in e['records']}
            for i, oid in enumerate(e['world']['ids']):
                r = id_to_record[oid+1]
                assert r[1:4] == [513+e['world']['attrs'][i][0], 521+e['world']['attrs'][i][1], 524+e['world']['attrs'][i][2]]
                assert r[4:6] == [e['world']['ids'][j]+1 for j in e['world']['relations'][i]]
    result = {'splits': {s: manifest(rows) for s, rows in splits.items()}, 'worlds_disjoint': True,
              'solver_matches_encoded_evidence': True, 'model_input_keys': ['records', 'question', 'mask'],
              'labels_passed_only_to_loss': True, 'answer_vocabulary': ANSWERS, 'token_vocabulary': VOCAB}
    (root/'dataset_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (root/'chinese_examples.json').write_text(json.dumps(splits['test'][:5], ensure_ascii=False, indent=2))
    return result
