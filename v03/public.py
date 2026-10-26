"""Official CLEVR-SG: scene records and raw question words, never program tokens."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import torch
from .data import collate, digest
from .run import write

BASE = Path('data/clevr/CLEVR_v1.0')
CACHE = Path('data/clevr/v03_subset.pt')
SOURCE = 'https://cs.stanford.edu/people/jcjohns/clevr/'


def execute(scene, program):
    """Offline label audit only. This function is not called by PublicData or Network."""
    values = []
    for instruction in program:
        function = instruction['function']; args = [values[i] for i in instruction['inputs']]
        parameter = instruction.get('value_inputs', [])
        if function == 'scene':
            result = list(range(len(scene['objects'])))
        elif function.startswith('filter_'):
            attribute = function[7:]; result = [i for i in args[0] if scene['objects'][i][attribute] == parameter[0]]
        elif function == 'unique':
            assert len(args[0]) == 1; result = args[0][0]
        elif function == 'relate':
            result = scene['relationships'][parameter[0]][args[0]]
        elif function.startswith('same_'):
            attribute = function[5:]; value = scene['objects'][args[0]][attribute]
            result = [i for i, o in enumerate(scene['objects']) if i != args[0] and o[attribute] == value]
        elif function in ('union', 'intersect'):
            result = sorted(set(args[0]) | set(args[1]) if function == 'union' else set(args[0]) & set(args[1]))
        elif function == 'count':
            result = len(args[0])
        elif function == 'exist':
            result = bool(args[0])
        elif function.startswith('query_'):
            result = scene['objects'][args[0]][function[6:]]
        elif function.startswith('equal_'):
            result = args[0] == args[1]
        elif function == 'less_than':
            result = args[0] < args[1]
        elif function == 'greater_than':
            result = args[0] > args[1]
        else:
            raise ValueError(f'Unknown audit-only function {function}')
        values.append(result)
    value = values[-1]
    return ('yes' if value else 'no') if isinstance(value, bool) else str(value)


def words(question):
    return re.findall(r"[a-z]+|[0-9]+|[^\w\s]", question.lower())


def scene_records(scene):
    colors = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
    shapes = ['cube', 'sphere', 'cylinder']; sizes = ['small', 'large']; materials = ['rubber', 'metal']
    records = []
    for i, obj in enumerate(scene['objects']):
        xyz = [int(np.clip(round((v+4)*7), 0, 63)) for v in obj['3d_coords']]
        records.append([2, 4+i, 14+colors.index(obj['color']), 22+shapes.index(obj['shape']),
                        25+sizes.index(obj['size']), 27+materials.index(obj['material']),
                        29+xyz[0], 93+xyz[1], 157+xyz[2]])
    for relation, direction in enumerate(('left', 'right', 'front', 'behind')):
        for source, targets in enumerate(scene['relationships'][direction]):
            for target in targets:
                records.append([3, 4+source, 221+relation, 4+target, 0, 0, 0, 0, 0])
    return records


def prepare():
    if CACHE.exists():
        return
    archive = Path('data/clevr/CLEVR_v1.0_no_images.zip')
    if not archive.exists():
        raise FileNotFoundError('Official CLEVR package missing; external task remains unverified')
    scenes = {split: {s['image_index']: s for s in json.loads((BASE/'scenes'/f'CLEVR_{split}_scenes.json').read_text())['scenes']}
              for split in ('train', 'val')}
    raw_train = json.loads((BASE/'questions/CLEVR_train_questions.json').read_text())['questions']
    raw_val = json.loads((BASE/'questions/CLEVR_val_questions.json').read_text())['questions']
    rng = np.random.default_rng(3119)
    indices = sorted(rng.choice(len(raw_train), 50000, replace=False).tolist())
    train = [raw_train[i] for i in indices]
    dev_scenes = {i for i in scenes['val'] if int(hashlib.sha256(f'clevr-val-{i}'.encode()).hexdigest(), 16) % 2 == 0}
    val_a = [q for q in raw_val if q['image_index'] in dev_scenes]
    val_b = [q for q in raw_val if q['image_index'] not in dev_scenes]
    dev = [val_a[i] for i in sorted(rng.choice(len(val_a), 1500, replace=False).tolist())]
    test = [val_b[i] for i in sorted(rng.choice(len(val_b), 3000, replace=False).tolist())]
    vocabulary = {w: 256+i for i, w in enumerate(sorted({w for q in train for w in words(q['question'])}))}
    answers = sorted({q['answer'] for q in train})
    audited = 0
    for split, group in (('train', train), ('val', dev+test)):
        for i in rng.choice(len(group), min(500, len(group)), replace=False):
            q = group[i]
            assert execute(scenes[split][q['image_index']], q['program']) == q['answer']
            audited += 1
    encoded_scenes = {}; lists = {}; memberships = {}
    for name, split, group in (('train', 'train', train), ('dev', 'val', dev), ('test', 'val', test)):
        rows = []; memberships[name] = {'scene_ids': sorted({q['image_index'] for q in group}),
                                      'question_ids': [q['question_index'] for q in group]}
        for q in group:
            scene_id = f"{split}:{q['image_index']}"; scene = scenes[split][q['image_index']]
            if scene_id not in encoded_scenes:
                encoded_scenes[scene_id] = scene_records(scene)
            depths = []
            for instruction in q['program']:
                depths.append(max((depths[i] for i in instruction['inputs']), default=0) + int(instruction['function'] == 'relate'))
            last = q['program'][-1]['function']
            family = 'compare' if last.startswith('equal') or last in ('greater_than', 'less_than') else ('count' if last == 'count' else ('exist' if last == 'exist' else 'retrieve'))
            rows.append({'scene_id': scene_id, 'world_hash': digest({'objects': scene['objects'], 'relationships': scene['relationships']}),
                         'question': [vocabulary.get(w, 1) for w in words(q['question'])], 'target': answers.index(q['answer']),
                         'raw_question': q['question'], 'family': family, 'objects': len(scene['objects']), 'depth': max(depths),
                         'irrelevant': 0., 'holdout': False, 'question_id': q['question_index']})
        lists[name] = rows
    sets = {k: {e['world_hash'] for e in rows} for k, rows in lists.items()}
    assert not (sets['train'] & sets['dev'] or sets['train'] & sets['test'] or sets['dev'] & sets['test'])
    manifest = {'name': 'CLEVR-SG derived, official v1.0 without images', 'source': SOURCE,
                'archive_url': 'https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0_no_images.zip', 'license': 'CC BY 4.0',
                'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'selection_seed': 3119,
                'train_questions': 50000, 'dev_questions': len(dev), 'test_questions': len(test),
                'worlds_disjoint': True, 'scene_split_rule': 'SHA256(clevr-val-ID) parity, before question sampling',
                'memberships': memberships, 'memberships_sha256': digest(memberships), 'audit_only_programs_checked': audited,
                'programs_are_neural_inputs': False, 'raw_question_text_retained': True, 'word_vocabulary': vocabulary,
                'answer_vocabulary': answers, 'input': 'independently encoded object records and relationship records, plus raw question words',
                'limitation': 'Ground truth object grouping and attributes; not image VQA; no CoGenT in this subset.'}
    temporary = CACHE.with_suffix('.tmp')
    torch.save({'scenes': encoded_scenes, 'examples': lists, 'manifest': manifest, 'answers': answers,
                'vocab_size': max(vocabulary.values())+1}, temporary); temporary.replace(CACHE)
    write('data/clevr/v03_manifest.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ('memberships', 'word_vocabulary')}), flush=True)


class PublicData:
    def __init__(self):
        if not CACHE.exists():
            raise FileNotFoundError('Run python -m v03.public before public training')
        value = torch.load(CACHE, map_location='cpu', weights_only=True)
        self.scenes = value['scenes']; self.examples = value['examples']; self.manifest = value['manifest']
        self.answers = value['answers']; self.vocab_size = value['vocab_size']
        self.dev = self.expand(self.examples['dev']); self.test = self.expand(self.examples['test'])

    def expand(self, rows):
        return [e | {'records': self.scenes[e['scene_id']]} for e in rows]

    def training_batch(self, rng, count):
        # Sampling from the fixed 50k subset with replacement; exposure budget is explicit.
        group = self.expand([self.examples['train'][i] for i in rng.integers(len(self.examples['train']), size=count)])
        return collate(group), [e['world_hash'] for e in group]


if __name__ == '__main__':
    prepare()
