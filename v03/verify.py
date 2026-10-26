"""Read-only verification of preserved archives, paired schedules and completed checkpoints."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .data import digest
from .model import Config, Network
from .run import ROOT, cases, write
from .baselines import identifier, Shortcut


def main():
    torch.set_num_threads(2)
    cfg = json.loads(Path('configs/v03/base.json').read_text())
    statuses = json.loads((ROOT/'conditions.json').read_text()); checked = []; schedules = defaultdict(set)
    baseline_targets = {}; source_sets = defaultdict(set)
    for stage in ('pilot', 'main', 'capacity', 'iterations', 'topology', 'public'):
        registered = [v['condition'] for v in statuses.values() if v.get('condition', {}).get('stage') == stage]
        assert {c['id'] for c in cases(stage, cfg)} <= {c['id'] for c in registered}
        for c in sorted(registered, key=lambda c: c['id']):
            folder = ROOT/c['id']; path = folder/'result.json'
            assert statuses[c['id']]['status'] == 'completed', f"Not completed: {c['id']}"
            r = json.loads(path.read_text()); protocol = r['protocol']; actual = protocol['condition']
            assert all(actual[k] == value for k, value in c.items())
            assert r['status'] == 'completed' and r['steps_completed'] == c['steps']
            assert r['episodes_seen'] == c['steps']*cfg['batch_size']
            assert r['training_worlds_disjoint_from_eval']
            assert r['data_schedule_sha256'] == digest(r['batch_sha256']) and len(r['batch_sha256']) == c['steps']
            domain = 'public' if stage == 'public' else ('pilot' if stage == 'pilot' else 'synthetic')
            schedules[(domain, c['seed'], c['steps'])].add(r['data_schedule_sha256'])
            for source, expected in protocol['source_hashes'].items():
                assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == expected, source
                source_sets[source].add(expected)
            best = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
            last = torch.load(folder/'last.pt', map_location='cpu', weights_only=True)
            assert best['protocol'] == last['protocol'] == protocol
            assert best['step'] == r['best_step'] and last['step'] == c['steps']
            model = Network(Config(**protocol['model'])); model.load_state_dict(best['model'])
            assert sum(p.numel() for p in model.parameters()) == r['ledger']['trainable_parameters']
            assert model.cfg.nodes*(model.cfg.state_dim+2) == r['ledger']['current_state_floats_per_sample']
            assert all(bool(p.isfinite().all()) for p in model.parameters())
            torch.testing.assert_close(model.initial_z.square().sum(-1), torch.ones(model.cfg.nodes), atol=2e-6, rtol=0)
            records = [json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
            validations = [row for row in records if 'validation_loss' in row]
            selected = min(validations, key=lambda row: row['validation_loss'])
            assert selected['step'] == r['best_step'] and selected['validation_loss'] == r['validation_loss']
            worlds = json.loads((folder/'training_world_hashes.json').read_text())
            assert len(worlds) == c['steps']*cfg['batch_size'] and len(set(worlds)) == r['training_worlds_unique']
            if stage != 'pilot':
                rows = r['test']['episode_results']; correct = [e['correct'] for e in rows]; losses = [e['loss'] for e in rows]
                assert len(rows) == r['test']['episodes']
                assert all(e['correct'] == int(e['prediction'] == e['target']) for e in rows)
                assert np.isfinite(losses).all() and abs(np.mean(correct)-r['test']['accuracy']) < 1e-12
                assert abs(np.mean(losses)-r['test']['loss']) < 1e-10
                signature = digest([(e['world_hash'], e['target']) for e in rows])
                if domain in baseline_targets:
                    assert signature == baseline_targets[domain], 'Unpaired evaluation worlds or labels'
                baseline_targets[domain] = signature
            if stage == 'capacity':
                assert r['ledger']['current_state_floats_per_sample'] == 8192
            if stage == 'topology':
                assert r['graph']['degree'] == 8
                assert r['graph']['components'] == (2 if actual['topology'] == 'cut' else 1)
            checked.append(str(path))
    assert all(len(hashes) == 1 for hashes in schedules.values()), 'Paired training schedule mismatch'
    old = json.loads(Path('matrix_lab/docs/RESULTS_MANIFEST.json').read_text())
    for path, expected in {**old['sources'], **old['artifacts']}.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, f'Preserved archive changed: {path}'
    for mode in ('question_only', 'evidence_only'):
        for seed in cfg['seeds']:
            folder = ROOT/identifier(mode, seed); r = json.loads((folder/'result.json').read_text())
            assert r['data_schedule_sha256'] in schedules[('synthetic', seed, 2048)]
            for source, expected in r['source_hashes'].items():
                assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == expected
            checkpoint = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
            selected = min(r['validation_history'], key=lambda row: row['validation_loss'])
            assert checkpoint['step'] == r['best_step'] == selected['step']
            model = Shortcut(mode); model.load_state_dict(checkpoint['model'])
    report = {'status': 'completed', 'training_conditions_verified': len(checked), 'all_source_hashes_current': True,
              'paired_data_schedules_equal': True, 'paired_test_worlds_and_labels_equal': True, 'shortcut_runs_verified': 6,
              'all_checkpoints_safe_load': True, 'best_checkpoints_match_validation_selection': True,
              'capacity_exactly_8192': True, 'degree_and_components_checked': True,
              'old_sources_preserved': len(old['sources']), 'old_artifacts_preserved': len(old['artifacts']),
              'checked_results': checked, 'schedule_groups': {str(k): list(v) for k, v in schedules.items()}}
    write(ROOT/'verification.json', report); print(json.dumps({k: v for k, v in report.items() if k not in ('checked_results', 'schedule_groups')}))


if __name__ == '__main__':
    main()
