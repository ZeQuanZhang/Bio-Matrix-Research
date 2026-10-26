"""Independent implementation probes; never starts the formal scan."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from matrix_lab.run import sync
from v04.run import now, write
from .data import collate, digest, inputs, quartet, training_batch, validate
from .model import Network
from .run import PROJECT, cases, check_other_training, model_config, select_device, source_hashes, train_update


def register(cfg, root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    if (root/'experiment.json').exists():
        identity = json.loads((root/'experiment.json').read_text())
        if identity['config'] != cfg or identity['sources'] != source_hashes():
            raise ValueError('Existing experiment differs; use a new result root')
    planned = cases(cfg)
    path = root/'conditions.json'; existing = json.loads(path.read_text()) if path.exists() else {}
    for case in planned:
        existing.setdefault(case['id'], dict(condition=case, status='not_executed', history=[]))
    write(path, existing)
    write(root/'plan.json', dict(config=cfg, sources=source_hashes(), conditions=planned,
                                status='Registered only; no formal updates executed by preflight'))
    from .report import report
    report(root)
    return len(planned)


def preflight(cfg, root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True); torch.set_num_threads(2)
    frozen_sources = source_hashes()
    diagnostic_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    validation = validate(root/'data', cfg | {'development_quartets': 1, 'test_quartets': 1})
    requested = select_device(cfg['device']); deferred = None
    try:
        check_other_training(requested, root)
        device = requested
    except RuntimeError as error:
        deferred = str(error); device = torch.device('cpu')
        print(f'GPU 检查延期：{deferred}\n继续执行独立 CPU 检查。', flush=True)
    generated = training_batch(np.random.default_rng(191), profile='history', keys=cfg['matrix_size'])
    rows = [r | {'events': r['events'][:8]} for r in generated]
    batch = collate(rows); records = []
    for n in tqdm(cfg['nodes'], desc='规模与梯度检查', unit='规模'):
        for path in ('shared', 'separated'):
            for variant in ('message', 'full'):
                case = dict(variant=variant, nodes=n, read_path=path)
                torch.manual_seed(11); mc = model_config(cfg, case); model = Network(mc).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=.0001)
                probe = inputs(batch, device); probe['value'] = probe['value'].detach().requires_grad_(True)
                sync(device); start = time.perf_counter()
                model.eval()
                logp, _, _ = model(**probe)
                mask = batch['query'].to(device)
                loss = F.nll_loss(logp[mask], batch['target'].to(device)[mask]); loss.backward()
                groups = {}
                for label, names in {'communication': ('coefficient', 'content_angles', 'message_'),
                                     'slow_rule': ('slow_basis', 'raw_slow_rate', 'gates'), 'readout': ('readout',)}.items():
                    groups[label] = sum(float(p.grad.detach().square().sum()) for name, p in model.named_parameters()
                                        if p.grad is not None and any(token in name for token in names))**.5
                input_gradient = float(probe['value'].grad[:, :1].norm())
                optimizer.step(); sync(device)
                elapsed = time.perf_counter()-start
                with torch.no_grad():
                    state = model.initial_state(1); state['x'] = torch.randn_like(state['x'])
                    before = float(state['x'].square().sum())
                    changed = model.tick(state, 0, damping=False)
                    norm_error = abs(float(changed['x'].square().sum())-before)/max(before, 1e-10)
                record = dict(nodes=n, read_path=path, variant=variant, config=asdict(mc), device=str(device),
                              nll=float(loss.detach()), gradient=groups, first_input_gradient=input_gradient,
                              carrier_relative_energy_error=norm_error, update_seconds=elapsed,
                              ledger=model.ledger(), sampled_process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform == 'darwin' else 1024))
                if device.type == 'mps':
                    record.update(sampled_mps_current_bytes=int(torch.mps.current_allocated_memory()),
                                  sampled_mps_driver_bytes=int(torch.mps.driver_allocated_memory()))
                records.append(record)
                write(root/'resource_and_gradient.json', dict(scope='Eight-event prefix; actual backward and one optimizer update per condition',
                     sources=frozen_sources, diagnostic_source_sha256=diagnostic_hash, data_sha256=digest(rows), records=records))
                del model, optimizer, probe, logp, state, changed
                if device.type == 'mps':
                    torch.mps.empty_cache()
    # A complete lifetime update also traverses consolidation and explicit short-state erasure.
    torch.manual_seed(11); case = dict(variant='full', nodes=max(cfg['nodes']), read_path='separated')
    model = Network(model_config(cfg, case)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=.0001)
    rng = np.random.default_rng(192)
    groups = [quartet(rng, 'delayed', cfg['matrix_size'], split='resource') for _ in range(12*cfg['batch_size']//4)]
    groups.sort(key=lambda group: len(group[0]['events']), reverse=True)
    lifetime = [row for group in groups for row in group][:cfg['batch_size']]
    print(f"完整资源检查：N={case['nodes']}，{len(lifetime[0]['events'])}事件，J={cfg['internal_steps']}，完整反传。", flush=True)
    start = time.perf_counter(); metrics = train_update(model, optimizer, collate(lifetime), cfg, device); sync(device)
    complete = dict(status='completed', events=len(lifetime[0]['events']), nodes=case['nodes'],
                    internal_steps=cfg['internal_steps'], credit=cfg['credit'],
                    seconds=time.perf_counter()-start, metrics=metrics, data_sha256=digest(lifetime),
                    sampled_process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform == 'darwin' else 1024))
    write(root/'complete_lifetime_update.json', complete)
    source_files = [PROJECT/'run_biomatrix_v04.py', PROJECT/'configs/v04/base.json']
    for name in ('matrix_lab', 'v03', 'v04'):
        source_files.extend((PROJECT/name).rglob('*.py'))
    old = {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    write(root/'preserved_old_sources.json', old)
    if source_hashes() != frozen_sources or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != diagnostic_hash:
        write(root/'error.json', dict(status='failed', reason='Sources changed during preflight; rerun in a fresh directory'))
        raise RuntimeError('Sources changed during preflight; these measurements are not accepted')
    result = dict(utc=now(), sources=frozen_sources, diagnostic_source_sha256=diagnostic_hash, data_validation=validation, device=str(device),
                  resource_probes=dict(status='completed', conditions=len(records), events=8),
                  complete_lifetime_update=complete,
                  gpu=dict(status='not_executed', reason=deferred or 'CPU selected or GPU unavailable') if device.type == 'cpu' else dict(status='completed'),
                  formal_scan=dict(status='not_executed', conditions=len(cases(cfg))),
                  public_chinese_llm=dict(status='not_executed'),
                  interpretation='Engineering/resource probes only; a convergence or scaling claim requires paired formal runs.')
    write(root/'acceptance.json', result)
    print(f'检查结果已保存：{root}', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--device', choices=('cpu', 'mps', 'cuda', 'auto'), default='cpu')
    parser.add_argument('--root', type=Path, default=PROJECT/'results/v05/preflight')
    args = parser.parse_args(); cfg = json.loads((PROJECT/'configs/v05/base.json').read_text()); cfg['device'] = args.device
    preflight(cfg, args.root)
