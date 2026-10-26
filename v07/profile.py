"""Device resource checks, excluded from formal capability results."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import gc
import platform
from pathlib import Path
import resource
import time

import torch

from .common import Console, PROJECT, RunLock, device_for, digest, source_hashes, state_bytes, sync, write
from .data import Corpus, ROOT, prepare
from .model import initialize, parameter_count
from .run import build, cases, load_config, make_training_stream, model_config, train_update, training_batch, validate_config


def preflight(cfg, root, data_root=ROOT):
    validate_config(cfg); root = Path(root); root.mkdir(parents=True, exist_ok=True)
    device = device_for(cfg['device']); torch.set_num_threads(cfg['cpu_threads'])
    ui = Console(root/'events.jsonl')
    rows = []
    with RunLock(PROJECT/'results/v07/.queue.lock'), (RunLock(PROJECT/'results/v06/.queue.lock', record_owner=False) if device.type != 'cpu' else nullcontext()):
        from v05.run import check_other_training
        check_other_training(device, root)
        manifest = prepare(data_root, cfg['vocab_size'], ui); corpus = Corpus(data_root)
        unique = {}
        for planned in cases(cfg):
            mc = model_config(cfg, planned)
            unique.setdefault(digest([asdict(mc), planned.get('bptt_tokens', cfg['bptt_tokens'])]), planned)
        for case in unique.values():
            n, variant = case['nodes'], case['variant']
            probe = cfg | {k: v for k, v in case.items() if k in cfg}
            torch.manual_seed(11); mc = model_config(cfg, case); model = build(mc); initialize(model, 11); model = model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'])
            stream = make_training_stream(corpus, probe, 11)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            ui.stage(f'{device.type.upper()} 资源预检',
                     f'N={n} {variant}；B={probe["batch_size"]} T={probe["sequence_tokens"]}，完整反向与一次更新', total=2, unit='测量')
            try:
                # First call includes allocator/kernel warmup; second is reported separately.
                timings = []; metrics = None; state = None
                for i in range(2):
                    values, loss_mask, reset_mask, sample = stream.next()
                    sync(device); started = time.perf_counter()
                    metrics, state = train_update(model, optimizer, values, probe, device, i,
                                                 state=state, reset_mask=reset_mask, loss_mask=loss_mask)
                    sync(device); timings.append(time.perf_counter()-started); ui.update(loss=f'{metrics["loss"]:.3f}')
                row = dict(case=case, status='completed', device=str(device), dtype='float32',
                           model_config=asdict(mc),
                           source_hashes=source_hashes(),
                           parameters=parameter_count(model), state_bytes=state_bytes(state),
                           warmup_update_seconds=timings[0], update_seconds=timings[1],
                           tokens_per_second=int(loss_mask.sum())/timings[1],
                           probe_sequence_tokens=probe['sequence_tokens'], probe_bptt_tokens=probe['bptt_tokens'],
                           formal_bptt_tokens=probe['bptt_tokens'], metrics=metrics, sample=sample,
                           process_cumulative_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system() == 'Darwin' else 1024))
                if device.type == 'mps':
                    row.update(mps_allocated_bytes=torch.mps.current_allocated_memory(),
                               mps_driver_bytes=torch.mps.driver_allocated_memory(),
                               recommended_max_bytes=torch.mps.recommended_max_memory())
                elif device.type == 'cuda':
                    row.update(cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                               cuda_reserved_bytes=torch.cuda.memory_reserved(device),
                               cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                               cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                               cuda_device=torch.cuda.get_device_name(device))
                rows.append(row)
                del state
            except Exception as error:
                rows.append(dict(case=case, status='failed', error=repr(error)))
                write(root/'preflight.json', dict(data_manifest=manifest, conditions=rows, formal_training=False))
                raise
            finally:
                del model, optimizer; gc.collect()
                if device.type == 'mps':
                    torch.mps.empty_cache()
                elif device.type == 'cuda':
                    torch.cuda.empty_cache()
            write(root/'preflight.json', dict(data_manifest=manifest, conditions=rows, formal_training=False,
                                              note='Two engineering optimizer updates per condition; no accuracy or scaling claim. RSS is process-cumulative, MPS figures are snapshots, not isolated peaks.'))
    ui.stage('资源预检完成', f'{len(rows)} 组；结果 {root}/preflight.json')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--device', default='auto'); parser.add_argument('--root', type=Path, default=PROJECT/'results/v07/aligned_preflight')
    args = parser.parse_args(); preflight(load_config() | dict(device=args.device), args.root)
