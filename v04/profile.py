"""One real training update per size/variant; sampled memory is not peak memory."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch
from tqdm import tqdm
from matrix_lab.run import sync
from .data import collate, training_batch
from .model import Network
from .run import PROJECT, model_config, select_device, train_update, write, source_hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='auto'); parser.add_argument('--nodes', nargs='+', type=int, default=[32, 128, 512])
    parser.add_argument('--events', type=int, default=8)
    parser.add_argument('--output', type=Path, default=PROJECT/'results/v04/implementation_checks/resource_profile.json')
    args = parser.parse_args(); device = select_device(args.device); torch.set_num_threads(4)
    cfg = json.loads((PROJECT/'configs/v04/base.json').read_text()); records = []
    if args.events < 8:
        raise ValueError('At least eight events are needed to include a scored query')
    batch = collate(training_batch(np.random.default_rng(191), profile='long' if args.events > 8 else 'history'))
    batch = {k: v[:, :args.events] for k, v in batch.items()}
    for nodes, variant in tqdm([(n, v) for n in args.nodes for v in ('graph', 'full')], desc='资源剖析', file=sys.stdout):
        torch.manual_seed(11); case = dict(nodes=nodes, variant=variant)
        model = Network(model_config(cfg, case)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'])
        sync(device); tick = time.perf_counter()
        train_update(model, optimizer, batch, cfg, device); sync(device)
        row = dict(nodes=nodes, variant=variant, events=batch['key'].shape[1], batch_size=cfg['batch_size'],
                   internal_steps=cfg['internal_steps'], seconds=time.perf_counter()-tick,
                   ledger=model.ledger(), device=str(device), torch=str(torch.__version__),
                   process_rss_highwater_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024))
        if device.type == 'mps':
            row.update(sampled_mps_current_bytes=int(torch.mps.current_allocated_memory()),
                       sampled_mps_driver_bytes=int(torch.mps.driver_allocated_memory()))
        records.append(row); write(args.output, dict(scope='One cold training update per condition; sampled memory is not peak memory', sources=source_hashes(), records=records))
        tqdm.write(f"{variant} N={nodes}: {row['seconds']:.2f}s/update, params={row['ledger']['parameters']}", file=sys.stdout)
        del model, optimizer
        if device.type == 'mps':
            torch.mps.empty_cache()


if __name__ == '__main__':
    main()
