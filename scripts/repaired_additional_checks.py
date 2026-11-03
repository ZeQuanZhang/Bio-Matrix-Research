"""Read-only arbitrary-boundary causality and full-length CPU/MPS gradient parity."""
from pathlib import Path
import sys
import torch
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from evaluations.repaired.model import build, NAMES
from evaluations.repaired.data import ROOT
from v07.data import Corpus
from v07.common import write
from transformers.utils import logging


def main():
    logging.set_verbosity_error(); torch.set_num_threads(4)
    corpus = Corpus(ROOT); values, _, _ = corpus.sample('train', np.random.default_rng(62003), 1, 512)
    x = torch.as_tensor(values[:, :-1]); y = torch.as_tensor(values[:, 1:]); changed = x.clone(); changed[:, 371:] = 17
    rows = {}
    for name in NAMES:
        m = build(name, 11).eval()
        with torch.no_grad():
            expected = m(x)[0]; altered = m(changed)[0]
            torch.testing.assert_close(expected[:, :371], altered[:, :371], atol=4e-5, rtol=4e-4)
            _, state, _ = m(x[:, :371]); continuation = m(x[:, 371:], state)[0]
            torch.testing.assert_close(expected[:, 371:], continuation, atol=4e-5, rtol=4e-4)
        logits = m(x)[0]
        loss = torch.nn.functional.cross_entropy(logits[:, -128:].reshape(-1, 4096), y[:, -128:].reshape(-1))
        loss.backward(); cpu_grad = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
        m.zero_grad(set_to_none=True); m.to('mps')
        metal = m(x.to('mps'))[0]
        metal_loss = torch.nn.functional.cross_entropy(metal[:, -128:].reshape(-1, 4096), y[:, -128:].reshape(-1).to('mps'))
        metal_loss.backward()
        torch.testing.assert_close(logits.detach(), metal.detach().cpu(), atol=4e-5, rtol=4e-4)
        differences = []
        for n, p in m.named_parameters():
            if n in cpu_grad:
                assert p.grad is not None
                torch.testing.assert_close(cpu_grad[n], p.grad.cpu(), atol=5e-5, rtol=1e-3)
                differences.append(float((cpu_grad[n]-p.grad.cpu()).abs().max()))
        rows[name] = dict(status='completed', sequence=512, causal_cut=371,
            logit_cpu_mps_max_abs=float((logits.detach()-metal.detach().cpu()).abs().max()),
            gradient_max_abs=max(differences), compared_gradients=len(differences))
        print(name, rows[name], flush=True)
        del m, logits, metal, loss, metal_loss, cpu_grad
        torch.mps.empty_cache()
    write(PROJECT/'results/repaired/local333k_corrected_20261103/additional_checks.json', dict(status='completed', conditions=rows,
        note='Separate read-only models; no production checkpoint or frozen source modifications'))


if __name__ == '__main__':
    main()
