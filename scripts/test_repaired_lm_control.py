"""A one-token predictor must ignore past-only substitutions in the LM assay."""
import hashlib
from pathlib import Path
import sys
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.repaired_lm_history_controls import token_losses
from v07.common import write


class TokenOnly(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(4096, 4)
        self.output = torch.nn.Linear(4, 4096, bias=False)

    def forward(self, x):
        return self.output(self.embed(x)), None, {}


if __name__ == '__main__':
    torch.manual_seed(63017)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = TokenOnly().to(device).eval()
    values = np.random.default_rng(63017).integers(3, 4096, (2, 513), dtype=np.int64)
    warm = token_losses(model, values, values, device)
    changed = values.copy(); changed[:, :383] = 17
    kept = token_losses(model, changed, values, device)
    np.testing.assert_allclose(warm, kept, atol=1e-6, rtol=0)
    changed[:, 383] = (values[:, 383]-2) % 4093+3
    current = token_losses(model, changed, values, device)
    np.testing.assert_allclose(warm[:, 1:], current[:, 1:], atol=1e-6, rtol=0)
    assert np.max(np.abs(warm[:, 0]-current[:, 0])) > 1e-4
    assert warm.dtype == np.float64 and warm.shape == (2, 128) and np.isfinite(warm).all()
    result = dict(status='completed', device=device, current_kept_invariance_max_abs=float(np.max(np.abs(warm-kept))),
        only_first_target_changes_when_current_input_changes=True, cpu_float64_aggregation=True,
        control_source_sha256=hashlib.sha256((PROJECT/'scripts/repaired_lm_history_controls.py').read_bytes()).hexdigest())
    write(PROJECT/'results/repaired/local333k_corrected_20261103/lm_control_smoke.json', result)
    print(result)
