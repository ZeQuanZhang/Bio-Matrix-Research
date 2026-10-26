"""The active, stateless content path of v0.9, with no Bio allocation."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from v07.common import digest
from v08.model import initialize as initialize_shared


@dataclass
class Config:
    vocab_size: int = 4096
    width: int = 64
    matrix_size: int = 8
    context_limit: int = 4096


class ContentOnly(nn.Module):
    """Same prediction map as the dense content path when Bio read gain is zero.

    Only the first half of v0.9's read matrix participates in that map.  There
    are no graph, credit, oscillator, address or dynamic-memory parameters.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.width)
        self.content_projection = nn.Linear(cfg.width, cfg.matrix_size**2, bias=False)
        self.read_projection = nn.Linear(cfg.matrix_size**2, cfg.width, bias=False)
        self.norm = nn.LayerNorm(cfg.width)
        self.ff = nn.Sequential(nn.Linear(cfg.width, 4*cfg.width), nn.GELU(),
                                nn.Linear(4*cfg.width, cfg.width))
        self.output_norm = nn.LayerNorm(cfg.width)

    def head(self, hidden):
        return self.output_norm(hidden+self.ff(self.norm(hidden)))

    def forward(self, tokens, state=None, reset_mask=None):
        hidden = self.read_projection(self.content_projection(self.embedding(tokens)))
        return F.linear(self.head(hidden), self.embedding.weight), {}, {}


@torch.no_grad()
def initialize(model, seed):
    """Preserve shared named initialization, including the sliced full read map."""
    initialize_shared(model, seed)
    generator = torch.Generator().manual_seed(int(digest([seed, 'read_projection.weight'])[:15], 16))
    width, columns = model.read_projection.weight.shape
    full_read = torch.randn((width, 2*columns), generator=generator)*.02
    model.read_projection.weight.copy_(full_read[:, :columns])
