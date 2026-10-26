"""Causal Transformer with an optional eight-state biochemical computation module."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


def load_checkpoint(source, device="cpu"):
    # Older local runs stored TorchVersion metadata; no arbitrary pickle globals are allowed.
    with torch.serialization.safe_globals([type(torch.__version__)]):
        return torch.load(source, map_location=device, weights_only=True)


def _scan(a, b, initial):
    """Inclusive affine scan: state[t] = a[t] * state[t-1] + b[t]."""
    step = 1
    while step < b.shape[1]:
        b = torch.cat((b[:, :step], b[:, step:] + a[:, step:] * b[:, :-step]), 1)
        a = torch.cat((a[:, :step], a[:, step:] * a[:, :-step]), 1)
        step *= 2
    return b + a * initial[:, None]


class AffineScan(torch.autograd.Function):
    # Save only the actual trajectory; retaining every parallel-scan stage wastes memory.
    @staticmethod
    def forward(ctx, a, b, initial):
        ctx.metal = a.device.type == "mps" and a.dtype == b.dtype == initial.dtype == torch.float32
        if ctx.metal:
            import metal_scan
            states, a, initial = metal_scan.forward(a, b, initial)
        else:
            states = _scan(a, b, initial)
        ctx.save_for_backward(a, states, initial)
        return states

    @staticmethod
    def backward(ctx, gradient):
        a, states, initial = ctx.saved_tensors
        if ctx.metal:
            import metal_scan
            return metal_scan.backward(a, states, initial, gradient)
        shifted = torch.cat((a[:, 1:], torch.zeros_like(a[:, :1])), 1)
        adjoint = _scan(shifted.flip(1), gradient.flip(1), torch.zeros_like(initial)).flip(1)
        previous = torch.cat((initial[:, None], states[:, :-1]), 1)
        return adjoint * previous, adjoint, adjoint[:, 0] * a[:, 0]


class BioState(nn.Module):
    """Eight coupled, bounded state trajectories per channel, with positive rates.

    These are phenomenological biochemical dynamics, not a full molecular simulator.
    State is explicit and belongs to a sequence, never to the shared model object.
    """
    names = ("calcium", "release", "resources", "receptor", "membrane",
             "adaptation", "eligibility", "plasticity")

    def __init__(self, width):
        super().__init__()
        # Each channel starts at a different time scale; training can change every rate.
        rates = torch.tensor([.12, .08, .04, .15, .25, .015, .035, .002])
        scale = torch.logspace(-.3, .3, width)
        self.log_rates = nn.Parameter(torch.log(rates[:, None] * scale))
        self.coupling = nn.Parameter(torch.zeros(5, width))
        self.readout = nn.Parameter(torch.full((3, width), .1))

    def forward(self, x, state=None, mode="bio"):
        batch, length, width = x.shape
        if state is None:
            state = x.new_zeros(batch, 8, width)
            state[:, 2] = 1  # A fresh sequence has a full releasable resource pool.
        if state.shape != (batch, 8, width):
            raise ValueError("Bio state must have shape [batch, 8, channels].")
        rates = self.log_rates.clamp(-9, 2).exp()
        coupling = self.coupling.sigmoid()
        tracks = []

        def evolve(index, equilibrium, rate):
            decay = torch.exp(-rate).expand_as(equilibrium)
            drive = (1 - decay) * equilibrium
            if mode == "stateless":
                value = decay * state[:, index:index + 1] + drive
            elif mode == "bio":
                value = AffineScan.apply(decay, drive, state[:, index])
            else:
                raise ValueError(f"Unknown Bio mode: {mode}")
            tracks.append(value)
            return value

        pre = x.tanh()
        stimulus = x.sigmoid()
        calcium = evolve(0, stimulus, rates[0])
        release = evolve(1, (stimulus + calcium) / 2, rates[1])
        depletion = rates[2] * release * calcium * (1 + coupling[0])
        resources = evolve(2, rates[2] / (rates[2] + depletion), rates[2] + depletion)
        previous_resources = (state[:, 2:3].expand_as(resources) if mode == "stateless" else
                              torch.cat((state[:, 2:3], resources[:, :-1]), 1))
        transmitter = release * calcium * previous_resources
        binding = rates[3] * transmitter * (1 + coupling[1])
        unbinding = rates[3] * (.05 + coupling[2])
        receptor = evolve(3, binding / (binding + unbinding), binding + unbinding)
        membrane = evolve(4, torch.tanh(pre + 2 * receptor - 1), rates[4])
        adaptation = evolve(5, membrane.sigmoid(), rates[5])
        post = torch.tanh(membrane - coupling[3] * (adaptation - .5))
        eligibility = evolve(6, pre * post, rates[6])
        # An input-dependent modulation signal gates consolidation of the eligibility trace.
        modulation = (stimulus - calcium).abs() * coupling[4] + .01
        plasticity = evolve(7, eligibility, rates[7] * modulation)
        gain = self.readout.tanh()
        output = x * (1 + gain[2] * plasticity)
        output = output + gain[0] * post + gain[1] * (receptor - adaptation)
        return output, torch.stack([s[:, -1] for s in tracks], 1)


@dataclass
class Config:
    vocab_size: int = 8192
    width: int = 256
    layers: int = 4
    heads: int = 4
    context: int = 256
    variant: str = "baseline"
    ff_width: int = 1024


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads
        self.norm1 = nn.LayerNorm(cfg.width)
        self.qkv = nn.Linear(cfg.width, cfg.width * 3, bias=False)
        self.projection = nn.Linear(cfg.width, cfg.width, bias=False)
        self.norm2 = nn.LayerNorm(cfg.width)
        self.up = nn.Linear(cfg.width, cfg.ff_width, bias=False)
        self.down = nn.Linear(cfg.ff_width, cfg.width, bias=False)
        self.bio = BioState(cfg.width) if cfg.variant != "baseline" else None

    def forward(self, x, state=None, mode="bio", attention_window=None):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q, k, v = [t.view(batch, length, self.heads, width // self.heads).transpose(1, 2)
                   for t in (q, k, v)]
        mask = None
        if attention_window is not None:
            positions = torch.arange(length, device=x.device)
            distance = positions[:, None] - positions[None, :]
            mask = (distance >= 0) & (distance < attention_window)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                is_causal=mask is None)
        x = x + self.projection(attended.transpose(1, 2).reshape(batch, length, width))
        transformed = self.down(F.gelu(self.up(self.norm2(x))))
        if self.bio is not None:
            transformed, state = self.bio(transformed, state, mode)
        return x + transformed, state


class LanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.width % cfg.heads:
            raise ValueError("width must be divisible by heads")
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.width)
        self.position = nn.Embedding(cfg.context, cfg.width)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.norm = nn.LayerNorm(cfg.width)
        # Standard Gaussian initialization; equal seeds do not imply equal differently shaped weights.
        for name, parameter in self.named_parameters():
            if parameter.ndim >= 2 and not any(s in name for s in (".bio.",)):
                nn.init.normal_(parameter, std=.02)

    def forward(self, tokens, states=None, mode="bio", attention_window=None):
        length = tokens.shape[1]
        if length > self.cfg.context:
            raise ValueError("Input exceeds configured context; use sequential chunks.")
        x = self.embedding(tokens) + self.position(torch.arange(length, device=tokens.device))
        updated = []
        for i, block in enumerate(self.blocks):
            x, state = block(x, None if states is None else states[i], mode, attention_window)
            updated.append(state)
        return F.linear(self.norm(x), self.embedding.weight), updated

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def detach_states(states):
    return [None if state is None else state.detach() for state in states]
