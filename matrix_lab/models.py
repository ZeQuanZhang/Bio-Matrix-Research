"""Explicit recurrent states, coupled reactions and factorized dynamic readout."""
from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass
class Config:
    cell: str = "reaction"
    readout: str = "dynamic"
    width: int = 32
    state: int = 32
    reaction_rank: int = 8
    matrix_rank: int = 16
    keys: int = 8
    values: int = 4


VARIANTS = {
    "gru_add": ("gru", "add"),
    "linear_add": ("linear", "add"),
    "gated_add": ("gated", "add"),
    "reaction_add": ("reaction", "add"),
    "gru_dynamic": ("gru", "dynamic"),
    "gated_dynamic": ("gated", "dynamic"),
    "reaction_dynamic": ("reaction", "dynamic"),
}


class MemoryModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, r, k, q = cfg.width, cfg.state, cfg.matrix_rank, cfg.reaction_rank
        if cfg.cell not in ("gru", "linear", "gated", "reaction") or cfg.readout not in ("add", "dynamic"):
            raise ValueError("Unknown state or readout mechanism")
        self.operation = nn.Embedding(3, d)
        self.key = nn.Embedding(cfg.keys + 1, d)
        self.value = nn.Embedding(cfg.values + 1, d)
        self.encoder = nn.Sequential(nn.Linear(d, d), nn.Tanh())
        self.base = nn.Linear(d, d)
        self.gate = nn.Linear(r, k, bias=False)
        self.v = nn.Linear(d, k, bias=False)
        self.u = nn.Linear(k, d, bias=False)
        self.head = nn.Sequential(nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, cfg.values))
        chunks = 3 if cfg.cell == "gru" else 2
        self.input_state = nn.Linear(d, chunks * r)
        if cfg.cell != "linear":
            self.state_state = nn.Linear(r, chunks * r, bias=cfg.cell == "gru")
        if cfg.cell == "reaction":
            self.pq = nn.Linear(r, 2 * q, bias=False)
            self.reaction = nn.Linear(q, r, bias=False)
            nn.init.normal_(self.reaction.weight, std=.05 / math.sqrt(q))
        if cfg.cell != "gru":
            with torch.no_grad():
                self.input_state.bias[r:].fill_(-2.)

    def step(self, encoded, state):
        r = self.cfg.state
        if self.cfg.cell == "gru":
            ir, iz, inn = encoded.chunk(3, -1)
            hr, hz, hn = self.state_state(state).chunk(3, -1)
            reset, retain = (ir + hr).sigmoid(), (iz + hz).sigmoid()
            candidate = (inn + reset * hn).tanh()
            return retain * state + (1 - retain) * candidate
        candidate, update = encoded[:, :r], encoded[:, r:]
        if self.cfg.cell != "linear":
            feedback, gate_feedback = self.state_state(state).chunk(2, -1)
            candidate, update = candidate + feedback, update + gate_feedback
        if self.cfg.cell == "reaction":
            p, q = self.pq(state).chunk(2, -1)
            candidate = candidate + self.reaction(p * q)
        update = update.sigmoid()
        return (1 - update) * state + update * candidate.tanh()

    def read(self, x, state):
        gate, query = self.gate(state).sigmoid(), self.v(x)
        # Both controls use the same active parameters and nonlinear output head.
        feature = gate * query if self.cfg.readout == "dynamic" else gate + query
        return self.head(self.base(x) + self.u(feature))

    def forward(self, events, queries=(), state=None, intervention="carry"):
        if intervention not in ("carry", "reset", "swap"):
            raise ValueError("Unknown state intervention")
        x = self.encoder(self.operation(events[..., 0]) + self.key(events[..., 1]) + self.value(events[..., 2]))
        encoded = self.input_state(x)
        if state is None:
            state = x.new_zeros(events.shape[0], self.cfg.state)
        if state.shape != (events.shape[0], self.cfg.state):
            raise ValueError("State belongs to another batch or configuration")
        answers = []
        for t in range(events.shape[1]):
            if t in queries:
                if intervention == "reset":
                    state = torch.zeros_like(state)
                elif intervention == "swap":
                    state = state.roll(1, 0)
            state = self.step(encoded[:, t], state)
            if t in queries:
                answers.append(self.read(x[:, t], state))
        logits = torch.stack(answers, 1) if answers else x.new_empty(events.shape[0], 0, self.cfg.values)
        return logits, state

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
