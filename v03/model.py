"""Role-conditioned sparse content networks with nonperiodic or circular auxiliaries."""
from dataclasses import dataclass, asdict
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .graph import graph

MODELS = ('C1', 'C2', 'C3')


@dataclass
class Config:
    model: str = 'C3'
    nodes: int = 32
    state_dim: int = 16
    width: int = 128
    bottleneck: int = 32
    rank: int = 8
    neighbors: int = 8
    iterations: int = 16
    graph_seed: int = 901
    role_seed: int = 902
    phase_seed: int = 903
    topology: str = 'connected'
    vocab: int = 640
    answers: int = 80
    phase_to_messages: bool = True
    phase_to_router: bool = False


def rotate(z, delta):
    a, b = z.unbind(-1); c, s = delta.cos(), delta.sin()
    return F.normalize(torch.stack((a*c-b*s, a*s+b*c), -1), dim=-1)


class Network(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg
        if cfg.model not in MODELS:
            raise ValueError('Unknown core model')
        n, d, w, p = cfg.nodes, cfg.state_dim, cfg.width, cfg.bottleneck
        self.register_buffer('neighbors', graph(n, cfg.graph_seed, cfg.topology, cfg.neighbors).clone())
        rng = np.random.default_rng(cfg.role_seed)
        role = torch.tensor(rng.normal(size=(n, 8)), dtype=torch.float32)
        self.register_buffer('roles', F.normalize(role, dim=-1))
        theta = torch.tensor(np.random.default_rng(cfg.phase_seed).uniform(-math.pi, math.pi, n), dtype=torch.float32)
        self.register_buffer('initial_z', torch.stack((theta.cos(), theta.sin()), -1))
        self.embedding = nn.Embedding(cfg.vocab, w, padding_idx=0)
        self.record_position = nn.Parameter(torch.randn(12, w) / math.sqrt(w))
        self.query_encoder = nn.GRU(w, w, batch_first=True)
        self.record_encoder = nn.Sequential(nn.Linear(w, w), nn.Tanh())
        self.input_query = nn.Linear(w + d + 8, p)
        self.evidence_key = nn.Linear(w, p, bias=False)
        self.evidence_value = nn.Linear(w, p, bias=False)
        self.input_write = nn.Linear(p + w + 8, 2*d)
        self.self_projection = nn.Sequential(nn.Linear(d, p, bias=False), nn.Linear(p, 2*d, bias=False))
        self.edge_q = nn.Linear(d + p + 8, p, bias=False)
        self.edge_k = nn.Linear(d + p + 8, p, bias=False)
        self.message_in = nn.Linear(d, p, bias=False)
        self.message_out = nn.Linear(p, 2*d, bias=False)
        self.aux_drive = nn.Linear(d + p + 8, 2)
        self.raw_dt = nn.Parameter(torch.tensor(math.log(.2/.8)))
        if cfg.model == 'C3':
            self.raw_coupling = nn.Parameter(torch.tensor(0.))
        self.read_query = nn.Linear(w, d + 8, bias=False)
        self.read_gate = nn.Linear(d + 1, cfg.rank)
        self.base = nn.Linear(w, w)
        self.v = nn.Linear(w, cfg.rank, bias=False)
        self.u = nn.Linear(cfg.rank, w, bias=False)
        self.head = nn.Sequential(nn.GELU(), nn.Linear(w, cfg.answers))
        nn.init.constant_(self.input_write.bias[d:], 1.)

    def encode(self, records, question, mask):
        if records.shape[-1] > len(self.record_position):
            raise ValueError('Too many record fields')
        e = (self.embedding(records) * self.record_position[:records.shape[-1]]).sum(-2)
        evidence = self.record_encoder(e)
        # Right padding is explicitly excluded from the question recurrent state.
        embedded = self.embedding(question)
        sequence, _ = self.query_encoder(embedded)
        length = (question != 0).sum(-1).clamp_min(1)
        h = sequence[torch.arange(len(question), device=question.device), length - 1]
        return h, self.evidence_key(evidence), self.evidence_value(evidence), mask

    def update(self, s, z, encoded, coupling=True, messages=True):
        h, ek, ev, mask = encoded; n, d = self.cfg.nodes, self.cfg.state_dim
        role = self.roles[None].expand(len(s), -1, -1)
        query = self.input_query(torch.cat((h[:, None].expand(-1, n, -1), s, role), -1))
        attention = (query @ ek.transpose(1, 2) / math.sqrt(ek.shape[-1])).masked_fill(~mask[:, None], -1e4).softmax(-1)
        inp = attention @ ev
        feature = torch.cat((s, inp, role), -1)
        eq, ek = self.edge_q(feature), self.edge_k(feature)
        neighbor_key = ek[:, self.neighbors]
        scores = (eq[:, :, None] * neighbor_key).sum(-1) / math.sqrt(eq.shape[-1])
        pi = scores.softmax(-1)
        zj = z[:, self.neighbors]
        if self.cfg.model == 'C1':
            chi = (z[:, :, None] * zj).sum(-1).sigmoid()
        else:
            chi = (1 + (z[:, :, None] * zj).sum(-1)) / 2
        gate = chi if self.cfg.phase_to_messages else torch.ones_like(chi)
        projected = self.message_in(s)
        message = (pi[..., None] * gate[..., None] * projected[:, self.neighbors]).sum(2)
        if not messages:
            message = message * 0
        pre = (self.input_write(torch.cat((inp, h[:, None].expand(-1, n, -1), role), -1))
               + self.self_projection(s) + self.message_out(message))
        candidate, retain = pre.chunk(2, -1)
        retain = retain.sigmoid()
        new_s = retain*s + (1-retain)*candidate.tanh()
        drive = self.aux_drive(feature).tanh()
        dt = .5*self.raw_dt.sigmoid()
        if self.cfg.model == 'C1':
            # Both nonperiodic coordinates participate in communication and readout.
            target = (drive + (pi[..., None] * zj).sum(2)).tanh()
            new_z = (1-dt)*z + dt*target
        else:
            force = torch.zeros_like(drive[..., 0])
            if self.cfg.model == 'C3' and coupling:
                signed = scores.tanh()
                signed = signed / (signed.abs().sum(-1, keepdim=True) + 1e-6)
                cross = z[:, :, None, 0]*zj[..., 1] - z[:, :, None, 1]*zj[..., 0]
                force = 2*self.raw_coupling.sigmoid()*(signed*cross).sum(-1)
            new_z = rotate(z, dt*(.1*drive[..., 0] + .5*drive[..., 1] + force))
        return new_s, new_z

    def read(self, h, s, z):
        feature = torch.cat((s, self.roles[None].expand(len(s), -1, -1)), -1)
        weight = (feature*self.read_query(h)[:, None]).sum(-1).softmax(-1)
        mean = (weight[..., None]*s).sum(1)
        coherence = (weight[..., None]*z).sum(1).square().sum(-1, keepdim=True).add(1e-8).sqrt()
        gate = self.read_gate(torch.cat((mean, coherence), -1)).sigmoid()
        return self.head(self.base(h) + self.u(gate*self.v(h)))

    def forward(self, records, question, mask, iterations=None, intervention=None, trace=False):
        encoded = self.encode(records, question, mask)
        s = encoded[0].new_zeros(len(records), self.cfg.nodes, self.cfg.state_dim)
        z = self.initial_z[None].expand(len(records), -1, -1)
        history = []
        count = self.cfg.iterations if iterations is None else iterations
        for step in range(count):
            if intervention and step == count // 2:
                s, z = self.intervene(s, z, intervention)
            s, z = self.update(s, z, encoded, coupling=intervention != 'no_coupling', messages=intervention != 'no_messages')
            if trace:
                history.append(z.detach())
        logits = self.read(encoded[0], s, z)
        return logits, (s, z, history)

    def intervene(self, s, z, kind):
        if kind == 'rotate':
            z = rotate(z, z.new_full(z.shape[:-1], .73))
        elif kind == 'phase_swap':
            z = z.roll(1, 1)
        elif kind == 'phase_scramble':
            angles = torch.arange(self.cfg.nodes, device=z.device, dtype=z.dtype)*2.399963
            z = rotate(z, angles[None].expand(z.shape[0], -1))
        elif kind == 'content_matched':
            disrupted = z.roll(1, 1)-z
            perturbation = s.roll(1, 1)-s
            energy = disrupted.square().sum((1, 2), keepdim=True).sqrt()
            norm = perturbation.square().sum((1, 2), keepdim=True).sqrt().clamp_min(1e-6)
            s = s + perturbation * (energy/norm)
        return s, z

    def ledger(self, records=32):
        c = self.cfg; n, d, p, w, k = c.nodes, c.state_dim, c.bottleneck, c.width, c.neighbors
        # Dense linear MACs plus attention/message dot products; excludes nonlinear/reduction operations.
        per_round = n*((w+d+8)*p + p*records*2 + (p+w+8)*2*d + 3*d*p
                       + 2*(d+p+8)*p + 2*p*k + d*p + 2*p*d + 2*(d+p+8))
        return {'config': asdict(c), 'trainable_parameters': sum(v.numel() for v in self.parameters()),
                'current_state_floats_per_sample': n*(d+2), 'role_buffer_floats': n*8,
                'input_caches_per_sample_floats': records*2*p+w,
                'graph_index_int64': n*k, 'activation_storage_is_additional': True,
                'dense_and_attention_macs_per_internal_round_estimate': per_round,
                'mac_estimate_record_count': records, 'mac_exclusions': 'embedding, GRU question encoder, readout, nonlinearities, autograd; measured full runtime is primary',
                'sparse_edge_gather_shape': ['B', n, k, p]}
