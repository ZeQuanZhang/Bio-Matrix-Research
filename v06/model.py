"""Causal LM with one evolving ball graph, bounded matrix memories and local plasticity.

Local feedback is a finite response-filter approximation, compared separately with
no_local/full-BPTT controls. It is not an exact online replacement for BPTT.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from v04.model import rotate
from v05.model import rotate_rows
from .graph import solid_ball
from .common import digest

VARIANTS = ('message', 'independent', 'full', 'no_local', 'no_slow', 'no_probability',
            'no_dynamic', 'single_timescale', 'no_threshold', 'transformer')


@dataclass
class Config:
    vocab_size: int = 4096
    variant: str = 'full'
    nodes: int = 32
    matrix_size: int = 8
    width: int = 64
    bands: int = 3
    modes: int = 3
    trace_bands: int = 2
    neighbors: int = 8
    shells: int = 4
    internal_steps: int = 4
    graph_seed: int = 6101
    topology: str = 'connected'
    checkpoint_events: bool = False
    transformer_width: int = 64
    transformer_layers: int = 2
    transformer_ff: int = 256
    transformer_heads: int = 4
    context_limit: int = 128

    def __post_init__(self):
        if self.variant not in VARIANTS or self.matrix_size < 4 or self.matrix_size % 2:
            raise ValueError('Unknown variant or matrix_size not even and >= 4')
        if min(self.vocab_size, self.width, self.bands, self.modes, self.trace_bands, self.internal_steps) < 1:
            raise ValueError('All dimensions must be positive')
        if self.context_limit < 1:
            raise ValueError('context_limit must be positive')


def bounded(matrix, limit=4.):
    return matrix/(torch.linalg.vector_norm(matrix, dim=(-1, -2), keepdim=True)/limit).clamp_min(1.)


def joseph(m, p, key, value, variance, weight, process):
    """Exact rank-one expansion of the scalar-measurement Joseph update: O(d²)."""
    eye = torch.eye(m.shape[-1], device=m.device, dtype=m.dtype)
    prior = p+process[..., None, None]*eye
    pk = (prior@key[..., None]).squeeze(-1)
    denominator = (pk*key).sum(-1)+variance
    gain = weight[..., None]*pk/denominator.clamp_min(1e-6)[..., None]
    error = value-(m@key[..., None]).squeeze(-1)
    updated = m+error[..., :, None]*gain[..., None, :]
    post = (prior-gain[..., :, None]*pk[..., None, :]-pk[..., :, None]*gain[..., None, :]
            +denominator[..., None, None]*gain[..., :, None]*gain[..., None, :])
    return updated, .5*(post+post.transpose(-1, -2))+1e-6*eye


def jump_gate(score, training=False, blend=1., temperature=.35):
    soft = torch.sigmoid(score/temperature)
    hard = (score >= 0).to(score.dtype)
    if training:
        proxy = hard.detach()-soft.detach()+soft
        return (1-blend)*soft+blend*proxy
    return hard


class Network(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg; self.gate_blend = 1.
        d, w, n = cfg.matrix_size, cfg.width, cfg.nodes
        self.oscillatory = cfg.variant != 'message'
        for name, value in solid_ball(n, cfg.neighbors, cfg.shells, cfg.graph_seed, cfg.topology).items():
            self.register_buffer(name, value)
        self.embedding = nn.Embedding(cfg.vocab_size, w)
        self.node_shape = nn.Parameter(torch.randn(n, d, d)*.02)
        self.input_projection = nn.Linear(w, 2*d)
        self.controls = nn.Sequential(nn.Linear(12, w//2), nn.SiLU(),
                                      nn.Linear(w//2, 5+4*d+(d+3*cfg.modes if self.oscillatory else 0)))
        self.read_projection = nn.Linear(2*d*d, w, bias=False)
        self.norm = nn.LayerNorm(w)
        self.ff = nn.Sequential(nn.Linear(w, 4*w), nn.GELU(), nn.Linear(4*w, w))
        self.output_norm = nn.LayerNorm(w)
        self.raw_tau = nn.Parameter(torch.linspace(math.log(8), math.log(512), cfg.bands))
        self.raw_trace_tau = nn.Parameter(torch.linspace(math.log(4), math.log(64), cfg.trace_bands))
        self.raw_slow_rate = nn.Parameter(torch.tensor(-2.))
        self.raw_feedback_tau = nn.Parameter(torch.tensor(1.5))
        self.input_diagonal = nn.Parameter(torch.zeros(d))
        self.output_diagonal = nn.Parameter(torch.zeros(d))
        if self.oscillatory:
            # Single modes and adjacent cross-modes, never a Cartesian frequency grid.
            self.fourier = nn.Linear(24, 1+2*(2*cfg.modes-1))
            self.raw_coupling = nn.Parameter(torch.tensor(-1.5)) if cfg.variant != 'independent' else None
        else:
            self.message = nn.Sequential(nn.Linear(d*d, w//2), nn.SiLU(), nn.Linear(w//2, d*d))
        phase = torch.arange(n)[:, None]*2.399963+torch.arange(cfg.modes)[None]*.731
        self.register_buffer('initial_phase', torch.stack((phase.cos(), phase.sin()), -1))
        nn.init.normal_(self.embedding.weight, std=.02)
        nn.init.normal_(self.read_projection.weight, std=.02)
        nn.init.zeros_(self.controls[-1].bias)

    def initial_state(self, batch):
        c = self.cfg; d = c.matrix_size; device = self.embedding.weight.device
        base = (batch, c.nodes)
        zeros = lambda *shape: torch.zeros(*shape, device=device, dtype=self.embedding.weight.dtype)
        return dict(m=self.node_shape[None].expand(batch, -1, -1, -1).clone(), p=torch.eye(d, device=device).expand(*base, d, d).clone(),
                    x=zeros(*base, c.bands, d, d), slow=zeros(*base, d, d),
                    trace=zeros(*base, c.trace_bands, d), response=zeros(*base, d, d),
                    support=zeros(*base, d), innovation=zeros(*base),
                    z=self.initial_phase[None].expand(batch, -1, -1, -1).clone(),
                    prediction=zeros(batch, c.width), evidence_count=torch.zeros(*base, dtype=torch.long, device=device))

    def features(self, state):
        m, x = state['m'], state['x']
        summaries = torch.stack((m.mean((-1, -2)), m.square().mean((-1, -2)),
                                x.square().mean((-1, -2, -3)), state['p'].diagonal(dim1=-2, dim2=-1).mean(-1),
                                state['slow'].square().mean((-1, -2)), state['innovation']), -1)
        phase = state['z'].mean(-2) if self.oscillatory else torch.zeros_like(summaries[..., :2])
        geometry = torch.cat((self.positions, self.shell[:, None]/(self.cfg.shells-1)), -1)
        return torch.cat((summaries, phase, geometry[None].expand(len(m), -1, -1)), -1)

    def external(self, matrix, u, v, diagonal):
        if self.cfg.variant == 'no_dynamic':
            return matrix
        # (I + uvᵀ) applied without constructing a dense dynamic mapping.
        u, v = F.normalize(u, dim=-1, eps=1e-5), F.normalize(v, dim=-1, eps=1e-5)
        projection = (v[..., None, :]*matrix).sum(-1)
        return matrix*(1+.1*diagonal.tanh())+.15*projection[..., :, None]*u[..., None, :]

    def local_update(self, state, embedding):
        if self.cfg.variant in ('no_local', 'no_slow'):
            return state['slow']
        d = self.cfg.matrix_size
        # Negative CE gradient in the tied embedding space, from the newly visible token.
        error = embedding-state['prediction']
        projected = F.linear(error, self.read_projection.weight.T).reshape(len(error), 2, d, d).mean(1)
        eligibility = state['trace'].mean(-2)
        support = state['support'].mean(-1)
        agreement = torch.exp(-state['innovation'].clamp(0, 20))
        trust = (support/(support+2))*agreement
        update = projected[:, None]*eligibility[..., None, :]*state['response']
        return bounded(state['slow']+.1*self.raw_slow_rate.sigmoid()*trust[..., None, None]*update)

    def weights(self, content, x, drive, controls, features, state):
        source = self.neighbors
        intensity = (x.square().mean((-1, -2, -3))+drive.square().mean(-1)+1e-12).sqrt()
        source_intensity = intensity[:, source]
        similarity = F.cosine_similarity(content[:, :, None].flatten(-2), content[:, source].flatten(-2), dim=-1, eps=1e-5)
        threshold = .02+.12*controls[..., 2].sigmoid()
        score = source_intensity+.05*similarity-threshold[:, :, None]
        outward = jump_gate(score, self.training, self.gate_blend)
        if self.cfg.variant == 'no_threshold':
            outward = torch.ones_like(outward)
        gate = torch.where(self.outward[None], outward, torch.ones_like(outward))
        weight = gate*self.edge_mask[None]*(source_intensity/(.02+source_intensity))
        weight = weight*(.5+.5*similarity.sigmoid())
        if self.oscillatory:
            zi, zj = state['z'][:, :, None], state['z'][:, source]
            co = (zi*zj).sum(-1)
            si = zi[..., 0]*zj[..., 1]-zi[..., 1]*zj[..., 0]
            cross_co = co[..., :-1]*co[..., 1:]+si[..., :-1]*si[..., 1:]
            cross_si = si[..., :-1]*co[..., 1:]-co[..., :-1]*si[..., 1:]
            harmonic = torch.cat((co, cross_co, si, cross_si), -1)
            coeff = self.fourier(torch.cat((features[:, :, None].expand(-1, -1, source.shape[1], -1), features[:, source]), -1))
            # Positive, bounded Fourier modulation; sign-bearing content is transported separately.
            weight = weight*(.5+(coeff[..., 0]+(coeff[..., 1:]*harmonic).mean(-1)).sigmoid())
        else:
            si = None
        return weight/weight.sum(-1, keepdim=True).clamp_min(.1), gate, si

    def tick(self, state, drive, embedding_packet):
        c = self.cfg; d = c.matrix_size
        features = self.features(state); control = self.controls(features)
        uin, vin, uout, vout = control[..., 5:5+4*d].chunk(4, -1)
        content = state['x'].mean(-3)+.3*state['m']+.2*state['slow']
        content = self.external(content, uout, vout, self.output_diagonal)
        z = state['z']
        if self.oscillatory:
            angle, frequency, amplitude, forcing = torch.split(control[..., 5+4*d:], [d, c.modes, c.modes, c.modes], -1)
            angle = .12*angle.tanh()
            content = rotate_rows(content, angle[..., :d//2])
            content = rotate_rows(content.transpose(-1, -2), angle[..., d//2:]).transpose(-1, -2)
        else:
            content = self.message(content.flatten(-2)).reshape_as(content).tanh()
        weight, gate, sine = self.weights(content, state['x'], drive, control, features, state)
        message = (weight[..., None, None]*content[:, self.neighbors]).sum(2)
        received = (weight[..., None]*drive[:, self.neighbors]).sum(2)
        drive = .25*drive+.75*received
        drive = torch.where(self.write_ports[None, :, None].bool(), embedding_packet[:, None], drive)
        mass = drive[..., -1].clamp_min(.05)
        value = (drive[..., :d]/mass[..., None]).clamp(-2, 2)
        key = F.normalize(drive[..., d:2*d], dim=-1, eps=1e-5)
        packet = value[..., :, None]*key[..., None, :]
        packet = self.external(packet, uin, vin, self.input_diagonal)*drive[..., -1, None, None]
        tau = self.raw_tau.clamp(math.log(2), math.log(4096)).exp()
        if c.variant == 'single_timescale':
            tau = tau.mean().expand_as(tau)
        decay = torch.exp(-1/(tau*c.internal_steps))
        mixing = .1+.2*control[..., 3].sigmoid()
        target = message+.25*packet/c.internal_steps
        x = state['x']*decay[None, None, :, None, None]
        x = (1-mixing[..., None, None, None])*x+mixing[..., None, None, None]*target[..., None, :, :]
        x = bounded(x)
        if self.oscillatory:
            coupling = torch.zeros_like(frequency)
            if self.raw_coupling is not None:
                coupling = .2*self.raw_coupling.sigmoid()*(weight[..., None]*sine).sum(2)
            speed = .02+.35*frequency.sigmoid()+.08*forcing.tanh()+coupling
            z = rotate(z, speed/c.internal_steps)
            # Phase modulates a norm-preserving rotation; damping stays in the explicit time bands.
            carrier_angle = .15*(z[..., 0]*amplitude.sigmoid()).mean(-1)
            angles = carrier_angle[..., None, None].expand(*x.shape[:-2], d//2)
            x = rotate_rows(x, angles)
        return state | dict(x=x, z=z), drive, weight, gate, mixing

    def response_filter(self, flows, mixings, batch):
        """Bounded adjoint of scalar transport; nonlinear mapping Jacobians are approximated."""
        response = self.read_ports[None].expand(batch, -1)
        index = self.neighbors.flatten()[None].expand(batch, -1)
        for flow, mixing in zip(reversed(flows), reversed(mixings)):
            contributions = (response[..., None]*mixing[..., None]*flow).flatten(1)
            returned = torch.zeros_like(response).scatter_add(1, index, contributions)
            response = (1-mixing)*response+returned
        return (response*self.cfg.nodes).clamp(0, 2.)[..., None, None]

    def step(self, token, state):
        c = self.cfg; d = c.matrix_size; b = len(token)
        embedding = self.embedding(token)
        state = state | {'slow': self.local_update(state, embedding)}
        value, key = self.input_projection(embedding).chunk(2, -1)
        value = value.tanh(); key = F.normalize(key, dim=-1, eps=1e-5)
        packet = torch.cat((value, key, torch.ones(b, 1, device=token.device)), -1)
        drive = self.write_ports[None, :, None]*packet[:, None]
        flows, gates, mixings = [], [], []
        for _ in range(c.internal_steps):
            state, drive, weight, gate, mixing = self.tick(state, drive, packet)
            flows.append(weight); gates.append(gate); mixings.append(mixing)
        arrived = drive[..., -1] > 1e-4
        local_key = F.normalize(drive[..., d:2*d], dim=-1, eps=1e-5)
        local_value = (drive[..., :d]/drive[..., -1, None].clamp_min(.05)).clamp(-2, 2)
        prediction = (state['m']@local_key[..., None]).squeeze(-1)
        innovation = (local_value-prediction).square().mean(-1)
        controls = self.controls(self.features(state))
        reliability = controls[..., 0].sigmoid()*arrived
        variance = .02+.3*controls[..., 4].sigmoid()
        if c.variant == 'no_probability':
            m = .95*state['m']+.05*reliability[..., None, None]*local_value[..., :, None]*local_key[..., None, :]
            p = state['p']
        else:
            m, p = joseph(state['m'], state['p'], local_key, local_value, variance, reliability, .002*arrived)
            # No new observation count inside internal ticks. Correlated paths are one event.
            m = torch.where(arrived[..., None, None], m, state['m'])
            p = torch.where(arrived[..., None, None], p, state['p'])
        tau = self.raw_trace_tau.clamp(math.log(1), math.log(2048)).exp()
        decay = torch.exp(-1/tau)
        trace = state['trace']*decay[None, None, :, None]+(1-decay)[None, None, :, None]*local_key[..., None, :]*arrived[..., None, None]
        response = self.response_filter(flows, mixings, b).expand(-1, -1, d, d)
        blend = self.raw_feedback_tau.sigmoid()
        state = state | dict(m=bounded(m), p=p, trace=trace,
                             response=blend*state['response']+(1-blend)*response,
                             support=(.999*state['support']+local_key.square()*arrived[..., None]).clamp_max(100),
                             innovation=.9*state['innovation']+.1*innovation*arrived,
                             evidence_count=state['evidence_count']+arrived.long())
        fast = state['x'].mean(-3)
        memory = state['m']
        if c.variant != 'no_slow':
            compatibility = torch.exp(-innovation/(variance+.05))*controls[..., 1].sigmoid()
            memory = memory+compatibility[..., None, None]*state['slow']
        pooled = torch.cat(((fast*self.read_ports[None, :, None, None]).sum(1).flatten(1),
                            (memory*self.read_ports[None, :, None, None]).sum(1).flatten(1)), -1)
        hidden = embedding+self.read_projection(pooled)
        hidden = self.output_norm(hidden+self.ff(self.norm(hidden)))
        logits = F.linear(hidden, self.embedding.weight)
        state = state | {'prediction': logits.softmax(-1)@self.embedding.weight}
        outward = self.outward[None].expand(b, -1, -1)*self.edge_mask[None]
        diag = dict(outward_open=(gates[-1]*outward).sum()/outward.sum().clamp_min(1),
                    reached=arrived.float().mean(), state_rms=state['x'].square().mean().sqrt(),
                    slow_rms=state['slow'].square().mean().sqrt(),
                    covariance_min_diagonal=state['p'].diagonal(dim1=-2, dim2=-1).min(),
                    phase_coherence=state['z'].mean(1).square().sum(-1).sqrt().mean())
        return logits, state, diag

    def forward(self, tokens, state=None):
        state = self.initial_state(len(tokens)) if state is None else state
        outputs = []
        for t in range(tokens.shape[1]):
            if self.cfg.checkpoint_events and self.training and torch.is_grad_enabled():
                logits, state, diag = checkpoint(self.step, tokens[:, t], state, use_reentrant=False)
            else:
                logits, state, diag = self.step(tokens[:, t], state)
            outputs.append(logits)
        return torch.stack(outputs, 1), state, diag


class AttentionBlock(nn.Module):
    def __init__(self, width, heads, ff):
        super().__init__(); self.heads = heads; self.head_width = width//heads
        if width % heads or self.head_width % 2:
            raise ValueError('RoPE needs even per-head width')
        self.norm1 = nn.LayerNorm(width); self.norm2 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3*width, bias=False); self.out = nn.Linear(width, width, bias=False)
        self.ff = nn.Sequential(nn.Linear(width, ff), nn.GELU(), nn.Linear(ff, width))
        self.register_buffer('inv_frequency', 10000**(-torch.arange(0, self.head_width, 2).float()/self.head_width))

    def rope(self, x, positions):
        angles = positions[:, None]*self.inv_frequency[None]
        a, b = x.reshape(*x.shape[:-1], self.head_width//2, 2).unbind(-1)
        return torch.stack((a*angles.cos()-b*angles.sin(), a*angles.sin()+b*angles.cos()), -1).flatten(-2)

    def forward(self, x, past, position, limit):
        b, t, w = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, t, 3, self.heads, self.head_width).permute(2, 0, 3, 1, 4).unbind(0)
        positions = torch.arange(position, position+t, device=x.device, dtype=x.dtype)
        q, k = self.rope(q, positions), self.rope(k, positions)
        if past is not None:
            keep = max(0, limit-t)
            old_k, old_v = (p[:, :, -keep:] if keep else p[:, :, :0] for p in past)
            k, v = torch.cat((old_k, k), 2), torch.cat((old_v, v), 2)
        offset = k.shape[2]-t
        mask = torch.arange(k.shape[2], device=x.device)[None] <= (offset+torch.arange(t, device=x.device)[:, None])
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.)
        x = x+self.out(attended.transpose(1, 2).reshape(b, t, w))
        return x+self.ff(self.norm2(x)), (k, v)


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg
        w = cfg.transformer_width
        self.embedding = nn.Embedding(cfg.vocab_size, w)
        self.blocks = nn.ModuleList([AttentionBlock(w, cfg.transformer_heads, cfg.transformer_ff)
                                     for _ in range(cfg.transformer_layers)])
        self.norm = nn.LayerNorm(w)
        nn.init.normal_(self.embedding.weight, std=.02)

    def initial_state(self, batch):
        return dict(kv=[None]*len(self.blocks), position=0)

    def forward(self, tokens, state=None):
        state = self.initial_state(len(tokens)) if state is None else state
        if tokens.shape[1] > self.cfg.context_limit:
            raise ValueError('Transformer chunk exceeds fixed context budget')
        x = self.embedding(tokens); kv = []
        for block, past in zip(self.blocks, state['kv']):
            x, cache = block(x, past, state['position'], self.cfg.context_limit); kv.append(cache)
        return F.linear(self.norm(x), self.embedding.weight), dict(kv=kv, position=state['position']+tokens.shape[1]), {}


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def initialize(model, seed):
    """Shared names/shapes get identical initialization across graph controls."""
    for name, parameter in model.named_parameters():
        if name.startswith('raw_') or name in ('input_diagonal', 'output_diagonal'):
            continue
        generator = torch.Generator().manual_seed(int(digest([seed, name])[:15], 16))
        if name == 'embedding.weight' or name == 'node_shape' or name == 'read_projection.weight':
            parameter.copy_(torch.randn(parameter.shape, generator=generator)*.02)
        elif name.endswith('bias'):
            parameter.zero_()
        elif parameter.ndim == 1:
            parameter.fill_(1.)
        else:
            bound = 1/math.sqrt(parameter.shape[-1])
            parameter.copy_(torch.empty(parameter.shape).uniform_(-bound, bound, generator=generator))


def transformer_shape(target, vocab, layers=2, heads=4):
    """Choose active standard FF/attention widths, no dummy parameters."""
    candidates = []
    for width in range(16, 257, 8):
        for multiple in (2, 3, 4):
            ff = multiple*width
            count = vocab*width+layers*(4*width*width+2*width*ff+5*width+ff)+2*width
            candidates.append((abs(count-target), width, ff, count))
    _, width, ff, count = min(candidates)
    return dict(transformer_width=width, transformer_ff=ff, transformer_layers=layers,
                transformer_heads=heads), count
