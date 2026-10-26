"""Bounded Bio matrices with a continuous lifetime and a calibrated local read rule.

Local feedback is the exact VJP of the previous explicit associative slow read,
conditional on its cached query and read weights. Indirect recurrent credit still
requires BPTT. Scalar reverse flow defines distributed slow pooling and a positive
step-size preconditioner; it is not the nonlinear network's recurrent derivative.
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
            'no_dynamic', 'single_timescale', 'no_threshold', 'bypass', 'stateless', 'transformer')


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
    context_limit: int = 256
    covariance_tau: float = 4096.
    covariance_process: float = .002
    local_write_normalization: str = 'participation'
    local_write_reference: int = 32
    credit_horizon: int = 8

    def __post_init__(self):
        if self.variant not in VARIANTS or self.matrix_size < 4 or self.matrix_size % 2:
            raise ValueError('Unknown variant or matrix_size not even and >= 4')
        if min(self.vocab_size, self.width, self.bands, self.modes, self.trace_bands, self.internal_steps) < 1:
            raise ValueError('All dimensions must be positive')
        if self.context_limit < 1:
            raise ValueError('context_limit must be positive')
        if (not math.isfinite(self.covariance_tau) or not 1 <= self.covariance_tau <= 65536
                or not math.isfinite(self.covariance_process) or self.covariance_process < 0):
            # Resolve the stationary-prior renewal coefficient reliably in FP32.
            raise ValueError('Covariance tau must be in [1,65536], with finite nonnegative process noise')
        if (self.local_write_normalization not in ('participation', 'none')
                or self.local_write_reference < 1 or not isinstance(self.credit_horizon, int) or self.credit_horizon < 1):
            raise ValueError('Invalid local-write normalization or credit horizon')


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


def jump_gate(score, training=False, blend=1., temperature=.01):
    soft = torch.sigmoid(score/temperature)
    hard = (score >= 0).to(score.dtype)
    if training:
        proxy = hard.detach()-soft.detach()+soft
        return (1-blend)*soft+blend*proxy
    return hard


def unit_rotate(phase, angle):
    """Retain circular phase coordinates after long float32 evolution."""
    return F.normalize(rotate(phase, angle), dim=-1, eps=1e-6)


def layer_norm_vjp(x, direction, layer):
    """Analytic input VJP of affine LayerNorm (population variance)."""
    centered = x-x.mean(-1, keepdim=True)
    inv = torch.rsqrt(centered.square().mean(-1, keepdim=True)+layer.eps)
    normalized = centered*inv
    weighted = direction*layer.weight
    return inv*(weighted-weighted.mean(-1, keepdim=True)
                -normalized*(weighted*normalized).mean(-1, keepdim=True))


def invalidate_feedback(state):
    """An optimizer update invalidates cached head derivatives, not runtime memory."""
    if state is not None and 'feedback_valid' in state:
        return state | {name: torch.zeros_like(state[name]) for name in ('feedback_valid', 'credit_valid')}
    return state


def reset_lanes(state, fresh, reset_mask):
    if reset_mask is None:
        return state
    if reset_mask.dtype != torch.bool or reset_mask.shape != (len(next(iter(fresh.values()))),):
        raise ValueError('reset_mask must be boolean [batch]')
    return {key: torch.where(reset_mask.reshape(-1, *([1]*(value.ndim-1))), fresh[key], value)
            for key, value in state.items()}


class Network(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg; self.gate_blend = 1.
        self.runtime_local_enabled = True; self.runtime_slow_enabled = True
        self.runtime_automatic_feedback = True
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
        history = c.credit_horizon-1
        return dict(m=bounded(self.node_shape)[None].expand(batch, -1, -1, -1).clone(),
                    p=torch.eye(d, device=device, dtype=self.embedding.weight.dtype).expand(*base, d, d).clone(),
                    x=zeros(*base, c.bands, d, d), slow=zeros(*base, d, d),
                    trace=zeros(*base, c.trace_bands, d), response=zeros(*base, d, d),
                    support=zeros(*base, d), innovation=zeros(*base),
                    z=self.initial_phase[None].expand(batch, -1, -1, -1).clone(),
                    prediction=zeros(batch, c.width), feedback_h=zeros(batch, c.width),
                    read_query=zeros(*base, d), read_weight=zeros(*base),
                    feedback_gain=zeros(*base), read_scale=torch.ones(batch, device=device, dtype=self.embedding.weight.dtype),
                    feedback_valid=torch.zeros(batch, dtype=torch.bool, device=device),
                    credit_prediction=zeros(batch, history, c.width), credit_h=zeros(batch, history, c.width),
                    credit_query=zeros(batch, history, c.nodes, d), credit_weight=zeros(batch, history, c.nodes),
                    credit_gain=zeros(batch, history, c.nodes), credit_scale=zeros(batch, history),
                    credit_valid=torch.zeros(batch, history, dtype=torch.bool, device=device),
                    evidence_count=torch.zeros(*base, dtype=torch.long, device=device))

    def reset_fast(self, state):
        """Clear transient dynamics/credit; retain slow associations and exposure trust."""
        return self.initial_state(len(state['slow'])) | {'slow': state['slow'], 'support': state['support']}

    def clear_slow(self, state):
        """A content ablation: retain fast dynamics/exposure, invalidate old credit."""
        return invalidate_feedback(state | {'slow': torch.zeros_like(state['slow'])})

    def features(self, state):
        m, x = state['m'], state['x']
        slow = state['slow'] if self.runtime_slow_enabled and self.cfg.variant != 'no_slow' else torch.zeros_like(state['slow'])
        uncertainty = state['p'].diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(0)
        innovation = state['innovation'].clamp_min(0)
        summaries = torch.stack((m.mean((-1, -2)), m.square().mean((-1, -2)),
                                x.square().mean((-1, -2, -3)), uncertainty/(1+uncertainty),
                                slow.square().mean((-1, -2)), innovation/(1+innovation)), -1)
        phase = state['z'].mean(-2) if self.oscillatory else torch.zeros_like(summaries[..., :2])
        geometry = torch.cat((self.positions, self.shell[:, None]/(self.cfg.shells-1)), -1)
        return torch.cat((summaries, phase, geometry[None].expand(len(m), -1, -1)), -1)

    def external(self, matrix, u, v, diagonal):
        if self.cfg.variant == 'no_dynamic':
            return matrix
        # Right mapping M[diag(1+.1tanh(diagonal)) + .15 v uᵀ], without a dense product.
        u, v = F.normalize(u, dim=-1, eps=1e-5), F.normalize(v, dim=-1, eps=1e-5)
        projection = (v[..., None, :]*matrix).sum(-1)
        return matrix*(1+.1*diagonal.tanh())+.15*projection[..., :, None]*u[..., None, :]

    def observe(self, value, key, controls):
        """State-conditioned input features in each node's fixed latent axes."""
        if self.cfg.variant != 'no_dynamic':
            d = self.cfg.matrix_size
            u, v = controls[..., 5:5+2*d].chunk(2, -1)
            u, v = F.normalize(u, dim=-1, eps=1e-5), F.normalize(v, dim=-1, eps=1e-5)
            diagonal = 1+.1*self.input_diagonal.tanh()
            value = diagonal*value+.15*u*(v*value).sum(-1, keepdim=True)
            key = diagonal*key+.15*u*(v*key).sum(-1, keepdim=True)
        return value.tanh(), F.normalize(key, dim=-1, eps=1e-5)

    def stationary_prior(self, state):
        """Mean-reverting latent matrix prior with an analytically bounded covariance."""
        rho = math.exp(-1/self.cfg.covariance_tau)
        anchor = bounded(self.node_shape)[None]
        mean = math.sqrt(rho)*state['m']+(1-math.sqrt(rho))*anchor
        eye = torch.eye(self.cfg.matrix_size, dtype=state['p'].dtype, device=state['p'].device)
        covariance = rho*state['p']+(1-rho)*eye
        return mean, covariance

    def covariance_upper_bound(self):
        rho = math.exp(-1/self.cfg.covariance_tau)
        return 1+(self.cfg.covariance_process+1e-6)/(1-rho)

    def head(self, hidden):
        return self.output_norm(hidden+self.ff(self.norm(hidden)))

    def head_only(self, tokens):
        return F.linear(self.head(self.embedding(tokens)), self.embedding.weight)

    def head_vjp(self, hidden, direction):
        """Input VJP through both LayerNorms and the complete residual FFN."""
        normalized = self.norm(hidden)
        activation = self.ff[0](normalized)
        residual = hidden+self.ff[2](F.gelu(activation))
        outer = layer_norm_vjp(residual, direction, self.output_norm)
        gelu_derivative = (.5*(1+torch.erf(activation/math.sqrt(2)))
                           +activation*torch.exp(-activation.square()/2)/math.sqrt(2*math.pi))
        through_ff = F.linear(F.linear(outer, self.ff[2].weight.T)*gelu_derivative, self.ff[0].weight.T)
        return outer+layer_norm_vjp(hidden, through_ff, self.norm)

    def slow_read(self, slow, query, weight):
        """Actual keyed read: S q qᵀ, pooled with a fixed node weight."""
        value = (slow@query[..., None]).squeeze(-1)
        return (weight[..., None, None]*value[..., :, None]*query[..., None, :]).sum(1)

    def read_distribution(self, response):
        """Along-graph slow pooling; the reverse scalar flow is an explicit read operator."""
        weight = self.read_ports[None]+response.mean((-1, -2)).clamp(0, 2)
        return weight/weight.sum(-1, keepdim=True).clamp_min(1e-6)

    def write_scale(self, weight):
        # Relative participation of actual reads; common confidence amplitude is retained.
        relative = weight/weight.amax(-1, keepdim=True).clamp_min(torch.finfo(weight.dtype).tiny)
        effective = relative.sum(-1).square()/relative.square().sum(-1).clamp_min(torch.finfo(weight.dtype).tiny)
        scale = effective/self.cfg.local_write_reference if self.cfg.local_write_normalization == 'participation' else torch.ones_like(effective)
        return scale, effective

    def local_direction(self, state, embedding):
        """Negative CE derivative of the cached direct slow-read path only."""
        d = self.cfg.matrix_size
        output_direction = embedding-state['prediction']
        hidden_direction = self.head_vjp(state['feedback_h'], output_direction)
        pooled_direction = F.linear(hidden_direction, self.read_projection.weight.T)
        matrix_direction = pooled_direction[:, d*d:].reshape(len(embedding), d, d)
        query = state['read_query']
        value_direction = (matrix_direction[:, None]@query[..., None]).squeeze(-1)
        return (state['read_weight'][..., None, None]*value_direction[..., :, None]
                *query[..., None, :]*state['feedback_valid'][:, None, None, None])

    def local_proposal(self, state, embedding):
        if self.cfg.variant in ('no_local', 'no_slow') or not self.runtime_local_enabled or not self.runtime_slow_enabled:
            return torch.zeros_like(state['slow'])
        direction = self.local_direction(state, embedding)
        # Read participation corrects population dilution; confidence is never divided out.
        gain = state['read_scale'][:, None]*state['feedback_gain']
        return .02*self.raw_slow_rate.sigmoid()*gain[..., None, None]*direction

    def local_update(self, state, embedding):
        update = bounded(self.local_proposal(state, embedding), .05)
        return self.project_slow_update(state['slow'], update, state['read_query'])[0]

    def project_slow_update(self, slow, update, query):
        """Only the actual query subspace spends remaining slow-memory capacity."""
        unit = F.normalize(query, dim=-1, eps=1e-12)
        previous = (slow@unit[..., None]).squeeze(-1)
        orthogonal = slow-previous[..., :, None]*unit[..., None, :]
        proposed = previous+(update@unit[..., None]).squeeze(-1)
        remaining = (16-orthogonal.square().sum((-1, -2))).clamp_min(0)
        # At zero slack, choose the finite zero subgradient of the radius.
        radius = torch.where(remaining > 0, remaining.clamp_min(torch.finfo(slow.dtype).tiny).sqrt(), torch.zeros_like(remaining))
        norm = proposed.norm(dim=-1)
        scale = (radius/norm.clamp_min(1e-8)).clamp_max(1)
        projected = orthogonal+(proposed*scale[..., None])[..., :, None]*unit[..., None, :]
        active = update.square().sum((-1, -2)) > 0
        return torch.where(active[..., None, None], projected, slow), active & (norm > radius)

    def feedback_frame(self, state, lag):
        if not isinstance(lag, int) or not 1 <= lag <= self.cfg.credit_horizon:
            raise ValueError('Feedback lag exceeds the fixed credit horizon')
        if lag == 1:
            return state
        names = {'prediction': 'credit_prediction', 'feedback_h': 'credit_h', 'read_query': 'credit_query',
                 'read_weight': 'credit_weight', 'feedback_gain': 'credit_gain', 'read_scale': 'credit_scale',
                 'feedback_valid': 'credit_valid'}
        return state | {current: state[past][:, lag-2] for current, past in names.items()}

    def _apply_feedback(self, state, token, lag=1, feedback_mask=None):
        if token.ndim != 1 or len(token) != len(state['slow']):
            raise ValueError('Feedback token must have shape [batch]')
        frame = self.feedback_frame(state, lag)
        mask = torch.ones(len(token), dtype=torch.bool, device=token.device) if feedback_mask is None else feedback_mask
        if mask.dtype != torch.bool or mask.shape != token.shape:
            raise ValueError('feedback_mask must be boolean [batch]')
        frame = frame | {'feedback_valid': frame['feedback_valid'] & mask}
        proposal = self.local_proposal(frame, self.embedding(token))
        bounded_proposal = bounded(proposal, .05)
        slow, capacity_clipped = self.project_slow_update(state['slow'], bounded_proposal, frame['read_query'])
        updated = state | {'slow': slow}
        if self.runtime_local_enabled and self.runtime_slow_enabled and self.cfg.variant not in ('no_local', 'no_slow'):
            if lag == 1:
                updated['feedback_valid'] = state['feedback_valid'] & ~mask
            else:
                valid = state['credit_valid'].clone()
                valid[:, lag-2] = valid[:, lag-2] & ~mask
                updated['credit_valid'] = valid
        amplitude_clipped = proposal.norm(dim=(-1, -2)) > .05
        ratio = ((slow-state['slow']).norm(dim=(-1, -2)).sum(-1)
                 /bounded_proposal.norm(dim=(-1, -2)).sum(-1).clamp_min(1e-12)).clamp(0, 1)
        return updated, dict(local_update_clip_fraction=(amplitude_clipped | capacity_clipped).float().mean(-1),
                             capacity_clip_fraction=capacity_clipped.float().mean(-1), local_effective_step_ratio=ratio)

    def apply_feedback(self, state, token, lag=1, feedback_mask=None):
        """Consume explicit bounded-lag credit once; never substitute the current query."""
        return self._apply_feedback(state, token, lag, feedback_mask)[0]

    def cache_feedback(self, state, hidden, logits, query, weight, scale, gain):
        names = {'prediction': 'credit_prediction', 'feedback_h': 'credit_h', 'read_query': 'credit_query',
                 'read_weight': 'credit_weight', 'feedback_gain': 'credit_gain', 'read_scale': 'credit_scale',
                 'feedback_valid': 'credit_valid'}
        history = {past: torch.cat((state[current][:, None], state[past]), 1)[:, :self.cfg.credit_horizon-1].detach()
                   for current, past in names.items()}
        return state | history | dict(prediction=(logits.softmax(-1)@self.embedding.weight).detach(),
                                      feedback_h=hidden.detach(), read_query=query.detach(), read_weight=weight.detach(),
                                      feedback_gain=gain.detach(), read_scale=scale.detach(),
                                      feedback_valid=torch.ones(len(logits), dtype=torch.bool, device=logits.device))

    def weights(self, content, x, drive, controls, features, state):
        source = self.neighbors
        # Routing key and event mass carry no semantic pulse energy.
        semantic = torch.cat((x.flatten(-3)/math.sqrt(x.shape[-3]*x.shape[-2]*x.shape[-1]),
                              drive[..., :self.cfg.matrix_size]/math.sqrt(self.cfg.matrix_size)), -1)
        intensity = torch.linalg.vector_norm(semantic, dim=-1)
        source_intensity = intensity[:, source]
        similarity = F.cosine_similarity(content[:, :, None].flatten(-2), content[:, source].flatten(-2), dim=-1, eps=1e-5)
        threshold = .001+.009*controls[..., 2].sigmoid()
        score = source_intensity-threshold[:, :, None]*(1-.25*similarity)
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
        slow = state['slow'] if self.runtime_slow_enabled and c.variant != 'no_slow' else torch.zeros_like(state['slow'])
        content = state['x'].mean(-3)+.3*state['m']+.2*slow
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
        value, key = self.observe(value, key, control)
        packet = value[..., :, None]*key[..., None, :]
        packet = packet*drive[..., -1, None, None]
        tau = self.raw_tau.clamp(math.log(2), math.log(4096)).exp()
        if c.variant == 'single_timescale':
            tau = tau.mean().expand_as(tau)
        decay = torch.exp(-1/(tau*c.internal_steps))
        mixing = .1+.2*control[..., 3].sigmoid()
        target = mixing[..., None, None]*message+.25*packet
        x = state['x']*decay[None, None, :, None, None]
        x = x+(1-decay)[None, None, :, None, None]*target[..., None, :, :]
        x = bounded(x)
        if self.oscillatory:
            coupling = torch.zeros_like(frequency)
            if self.raw_coupling is not None:
                coupling = .2*self.raw_coupling.sigmoid()*(weight[..., None]*sine).sum(2)
            speed = .02+.35*frequency.sigmoid()+.08*forcing.tanh()+coupling
            z = unit_rotate(z, speed/c.internal_steps)
            # Phase modulates a norm-preserving rotation; damping stays in the explicit time bands.
            carrier_angle = .15*(z[..., 0]*amplitude.sigmoid()).mean(-1)/c.internal_steps
            angles = carrier_angle[..., None, None].expand(*x.shape[:-2], d//2)
            x = rotate_rows(x, angles)
        observation = (value, key, drive[..., -1] > 1e-4, control[..., :5])
        return state | dict(x=x, z=z), drive, weight, gate, mixing, observation

    def response_filter(self, flows, mixings, batch):
        """Bounded reverse-flow read weights/preconditioner, not a recurrent gradient."""
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
        before_slow = state['slow']
        if self.runtime_automatic_feedback:
            state, local_diagnostics = self._apply_feedback(state, token)
        else:
            local_diagnostics = {name: torch.zeros(b, dtype=embedding.dtype, device=token.device)
                                 for name in ('local_update_clip_fraction', 'capacity_clip_fraction', 'local_effective_step_ratio')}
        local_update_rms = (state['slow']-before_slow).square().mean((-1, -2, -3)).sqrt()
        value, key = self.input_projection(embedding).chunk(2, -1)
        value = value.tanh(); key = F.normalize(key, dim=-1, eps=1e-5)
        packet = torch.cat((value, key, torch.ones(b, 1, device=token.device, dtype=embedding.dtype)), -1)
        drive = self.write_ports[None, :, None]*packet[:, None]
        flows, gates, mixings = [], [], []
        for _ in range(c.internal_steps):
            state, drive, weight, gate, mixing, observation = self.tick(state, drive, packet)
            flows.append(weight); gates.append(gate); mixings.append(mixing)
        local_value, local_key, arrived, controls = observation
        prior_m, prior_p = self.stationary_prior(state) if c.variant != 'no_probability' else (state['m'], state['p'])
        prediction = (prior_m@local_key[..., None]).squeeze(-1)
        innovation = (local_value-prediction).square().mean(-1)
        reliability = controls[..., 0].sigmoid()*arrived
        variance = .02+.3*controls[..., 4].sigmoid()
        if c.variant == 'no_probability':
            m = .95*state['m']+.05*reliability[..., None, None]*local_value[..., :, None]*local_key[..., None, :]
            p = state['p']
        else:
            m, p = joseph(prior_m, prior_p, local_key, local_value, variance, reliability, self.cfg.covariance_process*arrived)
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
                             evidence_count=(state['evidence_count']+arrived.long()).clamp_max(2**31-1))
        fast = state['x'].mean(-3)
        query = F.normalize(local_key+.25*trace.mean(-2), dim=-1, eps=1e-5)
        compatibility = torch.exp(-innovation/(variance+.05))*controls[..., 1].sigmoid()
        # Fast/probabilistic matrices use surface ports; slow memories use this
        # explicit graph-distributed read, whose conditional VJP is local_direction.
        distribution = self.read_distribution(state['response'])
        read_weight = compatibility*distribution
        read_scale, effective_nodes = self.write_scale(read_weight)
        memory = (state['m']*self.read_ports[None, :, None, None]).sum(1)
        if c.variant != 'no_slow' and self.runtime_slow_enabled:
            memory = memory+self.slow_read(state['slow'], query, read_weight)
        pooled = torch.cat(((fast*self.read_ports[None, :, None, None]).sum(1).flatten(1), memory.flatten(1)), -1)
        hidden = self.read_projection(pooled)
        if c.variant == 'bypass':
            hidden = hidden+embedding
        logits = F.linear(self.head(hidden), self.embedding.weight)
        exposure = state['support'].mean(-1)
        trust = exposure/(exposure+2)*torch.exp(-state['innovation'].clamp(0, 20))
        gain = state['response'].mean((-1, -2)).clamp(0, 2)*trust
        state = self.cache_feedback(state, hidden, logits, query, read_weight, read_scale, gain)
        outward = self.outward[None].expand(b, -1, -1)*self.edge_mask[None]
        diagonal = state['p'].diagonal(dim1=-2, dim2=-1)
        phase_norm = state['z'].norm(dim=-1)
        diag = dict(outward_open=(gates[-1]*outward).sum((-1, -2))/outward.sum((-1, -2)).clamp_min(1),
                    reached=arrived.float().mean(-1), state_rms=state['x'].square().mean((-1, -2, -3, -4)).sqrt(),
                    slow_rms=state['slow'].square().mean((-1, -2, -3)).sqrt(), local_update_rms=local_update_rms,
                    **local_diagnostics, local_write_scale=read_scale,
                    effective_read_nodes=effective_nodes,
                    covariance_min_diagonal=diagonal.amin((-1, -2)), covariance_max_diagonal=diagonal.amax((-1, -2)),
                    phase_norm_min=phase_norm.amin((-1, -2)), phase_norm_max=phase_norm.amax((-1, -2)),
                    phase_coherence=state['z'].mean(1).square().sum(-1).sqrt().mean(-1))
        return logits, state, diag

    def forward(self, tokens, state=None, reset_mask=None, diagnostic_mask=None):
        state = self.initial_state(len(tokens)) if state is None else state
        if reset_mask is not None:
            state = reset_lanes(state, self.initial_state(len(tokens)), reset_mask)
        mask = torch.ones_like(tokens, dtype=torch.bool) if diagnostic_mask is None else diagnostic_mask
        if mask.dtype != torch.bool or mask.shape != tokens.shape:
            raise ValueError('diagnostic_mask must be boolean [batch,time]')
        outputs, diagnostics = [], []
        for t in range(tokens.shape[1]):
            if self.cfg.checkpoint_events and self.training and torch.is_grad_enabled():
                logits, state, diag = checkpoint(self.step, tokens[:, t], state, use_reentrant=False)
            else:
                logits, state, diag = self.step(tokens[:, t], state)
            outputs.append(logits)
            diagnostics.append(diag)
        result = {}
        for name in diagnostics[0]:
            values = torch.stack([entry[name] for entry in diagnostics], 1)
            if name.endswith('_min') or name == 'covariance_min_diagonal':
                aggregate = values.masked_fill(~mask, float('inf')).amin()
            elif name.endswith('_max') or name == 'covariance_max_diagonal':
                aggregate = values.masked_fill(~mask, -float('inf')).amax()
            else:
                aggregate = (values*mask).sum()/mask.sum().clamp_min(1)
            result[name] = torch.where(mask.any(), aggregate, torch.zeros_like(aggregate))
        return torch.stack(outputs, 1), state, result


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
        angles = (positions.to(x.dtype)[..., None]*self.inv_frequency[None, None])[:, None]
        a, b = x.reshape(*x.shape[:-1], self.head_width//2, 2).unbind(-1)
        return torch.stack((a*angles.cos()-b*angles.sin(), a*angles.sin()+b*angles.cos()), -1).flatten(-2)

    def forward(self, x, past, position, limit, past_valid=None):
        b, t, w = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, t, 3, self.heads, self.head_width).permute(2, 0, 3, 1, 4).unbind(0)
        positions = position[:, None]+torch.arange(t, device=x.device)[None]
        q, k = self.rope(q, positions), self.rope(k, positions)
        if past is not None:
            # Every query sees the same sliding context, independent of chunk size.
            keep = max(0, limit-1)
            old_k, old_v = (p[:, :, -keep:] if keep else p[:, :, :0] for p in past)
            k, v = torch.cat((old_k, k), 2), torch.cat((old_v, v), 2)
        offset = k.shape[2]-t
        key_index = torch.arange(k.shape[2], device=x.device)[None]
        query_index = offset+torch.arange(t, device=x.device)[:, None]
        mask = (key_index <= query_index)&(key_index >= query_index-limit+1)
        if past_valid is not None:
            valid = torch.cat((past_valid[:, -offset:] if offset else past_valid[:, :0],
                               torch.ones(b, t, dtype=torch.bool, device=x.device)), -1)
            mask = mask[None, None]&valid[:, None, None]
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.)
        x = x+self.out(attended.transpose(1, 2).reshape(b, t, w))
        return x+self.ff(self.norm2(x)), (k[:, :, -limit:], v[:, :, -limit:])


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
        return dict(kv=[None]*len(self.blocks),
                    position=torch.zeros(batch, dtype=torch.long, device=self.embedding.weight.device), valid=None)

    def forward(self, tokens, state=None, reset_mask=None, diagnostic_mask=None):
        state = self.initial_state(len(tokens)) if state is None else state
        if tokens.shape[1] > self.cfg.context_limit:
            raise ValueError('Transformer chunk exceeds fixed context budget')
        valid = state['valid']
        position = state['position']
        if reset_mask is not None:
            if reset_mask.dtype != torch.bool or reset_mask.shape != (len(tokens),):
                raise ValueError('reset_mask must be boolean [batch]')
            if valid is not None:
                valid = valid&~reset_mask[:, None]
            position = torch.where(reset_mask, torch.zeros_like(position), position)
        x = self.embedding(tokens); kv = []
        for block, past in zip(self.blocks, state['kv']):
            x, cache = block(x, past, position, self.cfg.context_limit, valid); kv.append(cache)
        keep = max(0, self.cfg.context_limit-1)
        old_valid = valid[:, -keep:] if valid is not None and keep else torch.empty(len(tokens), 0, dtype=torch.bool, device=tokens.device)
        valid = torch.cat((old_valid, torch.ones_like(tokens, dtype=torch.bool)), -1)[:, -self.cfg.context_limit:]
        return F.linear(self.norm(x), self.embedding.weight), dict(kv=kv, position=position+tokens.shape[1], valid=valid), {}


class Stateless(nn.Module):
    """From-scratch same-width embedding/FFN baseline with no runtime memory."""
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg
        w = cfg.width
        self.embedding = nn.Embedding(cfg.vocab_size, w)
        self.norm = nn.LayerNorm(w)
        self.ff = nn.Sequential(nn.Linear(w, 4*w), nn.GELU(), nn.Linear(4*w, w))
        self.output_norm = nn.LayerNorm(w)

    def initial_state(self, batch):
        return {}

    def forward(self, tokens, state=None, reset_mask=None, diagnostic_mask=None):
        hidden = self.embedding(tokens)
        hidden = self.output_norm(hidden+self.ff(self.norm(hidden)))
        return F.linear(hidden, self.embedding.weight), {}, {}


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
