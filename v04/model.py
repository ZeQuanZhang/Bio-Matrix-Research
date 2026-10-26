"""Matrix-valued recurrent nodes, latent context slots and sparse Fourier operators."""
from dataclasses import asdict, dataclass
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from v03.graph import graph

VARIANTS = ('graph', 'matrix', 'probability', 'fourier', 'full')


@dataclass
class Config:
    variant: str = 'full'
    nodes: int = 32
    matrix_size: int = 8
    contexts: int = 3
    width: int = 48
    modes: int = 4
    operator_bases: int = 2
    neighbors: int = 8
    active_neighbors: int = 4
    write_nodes: int = 4
    read_nodes: int = 4
    read_path: str = 'shared'
    internal_steps: int = 12
    graph_seed: int = 1901
    role_seed: int = 1902
    seen_capacity: int = 64
    topology: str = 'connected'
    baseline_rank: int = 0
    checkpoint_ticks: bool = False

    def __post_init__(self):
        if self.variant not in VARIANTS or self.matrix_size < 4 or self.modes < 1:
            raise ValueError('Invalid variant, matrix_size or modes')
        if self.contexts < 2 or self.internal_steps < 1 or self.seen_capacity < 1:
            raise ValueError('At least two contexts and positive iteration/cache sizes required')
        if not 1 <= self.active_neighbors <= self.neighbors:
            raise ValueError('active_neighbors must lie in [1, neighbors]')
        if self.read_path not in ('shared', 'separated'):
            raise ValueError('read_path must be shared or separated')
        if min(self.write_nodes, self.read_nodes) < 1 or self.write_nodes+self.read_nodes > self.nodes//2:
            raise ValueError('Read/write port counts exceed available graph nodes')


def detach(state):
    return {k: v.detach() for k, v in state.items()}


def rotate(z, angle):
    x, y = z.unbind(-1); c, s = angle.cos(), angle.sin()
    return torch.stack((x*c-y*s, x*s+y*c), -1)


def gaussian_update(m, p, key, value, variance, weight, process):
    """Tempered Joseph update; called once per fresh external observation."""
    d = m.shape[-1]; eye = torch.eye(d, device=m.device, dtype=m.dtype)
    prior = p+process[..., None, None]*eye
    pk = (prior@key[..., None]).squeeze(-1)
    denominator = (pk*key).sum(-1)+variance
    gain = pk/denominator.clamp_min(1e-6)[..., None]
    effective = weight[..., None]*gain
    error = value-(m@key[..., None]).squeeze(-1)
    updated = m+error[..., :, None]*effective[..., None, :]
    a = eye-effective[..., :, None]*key[..., None, :]
    posterior = a@prior@a.transpose(-1, -2)+variance[..., None, None]*effective[..., :, None]*effective[..., None, :]
    return updated, .5*(posterior+posterior.transpose(-1, -2))


class Network(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg
        d, n, w, k = cfg.matrix_size, cfg.nodes, cfg.width, cfg.modes
        self.probability = cfg.variant in ('probability', 'full')
        self.fourier = cfg.variant in ('fourier', 'full')
        self.state_width = 3*d*d+4*k
        self.register_buffer('neighbors', graph(n, cfg.graph_seed, cfg.topology, cfg.neighbors).clone())
        rng = np.random.default_rng(cfg.role_seed)
        self.register_buffer('roles', torch.tensor(rng.normal(0, .5, (n, 8)), dtype=torch.float32))
        writes, reads = torch.zeros(d, n), torch.zeros(d, n)
        for key in range(d):
            candidates = rng.permutation(n//2 if cfg.read_path == 'separated' else n)
            wi = candidates[:cfg.write_nodes]
            ri = rng.permutation(n//2)[:cfg.read_nodes]+n//2 if cfg.read_path == 'separated' else wi[:cfg.read_nodes]
            if cfg.read_path == 'shared' and cfg.read_nodes != cfg.write_nodes:
                raise ValueError('shared ports require equal read/write counts')
            writes[key, wi] = 1.; reads[key, ri] = 1./len(ri)
        self.register_buffer('write_ports', writes); self.register_buffer('read_ports', reads)
        angle = torch.tensor(rng.uniform(-math.pi, math.pi, (n, k)), dtype=torch.float32)
        self.register_buffer('initial_z', torch.stack((angle.cos(), angle.sin()), -1))
        self.raw_cue_scale = nn.Parameter(torch.tensor(-1.5))
        self.new_context_bias = nn.Parameter(torch.tensor(-3.))
        self.raw_content_weight = nn.Parameter(torch.tensor(-2.))
        self.raw_temperature = nn.Parameter(torch.tensor(2.))
        self.value_read = nn.Linear(d, 4, bias=False)
        with torch.no_grad():
            self.value_read.weight.zero_(); self.value_read.weight[:, :4].copy_(torch.eye(4))
        self.role_feature = nn.Linear(8, w)
        self.input_feature = nn.Linear(2*d*d+5, w)
        self.read_correction = nn.Sequential(nn.Linear(2*d+8, w), nn.Tanh(), nn.Linear(w, 4))
        nn.init.zeros_(self.read_correction[-1].weight); nn.init.zeros_(self.read_correction[-1].bias)
        self.edge_query = nn.Linear(16, 8, bias=False)
        self.edge_key = nn.Linear(16, 8, bias=False)
        if cfg.variant == 'graph':
            rank = cfg.baseline_rank or w
            self.h_project = nn.Linear(self.state_width, rank)
            self.h_input = nn.Linear(w, rank)
            self.h_candidate = nn.Linear(rank, self.state_width)
            self.h_gate = nn.Linear(rank, self.state_width)
            self.h_decode = nn.Linear(rank, d*d)
        else:
            eye = torch.eye(d)
            self.left = nn.Parameter(.2*eye); self.right = nn.Parameter(.2*eye)
            self.external_left = nn.Parameter(eye.clone()); self.external_right = nn.Parameter(eye.clone())
            self.raw_dynamic = nn.Parameter(torch.tensor(-1.))
            self.raw_dt = nn.Parameter(torch.tensor(-1.))
            self.raw_process = nn.Parameter(torch.tensor(-5.))
            self.raw_x_read = nn.Parameter(torch.tensor(-2.))
            self.write_gate = nn.Sequential(nn.Linear(12, w), nn.Tanh(), nn.Linear(w, 1))
            self.node_drive = nn.Linear(w, d*d)
            self.role_diagonal = nn.Linear(8, 2*d)
            self.message_left = nn.Parameter(eye.repeat(cfg.operator_bases, 1, 1)+.02*torch.randn(cfg.operator_bases, d, d))
            self.message_right = nn.Parameter(eye.repeat(cfg.operator_bases, 1, 1)+.02*torch.randn(cfg.operator_bases, d, d))
            self.harmonics = 2*k-1
            self.coefficient = nn.Linear(32, cfg.operator_bases*(1+2*self.harmonics))
            self.oscillator = nn.Linear(16, 3*k)
            self.raw_coupling = nn.Parameter(torch.tensor(-1.))

    def initial_state(self, batch, device=None):
        device = device or self.roles.device; c = self.cfg; d = c.matrix_size
        shape = (batch, c.contexts, c.nodes)
        state = dict(cue=torch.zeros(batch, c.contexts, 4, device=device),
                     occupied=torch.zeros(batch, c.contexts, device=device, dtype=torch.bool),
                     rho=torch.zeros(batch, c.contexts, device=device),
                     age=torch.zeros(batch, c.contexts, device=device),
                     seen=torch.full((batch, c.seen_capacity), -2, device=device, dtype=torch.long),
                     cursor=torch.zeros(batch, device=device, dtype=torch.long))
        if c.variant == 'graph':
            state['h'] = torch.zeros(*shape, self.state_width, device=device)
        else:
            state.update(m=torch.zeros(*shape, d, d, device=device), x=torch.zeros(*shape, d, d, device=device),
                         p=torch.eye(d, device=device).expand(*shape, d, d).clone(),
                         z=self.initial_z[None, None].expand(batch, c.contexts, -1, -1, -1).clone(),
                         amplitude=torch.ones(*shape, c.modes, device=device)*.5,
                         omega=torch.ones(*shape, c.modes, device=device)*.2)
        return state

    def memory(self, state):
        if self.cfg.variant == 'graph':
            return self.h_decode(self.h_project(state['h'])).reshape(*state['h'].shape[:-1], self.cfg.matrix_size, self.cfg.matrix_size)
        return state['m']

    def prediction(self, state, key):
        d = self.cfg.matrix_size
        address = F.one_hot(key, d).to(self.roles.dtype)
        weights = self.write_ports[key]/self.cfg.write_nodes
        predicted = (self.memory(state)@address[:, None, None, :, None]).squeeze(-1)
        mean = (weights[:, None, :, None]*predicted).sum(2)
        if self.cfg.variant != 'graph' and self.probability:
            pk = (state['p']@address[:, None, None, :, None]).squeeze(-1)
            variance = (pk*address[:, None, None]).sum(-1)
            variance = (variance*weights[:, None]).sum(-1)
        else:
            variance = mean.new_ones(mean.shape[:2])
        return mean, variance

    def route_context(self, state, event, mean, variance):
        c = self.cfg; occupied = state['occupied']; valid = event['valid']
        scale = .15+.7*self.raw_cue_scale.sigmoid()
        cue_error = (state['cue']-event['cue'][:, None]).square().mean(-1)
        cue_ll = -.5*cue_error/scale.square()
        obs_variance = (variance+event['noise'][:, None].square()+.01).clamp_min(.01)
        error = (mean[..., :4]-event['value'][:, None]).square().mean(-1)
        ll = -.5*(error/obs_variance+obs_variance.log())
        prior = .8*state['rho']+.2*occupied.to(mean.dtype)/occupied.sum(-1, keepdim=True).clamp_min(1)
        old_base = (cue_ll+prior.clamp_min(1e-5).log()).masked_fill(~occupied, -1e4)
        old_score = old_base+self.raw_content_weight.sigmoid()*ll*event['observe'][:, None]
        # One new candidate competes with existing slots; replacement uses least recent activity.
        free = (~occupied).to(torch.long).argmax(-1)
        replacement = state['age'].argmax(-1)
        destination = torch.where(occupied.all(-1), replacement, free)
        new_variance = 1+event['noise'].square()+.01
        new_ll = -.5*(event['value'].square().mean(-1)/new_variance+new_variance.log())
        new_base = self.new_context_bias.expand(len(mean), 1)
        new_score = new_base+self.raw_content_weight.sigmoid()*new_ll[:, None]*event['observe'][:, None]
        probability = torch.cat((old_score, new_score), -1).softmax(-1)
        allocate = ((probability[:, -1] > .5) | ~occupied.any(-1)) & valid
        allocation = F.one_hot(destination, c.contexts).bool() & allocate[:, None]
        rho = probability[:, :-1]+probability[:, -1:]*F.one_hot(destination, c.contexts)*allocate[:, None]
        rho = rho/rho.sum(-1, keepdim=True).clamp_min(1e-8)
        rho = torch.where(valid[:, None], rho, state['rho'])
        state = dict(state)
        fresh = self.initial_state(len(mean))
        for name in ('m', 'x', 'p', 'z', 'amplitude', 'omega', 'h'):
            if name in state:
                mask = allocation.reshape(*allocation.shape, *([1]*(state[name].ndim-2)))
                state[name] = torch.where(mask, fresh[name], state[name])
        state['occupied'] = occupied | allocation
        anchor = torch.where(allocation[..., None], event['cue'][:, None], state['cue'])
        state['cue'] = anchor+.15*rho[..., None]*valid[:, None, None]*(event['cue'][:, None]-anchor)
        state['age'] = torch.where((rho > .5) & valid[:, None], 0., state['age']+valid[:, None])
        state['rho'] = rho
        predictive = -(torch.logsumexp(torch.cat((old_score, new_score), -1), -1)
                       -torch.logsumexp(torch.cat((old_base, new_base), -1), -1)).clamp(-20, 20)
        predictive = torch.where(occupied.any(-1) & event['observe'] & valid, predictive, 0.)
        return state, predictive

    def features(self, state):
        if self.cfg.variant == 'graph':
            h = state['h']; chunks = torch.tensor_split(h, 8, dim=-1)
            summary = torch.stack([v.mean(-1) for v in chunks], -1)
        else:
            m, x = state['m'], state['x']
            summary = torch.stack((m.mean((-1, -2)), x.mean((-1, -2)),
                                   m.square().mean((-1, -2)), x.square().mean((-1, -2)),
                                   state['p'].diagonal(dim1=-2, dim2=-1).mean(-1),
                                   state['amplitude'].mean(-1), state['omega'].mean(-1),
                                   state['z'][..., 0].mean(-1)), -1)
        return torch.cat((summary, self.roles[None, None].expand(*summary.shape[:-1], 8)), -1)

    def edge_weights(self, feature):
        q, k = self.edge_query(feature), self.edge_key(feature)
        score = (q[..., None, :]*k[:, :, self.neighbors]).sum(-1)/math.sqrt(8)
        selected = score.topk(self.cfg.active_neighbors, -1)
        sparse = torch.full_like(score, -1e4).scatter(-1, selected.indices, selected.values)
        return sparse.softmax(-1)

    def matrix_tick(self, state, injected, residual, intervention):
        c = self.cfg; d = c.matrix_size; m, x = state['m'], state['x']
        feature = self.features(state); pi = self.edge_weights(feature)
        other = feature[:, :, self.neighbors]
        pair = torch.cat((feature[..., None, :].expand_as(other), other), -1)
        coeff = self.coefficient(pair).reshape(*pi.shape, c.operator_bases, 1+2*self.harmonics)
        factor = coeff[..., 0]
        zj, zi = state['z'][:, :, self.neighbors], state['z'][..., None, :, :]
        cos = (zi*zj).sum(-1); sin = zi[..., 0]*zj[..., 1]-zi[..., 1]*zj[..., 0]
        strength = state['amplitude'][..., None, :]*state['amplitude'][:, :, self.neighbors]
        if c.modes > 1:
            cos, sin = (torch.cat((cos, cos[..., :-1]*cos[..., 1:]+sin[..., :-1]*sin[..., 1:]), -1),
                        torch.cat((sin, sin[..., :-1]*cos[..., 1:]-cos[..., :-1]*sin[..., 1:]), -1))
            strength = torch.cat((strength, strength[..., :-1]*strength[..., 1:]), -1)
        if self.fourier:
            factor = factor+(coeff[..., 1:1+self.harmonics]*(cos*strength)[..., None, :]).sum(-1)/math.sqrt(self.harmonics)
            factor = factor+(coeff[..., 1+self.harmonics:]*(sin*strength)[..., None, :]).sum(-1)/math.sqrt(self.harmonics)
        factor = factor.tanh()*pi[..., None]
        message = torch.zeros_like(x)
        for r in range(c.operator_bases):
            left = self.message_left[r]/self.message_left[r].norm().clamp_min(1.)
            right = self.message_right[r]/self.message_right[r].norm().clamp_min(1.)
            transmitted = left@x@right
            message = message+(factor[..., r, None, None]*transmitted[:, :, self.neighbors]).sum(3)
        if intervention == 'no_messages':
            message = message*0
        alpha = self.raw_dynamic.sigmoid() if intervention != 'fixed_external' else self.raw_dynamic*0
        effective = self.external_left+alpha*(m.tanh()+x.tanh())/math.sqrt(d)
        mapped = effective@injected@self.external_right
        diagonal_a, diagonal_b = self.role_diagonal(self.roles).tanh().chunk(2, -1)
        a = self.left/self.left.norm().clamp_min(1.)+torch.diag_embed(.1*diagonal_a)
        b = self.right/self.right.norm().clamp_min(1.)+torch.diag_embed(.1*diagonal_b)
        pre = a@x+x@b+mapped+residual+message+.2*m.tanh()
        dt = .3*self.raw_dt.sigmoid()*state['rho'][..., None, None, None]
        state = dict(state); state['x'] = (1-dt)*x+dt*pre.tanh()
        drive, speed, amplitude = self.oscillator(feature).chunk(3, -1)
        rate = state['rho'][..., None, None]*.15
        state['omega'] = (1-rate)*state['omega']+rate*(.03+.7*speed.sigmoid())
        state['amplitude'] = (1-rate)*state['amplitude']+rate*amplitude.sigmoid()
        signed_coupling = coeff[..., 0].mean(-1).tanh()*pi
        coupling = (signed_coupling[..., None]*sin[..., :c.modes]).sum(3)
        if intervention in ('no_messages', 'no_coupling'):
            coupling = coupling*0
        if self.fourier:
            angle = rate*(state['omega']+.1*drive.tanh()+self.raw_coupling.sigmoid()*coupling)
            state['z'] = rotate(state['z'], angle)
        else:
            target = torch.stack((drive.tanh(), speed.tanh()), -1)
            state['z'] = (1-rate[..., None])*state['z']+rate[..., None]*target
        return state

    def step(self, event, state, intervention=None):
        c = self.cfg; d = c.matrix_size
        seen = (state['seen'] == event['evidence_id'][:, None]).any(-1)
        fresh = event['observe'] & event['valid'] & ~seen
        mean, variance = self.prediction(state, event['key'])
        state, predictive = self.route_context(state, event | {'observe': fresh}, mean, variance)
        key = F.one_hot(event['key'], d).to(mean.dtype)
        value = F.pad(event['value'], (0, d-4))
        local_key = key[:, None, None].expand(-1, c.contexts, c.nodes, -1)
        prediction = (self.memory(state)@local_key[..., None]).squeeze(-1)
        error = value[:, None, None]-prediction
        port = self.write_ports[event['key']][:, None]*fresh[:, None, None]
        assignment = state['rho'][..., None]*port
        packet = value[:, None, None, :, None]*local_key[..., None, :]
        innovation = error[..., :, None]*local_key[..., None, :]
        packet = packet*assignment[..., None, None]
        innovation = innovation*assignment[..., None, None]
        if c.variant != 'graph':
            role = self.roles[None, None].expand(len(mean), c.contexts, -1, -1)
            gate_feature = torch.cat((error.square().mean(-1, keepdim=True),
                                      event['noise'][:, None, None, None].expand(-1, c.contexts, c.nodes, 1),
                                      state['p'].diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True),
                                      state['rho'][..., None, None].expand(-1, -1, c.nodes, 1), role), -1)
            gain_weight = assignment*(.1+.9*self.write_gate(gate_feature).sigmoid().squeeze(-1))
            if self.probability:
                p = state['p']; observation_variance = event['noise'].square()+.005
            else:
                p = torch.eye(d, device=mean.device).expand_as(state['p'])
                observation_variance = event['noise'].new_ones(len(mean))
            process = .05*self.raw_process.sigmoid()*assignment
            updated, covariance = gaussian_update(state['m'], p, local_key,
                                                  value[:, None, None].expand_as(local_key),
                                                  observation_variance[:, None, None], gain_weight, process)
            state['m'] = updated
            state['p'] = covariance if self.probability else p
            # Only fresh evidence is used for M/P; internal iterations evolve reaction/communication.
            metadata = torch.cat((event['cue'], event['noise'][:, None]), -1)
            encoded = self.input_feature(torch.cat((packet.flatten(-2), innovation.flatten(-2),
                                                    metadata[:, None, None].expand(-1, c.contexts, c.nodes, -1)), -1))
            encoded = encoded+self.role_feature(self.roles)[None, None]
            injected = packet+.1*self.node_drive(encoded).reshape_as(packet)*assignment[..., None, None]
            for _ in range(c.internal_steps):
                if c.checkpoint_ticks and self.training and torch.is_grad_enabled():
                    state = checkpoint(self.matrix_tick, state, injected, innovation, intervention,
                                       use_reentrant=False, preserve_rng_state=False)
                else:
                    state = self.matrix_tick(state, injected, innovation, intervention)
        else:
            metadata = torch.cat((event['cue'], event['noise'][:, None]), -1)
            encoded = self.input_feature(torch.cat((packet.flatten(-2), innovation.flatten(-2),
                                                    metadata[:, None, None].expand(-1, c.contexts, c.nodes, -1)), -1))
            encoded = encoded+self.role_feature(self.roles)[None, None]
            for _ in range(c.internal_steps):
                feature = self.features(state); pi = self.edge_weights(feature)
                projected = self.h_project(state['h'])
                message = (pi[..., None]*projected[:, :, self.neighbors]).sum(3)
                if intervention == 'no_messages':
                    message = message*0
                u = (projected+message+self.h_input(encoded)).tanh()
                gate = self.h_gate(u).sigmoid()*state['rho'][..., None, None]*.3
                state['h'] = (1-gate)*state['h']+gate*self.h_candidate(u).tanh()
        seen_ids = state['seen'].clone()
        previous = seen_ids.gather(1, state['cursor'][:, None]).squeeze(1)
        seen_ids.scatter_(1, state['cursor'][:, None], torch.where(fresh, event['evidence_id'], previous)[:, None])
        state['seen'] = seen_ids
        state['cursor'] = (state['cursor']+fresh.long()) % c.seen_capacity
        # Padded events have no effect on any persistent state.
        return state, predictive, fresh

    def read(self, state, qa, qb, combine):
        c = self.cfg; memory = self.memory(state)
        distributions = []
        for query in (qa, qb):
            key = F.one_hot(query, c.matrix_size).to(memory.dtype)
            m = (memory@key[:, None, None, :, None]).squeeze(-1)
            if c.variant == 'graph':
                x = m*0
            else:
                x = (state['x']@key[:, None, None, :, None]).squeeze(-1)
            weights = self.read_ports[query][:, None, :, None]
            content, reaction = (m*weights).sum(2), (x*weights).sum(2)
            role = (self.roles[None]*self.read_ports[query][..., None]).sum(1)
            correction = self.read_correction(torch.cat((content, reaction, role[:, None].expand(-1, c.contexts, -1)), -1))
            logits = self.value_read(content)+correction
            if c.variant != 'graph':
                logits = logits+self.raw_x_read.sigmoid()*self.value_read(reaction)
            distributions.append((logits*(1+5*self.raw_temperature.sigmoid())).softmax(-1))
        first, second = distributions
        # Known arithmetic readout is shared by all variants; this task tests memory, not learned arithmetic.
        joint = torch.stack([sum(first[..., j]*second[..., (i-j) % 4] for j in range(4)) for i in range(4)], -1)
        per_context = torch.where(combine[:, None, None], joint, first)
        probability = (state['rho'][..., None]*per_context).sum(1)
        probability = torch.where(state['rho'].sum(-1, keepdim=True) > 0, probability, torch.full_like(probability, .25))
        probability = probability.clamp_min(1e-8)
        return (probability/probability.sum(-1, keepdim=True)).log()

    def forward(self, key, value, noise, cue, source, evidence_id, observe, query_a, query_b, combine, valid,
                state=None, intervention=None, intervention_mask=None):
        if state is None:
            state = self.initial_state(len(key))
        log_probs = []; predictive = []; routing = []; fresh_counts = []
        for t in range(key.shape[1]):
            before = state
            if intervention in ('reset_history', 'swap_history') and intervention_mask is not None:
                mask = intervention_mask[:, t]
                changed = self.initial_state(len(key)) if intervention == 'reset_history' else {k: v.roll(1, 0) for k, v in state.items()}
                state = {k: torch.where(mask.reshape(len(key), *([1]*(v.ndim-1))), changed[k], v) for k, v in state.items()}
                before = state
            event = dict(key=key[:, t], value=value[:, t], noise=noise[:, t], cue=cue[:, t], source=source[:, t],
                         evidence_id=evidence_id[:, t], observe=observe[:, t], valid=valid[:, t])
            state, loss, fresh = self.step(event, state, intervention)
            state = {k: torch.where(valid[:, t].reshape(len(key), *([1]*(v.ndim-1))), v, before[k]) for k, v in state.items()}
            log_probs.append(self.read(state, query_a[:, t], query_b[:, t], combine[:, t]))
            predictive.append(loss); routing.append(state['rho']); fresh_counts.append(fresh)
        return torch.stack(log_probs, 1), state, dict(predictive=torch.stack(predictive, 1),
                    routing=torch.stack(routing, 1), fresh=torch.stack(fresh_counts, 1))

    def ledger(self):
        state = self.initial_state(1, 'cpu')
        return dict(config=asdict(self.cfg), parameters=sum(p.numel() for p in self.parameters()),
                    state_floats_per_stream=sum(v.numel() for v in state.values() if v.is_floating_point()),
                    state_integer_or_boolean_elements=sum(v.numel() for v in state.values() if not v.is_floating_point()),
                    node_floats=self.state_width, context_slots=self.cfg.contexts,
                    input_cache='Current event only; bounded provenance IDs; no historical observation replay',
                    activation_storage_is_additional=True,
                    matching='Same graph, ports, streams, internal iterations, context router and node-state float count; parameter counts and measured runtime reported separately')
