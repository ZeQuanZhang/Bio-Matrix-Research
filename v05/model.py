"""Matrix nodes with conditional orthogonal transport and local adaptive weights."""
from dataclasses import asdict, dataclass
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from v03.graph import graph
from v04.model import detach, gaussian_update, rotate

VARIANTS = ('message', 'full', 'no_fourier', 'single_timescale', 'no_traces',
            'no_consolidation', 'frequency_only', 'always_read')
PERSISTENT = ('slow', 'protect', 'cue', 'mass', 'rho', 'seen', 'cursor')
FEATURE_WIDTH = 21


@dataclass
class Config:
    variant: str = 'full'
    nodes: int = 32
    matrix_size: int = 8
    contexts: int = 3
    width: int = 32
    modes: int = 4
    bands: int = 3
    trace_bands: int = 2
    neighbors: int = 8
    write_nodes: int = 4
    read_nodes: int = 4
    read_path: str = 'shared'
    internal_steps: int = 16
    graph_seed: int = 2901
    role_seed: int = 2902
    seen_capacity: int = 128
    topology: str = 'connected'
    checkpoint_events: bool = True
    baseline_rank: int = 0

    def __post_init__(self):
        if self.variant not in VARIANTS or self.matrix_size < 4 or self.matrix_size % 2:
            raise ValueError('Need a known variant and an even matrix_size >= 4')
        if min(self.contexts, self.modes, self.bands, self.trace_bands, self.internal_steps, self.seen_capacity) < 1:
            raise ValueError('State dimensions and internal_steps must be positive')
        if self.contexts < 2 or self.topology not in ('connected', 'cut', 'rewired'):
            raise ValueError('Need at least two contexts and a known topology')
        if self.read_path not in ('shared', 'separated'):
            raise ValueError('Unknown read path')
        if min(self.write_nodes, self.read_nodes) < 1 or max(self.write_nodes, self.read_nodes) > self.nodes//2:
            raise ValueError('Ports exceed available nodes')
        if self.read_path == 'shared' and self.write_nodes != self.read_nodes:
            raise ValueError('Shared ports require equal read/write counts')


def matchings(neighbors):
    """Greedy edge coloring; each substep contains disjoint undirected pairs."""
    edges = sorted({tuple(sorted((i, j))) for i, row in enumerate(neighbors.tolist()) for j in row})
    colors, occupied = [], []
    for i, j in edges:
        for index, used in enumerate(occupied):
            if i not in used and j not in used:
                break
        else:
            index = len(colors); colors.append([]); occupied.append(set())
        colors[index].append((i, j)); occupied[index].update((i, j))
    return [torch.tensor(pairs, dtype=torch.long) for pairs in colors]


def pair_rotate(x, pairs, angle):
    a, b = x[:, :, pairs[:, 0]], x[:, :, pairs[:, 1]]
    while angle.ndim < a.ndim:
        angle = angle.unsqueeze(-1)
    co, si = angle.cos(), angle.sin()
    result = x.index_copy(2, pairs[:, 0], co*a-si*b)
    return result.index_copy(2, pairs[:, 1], si*a+co*b)


def rotate_rows(x, angle):
    d = x.shape[-1]
    a, b = x.reshape(*x.shape[:-2], d//2, 2, d).unbind(-2)
    co, si = angle.cos()[..., None], angle.sin()[..., None]
    return torch.stack((co*a-si*b, si*a+co*b), -2).reshape_as(x)


class Network(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg
        d, n, w = cfg.matrix_size, cfg.nodes, cfg.width
        self.fourier = cfg.variant != 'no_fourier'
        self.rank = 4*d
        neighbors = graph(n, cfg.graph_seed, cfg.topology, cfg.neighbors).clone()
        self.register_buffer('neighbors', neighbors)
        groups = matchings(neighbors); self.matching_count = len(groups)
        for i, pairs in enumerate(groups):
            self.register_buffer(f'pairs_{i}', pairs)
        rng = np.random.default_rng(cfg.role_seed)
        self.register_buffer('roles', torch.tensor(rng.normal(0, .5, (n, 8)), dtype=torch.float32))
        writes, reads = torch.zeros(d, n), torch.zeros(d, n)
        for key in range(d):
            limit = n//2 if cfg.read_path == 'separated' else n
            indices = rng.permutation(limit)[:cfg.write_nodes]
            writes[key, indices] = 1.
            selected = (rng.permutation(n//2)[:cfg.read_nodes]+n//2
                        if cfg.read_path == 'separated' else indices)
            reads[key, selected] = 1./cfg.read_nodes
        self.register_buffer('write_ports', writes); self.register_buffer('read_ports', reads)
        phase = torch.tensor(rng.uniform(-math.pi, math.pi, (n, cfg.modes)), dtype=torch.float32)
        self.register_buffer('initial_z', torch.stack((phase.cos(), phase.sin()), -1))
        basis = torch.zeros(self.rank, d, d)
        for row in range(4):
            for col in range(d):
                basis[row*d+col, row, col] = 1.
        self.slow_basis = nn.Parameter(basis)
        self.raw_cue_scale = nn.Parameter(torch.tensor(-.7))
        self.raw_process = nn.Parameter(torch.tensor(-4.))
        self.raw_slow_rate = nn.Parameter(torch.tensor(-.8))
        self.raw_dynamic = nn.Parameter(torch.tensor(-1.))
        self.raw_temperature = nn.Parameter(torch.tensor(1.))
        self.raw_input_gain = nn.Parameter(torch.zeros(cfg.bands))
        self.raw_tau = nn.Parameter(torch.linspace(math.log(8), math.log(512), cfg.bands))
        self.raw_trace_tau = nn.Parameter(torch.linspace(math.log(4), math.log(64), cfg.trace_bands))
        self.gates = nn.Sequential(nn.Linear(FEATURE_WIDTH, w), nn.Tanh(), nn.Linear(w, 3))
        self.oscillator = nn.Linear(FEATURE_WIDTH, 3*cfg.modes)
        self.readout = nn.Sequential(nn.Linear(4*d, w), nn.Tanh(), nn.Linear(w, 4))
        self.raw_read_scale = nn.Parameter(torch.tensor(0.))
        self.harmonics = 2*cfg.modes-1
        self.coefficient = nn.Linear(2*FEATURE_WIDTH, 1+2*self.harmonics)
        if cfg.variant == 'message':
            rank = cfg.baseline_rank or w
            self.message_in = nn.Linear(d*d, rank)
            self.message_out = nn.Linear(rank, d*d)
            self.message_gate = nn.Linear(FEATURE_WIDTH, 1)
        else:
            self.content_angles = nn.Linear(FEATURE_WIDTH, d)
            with torch.no_grad():
                self.content_angles.weight.mul_(.02); self.content_angles.bias.zero_()
        with torch.no_grad():
            self.coefficient.weight.mul_(.3)

    def initial_state(self, batch, device=None):
        device = device or self.roles.device; c = self.cfg; d = c.matrix_size
        base = (batch, c.contexts, c.nodes)
        state = dict(m=torch.zeros(*base, d, d, device=device),
                     p=torch.eye(d, device=device).expand(*base, d, d).clone(),
                     x=torch.zeros(*base, c.bands, d, d, device=device),
                     slow=torch.zeros(*base, self.rank, device=device),
                     trace=torch.zeros(*base, c.trace_bands, self.rank, device=device),
                     support=torch.zeros(*base, d, device=device), utility=torch.zeros(*base, d, device=device),
                     app=torch.ones(*base, d, device=device), protect=torch.zeros(*base, d, device=device),
                     innovation=torch.zeros(*base, 4, device=device),
                     z=self.initial_z[None, None].expand(*base, c.modes, 2).clone(),
                     amplitude=torch.ones(*base, c.modes, device=device)*.5,
                     omega=torch.ones(*base, c.modes, device=device)*.2,
                     cue=torch.zeros(batch, c.contexts, 4, device=device),
                     mass=torch.zeros(batch, c.contexts, device=device),
                     rho=torch.full((batch, c.contexts), 1./c.contexts, device=device),
                     seen=torch.full((batch, c.seen_capacity), -2, dtype=torch.long, device=device),
                     cursor=torch.zeros(batch, dtype=torch.long, device=device))
        return state

    def reset_fast(self, state, mask):
        fresh = self.initial_state(len(mask))
        return {k: v if k in PERSISTENT else torch.where(mask.reshape(len(mask), *([1]*(v.ndim-1))), fresh[k], v)
                for k, v in state.items()}

    def basis(self):
        return self.slow_basis/self.slow_basis.flatten(1).norm(dim=-1).clamp_min(1.)[:, None, None]

    def slow_matrix(self, state):
        return torch.einsum('bcnr,rde->bcnde', state['slow'], self.basis())

    def memory(self, state, intervention=None):
        slow = self.slow_matrix(state)
        precision = state['p'].diagonal(dim1=-2, dim2=-1)
        gate = state['app']*state['protect']/(1+state['protect'])*precision/(precision+.15)
        if self.cfg.variant == 'always_read':
            gate = torch.ones_like(gate)
        if intervention == 'clear_slow' or self.cfg.variant == 'no_consolidation':
            gate = gate*0
        return state['m']+slow*gate[..., None, :]

    def features(self, state):
        summary = torch.stack((state['m'].mean((-1, -2)), state['m'].square().mean((-1, -2)),
                               state['x'].mean((-1, -2, -3)), state['x'].square().mean((-1, -2, -3)),
                               state['p'].diagonal(dim1=-2, dim2=-1).mean(-1),
                               state['slow'].square().mean(-1),
                               state['amplitude'].mean(-1)), -1)
        return torch.cat((summary, state['innovation'], state['z'].mean(-2),
                          self.roles[None, None].expand(*summary.shape[:-1], 8)), -1)

    def route(self, state, cue, valid):
        scale = .2+.8*self.raw_cue_scale.sigmoid()
        error = (state['cue']-cue[:, None]).square().mean(-1)
        used = (state['mass']+.001)/(state['mass'].sum(-1, keepdim=True)+.001*self.cfg.contexts)
        old = (-error/scale.square()+used.log()+.2*state['rho'].clamp_min(1e-6).log()).softmax(-1)
        novelty = (((old*error).sum(-1)-4*scale.square())/(scale.square()+.02)*2).sigmoid()
        cold = (-state['mass'].sum(-1)).exp()
        novelty = cold+(1-cold)*novelty
        priority = -torch.arange(self.cfg.contexts, device=cue.device, dtype=cue.dtype)*.6
        allocate = ((-2*state['mass']+priority)/.15).softmax(-1)
        rho = (1-novelty[:, None])*old+novelty[:, None]*allocate
        rho = torch.where(valid[:, None], rho, state['rho'])
        result = dict(state); result['rho'] = rho
        update = (.3*rho+.7*rho.square())*valid[:, None]
        result['cue'] = state['cue']+update[..., None]*(cue[:, None]-state['cue'])
        result['mass'] = state['mass']+rho*valid[:, None]
        return result

    def plasticity(self, state, event, key, value, assignment, fresh, feature):
        """Local approximate eligibility learning; no scored labels or autograd at use time."""
        c = self.cfg; d = c.matrix_size
        tau = self.raw_trace_tau.clamp(math.log(1), math.log(2048)).exp()
        trace = state['trace']*torch.exp(-1/tau)[None, None, None, :, None]
        query_key = F.one_hot(event['query_a'], d).to(value.dtype)
        query_port = self.write_ports[event['query_a']][:, None]*state['rho'][..., None]*event['request'][:, None, None]
        new_trace = torch.zeros_like(trace)
        for query, active in ((event['query_a'], event['request']),
                              (event['query_b'], event['request'] & event['combine'])):
            address = F.one_hot(query, d).to(value.dtype)
            derivative = torch.einsum('rde,be->brd', self.basis(), address)[..., :4]
            eligibility = derivative.square().sum(-1).clamp_max(2.)
            port = self.write_ports[query][:, None]*state['rho'][..., None]*active[:, None, None]
            new_trace = new_trace+port[..., None, None]*eligibility[:, None, None, None]/c.trace_bands
        trace = trace+new_trace
        trace = trace.clamp_max(4.)
        if c.variant == 'no_traces':
            trace = new_trace
        before = (state['m']@key[:, None, None, :, None]).squeeze(-1)
        slow_before = (self.slow_matrix(state)@key[:, None, None, :, None]).squeeze(-1)
        err = (before[..., :4]-value[:, None, None, :4]).square().mean(-1)
        slow_err = (slow_before[..., :4]-value[:, None, None, :4]).square().mean(-1)
        variance = (state['p'].diagonal(dim1=-2, dim2=-1)*key[:, None, None]).sum(-1)
        standardized = err/(variance+event['noise'][:, None, None].square()+.02)
        consistent = (-.5*standardized.clamp_max(30)).exp()
        quality = 1/(1+10*event['noise'][:, None, None].square())
        support_at = (state['support']*key[:, None, None]).sum(-1)
        gain = (slow_err-err).clamp(-1, 1)
        utility_at = (state['utility']*key[:, None, None]).sum(-1)
        utility = utility_at+.25*assignment*(gain-utility_at)
        gate = (2*(support_at-1)).sigmoid()*(4*utility).sigmoid()*consistent*quality
        gate = gate*(.2+.8*self.gates(feature)[..., 1].sigmoid())*assignment
        chosen_key, candidate = key, before
        if c.variant == 'frequency_only':
            chosen_key = torch.where(event['request'][:, None], query_key, key)
            candidate = (state['m']@chosen_key[:, None, None, :, None]).squeeze(-1)
            gate = torch.where(event['request'][:, None, None], query_port, assignment)*.75
            trace = torch.ones_like(trace)
        if c.variant == 'no_consolidation':
            gate = gate*0
        basis_feature = torch.einsum('rde,be->brd', self.basis(), chosen_key)[..., :4]
        prediction = (self.slow_matrix(state)@chosen_key[:, None, None, :, None]).squeeze(-1)
        local_gradient = torch.einsum('bcnd,brd->bcnr', (prediction-candidate)[..., :4], basis_feature)
        eligibility_factor = trace.sum(-2).clamp_max(4.)
        protection = (state['protect']*chosen_key[:, None, None]).sum(-1)
        rate = .3*self.raw_slow_rate.sigmoid()/(1+.2*protection)
        slow = state['slow']-rate[..., None]*gate[..., None]*eligibility_factor*local_gradient
        slow = slow/(slow.norm(dim=-1, keepdim=True)/8).clamp_min(1.)
        # A conflict lowers protection and applicability before slow weights are rewritten.
        conflict = ((slow_err-.12)/(.12+event['noise'][:, None, None].square())).clamp(0, 1)
        protect = state['protect']*(1-.5*assignment[..., None]*conflict[..., None]*key[:, None, None])
        protect = (protect+gate[..., None]*chosen_key[:, None, None]).clamp_max(100.)
        app_target = (-slow_err/(.1+event['noise'][:, None, None].square())).exp()
        app = state['app']+assignment[..., None]*(.15+.65*conflict[..., None])*key[:, None, None]*(app_target[..., None]-state['app'])
        result = dict(state)
        result.update(slow=slow, trace=trace, protect=protect, app=app,
                      support=(state['support']+assignment[..., None]*quality[..., None]*consistent[..., None]*key[:, None, None]).clamp_max(100.),
                      utility=state['utility']+key[:, None, None]*(utility-utility_at)[..., None],
                      innovation=.8*state['innovation']+.2*torch.stack((standardized.clamp_max(20), conflict, gain, quality.expand_as(gain)), -1)*assignment[..., None])
        return result

    def tick(self, state, index, intervention=None, damping=True):
        c = self.cfg; d = c.matrix_size; result = dict(state)
        feature = self.features(state); pairs = getattr(self, f'pairs_{index % self.matching_count}')
        i, j = pairs[:, 0], pairs[:, 1]
        first, second = feature[:, :, i], feature[:, :, j]
        zi, zj = state['z'][:, :, i], state['z'][:, :, j]
        cosine, sine = (zi*zj).sum(-1), zi[..., 0]*zj[..., 1]-zi[..., 1]*zj[..., 0]
        coeff = self.coefficient(torch.cat((first, second), -1))
        co, si = cosine, sine
        strength = state['amplitude'][:, :, i]*state['amplitude'][:, :, j]
        if c.modes > 1:
            co, si = (torch.cat((cosine, cosine[..., :-1]*cosine[..., 1:]+sine[..., :-1]*sine[..., 1:]), -1),
                      torch.cat((sine, sine[..., :-1]*cosine[..., 1:]-cosine[..., :-1]*sine[..., 1:]), -1))
            strength = torch.cat((strength, strength[..., :-1]*strength[..., 1:]), -1)
        angle = coeff[..., 0]
        if self.fourier:
            angle = angle+(coeff[..., 1:1+self.harmonics]*co*strength).sum(-1)/math.sqrt(self.harmonics)
            angle = angle+(coeff[..., 1+self.harmonics:]*si*strength).sum(-1)/math.sqrt(self.harmonics)
        angle = .8*angle.tanh()
        if c.variant == 'message':
            projected = self.message_in(state['x'].flatten(-2))
            edge_gain = (.5+.5*angle.tanh())[..., None, None]
            neighbor = projected.index_copy(2, i, edge_gain*projected[:, :, j]).index_copy(2, j, edge_gain*projected[:, :, i])
            gate = .2*self.message_gate(feature).sigmoid()[..., None, None]
            target = self.message_out((projected+neighbor).tanh()).reshape_as(state['x']).tanh()
            x = (1-gate)*state['x']+gate*target if intervention != 'no_messages' else state['x']
        else:
            angles = .15*self.content_angles(feature).tanh().reshape(*feature.shape[:-1], 1, 2, d//2)
            x = rotate_rows(state['x'], angles[..., 0, :])
            x = rotate_rows(x.transpose(-1, -2), angles[..., 1, :]).transpose(-1, -2)
            if intervention != 'no_messages':
                x = pair_rotate(x, pairs, angle)
        if damping:
            tau = self.raw_tau.clamp(math.log(2), math.log(4096)).exp()
            if c.variant == 'single_timescale':
                tau = tau.mean().expand_as(tau)
            x = x*torch.exp(-1/(tau*c.internal_steps))[None, None, None, :, None, None]
        result['x'] = x
        drive, speed, amp = self.oscillator(feature).chunk(3, -1)
        rate = .1*state['rho'][..., None, None]
        coupling = torch.zeros_like(state['omega'])
        if intervention not in ('no_coupling', 'no_messages'):
            coupling = coupling.index_copy(2, i, .1*sine).index_copy(2, j, -.1*sine)
        result['omega'] = (1-rate)*state['omega']+rate*(.03+.7*speed.sigmoid())
        result['amplitude'] = (1-rate)*state['amplitude']+rate*amp.sigmoid()
        if self.fourier:
            result['z'] = rotate(state['z'], rate*(result['omega']+.1*drive.tanh()+coupling))
        else:
            relaxation = rate*(.5+result['omega'])
            result['z'] = (1-relaxation[..., None])*state['z']+relaxation[..., None]*torch.stack((drive.tanh(), speed.tanh()), -1)
        return result

    def step(self, event, state, intervention=None):
        c = self.cfg; d = c.matrix_size
        if intervention == 'clear_slow':
            state = state | {'slow': torch.zeros_like(state['slow']), 'protect': torch.zeros_like(state['protect'])}
        state = self.reset_fast(state, event['restart'] & event['valid'])
        state = self.route(state, event['cue'], event['valid'])
        fresh = event['observe'] & event['valid'] & ~(state['seen'] == event['evidence_id'][:, None]).any(-1)
        key = F.one_hot(event['key'], d).to(state['m'].dtype)
        value = F.pad(event['value'], (0, d-4))
        local_key = key[:, None, None].expand(-1, c.contexts, c.nodes, -1)
        mean = (state['m']@local_key[..., None]).squeeze(-1)
        error = value[:, None, None]-mean
        port = self.write_ports[event['key']][:, None]*fresh[:, None, None]
        assignment = state['rho'][..., None]*port
        feature = self.features(state)
        predictive = ((error[..., :4].square().mean(-1)*assignment).sum((1, 2))/assignment.sum((1, 2)).clamp_min(1))
        state = self.plasticity(state, event, key, value, assignment, fresh, feature)
        if intervention == 'clear_slow':
            state['slow'] = torch.zeros_like(state['slow']); state['protect'] = torch.zeros_like(state['protect'])
        weight = assignment*(.1+.9*self.gates(feature)[..., 0].sigmoid())
        state['m'], state['p'] = gaussian_update(state['m'], state['p'], local_key,
            value[:, None, None].expand_as(local_key), event['noise'][:, None, None].square()+.005,
            weight, .02*self.raw_process.sigmoid()*assignment)
        packet = value[:, None, None, :, None]*local_key[..., None, :]*assignment[..., None, None]
        residual = error[..., :, None]*local_key[..., None, :]*assignment[..., None, None]
        alpha = self.raw_dynamic.sigmoid() if intervention != 'fixed_external' else self.raw_dynamic*0
        dynamic = (state['m']+state['x'].mean(-3)+self.slow_matrix(state)).tanh()/math.sqrt(d)
        injected = packet+alpha*(dynamic@packet)+.2*residual
        memory = self.memory(state, intervention)
        for query, active in ((event['query_a'], event['request']),
                              (event['query_b'], event['request'] & event['combine'])):
            address = F.one_hot(query, d).to(value.dtype)
            response = (memory@address[:, None, None, :, None]).squeeze(-1)
            query_port = self.write_ports[query][:, None]*state['rho'][..., None]*active[:, None, None]
            injected = injected+response[..., :, None]*address[:, None, None, None, :]*query_port[..., None, None]
        injected = injected/(injected.norm(dim=(-1, -2), keepdim=True)/3).clamp_min(1.)
        target_activity = .1+.9*self.gates(feature)[..., 2].sigmoid()
        activity = state['x'].square().mean((-1, -2, -3))
        homeostatic_gain = (2*target_activity/(target_activity+activity+1e-6)).sqrt().clamp(.25, 1.5)
        injected = injected*homeostatic_gain[..., None, None]
        tau = self.raw_tau.clamp(math.log(2), math.log(4096)).exp()
        if c.variant == 'single_timescale':
            tau = tau.mean().expand_as(tau)
        gain = (1-torch.exp(-2/tau)).sqrt()*(.5+self.raw_input_gain.sigmoid())
        gain = gain/gain.norm().clamp_min(1e-6)
        state['x'] = state['x']+injected[..., None, :, :]*gain[None, None, None, :, None, None]
        for tick in range(c.internal_steps):
            state = self.tick(state, tick, intervention)
        seen = state['seen'].clone(); old = seen.gather(1, state['cursor'][:, None]).squeeze(1)
        seen.scatter_(1, state['cursor'][:, None], torch.where(fresh, event['evidence_id'], old)[:, None])
        state['seen'] = seen; state['cursor'] = (state['cursor']+fresh.long()) % c.seen_capacity
        return state, predictive, fresh

    def read(self, state, qa, qb, combine, intervention=None):
        c = self.cfg; memory = self.memory(state, intervention); distributions = []
        for query in (qa, qb):
            address = F.one_hot(query, c.matrix_size).to(memory.dtype)
            m = (memory@address[:, None, None, :, None]).squeeze(-1)
            x = (state['x']@address[:, None, None, None, :, None]).squeeze(-1)
            weights = self.read_ports[query][:, None, :, None]
            content = (m*weights).sum(2)
            carrier = (x*weights[..., None]).sum(2)
            band_weights = torch.linspace(1, 2, c.bands, device=carrier.device, dtype=carrier.dtype)
            fast = (carrier*band_weights[None, None, :, None]).sum(-2)/band_weights.norm()
            slow = (carrier/band_weights[None, None, :, None]).sum(-2)/(1/band_weights).norm()
            scale = (.5+self.raw_read_scale.sigmoid())*math.sqrt(c.nodes/32)
            features = torch.cat((content, scale*fast, scale*slow, address[:, None].expand(-1, c.contexts, -1)), -1)
            distributions.append((self.readout(features)*(1+5*self.raw_temperature.sigmoid())).softmax(-1))
        first, second = distributions
        joint = torch.stack([sum(first[..., j]*second[..., (i-j) % 4] for j in range(4)) for i in range(4)], -1)
        probability = (state['rho'][..., None]*torch.where(combine[:, None, None], joint, first)).sum(1).clamp_min(1e-8)
        return (probability/probability.sum(-1, keepdim=True)).log()

    def forward(self, key, value, noise, cue, source, evidence_id, observe, query_a, query_b,
                combine, request, restart, valid, state=None, intervention=None, intervention_mask=None):
        state = self.initial_state(len(key)) if state is None else state
        outputs, predictions, routing, freshs, slow_norms = [], [], [], [], []
        for t in range(key.shape[1]):
            if intervention in ('reset_history', 'swap_history', 'reset_fast') and intervention_mask is not None:
                mask = intervention_mask[:, t]
                if intervention == 'reset_fast':
                    state = self.reset_fast(state, mask)
                else:
                    changed = self.initial_state(len(key)) if intervention == 'reset_history' else {k: v.roll(1, 0) for k, v in state.items()}
                    state = {k: torch.where(mask.reshape(len(key), *([1]*(v.ndim-1))), changed[k], v) for k, v in state.items()}
            before = state
            event = dict(key=key[:, t], value=value[:, t], noise=noise[:, t], cue=cue[:, t], source=source[:, t],
                         evidence_id=evidence_id[:, t], observe=observe[:, t], query_a=query_a[:, t], query_b=query_b[:, t],
                         combine=combine[:, t], request=request[:, t], restart=restart[:, t], valid=valid[:, t])
            if self.cfg.checkpoint_events and self.training and torch.is_grad_enabled():
                state, predictive, fresh = checkpoint(self.step, event, state, intervention, use_reentrant=False, preserve_rng_state=False)
            else:
                state, predictive, fresh = self.step(event, state, intervention)
            state = {k: torch.where(valid[:, t].reshape(len(key), *([1]*(v.ndim-1))), v, before[k]) for k, v in state.items()}
            outputs.append(self.read(state, query_a[:, t], query_b[:, t], combine[:, t], intervention))
            predictions.append(predictive); routing.append(state['rho']); freshs.append(fresh)
            slow_norms.append(state['slow'].square().mean((1, 2, 3)).sqrt())
        return torch.stack(outputs, 1), state, dict(predictive=torch.stack(predictions, 1), routing=torch.stack(routing, 1),
                                                  fresh=torch.stack(freshs, 1), slow_norm=torch.stack(slow_norms, 1))

    def ledger(self):
        state = self.initial_state(1, 'cpu')
        floating = {k: v.numel() for k, v in state.items() if v.is_floating_point()}
        return dict(config=asdict(self.cfg), parameters=sum(p.numel() for p in self.parameters()),
                    adaptive_local_weight_elements=state['slow'].numel(), state_floats_per_stream=sum(floating.values()),
                    state_breakdown=floating, state_integer_or_boolean_elements=sum(v.numel() for v in state.values() if not v.is_floating_point()),
                    matching_count=self.matching_count, persistent_on_fast_reset=list(PERSISTENT),
                    matching='Same graph, ports, allocated state and event schedule; useful global parameters matched for message control. Inactive reserved ablation state is disclosed; runtime measured.',
                    local_learning='Per-lifetime functional synaptic coefficients; AdamW learns shared rules. Local weights are checkpointable tensors, not cross-batch nn.Parameter mutation.',
                    evidence='Only fresh external reports update M/P and verified slow weights. Queries can update frequency_only control.',
                    global_control='Routing sees cues and cue history only, not observation values or prediction errors.',
                    guarantee='Orthogonal carrier substep at fixed angles; full state-dependent Jacobian and input/damping are measured, not declared unitary.')
