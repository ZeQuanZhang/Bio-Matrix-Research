"""Full-matrix v0.9 core: legal content carrier and reusable block algebra.

content_base=False,address_size=0 is the v0.8 dynamics control. Reuse of fixed
features/linear terms and splitting Fourier endpoints are algebraic execution
changes. The content carrier and low-dimensional addresses are new dynamics.
Graph views are provided by graph_buffer/node_anchor so group selection can
compact actual execution without slicing the full persistent state here.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from v05.model import rotate_rows
from v07.model import bounded, jump_gate, unit_rotate
from v08.model import (Config as PreviousConfig, Network as PreviousNetwork,
                       initialize as previous_initialize, rotate_vector)


@dataclass
class Config(PreviousConfig):
    implementation: str = 'sequential'
    content_base: bool = True
    reuse_controls: bool = True
    address_size: int = 8

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.address_size, int) or isinstance(self.address_size, bool) or not 0 <= self.address_size <= self.matrix_size**2:
            raise ValueError('address_size must be an integer in [0,d*d]')
        if not isinstance(self.content_base, bool) or not isinstance(self.reuse_controls, bool):
            raise ValueError('content_base and reuse_controls must be boolean')


class Network(PreviousNetwork):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.runtime_bio_read_enabled = True
        if cfg.address_size:
            self.address_projection = nn.Linear(cfg.matrix_size**2, cfg.address_size, bias=False)
            nn.init.orthogonal_(self.address_projection.weight)

    def graph_buffer(self, name):
        return getattr(self, name)

    def node_anchor(self):
        return self.node_shape

    def phase_initializer(self):
        return self.initial_phase

    def initial_state(self, batch):
        c = self.cfg; anchor = self.node_anchor(); n, d = anchor.shape[0], c.matrix_size
        zeros = lambda *shape: anchor.new_zeros(*shape)
        base = (batch, n); history = c.credit_horizon-1
        return dict(m=bounded(anchor)[None].expand(batch, -1, -1, -1).clone(),
                    p=torch.eye(d, device=anchor.device, dtype=anchor.dtype).expand(*base, d, d).clone(),
                    x=zeros(*base, c.bands, d, d), slow=zeros(*base, d, d),
                    trace=zeros(*base, c.trace_bands, d), response=zeros(*base, d, d),
                    support=zeros(*base, d), innovation=zeros(*base),
                    z=self.phase_initializer()[None].expand(batch, -1, -1, -1).clone(),
                    prediction=zeros(batch, c.width), feedback_h=zeros(batch, c.width),
                    read_query=zeros(*base, d), read_weight=zeros(*base), feedback_gain=zeros(*base),
                    read_scale=anchor.new_ones(batch), feedback_valid=torch.zeros(batch, dtype=torch.bool, device=anchor.device),
                    credit_prediction=zeros(batch, history, c.width), credit_h=zeros(batch, history, c.width),
                    credit_query=zeros(batch, history, n, d), credit_weight=zeros(batch, history, n),
                    credit_gain=zeros(batch, history, n), credit_scale=zeros(batch, history),
                    credit_valid=torch.zeros(batch, history, dtype=torch.bool, device=anchor.device),
                    evidence_count=torch.zeros(*base, dtype=torch.long, device=anchor.device))

    def features(self, state):
        m, x = state['m'], state['x']
        slow = state['slow'] if self.runtime_slow_enabled else torch.zeros_like(state['slow'])
        uncertainty = state['p'].diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(0)
        innovation = state['innovation'].clamp_min(0)
        summaries = torch.stack((m.mean((-1, -2)), m.square().mean((-1, -2)),
                                x.square().mean((-1, -2, -3)), uncertainty/(1+uncertainty),
                                slow.square().mean((-1, -2)), innovation/(1+innovation)), -1)
        geometry = torch.cat((self.graph_buffer('positions'), self.graph_buffer('shell')[:, None]/(self.cfg.shells-1)), -1)
        return torch.cat((summaries, state['z'].mean(-2), geometry[None].expand(len(m), -1, -1)), -1)

    def stationary_prior(self, state):
        rho = math.exp(-1/self.cfg.covariance_tau)
        mean = math.sqrt(rho)*state['m']+(1-math.sqrt(rho))*bounded(self.node_anchor())[None]
        eye = torch.eye(self.cfg.matrix_size, dtype=state['p'].dtype, device=state['p'].device)
        return mean, rho*state['p']+(1-rho)*eye

    def read_distribution(self, response):
        weight = self.graph_buffer('read_ports')[None]+response.mean((-1, -2)).clamp(0, 2)
        return weight/weight.sum(-1, keepdim=True).clamp_min(1e-6)

    def response_filter(self, flows, mixings, batch):
        response = self.graph_buffer('read_ports')[None].expand(batch, -1)
        source = self.graph_buffer('neighbors')
        index = source.flatten()[None].expand(batch, -1)
        for flow, mixing in zip(reversed(flows), reversed(mixings)):
            contributions = (response[..., None]*mixing[..., None]*flow).flatten(1)
            returned = torch.zeros_like(response).scatter_add(1, index, contributions)
            response = (1-mixing)*response+returned
        # Group execution deliberately calibrates this preconditioner to active N.
        return (response*response.shape[1]).clamp(0, 2.)[..., None, None]

    def local_direction(self, state, embedding):
        direction = super().local_direction(state, embedding)
        if self.cfg.content_base:
            direction = direction*self.bio_read_gain().detach()
        return direction

    def bio_read_gain(self):
        if not self.runtime_bio_read_enabled:
            return torch.zeros_like(self.raw_content_gain)
        return .3*self.raw_content_gain.sigmoid()

    def _fourier_coefficients(self, features):
        source = self.graph_buffer('neighbors')
        if self.cfg.reuse_controls:
            receiver = F.linear(features, self.fourier.weight[:, :12], self.fourier.bias)
            sender = F.linear(features, self.fourier.weight[:, 12:], None)
            return receiver[:, :, None]+sender[:, source]
        pair = torch.cat((features[:, :, None].expand(-1, -1, source.shape[1], -1),
                          features[:, source]), -1)
        return self.fourier(pair)

    def matrix_attention(self, receiver, sender, drive, controls, features, phase):
        source = self.graph_buffer('neighbors')
        query, key = receiver.flatten(-2), sender.flatten(-2)
        if self.cfg.address_size:
            query = self.address_projection(query)
            key = self.address_projection(key)
        query = F.normalize(query, dim=-1, eps=1e-5)
        key = F.normalize(key, dim=-1, eps=1e-5)
        similarity = (query[:, :, None]*key[:, source]).sum(-1)
        temperature = .1+.9*self.raw_attention_temperature.sigmoid()
        score = similarity/temperature
        zi, zj = phase[:, :, None], phase[:, source]
        co = (zi*zj).sum(-1)
        sine = zi[..., 0]*zj[..., 1]-zi[..., 1]*zj[..., 0]
        cross_co = co[..., :-1]*co[..., 1:]+sine[..., :-1]*sine[..., 1:]
        cross_si = sine[..., :-1]*co[..., 1:]-co[..., :-1]*sine[..., 1:]
        harmonic = torch.cat((co, cross_co, sine, cross_si), -1)
        coeff = self._fourier_coefficients(features)
        score = score+.25*(coeff[..., 0]+(coeff[..., 1:]*harmonic).mean(-1)).tanh()
        semantic = torch.cat((sender.flatten(-2)/self.cfg.matrix_size,
                              drive[..., :self.cfg.matrix_size]/math.sqrt(self.cfg.matrix_size)), -1)
        intensity = semantic.norm(dim=-1)[:, source]
        threshold = .001+.009*controls[..., 2].sigmoid()
        outward = jump_gate(intensity-threshold[:, :, None], self.training, self.gate_blend)
        mask = self.graph_buffer('edge_mask')
        gate = torch.where(self.graph_buffer('outward')[None], outward, torch.ones_like(outward))*mask[None]
        score = score.masked_fill(~mask[None].bool(), -torch.inf)
        competition = torch.cat((score, torch.zeros_like(score[..., :1])), -1).softmax(-1)[..., :-1]
        return competition*gate*(intensity/(.02+intensity)), gate, sine

    def base_parent(self):
        source = self.graph_buffer('neighbors'); shell = self.graph_buffer('shell')
        valid = self.graph_buffer('edge_mask').bool() & (shell[source] == shell[:, None]-1)
        slot = valid.to(torch.int64).argmax(-1)
        parent = source.gather(1, slot[:, None]).squeeze(1)
        return parent, valid.any(-1)

    def advance_base(self, content, mass, fresh, parent, valid):
        write = self.graph_buffer('write_ports')
        received_mass = mass[:, parent]*valid[None].to(mass.dtype)
        received = content[:, parent]*received_mass[..., None, None]
        content = torch.where(write[None, :, None, None].bool(), fresh[:, None], received)
        mass = torch.where(write[None].bool(), torch.ones_like(received_mass), received_mass)
        return content, mass

    def block_coefficients(self, tokens, state):
        c = self.cfg; b, t = tokens.shape; n, d = state['m'].shape[1], c.matrix_size
        write = self.graph_buffer('write_ports')
        source = self.graph_buffer('neighbors')
        embedding = self.embedding(tokens)
        value, key = self.input_projection(embedding).chunk(2, -1)
        packet = torch.cat((value.tanh(), F.normalize(key, dim=-1, eps=1e-5),
                            torch.ones(b, t, 1, dtype=embedding.dtype, device=tokens.device)), -1).flatten(0, 1)
        fresh_content = self.content_projection(embedding).reshape(b*t, d, d)
        drive = write[None, :, None]*packet[:, None]
        if c.reuse_controls:
            old = state['x'].mean(-3)+.3*state['m']
            if self.runtime_slow_enabled:
                old = old+.2*state['slow']
            old = old[:, None].expand(-1, t, -1, -1, -1).flatten(0, 1)
            response = state['response']
            snapshot = {'response': response[:, None].expand(-1, t, -1, -1, -1).flatten(0, 1)}
        else:
            snapshot = {name: state[name][:, None].expand(-1, t, *state[name].shape[1:]).flatten(0, 1)
                        for name in ('m', 'p', 'x', 'slow', 'innovation', 'z', 'response')}
            old = snapshot['x'].mean(-3)+.3*snapshot['m']
            if self.runtime_slow_enabled:
                old = old+.2*snapshot['slow']
        base_features = self.features(state)
        base_control = self.controls(base_features)
        phase_speed = .02+.35*base_control[..., 5+5*d:5+5*d+c.modes].sigmoid()
        time = torch.arange(t, dtype=embedding.dtype, device=tokens.device)[None, :, None, None]
        phase = unit_rotate(state['z'][:, None], time*phase_speed[:, None]).flatten(0, 1)
        if c.reuse_controls:
            fixed_features = torch.cat((base_features[..., :6], torch.zeros_like(base_features[..., 6:8]), base_features[..., 8:]), -1)
            fixed_projection = self.controls[0](fixed_features)
            fixed_features = fixed_features[:, None].expand(-1, t, -1, -1).flatten(0, 1)
            fixed_projection = fixed_projection[:, None].expand(-1, t, -1, -1).flatten(0, 1)
        base_content = write[None, :, None, None]*fresh_content[:, None]
        base_mass = write[None].expand(b*t, -1)
        parent, parent_valid = self.base_parent() if c.content_base else (None, None)
        transport = torch.zeros_like(old)
        flows, mixings, gates = [], [], []
        for _ in range(c.internal_steps):
            if c.reuse_controls:
                phase_features = phase.mean(-2)
                features = torch.cat((fixed_features[..., :6], phase_features, fixed_features[..., 8:]), -1)
                preactivation = fixed_projection+F.linear(phase_features, self.controls[0].weight[:, 6:8], None)
                control = self.controls[2](self.controls[1](preactivation))+self.drive_controls(drive)
            else:
                features = self.features(snapshot | {'z': phase})
                control = self.controls(features)+self.drive_controls(drive)
            uin, vin, uout, vout = control[..., 5:5+4*d].chunk(4, -1)
            mass = drive[..., -1].clamp_min(.05)
            local_value = (drive[..., :d]/mass[..., None]).clamp(-2, 2)
            local_key = F.normalize(drive[..., d:2*d], dim=-1, eps=1e-5)
            local_value, local_key = self.observe(local_value, local_key, control)
            local_packet = local_value[..., :, None]*local_key[..., None, :]*drive[..., -1, None, None]
            fresh = base_content if c.content_base else write[None, :, None, None]*fresh_content[:, None]
            communication = old+.5*transport+.5*local_packet+fresh
            receiver = self.external(communication, uin, vin, self.input_diagonal)
            sender = self.external(communication, uout, vout, self.output_diagonal)
            angle, frequency, amplitude, forcing = torch.split(control[..., 5+4*d:], [d, c.modes, c.modes, c.modes], -1)
            angles = .12*angle.tanh()
            sender = rotate_rows(sender, angles[..., :d//2])
            sender = rotate_rows(sender.transpose(-1, -2), angles[..., d//2:]).transpose(-1, -2)
            weight, gate, sine = self.matrix_attention(receiver, sender, drive, control, features, phase)
            transport = (weight[..., None, None]*sender[:, source]).sum(2)
            received = (weight[..., None]*drive[:, source]).sum(2)
            drive = .25*drive+.75*received
            drive = torch.where(write[None, :, None].bool(), packet[:, None], drive)
            coupling = .2*self.raw_coupling.sigmoid()*(weight[..., None]*sine).sum(2)
            speed = .02+.35*frequency.sigmoid()+.08*forcing.tanh()+coupling
            phase = unit_rotate(phase, speed/c.internal_steps)
            mixing = .1+.2*control[..., 3].sigmoid()
            flows.append(weight); mixings.append(mixing); gates.append(gate)
            if c.content_base:
                base_content, base_mass = self.advance_base(base_content, base_mass, fresh_content, parent, parent_valid)
        local_value, local_key, arrived = self.final_observation(drive, control)
        delta_key = F.normalize(local_key+.25*transport.mean(-2), dim=-1, eps=1e-5)
        delta_value = (local_value+mixing[..., None]*(transport@delta_key[..., None]).squeeze(-1)).tanh()
        carrier = .15*(phase[..., 0]*amplitude.sigmoid()).mean(-1)
        carrier = carrier.reshape(b, t, n).cumsum(1)[..., None].expand(-1, -1, -1, d//2)
        rotating_value = rotate_vector(delta_value.reshape(b, t, n, d), -carrier)
        keys = delta_key.reshape(b, t, n, d).permute(0, 2, 1, 3)[:, :, None].expand(-1, -1, c.bands, -1, -1)
        values = rotating_value.permute(0, 2, 1, 3)[:, :, None].expand_as(keys)
        tau = self.raw_tau.clamp(math.log(2), math.log(4096)).exp()
        alpha = torch.exp(-1/tau)[None, None, :, None].expand(b, n, -1, t)
        beta = (.02+.28*control[..., 0].sigmoid()).reshape(b, t, n).permute(0, 2, 1)
        beta = beta[:, :, None].expand(-1, -1, c.bands, -1)
        response = self.response_filter(flows, mixings, b*t).expand(-1, -1, d, d)
        blend = self.raw_feedback_tau.sigmoid()
        response = blend*snapshot['response']+(1-blend)*response
        return dict(key=keys, value=values, alpha=alpha, beta=beta, carrier=carrier,
                    local_value=local_value, local_key=local_key, arrived=arrived,
                    control=control[..., :5], response=response, phase=phase,
                    gate=gates[-1], attention=flows[-1], transport=transport,
                    content_base_transport=base_content if c.content_base else torch.zeros_like(base_content),
                    content_base_mass=base_mass if c.content_base else torch.zeros_like(base_mass))

    def read_block(self, states, coefficients, state, b, t):
        c = self.cfg; n, d = state['m'].shape[1], c.matrix_size
        local_key, local_value = coefficients['local_key'], coefficients['local_value']
        arrived, controls = coefficients['arrived'], coefficients['control']
        expand = lambda tensor: tensor[:, None].expand(-1, t, *tensor.shape[1:]).flatten(0, 1)
        prior_m, _ = self.stationary_prior(state)
        innovation = (local_value-(expand(prior_m)@local_key[..., None]).squeeze(-1)).square().mean(-1)
        variance = .02+.3*controls[..., 4].sigmoid()
        trace_tau = self.raw_trace_tau.clamp(math.log(1), math.log(2048)).exp()
        decay = torch.exp(-1/trace_tau)
        trace = (expand(state['trace'])*decay[None, None, :, None]
                 +(1-decay)[None, None, :, None]*local_key[..., None, :]*arrived[..., None, None])
        query = F.normalize(local_key+.25*trace.mean(-2), dim=-1, eps=1e-5)
        compatibility = torch.exp(-innovation/(variance+.05))*controls[..., 1].sigmoid()
        read_weight = compatibility*self.read_distribution(coefficients['response'])
        read_scale, effective = self.write_scale(read_weight)
        ports = self.graph_buffer('read_ports')
        memory = (expand(state['m'])*ports[None, :, None, None]).sum(1)
        if self.runtime_slow_enabled:
            memory = memory+self.slow_read(expand(state['slow']), query, read_weight)
        physical = states.permute(0, 3, 1, 2, 4, 5)
        carrier = coefficients['carrier'][:, :, :, None].expand(-1, -1, -1, c.bands, -1)
        physical = bounded(rotate_rows(physical, carrier))
        gamma = self.bio_read_gain()
        fast = physical.mean(-3).flatten(0, 1)+gamma*coefficients['transport']
        fast = (fast*ports[None, :, None, None]).sum(1)
        if c.content_base:
            mass = coefficients['content_base_mass']*ports[None]
            weights = mass/mass.sum(-1, keepdim=True).clamp_min(1e-6)
            base = (weights[..., None, None]*coefficients['content_base_transport']).sum(1)
            fast = base+gamma*bounded(fast)
            memory = gamma*memory
        pooled = torch.cat((fast.flatten(1), memory.flatten(1)), -1)
        hidden = self.read_projection(pooled)
        logits = F.linear(self.head(hidden), self.embedding.weight)
        support = (.999*expand(state['support'])+local_key.square()*arrived[..., None]).clamp_max(100)
        exposure = support.mean(-1)
        trust = exposure/(exposure+2)*torch.exp(-innovation.clamp(0, 20))
        gain = coefficients['response'].mean((-1, -2)).clamp(0, 2)*trust
        return logits.reshape(b, t, -1), physical, dict(hidden=hidden, query=query, weight=read_weight,
                            scale=read_scale, gain=gain, effective=effective, innovation=innovation)

    def commit_block(self, tokens, state, coefficients, reads, logits, physical):
        state, diagnostics = super().commit_block(tokens, state, coefficients, reads, logits, physical)
        mass = coefficients['content_base_mass']
        diagnostics['content_base_surface_mass'] = (mass*self.graph_buffer('read_ports')[None]).sum(-1).mean()
        diagnostics['active_nodes'] = logits.new_tensor(state['m'].shape[1])
        return state, diagnostics


Core = Network


@torch.no_grad()
def initialize(model, seed):
    previous_initialize(model, seed)
    if hasattr(model, 'address_projection'):
        generator = torch.Generator().manual_seed(seed+900001)
        matrix = torch.empty(model.address_projection.weight.shape, dtype=torch.float64)
        nn.init.orthogonal_(matrix, generator=generator)
        model.address_projection.weight.copy_(matrix)
