"""Bio v0.8 C candidate: directed matrix attention and block delta dynamics.

This is a new model, not an equivalent execution of v0.7. Slow context and the
old fast matrix are frozen while a block's coefficients are generated. All
tokens use the same snapshot, their own input, and J synchronous graph rounds.
No coefficient reads a fast state that is still to be computed inside the block.
The arrival-gated-observation rule suppresses fresh value/key components below
the same arrival threshold used by Joseph updates; historical transport stays on.

Fast matrices evolve in co-rotating coordinates. Norm bounds apply to the read
view and the block boundary, not inside the affine recurrence. Observed-token
feedback and Joseph observations are consumed sequentially at the boundary;
their new slow context affects the next block. The last prediction stays pending.
Calling with complete blocks is chunk invariant. Ending a call with a partial
block commits an early boundary, and is intentionally a different schedule.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.profiler import record_function

from v04.model import rotate
from v05.model import rotate_rows
from v07.model import (Config as PreviousConfig, Network as PreviousNetwork,
                       initialize as previous_initialize,
                       bounded, joseph, jump_gate, reset_lanes, unit_rotate)
from .recurrence import sequential_delta, parallel_delta, triangular_delta


@dataclass
class Config(PreviousConfig):
    block_size: int = 16
    implementation: str = 'triangular'

    def __post_init__(self):
        super().__post_init__()
        if self.variant != 'full':
            raise ValueError('v0.8 currently implements the complete full variant only')
        if not isinstance(self.block_size, int) or isinstance(self.block_size, bool) or self.block_size < 1:
            raise ValueError('block_size must be a positive integer')
        if self.implementation not in ('sequential', 'parallel', 'triangular'):
            raise ValueError('implementation must be sequential, parallel or triangular')
        if self.checkpoint_events:
            raise ValueError('v0.8 block checkpointing is not implemented')


def rotate_vector(value, angle):
    paired = value.reshape(*value.shape[:-1], value.shape[-1]//2, 2)
    return rotate(paired, angle).flatten(-2)


class Network(PreviousNetwork):
    def __init__(self, cfg):
        super().__init__(cfg)
        controls = self.controls[-1].out_features
        self.drive_controls = nn.Linear(2*cfg.matrix_size+1, controls, bias=False)
        self.content_projection = nn.Linear(cfg.width, cfg.matrix_size**2, bias=False)
        nn.init.orthogonal_(self.content_projection.weight)
        self.raw_content_gain = nn.Parameter(torch.tensor(math.log(.5)))
        self.raw_attention_temperature = nn.Parameter(torch.tensor(-1.))

    def local_direction(self, state, embedding):
        # F1 eligibility is a local rule, not a higher-order head derivative.
        return super().local_direction(state, embedding).detach()

    def final_observation(self, drive, control):
        d = self.cfg.matrix_size
        mass = drive[..., -1].clamp_min(.05)
        value = (drive[..., :d]/mass[..., None]).clamp(-2, 2)
        key = F.normalize(drive[..., d:2*d], dim=-1, eps=1e-5)
        value, key = self.observe(value, key, control)
        arrived = drive[..., -1] > 1e-4
        # Unobserved fresh pulses must not acquire 1/eps eligibility via unit keys.
        value = torch.where(arrived[..., None], value, torch.zeros_like(value))
        key = torch.where(arrived[..., None], key, torch.zeros_like(key))
        return value, key, arrived

    def matrix_attention(self, receiver, sender, drive, controls, features, phase):
        """Incoming competition with a null option; no all-masked softmax row."""
        source = self.neighbors
        query = F.normalize(receiver.flatten(-2), dim=-1, eps=1e-5)
        key = F.normalize(sender.flatten(-2), dim=-1, eps=1e-5)
        similarity = (query[:, :, None]*key[:, source]).sum(-1)
        temperature = .1+.9*self.raw_attention_temperature.sigmoid()
        score = similarity/temperature
        zi, zj = phase[:, :, None], phase[:, source]
        co = (zi*zj).sum(-1)
        sine = zi[..., 0]*zj[..., 1]-zi[..., 1]*zj[..., 0]
        cross_co = co[..., :-1]*co[..., 1:]+sine[..., :-1]*sine[..., 1:]
        cross_si = sine[..., :-1]*co[..., 1:]-co[..., :-1]*sine[..., 1:]
        harmonic = torch.cat((co, cross_co, sine, cross_si), -1)
        pair = torch.cat((features[:, :, None].expand(-1, -1, source.shape[1], -1),
                          features[:, source]), -1)
        coeff = self.fourier(pair)
        score = score+.25*(coeff[..., 0]+(coeff[..., 1:]*harmonic).mean(-1)).tanh()
        semantic = torch.cat((sender.flatten(-2)/self.cfg.matrix_size,
                              drive[..., :self.cfg.matrix_size]/math.sqrt(self.cfg.matrix_size)), -1)
        intensity = semantic.norm(dim=-1)[:, source]
        threshold = .001+.009*controls[..., 2].sigmoid()
        outward = jump_gate(intensity-threshold[:, :, None], self.training, self.gate_blend)
        gate = torch.where(self.outward[None], outward, torch.ones_like(outward))*self.edge_mask[None]
        score = score.masked_fill(~self.edge_mask[None].bool(), -torch.inf)
        null = torch.zeros_like(score[..., :1])
        competition = torch.cat((score, null), -1).softmax(-1)[..., :-1]
        # An absolute amplitude gate can reject every neighbor, including at degree one.
        weight = competition*gate*(intensity/(.02+intensity))
        return weight, gate, sine

    def block_coefficients(self, tokens, state):
        """Batch T positions spatially; no cross-token mixing in this stage."""
        c = self.cfg; b, t = tokens.shape; n, d = c.nodes, c.matrix_size
        embedding = self.embedding(tokens)
        value, key = self.input_projection(embedding).chunk(2, -1)
        packet = torch.cat((value.tanh(), F.normalize(key, dim=-1, eps=1e-5),
                            torch.ones(b, t, 1, dtype=embedding.dtype, device=tokens.device)), -1)
        packet = packet.flatten(0, 1)
        fresh_content = self.content_projection(embedding).reshape(b*t, d, d)
        drive = self.write_ports[None, :, None]*packet[:, None]
        snapshot = {name: state[name][:, None].expand(-1, t, *state[name].shape[1:]).flatten(0, 1)
                    for name in ('m', 'p', 'x', 'slow', 'innovation', 'z', 'response')}
        old = snapshot['x'].mean(-3)+.3*snapshot['m']
        if self.runtime_slow_enabled:
            old = old+.2*snapshot['slow']
        base_features = self.features(state)
        base_control = self.controls(base_features)
        frequency = base_control[..., 5+5*d:5+5*d+c.modes]
        phase_speed = .02+.35*frequency.sigmoid()
        time = torch.arange(t, dtype=embedding.dtype, device=tokens.device)[None, :, None, None]
        phase = unit_rotate(state['z'][:, None], time*phase_speed[:, None]).flatten(0, 1)
        transport = torch.zeros_like(old)
        flows, mixings, gates = [], [], []
        for _ in range(c.internal_steps):
            frame = snapshot | {'z': phase}
            features = self.features(frame)
            control = self.controls(features)+self.drive_controls(drive)
            uin, vin, uout, vout = control[..., 5:5+4*d].chunk(4, -1)
            mass = drive[..., -1].clamp_min(.05)
            local_value = (drive[..., :d]/mass[..., None]).clamp(-2, 2)
            local_key = F.normalize(drive[..., d:2*d], dim=-1, eps=1e-5)
            local_value, local_key = self.observe(local_value, local_key, control)
            local_packet = local_value[..., :, None]*local_key[..., None, :]*drive[..., -1, None, None]
            communication = (old+.5*transport+.5*local_packet
                             +self.write_ports[None, :, None, None]*fresh_content[:, None])
            receiver = self.external(communication, uin, vin, self.input_diagonal)
            sender = self.external(communication, uout, vout, self.output_diagonal)
            angle, frequency, amplitude, forcing = torch.split(control[..., 5+4*d:],
                                                               [d, c.modes, c.modes, c.modes], -1)
            angles = .12*angle.tanh()
            sender = rotate_rows(sender, angles[..., :d//2])
            sender = rotate_rows(sender.transpose(-1, -2), angles[..., d//2:]).transpose(-1, -2)
            weight, gate, sine = self.matrix_attention(receiver, sender, drive, control, features, phase)
            transport = (weight[..., None, None]*sender[:, self.neighbors]).sum(2)
            received = (weight[..., None]*drive[:, self.neighbors]).sum(2)
            drive = .25*drive+.75*received
            drive = torch.where(self.write_ports[None, :, None].bool(), packet[:, None], drive)
            coupling = .2*self.raw_coupling.sigmoid()*(weight[..., None]*sine).sum(2)
            speed = .02+.35*frequency.sigmoid()+.08*forcing.tanh()+coupling
            phase = unit_rotate(phase, speed/c.internal_steps)
            mixing = .1+.2*control[..., 3].sigmoid()
            flows.append(weight); mixings.append(mixing); gates.append(gate)
        local_value, local_key, arrived = self.final_observation(drive, control)
        # Content travels only through nodes/edges; there is no embedding-to-head bypass.
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
                    gate=gates[-1], attention=flows[-1], transport=transport)

    def read_block(self, states, coefficients, state, b, t):
        c = self.cfg; n, d = c.nodes, c.matrix_size
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
        memory = (expand(state['m'])*self.read_ports[None, :, None, None]).sum(1)
        if self.runtime_slow_enabled:
            memory = memory+self.slow_read(expand(state['slow']), query, read_weight)
        physical = states.permute(0, 3, 1, 2, 4, 5)
        carrier = coefficients['carrier'][:, :, :, None].expand(-1, -1, -1, c.bands, -1)
        physical = bounded(rotate_rows(physical, carrier))
        fast = (physical.mean(-3).flatten(0, 1)
                +.3*self.raw_content_gain.sigmoid()*coefficients['transport'])
        pooled = torch.cat(((fast*self.read_ports[None, :, None, None]).sum(1).flatten(1),
                            memory.flatten(1)), -1)
        hidden = self.read_projection(pooled)
        logits = F.linear(self.head(hidden), self.embedding.weight)
        support = (.999*expand(state['support'])+local_key.square()*arrived[..., None]).clamp_max(100)
        exposure = support.mean(-1)
        trust = exposure/(exposure+2)*torch.exp(-innovation.clamp(0, 20))
        gain = coefficients['response'].mean((-1, -2)).clamp(0, 2)*trust
        return logits.reshape(b, t, -1), physical, dict(hidden=hidden, query=query, weight=read_weight,
                            scale=read_scale, gain=gain, effective=effective, innovation=innovation)

    def commit_block(self, tokens, state, coefficients, reads, logits, physical):
        """Ordered bookkeeping remains serial and is reported as such in diagnostics."""
        b, t = tokens.shape; d = self.cfg.matrix_size
        view = lambda tensor: tensor.reshape(b, t, *tensor.shape[1:])
        cv = {name: view(coefficients[name]) for name in ('local_key', 'local_value', 'arrived', 'control', 'response', 'phase')}
        rv = {name: view(value) for name, value in reads.items()}
        local_updates, clips = [], []
        tau = self.raw_trace_tau.clamp(math.log(1), math.log(2048)).exp()
        decay = torch.exp(-1/tau)
        for position in range(t):
            previous_slow = state['slow']
            if position and self.runtime_automatic_feedback:
                state, diag = self._apply_feedback(state, tokens[:, position])
                clips.append(diag['capacity_clip_fraction'])
            local_updates.append((state['slow']-previous_slow).square().mean((-1, -2, -3)).sqrt())
            key, value, arrived = cv['local_key'][:, position], cv['local_value'][:, position], cv['arrived'][:, position]
            control = cv['control'][:, position]
            prior_m, prior_p = self.stationary_prior(state)
            innovation = (value-(prior_m@key[..., None]).squeeze(-1)).square().mean(-1)
            reliability = control[..., 0].sigmoid()*arrived
            variance = .02+.3*control[..., 4].sigmoid()
            m, p = joseph(prior_m, prior_p, key, value, variance, reliability,
                          self.cfg.covariance_process*arrived)
            m = torch.where(arrived[..., None, None], bounded(m), state['m'])
            p = torch.where(arrived[..., None, None], p, state['p'])
            trace = (state['trace']*decay[None, None, :, None]
                     +(1-decay)[None, None, :, None]*key[..., None, :]*arrived[..., None, None])
            state = state | dict(m=m, p=p, trace=trace, response=cv['response'][:, position],
                      support=(.999*state['support']+key.square()*arrived[..., None]).clamp_max(100),
                      innovation=.9*state['innovation']+.1*innovation*arrived,
                      evidence_count=(state['evidence_count']+arrived.long()).clamp_max(2**31-1))
            state = self.cache_feedback(state, rv['hidden'][:, position], logits[:, position],
                         rv['query'][:, position], rv['weight'][:, position],
                         rv['scale'][:, position], rv['gain'][:, position])
        state = state | dict(x=physical[:, -1], z=cv['phase'][:, -1])
        zero = logits.new_zeros(())
        diagonal = state['p'].diagonal(dim1=-2, dim2=-1)
        diagnostics = dict(local_update_rms=torch.stack(local_updates, 1).mean(),
                           capacity_clip_fraction=torch.stack(clips, 1).mean() if clips else zero,
                           reached=coefficients['arrived'].to(logits.dtype).mean(),
                           effective_read_nodes=reads['effective'].mean(),
                           state_rms=state['x'].square().mean().sqrt(),
                           slow_rms=state['slow'].square().mean().sqrt(),
                           covariance_min_diagonal=diagonal.amin(), covariance_max_diagonal=diagonal.amax(),
                           phase_norm_min=state['z'].norm(dim=-1).amin(), phase_norm_max=state['z'].norm(dim=-1).amax(),
                           attention_mass=coefficients['attention'].sum(-1).mean(),
                           serial_feedback_steps=logits.new_tensor(t),
                           spatial_rounds=logits.new_tensor(self.cfg.internal_steps))
        return state, diagnostics

    def block(self, tokens, state):
        if self.runtime_automatic_feedback:
            state, _ = self._apply_feedback(state, tokens[:, 0])
        with record_function('v08.coefficients'):
            coefficients = self.block_coefficients(tokens, state)
        recurrence = {'sequential': sequential_delta, 'parallel': parallel_delta,
                      'triangular': triangular_delta}[self.cfg.implementation]
        with record_function('v08.fast_recurrence'):
            states, _ = recurrence(state['x'], coefficients['key'], coefficients['value'],
                                   coefficients['alpha'], coefficients['beta'])
        with record_function('v08.batched_read'):
            logits, physical, reads = self.read_block(states, coefficients, state, *tokens.shape)
        with record_function('v08.serial_slow_commit'):
            state, diag = self.commit_block(tokens, state, coefficients, reads, logits, physical)
        return logits, state, diag

    def step(self, token, state):
        raise NotImplementedError('use forward with explicit block boundaries')

    def forward(self, tokens, state=None, reset_mask=None, diagnostic_mask=None):
        if tokens.ndim != 2 or not tokens.shape[0] or not tokens.shape[1]:
            raise ValueError('tokens must be nonempty [batch,time]')
        if tokens.dtype != torch.long:
            raise ValueError('tokens must have dtype torch.long')
        if diagnostic_mask is not None:
            if diagnostic_mask.dtype != torch.bool or diagnostic_mask.shape != tokens.shape:
                raise ValueError('diagnostic_mask must be boolean [batch,time]')
            if not bool(diagnostic_mask.all()):
                raise ValueError('v0.8 block diagnostics currently require an all-valid mask')
        state = self.initial_state(len(tokens)) if state is None else state
        if reset_mask is not None:
            state = reset_lanes(state, self.initial_state(len(tokens)), reset_mask)
        outputs, diagnostics, lengths = [], [], []
        for start in range(0, tokens.shape[1], self.cfg.block_size):
            block = tokens[:, start:start+self.cfg.block_size]
            logits, state, diag = self.block(block, state)
            outputs.append(logits); diagnostics.append(diag); lengths.append(block.shape[1])
        result = {}
        for name in diagnostics[0]:
            values = torch.stack([row[name] for row in diagnostics])
            if name.endswith('_min') or name == 'covariance_min_diagonal':
                result[name] = values.amin()
            elif name.endswith('_max') or name == 'covariance_max_diagonal':
                result[name] = values.amax()
            elif name in ('serial_feedback_steps', 'spatial_rounds'):
                result[name] = values.sum()
            else:
                result[name] = (values*values.new_tensor(lengths)).sum()/sum(lengths)
        return torch.cat(outputs, 1), state, result


@torch.no_grad()
def initialize(model, seed):
    """Reuse shared v0.7 initialization and preserve R1's orthogonal entrance."""
    previous_initialize(model, seed)
    if hasattr(model, 'content_projection'):
        # Initialize on CPU so a fixed seed gives the same entrance on MPS/CUDA.
        generator = torch.Generator().manual_seed(seed+800001)
        matrix = torch.empty(model.content_projection.weight.shape, dtype=torch.float64)
        nn.init.orthogonal_(matrix, generator=generator)
        model.content_projection.weight.copy_(matrix)
