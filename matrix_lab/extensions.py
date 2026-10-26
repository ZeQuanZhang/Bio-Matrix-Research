"""Explicit approximate Gaussian filtering and bounded oscillatory state extensions."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .models import Config, MemoryModel


class Oscillator(nn.Module):
    def __init__(self, mode, width, state):
        super().__init__()
        if state % 2:
            raise ValueError('Oscillatory state needs pairs of coordinates')
        frequencies = torch.logspace(-2, math.log10(math.pi / 2), state // 2)
        if mode == 'spectral_zero':
            frequencies.zero_()
        if mode in ('spectral_learned', 'full'):
            self.raw_frequency = nn.Parameter(torch.logit((frequencies / math.pi).clamp(.0001, .9999)))
        else:
            self.register_buffer('fixed_frequency', frequencies)
        self.raw_decay = nn.Parameter(torch.full((state // 2,), 2.))
        self.drive = nn.Linear(width, state)
        self.gate = nn.Linear(width, state // 2)

    def forward(self, state, x):
        omega = self.raw_frequency.sigmoid()*math.pi if hasattr(self, 'raw_frequency') else self.fixed_frequency
        rho = .9 + .0999*self.raw_decay.sigmoid()
        pairs = state.view(len(state), -1, 2)
        a, b = pairs.unbind(-1)
        rotated = torch.stack((omega.cos()*a - omega.sin()*b, omega.sin()*a + omega.cos()*b), -1)
        update = self.gate(x).sigmoid()[:, :, None]
        drive = self.drive(x).tanh().view_as(pairs)
        return ((1-update)*rho[None, :, None]*rotated + update*drive).flatten(1)


class ExtendedMemory(nn.Module):
    def __init__(self, mode, width=32, state=32):
        super().__init__()
        if mode not in ('bayes', 'spectral_zero', 'spectral_fixed', 'spectral_learned', 'full'):
            raise ValueError('Unknown extension')
        self.mode, self.r = mode, state
        self.bayesian = mode in ('bayes', 'full')
        self.spectral = mode.startswith('spectral') or mode == 'full'
        self.core = MemoryModel(Config(width=width, state=state, cell='reaction', readout='dynamic'))
        if self.bayesian:
            self.observation = nn.Linear(width, state)
            self.observation_noise = nn.Linear(width, state)
            self.process_noise = nn.Parameter(torch.full((state,), -7.))
            nn.init.zeros_(self.observation_noise.weight)
            nn.init.constant_(self.observation_noise.bias, -2.)
        if self.spectral:
            self.oscillator = Oscillator(mode, width, state)
            self.fusion = nn.Linear(state, state, bias=False)
            nn.init.normal_(self.fusion.weight, std=.05 / math.sqrt(state))

    @property
    def state_floats(self):
        return self.r * (1 + self.bayesian + self.spectral)

    def initial_state(self, batch, reference):
        state = reference.new_zeros(batch, self.state_floats)
        if self.bayesian:
            state[:, self.r:2*self.r] = 1.
        return state

    def jacobian(self, state, encoded):
        """Exact Jacobian of the reaction cell, for a diagonal EKF covariance approximation."""
        core, r = self.core, self.r
        feedback = core.state_state(state)
        p, q = core.pq(state).chunk(2, -1)
        z = encoded[:, :r] + feedback[:, :r] + core.reaction(p*q)
        alpha = (encoded[:, r:] + feedback[:, r:]).sigmoid()
        candidate = z.tanh()
        pm, qm = core.pq.weight.chunk(2, 0)
        dz = core.state_state.weight[:r][None] + core.reaction.weight[None] @ (q[:, :, None]*pm[None] + p[:, :, None]*qm[None])
        da = (alpha*(1-alpha))[:, :, None] * core.state_state.weight[r:][None]
        return (torch.diag_embed(1-alpha) + (candidate-state)[:, :, None]*da
                + (alpha*(1-candidate.square()))[:, :, None]*dz)

    def oscillator_step(self, state, x):
        return self.oscillator(state, x)

    def forward(self, events, queries=(), state=None, intervention='carry'):
        if intervention not in ('carry', 'reset', 'swap'):
            raise ValueError('Unknown intervention')
        core, r = self.core, self.r
        x = core.encoder(core.operation(events[..., 0]) + core.key(events[..., 1]) + core.value(events[..., 2]))
        if self.bayesian:
            # The prior sees operation/key controls, not the newly observed value.
            control_value = torch.where(events[..., 0] == 0, core.cfg.values, events[..., 2])
            control = core.encoder(core.operation(events[..., 0]) + core.key(events[..., 1])
                                   + core.value(control_value))
            encoded = core.input_state(control)
        else:
            encoded = core.input_state(x)
        if state is None:
            state = self.initial_state(events.shape[0], x)
        if state.shape != (events.shape[0], self.state_floats):
            raise ValueError('Wrong extended-state shape')
        answers = []
        for t in range(events.shape[1]):
            if t in queries:
                if intervention == 'reset':
                    state = self.initial_state(events.shape[0], x)
                elif intervention == 'swap':
                    state = state.roll(1, 0)
            mean = state[:, :r]
            predicted = core.step(encoded[:, t], mean)
            parts = []
            if self.bayesian:
                variance = state[:, r:2*r]
                jac = self.jacobian(mean, encoded[:, t])
                prior_variance = (jac.square()*variance[:, None, :]).sum(-1) + F.softplus(self.process_noise) + 1e-5
                prior_variance = prior_variance.clamp(1e-6, 100.)
                observation = self.observation(x[:, t]).tanh()
                observation_variance = F.softplus(self.observation_noise(x[:, t])) + 1e-4
                gain = prior_variance / (prior_variance + observation_variance)
                # Identity observation model; assimilate observed values only on write events.
                observed = (events[:, t, 0] == 0)[:, None]
                mean = torch.where(observed, predicted + gain*(observation-predicted), predicted)
                posterior_variance = torch.where(observed, (1-gain)*prior_variance, prior_variance)
                parts = [mean, posterior_variance]
            else:
                mean = predicted; parts = [mean]
            read_state = mean
            if self.spectral:
                old_oscillator = state[:, -r:]
                oscillator = self.oscillator_step(old_oscillator, x[:, t])
                parts.append(oscillator)
                read_state = mean + self.fusion(oscillator)
            state = torch.cat(parts, -1)
            if t in queries:
                answers.append(core.read(x[:, t], read_state))
        logits = torch.stack(answers, 1) if answers else x.new_empty(len(events), 0, core.cfg.values)
        return logits, state

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
