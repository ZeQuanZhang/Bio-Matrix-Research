"""Batch frame-local feedback work; retain ordered slow/probability updates.

The block's logits/read metadata are already fixed before commit. Their expected
embeddings and detached F1 directions can therefore be evaluated together. Slow
capacity projections and Joseph observations retain the v0.8 event order. Only
the final prediction remains pending when automatic local feedback is enabled.
"""
import math

import torch

from v07.model import bounded, joseph


class FeedbackMixin:
    """Place before v08.Network in the MRO; cfg.batch_feedback=False is the oracle."""

    def commit_block(self, tokens, state, coefficients, reads, logits, physical):
        if not getattr(self.cfg, 'batch_feedback', True):
            return super().commit_block(tokens, state, coefficients, reads, logits, physical)

        b, t = tokens.shape
        view = lambda value: value.reshape(b, t, *value.shape[1:])
        cv = {name: view(coefficients[name]) for name in
              ('local_key', 'local_value', 'arrived', 'control', 'response', 'phase')}
        rv = {name: view(value) for name, value in reads.items()}
        before = state
        with torch.no_grad():
            frames = dict(prediction=logits.softmax(-1)@self.embedding.weight,
                          feedback_h=rv['hidden'].detach(), read_query=rv['query'].detach(),
                          read_weight=rv['weight'].detach(), read_scale=rv['scale'].detach(),
                          feedback_gain=rv['gain'].detach(),
                          feedback_valid=torch.ones(b, t, dtype=torch.bool, device=tokens.device))

        local_enabled = (self.runtime_local_enabled and self.runtime_slow_enabled
                         and self.cfg.variant not in ('no_local', 'no_slow'))
        updates = None
        if t > 1 and self.runtime_automatic_feedback:
            frame = {name: value[:, :-1].flatten(0, 1) for name, value in frames.items()}
            # local_proposal reads slow only to size the disabled-rule zero result.
            frame['slow'] = state['slow'][:, None].expand(-1, t-1, *state['slow'].shape[1:]).flatten(0, 1)
            proposal = self.local_proposal(frame, self.embedding(tokens[:, 1:]).flatten(0, 1))
            updates = bounded(proposal, .05).reshape(b, t-1, *state['slow'].shape[1:])

        tau = self.raw_trace_tau.clamp(math.log(1), math.log(2048)).exp()
        decay = torch.exp(-1/tau)
        local_updates, clips = [], []
        for position in range(t):
            previous_slow = state['slow']
            if updates is not None and position:
                slow, clipped = self.project_slow_update(state['slow'], updates[:, position-1],
                                                         frames['read_query'][:, position-1])
                state = state | {'slow': slow}
                clips.append(clipped.float().mean(-1))
            local_updates.append((state['slow']-previous_slow).square().mean((-1, -2, -3)).sqrt())
            key = cv['local_key'][:, position]
            value = cv['local_value'][:, position]
            arrived = cv['arrived'][:, position]
            control = cv['control'][:, position]
            prior_m, prior_p = self.stationary_prior(state)
            innovation = (value-(prior_m@key[..., None]).squeeze(-1)).square().mean(-1)
            reliability = control[..., 0].sigmoid()*arrived
            variance = .02+.3*control[..., 4].sigmoid()
            m, p = joseph(prior_m, prior_p, key, value, variance, reliability,
                          self.cfg.covariance_process*arrived)
            state = state | dict(
                m=torch.where(arrived[..., None, None], bounded(m), state['m']),
                p=torch.where(arrived[..., None, None], p, state['p']),
                trace=(state['trace']*decay[None, None, :, None]
                       +(1-decay)[None, None, :, None]*key[..., None, :]*arrived[..., None, None]),
                response=cv['response'][:, position],
                support=(.999*state['support']+key.square()*arrived[..., None]).clamp_max(100),
                innovation=.9*state['innovation']+.1*innovation*arrived,
                evidence_count=(state['evidence_count']+arrived.long()).clamp_max(2**31-1))

        names = {'prediction': 'credit_prediction', 'feedback_h': 'credit_h',
                 'read_query': 'credit_query', 'read_weight': 'credit_weight',
                 'feedback_gain': 'credit_gain', 'read_scale': 'credit_scale',
                 'feedback_valid': 'credit_valid'}
        history = self.cfg.credit_horizon-1
        cached = {}
        for current, past in names.items():
            if history:
                previous_frames = frames[current][:, :-1].flip(1)
                if current == 'feedback_valid' and self.runtime_automatic_feedback and local_enabled:
                    previous_frames = torch.zeros_like(previous_frames)
                cached[past] = torch.cat((previous_frames, before[current][:, None], before[past]), 1)[:, :history].detach()
            else:
                cached[past] = before[past][:, :0].detach()
            cached[current] = frames[current][:, -1].detach()
        state = state | cached | dict(x=physical[:, -1], z=cv['phase'][:, -1])

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
