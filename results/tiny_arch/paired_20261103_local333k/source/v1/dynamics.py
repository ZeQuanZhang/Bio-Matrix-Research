"""Finite segment-level Bio field used inside each Transformer memory branch.

The token-level delta memory lives in model.py. This field reads only the last
committed segment and writes once at a caller-defined segment boundary. Soft
overlapping groups change actual messages. Optional sparse execution gathers a
legal induced subgraph before node computation; the default executes all nodes.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from v06.model import bounded, joseph
from v07.graph import solid_ball


def _select(mask, new, old):
    return torch.where(mask.reshape(mask.shape+(1,)*(new.ndim-1)), new, old)


def _rotate(value, angle):
    """Real paired rotation, broadcasting an angle over the vector pairs."""
    pairs = value.reshape(*value.shape[:-1], -1, 2)
    c, s = angle.cos()[..., None], angle.sin()[..., None]
    a, b = pairs.unbind(-1)
    return torch.stack((c*a-s*b, s*a+c*b), -1).flatten(-2)


class BioField(nn.Module):
    credit_horizon = 8

    def __init__(self, width, heads, dim, bands=3, nodes=32, internal_steps=2,
                 phase_enabled=True, probability_enabled=True, dynamic_enabled=True,
                 local_feedback_enabled=True, coupling_enabled=True,
                 threshold_enabled=True, active_fraction=1.0):
        super().__init__()
        if min(width, heads, bands, internal_steps) < 1 or dim < 2 or dim % 2 or nodes < 8:
            raise ValueError('Need positive dimensions, even dim >= 2, and at least 8 nodes')
        if not math.isfinite(active_fraction) or not 0 < active_fraction <= 1:
            raise ValueError('active_fraction must be in (0,1]')
        self.width, self.heads, self.dim = width, heads, dim
        self.bands, self.nodes, self.internal_steps = bands, nodes, internal_steps
        self.phase_enabled, self.probability_enabled = phase_enabled, probability_enabled
        self.dynamic_enabled, self.local_feedback_enabled = dynamic_enabled, local_feedback_enabled
        self.coupling_enabled, self.threshold_enabled = coupling_enabled, threshold_enabled
        self.active_fraction = active_fraction
        self.automatic_feedback = True
        for name, value in solid_ball(nodes, min(8, nodes-1)).items():
            self.register_buffer(name, value)
        # One complete legal center-to-surface chain is always included in a route.
        target = int(self.read_ports.argmax())
        required = [target]
        while int(self.shell[target]) > 0:
            candidates = self.neighbors[target][self.edge_mask[target].bool()]
            parent = candidates[self.shell[candidates] == self.shell[target]-1]
            if not len(parent):
                raise ValueError('Ball graph lacks a radial parent')
            target = int(parent[0]); required.append(target)
        self.register_buffer('required_path', torch.tensor(sorted(set(required)), dtype=torch.long))
        self.node_shape = nn.Parameter(torch.randn(nodes, heads, dim, dim)*.02)
        self.input_diagonal = nn.Parameter(torch.zeros(heads, dim))
        self.input_rank = nn.Parameter(torch.randn(heads, dim)*.05)
        self.output_diagonal = nn.Parameter(torch.zeros(heads, dim))
        self.output_rank = nn.Parameter(torch.randn(heads, dim)*.05)
        self.input_projection = nn.Linear(width, heads*dim, bias=False)
        self.query_projection = nn.Linear(width, heads*dim, bias=False)
        self.node_address = nn.Parameter(torch.randn(nodes, heads, dim)/math.sqrt(dim))
        self.group_keys = nn.Parameter(torch.randn(4, heads, dim)/math.sqrt(dim))
        self.raw_tau = nn.Parameter(torch.linspace(math.log(8), math.log(512), bands))
        self.raw_trace_tau = nn.Parameter(torch.tensor([math.log(4.), math.log(64.)]))
        self.raw_frequency = nn.Parameter(torch.linspace(-2., 0., bands)[None].expand(nodes, -1).clone())
        self.raw_amplitude = nn.Parameter(torch.zeros(nodes, bands))
        self.raw_coupling = nn.Parameter(torch.tensor(-1.5))
        self.fourier = nn.Parameter(torch.tensor([.2, .1, .1]))
        self.raw_threshold = nn.Parameter(torch.tensor(-3.))
        self.raw_reliability = nn.Parameter(torch.zeros(heads))
        self.raw_variance = nn.Parameter(torch.full((heads,), -2.))
        self.raw_local_rate = nn.Parameter(torch.tensor(-2.))
        self.raw_slow_decay = nn.Parameter(torch.tensor(-5.))
        self.band_logits = nn.Parameter(torch.zeros(bands))
        self.register_buffer('consolidation_version', torch.zeros((), dtype=torch.long))
        angles = torch.arange(nodes)[:, None]*2.399963 + torch.arange(bands)[None]*.7
        self.register_buffer('initial_phase', torch.stack((angles.cos(), angles.sin()), -1))

    def initial_state(self, batch, device=None, dtype=None):
        device = self.node_shape.device if device is None else device
        dtype = self.node_shape.dtype if dtype is None else dtype
        n, h, d, s, c = self.nodes, self.heads, self.dim, self.bands, self.credit_horizon
        zeros = lambda *shape: torch.zeros(*shape, device=device, dtype=dtype)
        anchor = bounded(self.node_shape.to(device=device, dtype=dtype))
        return dict(m=anchor[None].expand(batch, -1, -1, -1, -1).clone(),
                    p=torch.eye(d, device=device, dtype=dtype).expand(batch, n, h, d, d).clone(),
                    fast=zeros(batch, n, s, h, d, d), slow=zeros(batch, n, h, d, d),
                    trace=zeros(batch, n, 2, h, d), support=zeros(batch, n, h),
                    innovation=zeros(batch, n, h),
                    phase=self.initial_phase.to(device=device, dtype=dtype)[None].expand(batch, -1, -1, -1).clone(),
                    credit_key=zeros(batch, c, n, h, d), credit_value=zeros(batch, c, n, h, d),
                    credit_prediction=zeros(batch, c, n, h, d), credit_weight=zeros(batch, c, n, h),
                    credit_valid=torch.zeros(batch, c, device=device, dtype=torch.bool),
                    elapsed=torch.zeros(batch, device=device, dtype=torch.long),
                    version=self.consolidation_version.to(device).expand(batch).clone())

    def empty_summary(self, batch, device=None, dtype=None):
        device = self.node_shape.device if device is None else device
        dtype = self.node_shape.dtype if dtype is None else dtype
        return {name: torch.zeros(batch, self.heads, self.dim, device=device, dtype=dtype)
                for name in ('key', 'value', 'query', 'hidden')} | {
                    'count': torch.zeros(batch, device=device, dtype=dtype)}

    def accumulate(self, summary, controls, hidden=None, mask=None):
        key, value, query = (controls[name] for name in ('key', 'value', 'query'))
        if key.ndim != 5 or key.shape != value.shape or key.shape != query.shape:
            raise ValueError('Field controls must share [batch,time,bands,heads,dim]')
        b, t, s, h, d = key.shape
        if (s, h, d) != (self.bands, self.heads, self.dim):
            raise ValueError('Field control dimensions differ from the module')
        weight = controls.get('band_weights')
        if weight is None:
            weight = key.new_full((b, t, s, h), 1/s)
        weight = weight/weight.sum(2, keepdim=True).clamp_min(1e-6)
        values = {name: (tensor*weight[..., None]).sum(2)
                  for name, tensor in (('key', key), ('value', value), ('query', query))}
        values['hidden'] = (self.input_projection(hidden).reshape(b, t, h, d).tanh()
                            if hidden is not None else torch.zeros_like(values['value']))
        valid = torch.ones(b, t, device=key.device, dtype=torch.bool) if mask is None else mask.bool()
        if valid.shape != (b, t):
            raise ValueError('Summary mask must have [batch,time] shape')
        result = dict(summary)
        # Fixed token order makes arbitrary caller splits preserve the same sums.
        for at in range(t):
            for name in values:
                result[name] = result[name]+values[name][:, at]*valid[:, at, None, None]
            result['count'] = result['count']+valid[:, at].to(key.dtype)
        return result

    def _mapped(self, value, state, output=False):
        if not self.dynamic_enabled:
            return value
        diagonal = self.output_diagonal if output else self.input_diagonal
        rank = self.output_rank if output else self.input_rank
        context = state['m']+.25*state['slow']
        diagonal = .15*torch.tanh(diagonal+context.diagonal(dim1=-2, dim2=-1))
        direction = F.normalize(context.mean(-2)+rank, dim=-1, eps=1e-4)
        while diagonal.ndim < value.ndim:
            diagonal, direction = diagonal.unsqueeze(1), direction.unsqueeze(1)
        return value*(1+diagonal)+.05*direction*(value*direction).sum(-1, keepdim=True)

    def _groups(self, state):
        address = self.node_address[None]+.2*state['m'].mean(-2)
        score = torch.einsum('bnhd,ghd->bng', address, self.group_keys)/math.sqrt(self.heads*self.dim)
        if self.phase_enabled:
            score = score+.2*state['phase'][..., 0].mean(-1)[..., None]*torch.arange(
                1, 5, device=score.device, dtype=score.dtype)[None, None]/4
        return score.softmax(-1)

    def route(self, state, lane=0):
        """Historical top-k plus a legal mandatory path and one rotating explorer."""
        if self.active_fraction == 1:
            return torch.arange(self.nodes, device=state['m'].device)
        with torch.no_grad():
            score = (state['m'][lane].square().mean((-1, -2, -3))
                     +state['slow'][lane].square().mean((-1, -2, -3))
                     +.02*state['support'][lane].mean(-1))
            if self.phase_enabled:
                score = score+.01*state['phase'][lane, ..., 0].mean(-1)
            chosen = set(self.required_path.cpu().tolist())
            elapsed = int(state['elapsed'][lane])
            offset = (elapsed ^ (elapsed >> 5) ^ (elapsed >> 10)) % self.nodes
            explorer = next((at % self.nodes for at in range(offset, offset+self.nodes)
                             if at % self.nodes not in chosen), None)
            if explorer is not None:
                chosen.add(explorer)
            budget = max(len(chosen), math.ceil(self.nodes*self.active_fraction))
            # CPU scalar routing is included in execution cost; no speed claim is made.
            scores = score.cpu().tolist()
            for at in sorted(range(self.nodes), key=lambda i: (-scores[i], i)):
                if len(chosen) >= budget:
                    break
                chosen.add(at)
            return torch.tensor(sorted(chosen), device=score.device, dtype=torch.long)

    def induced_view(self, indices):
        """Short-lived parameter-preserving view; expensive node work uses only K nodes."""
        view = copy.copy(self)
        view._parameters, view._buffers, view._modules = dict(self._parameters), dict(self._buffers), dict(self._modules)
        view.nodes, view.active_fraction = len(indices), 1.
        for name in ('node_shape', 'node_address', 'raw_frequency', 'raw_amplitude'):
            view._parameters[name] = self._parameters[name].index_select(0, indices)
        for name in ('positions', 'shell', 'write_ports', 'read_ports', 'initial_phase'):
            view._buffers[name] = self._buffers[name].index_select(0, indices)
        view._buffers['read_ports'] = view.read_ports/view.read_ports.sum().clamp_min(1e-6)
        inverse = indices.new_full((self.nodes,), -1)
        inverse[indices] = torch.arange(len(indices), device=indices.device)
        neighbors = inverse[self.neighbors[indices]]
        view._buffers['edge_mask'] = self.edge_mask[indices]*(neighbors >= 0)
        view._buffers['neighbors'] = neighbors.clamp_min(0)
        view._buffers['outward'] = self.outward[indices]
        view._buffers['required_path'] = inverse[self.required_path]
        return view

    @staticmethod
    def _node_axis(name):
        if name in ('credit_key', 'credit_value', 'credit_prediction', 'credit_weight'):
            return 2
        if name in ('m', 'p', 'fast', 'slow', 'trace', 'support', 'innovation', 'phase'):
            return 1
        return None

    def _gather(self, state, lane, indices):
        return {name: (value[lane:lane+1].index_select(self._node_axis(name), indices)
                       if self._node_axis(name) is not None else value[lane:lane+1])
                for name, value in state.items()}

    def _sparse_read(self, hidden, query, state):
        outputs, reports = [], []
        for lane in range(len(hidden)):
            indices = self.route(state, lane)
            view = self.induced_view(indices)
            result, report = view.read(hidden[lane:lane+1], query[lane:lane+1],
                                       self._gather(state, lane, indices))
            outputs.append(result)
            reports.append(report | {'active_fraction': result.new_tensor(len(indices)/self.nodes),
                                     'executed_nodes': result.new_tensor(len(indices))})
        return torch.cat(outputs), {name: torch.stack([row[name] for row in reports]).mean()
                                    for name in reports[0]}

    def _sparse_commit(self, summary, state, commit_mask):
        rows = []
        for lane in range(len(summary['count'])):
            old = {name: value[lane:lane+1] for name, value in state.items()}
            enabled = bool(summary['count'][lane] > 0) and (commit_mask is None or bool(commit_mask[lane]))
            if not enabled:
                rows.append(old); continue
            indices = self.route(state, lane)
            view = self.induced_view(indices)
            local = view.commit({name: value[lane:lane+1] for name, value in summary.items()},
                                self._gather(state, lane, indices))
            row = {}
            for name, before in old.items():
                axis = self._node_axis(name)
                if axis is None:
                    row[name] = local[name]
                else:
                    # All nodes share the real segment clock, including inactive credit slots.
                    baseline = (torch.cat((torch.zeros_like(before[:, :1]), before[:, :-1]), 1)
                                if axis == 2 else before)
                    row[name] = baseline.index_copy(axis, indices, local[name])
            rows.append(row)
        return {name: torch.cat([row[name] for row in rows]) for name in state}

    def read(self, hidden, query, state):
        if hidden.shape[:2] != query.shape[:2] or query.shape[-2:] != (self.heads, self.dim):
            raise ValueError('Field read requires hidden[B,T,W], query[B,T,H,d]')
        if self.active_fraction < 1:
            return self._sparse_read(hidden, query, state)
        q = query+.05*self.query_projection(hidden).reshape(*query.shape)
        q = q[:, :, None].expand(-1, -1, self.nodes, -1, -1)
        q = self._mapped(q, state)+.05*state['trace'].mean(2)[:, None]
        q = F.normalize(q, dim=-1, eps=1e-4)
        if self.phase_enabled:
            angle = torch.atan2(state['phase'][..., 1], state['phase'][..., 0]).mean(-1)
            q = _rotate(q, angle[:, None, :, None])
        fast = (state['fast']*self.band_logits.softmax(-1)[None, None, :, None, None, None]).sum(2)
        matrix = state['m']+state['slow']+fast
        value = torch.einsum('bnhde,btnhe->btnhd', matrix, q)
        value = self._mapped(value, state, output=True)
        address = F.normalize(self.node_address[None]+state['m'].mean(-2), dim=-1, eps=1e-4)
        score = (q*address[:, None]).sum(-1)/math.sqrt(self.dim)
        if self.probability_enabled:
            variance = torch.einsum('btnhd,bnhde,btnhe->btnh', q, state['p'], q).clamp_min(0)
            score = score-.25*torch.log1p(variance)-.1*state['innovation'][:, None]
        groups = self._groups(state)
        choices = torch.einsum('bthd,ghd->btg', query, self.group_keys).softmax(-1)
        membership = torch.einsum('btg,bng->btn', choices, groups).clamp_min(1e-5)
        # Interior memories remain accessible; surface ports provide a radial prior.
        ports = .2/self.nodes+.8*self.read_ports
        score = score+ports.log()[None, None, :, None]+membership.log()[..., None]
        weight = score.softmax(2)
        result = (weight[..., None]*value).sum(2)
        effective = weight.square().sum(2).reciprocal().mean()
        return result, dict(effective_nodes=effective, active_fraction=query.new_tensor(1.),
                            slow_rms=state['slow'].square().mean().sqrt(),
                            innovation=state['innovation'].mean())

    def _flow(self, packet, mass, state, phase):
        source, valid = self.neighbors, self.edge_mask
        neighbor = packet[:, source]
        energy = torch.linalg.vector_norm(neighbor, dim=-1)/math.sqrt(self.dim)
        threshold = .001+.02*self.raw_threshold.sigmoid()
        if self.threshold_enabled:
            soft = torch.sigmoid((energy-threshold)/.02)
            hard = (energy >= threshold).to(packet.dtype)
            gate = hard.detach()-soft.detach()+soft if self.training else hard
            # Keep absolute pulse strength after the threshold decision.
            gate = gate*energy/(energy+threshold)
            gate = torch.where(self.outward[None, :, :, None], gate, torch.ones_like(gate))
        else:
            gate = torch.ones_like(energy)
        groups = self._groups(state)
        affinity = (groups[:, :, None]*groups[:, source]).sum(-1)
        raw = valid[None, :, :, None]*gate*(.5+affinity[..., None])
        if self.phase_enabled:
            receiver, sender = phase[:, :, None], phase[:, source]
            cosine = (receiver*sender).sum(-1).mean(-1)
            sine = (receiver[..., 0]*sender[..., 1]-receiver[..., 1]*sender[..., 0]).mean(-1)
            harmonic = (self.fourier[0]*cosine+self.fourier[1]*sine+
                        self.fourier[2]*(2*cosine.square()-1)).tanh()
            raw = raw*(1+.4*harmonic[..., None])
        # The denominator excludes confidence/gates, retaining absolute attenuation.
        denominator = valid.sum(-1).clamp_min(1)[None, :, None, None]
        message = (raw[..., None]*neighbor).sum(2)/denominator
        arrived = (raw*mass[:, source, None]).sum(2)/valid.sum(-1).clamp_min(1)[None, :, None]
        return message, arrived, raw

    def commit(self, summary, state, commit_mask=None):
        if self.active_fraction < 1:
            return self._sparse_commit(summary, state, commit_mask)
        count = summary['count']
        mask = count > 0
        if commit_mask is not None:
            mask = mask & commit_mask.bool()
        denominator = count.clamp_min(1)[:, None, None]
        key = F.normalize(summary['key']/denominator, dim=-1, eps=1e-4)
        value = (summary['value']/denominator+.1*summary['hidden']/denominator).tanh()
        query = F.normalize(summary['query']/denominator, dim=-1, eps=1e-4)
        b, n, h, d = len(count), self.nodes, self.heads, self.dim
        node_key = self._mapped(key[:, None].expand(-1, n, -1, -1), state)
        node_value = self._mapped(value[:, None].expand(-1, n, -1, -1), state)
        node_key = F.normalize(node_key, dim=-1, eps=1e-4)
        # A newly observed latent result corrects only earlier, already-issued predictions.
        feedback_state = (self.apply_feedback(state, node_value, feedback_mask=mask)
                          if self.automatic_feedback else state)
        slow = feedback_state['slow']
        # Forgetting is explicit and separate from query-directed correction.
        slow = slow*torch.exp(-.001*self.raw_slow_decay.sigmoid()*count[:, None, None, None, None])
        packet = node_value*self.write_ports[None, :, None, None]
        mass = self.write_ports[None].expand(b, -1)
        phase = state['phase']
        carrier = state['fast'].mean(2)
        edges = None
        for _ in range(self.internal_steps):
            signal = packet+.1*(carrier@node_key[..., None]).squeeze(-1)
            message, arrival, edges = self._flow(signal, mass, state, phase)
            packet = .5*packet+.5*message
            mass = (mass+arrival.mean(-1)).clamp(0, 1)
            if self.phase_enabled:
                frequency = .01+.2*self.raw_frequency.sigmoid()
                uncertainty = state['p'].diagonal(dim1=-2, dim2=-1).mean((-1, -2))
                confidence = 1/(1+uncertainty)
                activity = packet.square().mean((-1, -2))
                modulation = 1+.1*torch.tanh(activity+confidence-state['innovation'].mean(-1))
                angle = frequency[None]*modulation[..., None]*count[:, None, None]/self.internal_steps
                if self.coupling_enabled:
                    source_phase = phase[:, self.neighbors]
                    diff = (phase[:, :, None, :, 0]*source_phase[..., 1]
                            -phase[:, :, None, :, 1]*source_phase[..., 0])
                    coupling = (diff*self.edge_mask[None, :, :, None]).sum(2)/self.edge_mask.sum(-1).clamp_min(1)[None, :, None]
                    angle = angle+.2*self.raw_coupling.sigmoid()*coupling
                cosine, sine = angle.cos(), angle.sin()
                phase = torch.stack((phase[..., 0]*cosine-phase[..., 1]*sine,
                                     phase[..., 1]*cosine+phase[..., 0]*sine), -1)
                phase = F.normalize(phase, dim=-1, eps=1e-5)
        reached = mass[..., None].expand(-1, -1, h)
        observed_value = packet/(mass[..., None, None].clamp_min(.05))
        prior_decay = torch.exp(-count/4096.)[:, None, None, None, None]
        anchor = bounded(self.node_shape)[None]
        prior_m = prior_decay.sqrt()*state['m']+(1-prior_decay.sqrt())*anchor
        eye = torch.eye(d, device=key.device, dtype=key.dtype)
        prior_p = prior_decay*state['p']+(1-prior_decay)*eye
        reliability = reached*self.raw_reliability.sigmoid()[None, None]
        variance = .02+.3*self.raw_variance.sigmoid()[None, None]
        if self.probability_enabled:
            new_m, new_p = joseph(prior_m, prior_p, node_key, observed_value, variance, reliability, .002*reached)
        else:
            error_m = observed_value-(prior_m@node_key[..., None]).squeeze(-1)
            new_m = prior_m+.1*reliability[..., None, None]*error_m[..., :, None]*node_key[..., None, :]
            new_p = state['p']
        update_m = reached > 1e-6
        new_m = torch.where(update_m[..., None, None], bounded(new_m), state['m'])
        new_p = torch.where(update_m[..., None, None], new_p, state['p'])
        tau = self.raw_tau.clamp(math.log(1), math.log(65536)).exp()
        decay = torch.exp(-count[:, None]/tau[None])[:, None, :, None, None, None]
        outer = observed_value[..., :, None]*node_key[..., None, :]
        fast = decay*state['fast']+(1-decay)*outer[:, :, None]
        if self.phase_enabled:
            old_phase = state['phase']
            cross = old_phase[..., 0]*phase[..., 1]-old_phase[..., 1]*phase[..., 0]
            dot = (old_phase*phase).sum(-1)
            angle = torch.atan2(cross, dot)[:, :, :, None, None]
            fast = _rotate(fast, angle.expand(-1, -1, -1, h, d))
            confidence = 1/(1+state['p'].diagonal(dim1=-2, dim2=-1).mean((-1, -2)))
            modulation = .2*torch.tanh(packet.square().mean((-1, -2))+confidence-state['innovation'].mean(-1))
            amplitude = .5+(self.raw_amplitude[None]+modulation[..., None]).sigmoid()[..., None, None, None]
            fast = fast*amplitude
        fast = bounded(fast)
        trace_decay = torch.exp(-count[:, None]/self.raw_trace_tau.exp()[None])[:, None, :, None, None]
        trace = trace_decay*state['trace']+(1-trace_decay)*node_key[:, :, None]*reached[:, :, None, :, None]
        innovation = (observed_value-(prior_m@node_key[..., None]).squeeze(-1)).square().mean(-1)
        support = (.999*state['support']+reached/(1+innovation)).clamp_max(1000)
        # Predict the next segment in the same latent coordinate, using only current state.
        credit_query = F.normalize(node_key+.25*trace.mean(2), dim=-1, eps=1e-4)
        prediction = ((new_m+slow+.1*fast.mean(2))@credit_query[..., None]).squeeze(-1)
        credit_weight = reached*support/(support+2)
        prepend = lambda current, old: torch.cat((current[:, None], old[:, :-1]), 1)
        new = dict(m=new_m, p=new_p, fast=fast, slow=slow, trace=trace, support=support,
                   innovation=.9*state['innovation']+.1*innovation*reached, phase=phase,
                   credit_key=prepend(credit_query, state['credit_key']),
                   credit_value=prepend(node_value, state['credit_value']),
                   credit_prediction=prepend(prediction, state['credit_prediction']),
                   credit_weight=prepend(credit_weight, state['credit_weight']),
                   credit_valid=prepend(torch.ones_like(mask), feedback_state['credit_valid']),
                   elapsed=state['elapsed']+count.long(),
                   version=self.consolidation_version.expand(b).clone())
        return {name: _select(mask, value, state[name]) for name, value in new.items()}

    write_summary = commit

    def apply_feedback(self, state, observed, lag=1, feedback_mask=None, mask=None):
        """Consume one known-lag latent outcome once; not a semantic credit parser."""
        if not isinstance(lag, int) or not 1 <= lag <= self.credit_horizon:
            raise ValueError('Feedback lag outside the finite credit horizon')
        if observed.shape == (len(state['m']), self.heads, self.dim):
            observed = observed[:, None].expand(-1, self.nodes, -1, -1)
        if observed.shape != state['credit_prediction'][:, lag-1].shape:
            raise ValueError('Observed latent feedback must have [batch,nodes,heads,dim]')
        if feedback_mask is not None and mask is not None:
            raise ValueError('Provide only one feedback mask')
        feedback_mask = mask if mask is not None else feedback_mask
        if not self.local_feedback_enabled:
            return state
        at = lag-1
        valid = state['credit_valid'][:, at] & (state['version'] == self.consolidation_version)
        if feedback_mask is not None:
            valid = valid & feedback_mask.bool()
        error = observed-state['credit_prediction'][:, at]
        key = state['credit_key'][:, at]
        reliability = state['credit_weight'][:, at]/(1+error.square().mean(-1))
        update = (.05*self.raw_local_rate.sigmoid()*reliability[..., None, None]
                  *error[..., :, None]*key[..., None, :])
        slow = state['slow']
        old_projection = (slow@key[..., None]).squeeze(-1)
        orthogonal = slow-old_projection[..., :, None]*key[..., None, :]
        proposed = old_projection+(update@key[..., None]).squeeze(-1)
        remaining = (16-orthogonal.square().sum((-1, -2))).clamp_min(0)
        radius = torch.where(remaining > 0, remaining.clamp_min(1e-12).sqrt(), torch.zeros_like(remaining))
        factor = (radius/proposed.norm(dim=-1).clamp_min(1e-6)).clamp_max(1)
        result = orthogonal+(factor[..., None]*proposed)[..., :, None]*key[..., None, :]
        credit = state['credit_valid'].clone()
        credit[:, at] = credit[:, at] & ~valid
        return state | {'slow': _select(valid, result, slow), 'credit_valid': credit}

    def write(self, hidden, controls, state):
        summary = self.accumulate(self.empty_summary(hidden.shape[0], hidden.device, hidden.dtype), controls, hidden)
        return self.commit(summary, state)

    @torch.no_grad()
    def consolidation_proposal(self, state, *, allow_multilane=False, minimum_support=2.,
                               maximum_innovation=.25, rate=.01, max_update=.01):
        if (not all(math.isfinite(x) for x in (rate, max_update, minimum_support, maximum_innovation))
                or not 0 < rate <= 1 or max_update <= 0 or minimum_support < 0 or maximum_innovation < 0):
            raise ValueError('Invalid consolidation thresholds or rate')
        if len(state['m']) != 1 and not allow_multilane:
            raise ValueError('Multiple sessions require explicit allow_multilane=True aggregation')
        eligible = (state['support'] >= minimum_support) & (state['innovation'] <= maximum_innovation)
        weight = eligible*state['support']/(1+state['innovation'])
        target = ((state['m']+state['slow'])*weight[..., None, None]).sum(0)/weight.sum(0).clamp_min(1)[..., None, None]
        valid = weight.sum(0) > 0
        delta = bounded(rate*(target-self.node_shape), max_update)*valid[..., None, None]
        return dict(delta=delta.detach().clone(), version=self.consolidation_version.clone(),
                    eligible=valid, sessions=len(state['m']))

    @torch.no_grad()
    def apply_consolidation(self, proposal, optimizer=None):
        if int(proposal['version']) != int(self.consolidation_version):
            raise ValueError('Stale consolidation proposal')
        delta = proposal['delta']
        if delta.shape != self.node_shape.shape or not bool(torch.isfinite(delta).all()):
            raise ValueError('Invalid consolidation delta')
        changed = bool((delta != 0).any())
        if changed:
            self.node_shape.add_(delta)
            self.consolidation_version.add_(1)
            if optimizer is not None:
                optimizer.state.pop(self.node_shape, None)
        return dict(consolidation_version=int(self.consolidation_version),
                    update_norm=float(delta.norm()), eligible=int(proposal['eligible'].sum()),
                    sessions=proposal['sessions'])

    consolidate_proposal = consolidation_proposal

    def invalidate_credit(self, state):
        return state | {'credit_valid': torch.zeros_like(state['credit_valid']),
                        'version': self.consolidation_version.expand_as(state['version']).clone()}

    @torch.no_grad()
    def consolidate(self, state, optimizer=None, **kwargs):
        """Explicit offline transaction; callers must finish backward/step first."""
        self.apply_consolidation(self.consolidation_proposal(state, **kwargs), optimizer)
        return self.invalidate_credit(state)
