"""Full Bio network with compact group execution and causal block streaming.

Idle nodes freeze between observations. Old pending credit is consumed on the
full node space before selecting a new active set. Group summaries are bounded
historical matrices; selection never sees the current block's tokens. Static
partitions are shared by connected/cut graphs so global selection cannot carry
information across the cut. Different batch lanes choose independently.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.profiler import record_function

from v07.common import sync
from v07.model import bounded
from v08.recurrence import sequential_delta, parallel_delta, triangular_delta
from .core import Config as CoreConfig, Network as CoreNetwork, initialize as initialize_core
from .feedback import FeedbackMixin


NODE_AXES = {name: 1 for name in ('m', 'p', 'x', 'slow', 'trace', 'response',
             'support', 'innovation', 'z', 'read_query', 'read_weight',
             'feedback_gain', 'evidence_count')}
NODE_AXES.update({name: 2 for name in ('credit_query', 'credit_weight', 'credit_gain')})
FRAME_NODES = {'read_query', 'read_weight', 'feedback_gain'}
CREDIT_CURRENT = {'credit_query': 'read_query', 'credit_weight': 'read_weight',
                  'credit_gain': 'feedback_gain'}


@dataclass
class Config(CoreConfig):
    batch_feedback: bool = True
    active_fraction: float = 1.
    group_size: int = 16
    route_mode: str = 'summary'

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.batch_feedback, bool):
            raise ValueError('batch_feedback must be boolean')
        if not math.isfinite(self.active_fraction) or not 0 < self.active_fraction <= 1:
            raise ValueError('active_fraction must be in (0,1]')
        if not isinstance(self.group_size, int) or isinstance(self.group_size, bool) or self.group_size < 1:
            raise ValueError('group_size must be a positive integer')
        if self.route_mode not in ('summary', 'cyclic'):
            raise ValueError('route_mode must be summary or cyclic')


class Network(FeedbackMixin, CoreNetwork):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.active_fraction = cfg.active_fraction
        self._active_indices = None
        self._graph_view = None
        self.last_routes = []
        self.profile_enabled = False
        self.profile_records = []
        self._make_groups()
        reached = self.write_ports.bool().clone()
        for _ in range(cfg.nodes):
            expanded = reached | (reached[self.neighbors] & self.edge_mask.bool()).any(-1)
            if torch.equal(expanded, reached):
                break
            reached = expanded
        self.register_buffer('feedback_reachable', reached)
        address = max(1, cfg.address_size or cfg.matrix_size)
        self.route_key = nn.Linear(cfg.matrix_size**2, address, bias=False)
        self.route_query = nn.Linear(cfg.matrix_size**2, address, bias=False)

    def _make_groups(self):
        # The same static partition is used with/without the graph-cut intervention.
        boundary = self.cfg.shells//2
        partition = (self.shell >= boundary).long()
        groups, components = [], []
        for component in (0, 1):
            members = torch.where(partition == component)[0].tolist()
            for start in range(0, len(members), self.cfg.group_size):
                groups.append(tuple(members[start:start+self.cfg.group_size]))
                components.append(component)
        membership = torch.empty(self.cfg.nodes, dtype=torch.long)
        for group, nodes in enumerate(groups):
            membership[list(nodes)] = group
        self.groups = tuple(groups)
        self.components = tuple(tuple(i for i, c in enumerate(components) if c == part)
                                for part in (0, 1))
        self.register_buffer('node_group', membership)
        self.register_buffer('group_counts', torch.tensor([len(g) for g in groups], dtype=torch.float32))
        # Follow one actual radial-parent path to an actual read port. The path's
        # nodes stay mandatory under cut, but cut edges are never repaired.
        parents = []
        for target in range(self.cfg.nodes):
            choices = [int(source) for source in self.neighbors[target].tolist()
                       if int(self.shell[source]) == int(self.shell[target])-1]
            parents.append(choices[0] if choices else -1)
        read_nodes = torch.where(self.read_ports > 0)[0].tolist()
        path = []
        if read_nodes:
            node = read_nodes[0]
            seen = set()
            while node >= 0 and node not in seen:
                seen.add(node); path.append(node)
                if bool(self.write_ports[node]):
                    break
                node = parents[node]
        path.extend(torch.where(self.write_ports > 0)[0].tolist())
        self.mandatory_groups = frozenset(int(membership[i]) for i in path)

    def graph_buffer(self, name):
        return self._graph_view[name] if self._graph_view is not None and name in self._graph_view else super().graph_buffer(name)

    def node_anchor(self):
        return self.node_shape if self._active_indices is None else self.node_shape.index_select(0, self._active_indices)

    def phase_initializer(self):
        return self.initial_phase if self._active_indices is None else self.initial_phase.index_select(0, self._active_indices)

    def _summaries(self, state, indices=None):
        matrix = state['x'].mean(-3)+.3*state['m']
        if self.runtime_slow_enabled:
            matrix = matrix+.2*state['slow']
        matrix = bounded(matrix).flatten(-2).detach()
        groups = self.node_group if indices is None else self.node_group.index_select(0, indices)
        result = matrix.new_zeros(len(matrix), len(self.groups), matrix.shape[-1])
        result = result.scatter_add(1, groups[None, :, None].expand_as(matrix), matrix)
        return result/self.group_counts[None, :, None]

    def initial_state(self, batch):
        state = super().initial_state(batch)
        return state | dict(group_summary=self._summaries(state),
                            route_step=torch.zeros(batch, dtype=torch.long, device=state['m'].device))

    def reset_fast(self, state):
        fresh = super().reset_fast(state)
        return fresh | {'group_summary': self._summaries(fresh)}

    def clear_slow(self, state):
        cleared = super().clear_slow(state)
        return cleared | {'group_summary': self._summaries(cleared)}

    def local_proposal(self, state, embedding):
        proposal = super().local_proposal(state, embedding)
        # Observed-token supervision must respect the same graph-cut boundary.
        return proposal*self.graph_buffer('feedback_reachable')[None, :, None, None]

    def _apply_feedback(self, state, token, lag=1, feedback_mask=None):
        updated, diag = super()._apply_feedback(state, token, lag, feedback_mask)
        if self._active_indices is None and 'group_summary' in updated:
            # Pending credit may write a group that the next route leaves idle.
            updated = updated | {'group_summary': self._summaries(updated)}
        return updated, diag

    def training_progress(self, completed_tokens, total_tokens):
        """Soft gates first; postpone actual sparse execution until input warmup."""
        if not 0 <= completed_tokens <= total_tokens or total_tokens <= 0:
            raise ValueError('need 0 <= completed_tokens <= total_tokens and total_tokens > 0')
        progress = completed_tokens/total_tokens
        self.gate_blend = min(1., progress/.1)
        sparse_progress = min(1., max(0., (progress-.1)/.2))
        self.active_fraction = 1.-sparse_progress*(1.-self.cfg.active_fraction)
        return dict(gate_blend=self.gate_blend, active_fraction=self.active_fraction)

    @contextmanager
    def _compact_graph(self, indices):
        old_indices, old_view = self._active_indices, self._graph_view
        self._active_indices = indices
        source = self.neighbors.index_select(0, indices)
        lookup = torch.full((self.cfg.nodes,), -1, dtype=torch.long, device=indices.device)
        lookup[indices] = torch.arange(len(indices), device=indices.device)
        mapped = lookup[source]
        view = {name: getattr(self, name).index_select(0, indices)
                for name in ('positions', 'shell', 'write_ports', 'read_ports', 'outward', 'feedback_reachable')}
        view['neighbors'] = mapped.clamp_min(0)
        view['edge_mask'] = self.edge_mask.index_select(0, indices)*(mapped >= 0)
        self._graph_view = view
        try:
            yield
        finally:
            self._active_indices, self._graph_view = old_indices, old_view

    def _route(self, state):
        """Return per-lane IDs and differentiable gains from old summaries only."""
        b = len(state['m']); device = state['m'].device
        if self.active_fraction >= 1:
            indices = torch.arange(self.cfg.nodes, device=device)
            return [(tuple(range(len(self.groups))), indices, None) for _ in range(b)]
        summaries = state['group_summary'].detach()
        keys = F.normalize(self.route_key(summaries), dim=-1, eps=1e-5)
        scores = summaries.new_zeros(b, len(self.groups))
        for component in self.components:
            if component:
                group_ids = torch.tensor(component, device=device)
                context = summaries.index_select(1, group_ids).mean(1)
                query = F.normalize(self.route_query(context), dim=-1, eps=1e-5)
                scores[:, group_ids] = (keys[:, group_ids]*query[:, None]).sum(-1)
        detached_scores = scores.detach().cpu()
        clocks = state['route_step'].detach().cpu().tolist()
        plans = []
        for lane in range(b):
            chosen = set()
            for component in self.components:
                if not component:
                    continue
                required = self.mandatory_groups.intersection(component)
                budget = min(len(component), max(math.ceil(len(component)*self.active_fraction),
                             len(required)+(len(required) < len(component))))
                selected = set(required)
                # A deterministic exploration slot avoids permanent starvation.
                explore = [g for g in component if g not in required]
                if explore:
                    selected.add(explore[clocks[lane] % len(explore)])
                remaining = [g for g in component if g not in selected]
                if self.cfg.route_mode == 'summary':
                    remaining.sort(key=lambda g: (-float(detached_scores[lane, g]), g))
                else:
                    remaining.sort(key=lambda g: (g-clocks[lane]) % len(self.groups))
                selected.update(remaining[:max(0, budget-len(selected))])
                chosen.update(selected)
            groups = tuple(sorted(chosen))
            nodes = sorted(node for group in groups for node in self.groups[group])
            indices = torch.tensor(nodes, dtype=torch.long, device=device)
            gain = None
            if self.cfg.route_mode == 'summary':
                # Train selected addresses through the bounded dynamic correction;
                # the amplitude-preserving current-content carrier stays separate.
                gain = .5+scores[lane].sigmoid().index_select(0, self.node_group.index_select(0, indices))
            plans.append((groups, indices, gain))
        return plans

    @staticmethod
    def _gather(state, lanes, indices):
        result = {}
        for name, value in state.items():
            value = value.index_select(0, lanes)
            result[name] = value.index_select(NODE_AXES[name], indices) if name in NODE_AXES else value
        return result

    def _scatter(self, original, compact, indices, length):
        result = {}
        for name, value in compact.items():
            if name not in NODE_AXES:
                result[name] = value
                continue
            axis = NODE_AXES[name]
            if name in FRAME_NODES:
                base = torch.zeros_like(original[name])
            elif name in CREDIT_CURRENT:
                history = self.cfg.credit_horizon-1
                old = torch.cat((original[CREDIT_CURRENT[name]][:, None], original[name]), 1)
                new = original[name].new_zeros(len(old), max(0, length-1), *old.shape[2:])
                base = torch.cat((new, old), 1)[:, :history].detach()
            else:
                base = original[name]
            result[name] = base.index_copy(axis, indices, value)
        selected = torch.zeros(len(self.groups), dtype=torch.bool, device=indices.device)
        selected[self.node_group.index_select(0, indices)] = True
        summary = self._summaries(compact, indices)
        result['group_summary'] = torch.where(selected[None, :, None], summary, original['group_summary'])
        result['route_step'] = original['route_step']+1
        return result

    @contextmanager
    def _stage(self, name):
        device = self.embedding.weight.device
        if self.profile_enabled:
            sync(device); started = time.perf_counter()
        with record_function('v09.'+name):
            yield
        if self.profile_enabled:
            sync(device)
            self.profile_records.append(dict(stage=name, seconds=time.perf_counter()-started,
                                             active_nodes=self.node_anchor().shape[0]))

    def _execute(self, tokens, state, gain=None):
        with self._stage('coefficients'):
            coefficients = self.block_coefficients(tokens, state)
            if gain is not None:
                b, t = tokens.shape
                multiplier = gain[:, None].expand(-1, t, -1).flatten(0, 1)
                coefficients['transport'] = coefficients['transport']*multiplier[..., None, None]
        method = dict(sequential=sequential_delta, parallel=parallel_delta, triangular=triangular_delta)[self.cfg.implementation]
        with self._stage('fast_recurrence'):
            states, _ = method(state['x'], coefficients['key'], coefficients['value'], coefficients['alpha'], coefficients['beta'])
        with self._stage('batched_read'):
            logits, physical, reads = self.read_block(states, coefficients, state, *tokens.shape)
        with self._stage('serial_slow_commit'):
            state, diag = self.commit_block(tokens, state, coefficients, reads, logits, physical)
        if 'content_base_mass' in coefficients:
            ports = self.graph_buffer('read_ports')
            diag['content_base_surface_mass'] = (coefficients['content_base_mass']*ports[None]).sum(-1).mean()
        return logits, state, diag

    def block(self, tokens, state):
        with self._stage('pending_and_route'):
            if self.runtime_automatic_feedback:
                state, _ = self._apply_feedback(state, tokens[:, 0])
            plans = self._route(state)
        self.last_routes.append([dict(active_ids=ids.detach().clone(), groups=groups) for groups, ids, _ in plans])
        if all(len(ids) == self.cfg.nodes for _, ids, _ in plans):
            # Mandatory paths can fill a small graph; avoid gathering the whole
            # graph just to scatter it back, while retaining the router gain.
            gain = None if plans[0][2] is None else torch.stack([item[2] for item in plans])
            logits, end, diag = self._execute(tokens, state, gain)
            end = end | dict(group_summary=self._summaries(end), route_step=state['route_step']+1)
            diag.update(active_nodes=logits.new_tensor(self.cfg.nodes), active_fraction=logits.new_tensor(1.))
            return logits, end, diag
        # Lanes sharing a route retain GEMM batching. Divergent routes form
        # separate compact batches rather than coupling independent streams.
        batches = {}
        for lane, (groups, ids, gain) in enumerate(plans):
            batches.setdefault(groups, []).append(lane)
        outputs = [None]*len(tokens); states = [None]*len(tokens); diagnostics = []
        for groups, lane_list in batches.items():
            lanes = torch.tensor(lane_list, device=tokens.device)
            indices = plans[lane_list[0]][1]
            gain = None if plans[lane_list[0]][2] is None else torch.stack([plans[i][2] for i in lane_list])
            original = {name: value.index_select(0, lanes) for name, value in state.items()}
            compact = self._gather(state, lanes, indices)
            with self._compact_graph(indices):
                logits, compact, diag = self._execute(tokens.index_select(0, lanes), compact, gain)
            with self._stage('scatter_and_summary'):
                end = self._scatter(original, compact, indices, tokens.shape[1])
            for i, lane in enumerate(lane_list):
                outputs[lane] = logits[i:i+1]
                states[lane] = {name: value[i:i+1] for name, value in end.items()}
            diag.update(active_nodes=logits.new_tensor(len(indices)),
                        active_fraction=logits.new_tensor(len(indices)/self.cfg.nodes))
            diagnostics.append((len(lane_list), diag))
        merged = {name: torch.cat([s[name] for s in states], 0) for name in states[0]}
        diag = {}
        for name in diagnostics[0][1]:
            values = torch.stack([row[name] for _, row in diagnostics])
            if name.endswith('_min') or name == 'covariance_min_diagonal':
                diag[name] = values.amin()
            elif name.endswith('_max') or name == 'covariance_max_diagonal':
                diag[name] = values.amax()
            else:
                diag[name] = sum(weight*row[name] for weight, row in diagnostics)/len(tokens)
        return torch.cat(outputs, 0), merged, diag

    def forward(self, tokens, state=None, reset_mask=None, diagnostic_mask=None):
        self.last_routes = []
        return super().forward(tokens, state, reset_mask, diagnostic_mask)

    def initial_stream_state(self, batch):
        return dict(committed=self.initial_state(batch),
                    tokens=torch.empty(batch, 0, dtype=torch.long, device=self.embedding.weight.device),
                    parameter_versions=tuple(p._version for p in self.parameters()),
                    protocol=self._stream_protocol())

    def _stream_protocol(self):
        return dict(config=asdict(self.cfg), active_fraction=self.active_fraction, gate_blend=self.gate_blend,
                    local=self.runtime_local_enabled, slow=self.runtime_slow_enabled,
                    automatic_feedback=self.runtime_automatic_feedback,
                    bio_read=getattr(self, 'runtime_bio_read_enabled', True),
                    device=str(self.embedding.weight.device), dtype=str(self.embedding.weight.dtype))

    def _check_stream(self, state):
        if not isinstance(state, dict) or 'committed' not in state:
            raise ValueError('use initial_stream_state(batch) for token streaming')
        if tuple(p._version for p in self.parameters()) != state['parameter_versions']:
            raise RuntimeError('parameters changed during stream; start a new stream')
        if state['protocol'] != self._stream_protocol():
            raise RuntimeError('execution protocol changed during stream; start a new stream')

    @torch.no_grad()
    def step(self, token, state):
        """Exact block schedule via prefix replay; eval reference, not fast decode."""
        if self.training:
            raise RuntimeError('stream step requires eval(); training uses forward')
        self._check_stream(state)
        if token.dtype != torch.long or token.ndim != 1 or len(token) != len(state['tokens']):
            raise ValueError('stream token must be long [batch]')
        prefix = torch.cat((state['tokens'], token[:, None]), 1)
        logits, candidate, diag = self(prefix, state['committed'])
        complete = prefix.shape[1] == self.cfg.block_size
        new = state | dict(committed=candidate if complete else state['committed'],
                           tokens=prefix[:, :0] if complete else prefix, candidate=candidate)
        diag = diag | dict(replayed_tokens=logits.new_tensor(prefix.shape[1]))
        return logits[:, -1], new, diag

    @torch.no_grad()
    def flush_stream(self, state):
        """Explicit end-of-document boundary; commit a partial prefix once."""
        self._check_stream(state)
        if state['tokens'].shape[1]:
            return state | dict(committed=state['candidate'], tokens=state['tokens'][:, :0])
        return state


def initialize(model, seed):
    initialize_core(model, seed)
