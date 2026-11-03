"""Bio-Matrix v1.1: Transformer geometry with trainable matrix-memory branches.

Each segment reads its incoming snapshot and commits writes at its boundary.
The segment clock is persistent, so Python call boundaries do not define memory
semantics. Runtime state is never detached inside the model.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from v07.model import AttentionBlock


@dataclass
class Config:
    vocab_size: int = 4096
    width: int = 64
    layers: int = 2
    heads: int = 4
    ff: int = 256
    context_limit: int = 128
    segment_tokens: int = 64
    memory_heads: int | None = None
    memory_dim: int | None = None
    bands: int = 3
    memory_enabled: bool = True
    phase_enabled: bool = True
    full_dynamics: bool = True
    nodes: int = 32
    internal_steps: int = 3
    probability_enabled: bool = True
    dynamic_enabled: bool = True
    local_feedback_enabled: bool = True
    coupling_enabled: bool = True
    threshold_enabled: bool = True
    active_fraction: float = 1.
    memory_init: str = 'backbone'
    field_write_mode: str = 'ordered'

    def __post_init__(self):
        if self.memory_heads is None:
            self.memory_heads = self.heads
        if self.memory_dim is None and isinstance(self.width, int) and isinstance(self.heads, int) and self.heads > 0:
            self.memory_dim = self.width//self.heads
        for name in ('vocab_size', 'width', 'layers', 'heads', 'ff', 'context_limit',
                     'segment_tokens', 'memory_heads', 'memory_dim', 'bands', 'nodes', 'internal_steps'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('memory_enabled', 'phase_enabled', 'full_dynamics', 'probability_enabled',
                     'dynamic_enabled', 'local_feedback_enabled', 'coupling_enabled', 'threshold_enabled'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'{name} must be boolean')
        if self.width % self.heads or (self.width//self.heads) % 2:
            raise ValueError('Transformer RoPE requires even per-head width')
        if self.memory_dim % 2:
            raise ValueError('Memory phase addressing requires an even memory_dim')
        if self.memory_init not in ('backbone', 'random'):
            raise ValueError('memory_init must be backbone or random')
        if self.field_write_mode not in ('ordered', 'summary'):
            raise ValueError('field_write_mode must be ordered or summary')
        if self.memory_enabled and (self.memory_heads != self.heads or self.memory_dim != self.width//self.heads):
            raise ValueError('v1.1 memory requires exact backbone head geometry; low-rank transfer is not implemented')
        if self.segment_tokens > self.context_limit:
            raise ValueError('segment_tokens cannot exceed the current-text context limit')
        if not math.isfinite(self.active_fraction) or not 0 < self.active_fraction <= 1:
            raise ValueError('active_fraction must be in (0,1]')


class MatrixMemory(nn.Module):
    """Multiple complete matrix time bands with query-dependent reads and writes."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        h, d, bands = cfg.memory_heads, cfg.memory_dim, cfg.bands
        # prepare() already receives this block's pre-attention normalized input.
        self.norm = nn.Identity()
        self.query = nn.Linear(cfg.width, h*d, bias=False)
        self.key = nn.Linear(cfg.width, h*d, bias=False)
        self.value = nn.Linear(cfg.width, h*d, bias=False)
        self.decay = nn.Linear(cfg.width, bands*h)
        self.write_gate = nn.Linear(cfg.width, bands*h)
        self.band_gate = nn.Linear(cfg.width, bands*h)
        self.output_gate = nn.Linear(cfg.width, h*d)
        self.output = nn.Linear(h*d, cfg.width, bias=False)
        self.log_tau = nn.Parameter(torch.linspace(math.log(16), math.log(1024), bands))
        self.innovation_gain = nn.Parameter(torch.zeros(bands, h))
        self.raw_gain = nn.Parameter(torch.tensor(0.))
        if cfg.phase_enabled:
            periods = torch.logspace(math.log10(128), math.log10(2048), bands)
            frequency = (2*math.pi/periods)[:, None, None].expand(bands, h, d//2)
            self.raw_frequency = nn.Parameter(torch.logit(frequency/.25).clone())
        else:
            self.register_parameter('raw_frequency', None)
        nn.init.zeros_(self.decay.weight)
        nn.init.constant_(self.decay.bias, math.log(math.expm1(1.)))
        nn.init.zeros_(self.write_gate.bias)
        nn.init.zeros_(self.band_gate.bias)
        nn.init.constant_(self.output_gate.bias, -1.)
        nn.init.normal_(self.output.weight, std=.02)
        self.field = None
        if cfg.full_dynamics:
            from .dynamics import BioField
            self.field = BioField(cfg.width, h, d, bands=bands, nodes=cfg.nodes,
                                  internal_steps=cfg.internal_steps, phase_enabled=cfg.phase_enabled,
                                  probability_enabled=cfg.probability_enabled,
                                  dynamic_enabled=cfg.dynamic_enabled,
                                  local_feedback_enabled=cfg.local_feedback_enabled,
                                  coupling_enabled=cfg.coupling_enabled,
                                  threshold_enabled=cfg.threshold_enabled,
                                  active_fraction=cfg.active_fraction,
                                  field_write_mode=cfg.field_write_mode,
                                  segment_tokens=cfg.segment_tokens)

    @property
    def state_dtype(self):
        return torch.float64 if self.query.weight.dtype == torch.float64 else torch.float32

    def initial_state(self, batch):
        c = self.cfg
        return torch.zeros(batch, c.bands, c.memory_heads, c.memory_dim, c.memory_dim,
                           dtype=self.state_dtype, device=self.query.weight.device)

    def _rotate(self, vector, positions):
        if self.raw_frequency is None:
            return vector, vector.new_zeros(())
        frequency = .25*self.raw_frequency.to(vector.dtype).sigmoid()
        # FP32 absolute positions eventually lose unit-token precision at 2**24.
        angle = positions.to(vector.dtype)[:, :, None, None, None]*frequency[None, None]
        cosine, sine = angle.cos(), angle.sin()
        first, second = vector.reshape(*vector.shape[:-1], -1, 2).unbind(-1)
        rotated = torch.stack((first*cosine-second*sine, first*sine+second*cosine), -1).flatten(-2)
        error = ((cosine.square()+sine.square())-1).abs().amax()
        return rotated, error

    def prepare(self, hidden, snapshot, positions):
        """All control coefficients use the incoming snapshot, never pending writes."""
        c = self.cfg
        batch, length = hidden.shape[:2]
        normalized = self.norm(hidden)
        shape = (batch, length, c.memory_heads, c.memory_dim)
        query = F.normalize(self.query(normalized).reshape(shape).to(self.state_dtype), dim=-1, eps=1e-6)
        key = F.normalize(self.key(normalized).reshape(shape).to(self.state_dtype), dim=-1, eps=1e-6)
        value = self.value(normalized).reshape(shape).to(self.state_dtype).tanh()
        query = query[:, :, None].expand(-1, -1, c.bands, -1, -1)
        key = key[:, :, None].expand_as(query)
        value = value[:, :, None].expand_as(query)
        query, radius_error = self._rotate(query, positions)
        key, _ = self._rotate(key, positions)
        predicted = torch.einsum('bshvk,btshk->btshv', snapshot, key)
        innovation = (value-predicted).square().mean(-1).log1p()
        gate_shape = (batch, length, c.bands, c.memory_heads)
        write = self.write_gate(normalized).reshape(gate_shape).to(self.state_dtype)
        eta = .5*(write+self.innovation_gain.sigmoid()[None, None]*innovation).sigmoid()
        tau = self.log_tau.to(self.state_dtype).clamp(math.log(2), math.log(1e6)).exp()
        decay = F.softplus(self.decay(normalized).reshape(gate_shape).to(self.state_dtype))
        alpha = torch.exp(-decay/tau[None, None, :, None])
        weights = self.band_gate(normalized).reshape(gate_shape).to(self.state_dtype).softmax(2)
        output_gate = self.output_gate(normalized).reshape(shape).to(self.state_dtype).sigmoid()
        return dict(query=query, key=key, value=value, alpha=alpha, eta=eta,
                    band_weights=weights, output_gate=output_gate,
                    innovation=innovation, phase_radius_error=radius_error)

    def read(self, controls, snapshot):
        reads = torch.einsum('bshvk,btshk->btshv', snapshot, controls['query'])
        reads = (reads*controls['band_weights'][..., None]).sum(2)/math.sqrt(self.cfg.memory_dim)
        if 'field_read' in controls:
            reads = reads+controls['field_read']
        # This bounded RMS rescaling never amplifies nearly empty memory.
        reads = reads*torch.rsqrt(1+reads.square().mean(-1, keepdim=True))
        reads = (reads*controls['output_gate']).flatten(-2)
        return .1*self.raw_gain.sigmoid()*self.output(reads.to(self.output.weight.dtype))

    @staticmethod
    def write(controls, pending):
        for index in range(controls['key'].shape[1]):
            key, value = controls['key'][:, index], controls['value'][:, index]
            alpha, eta = controls['alpha'][:, index], controls['eta'][:, index]
            previous = (pending@key[..., None]).squeeze(-1)
            pending = (alpha[..., None, None]*pending
                       +eta[..., None, None]*(value-alpha[..., None]*previous)[..., None]*key[..., None, :])
        return pending


class Network(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.width)
        self.blocks = nn.ModuleList([AttentionBlock(cfg.width, cfg.heads, cfg.ff) for _ in range(cfg.layers)])
        self.norm = nn.LayerNorm(cfg.width)
        nn.init.normal_(self.embedding.weight, std=.02)
        self.memories = nn.ModuleList([MatrixMemory(cfg) for _ in self.blocks] if cfg.memory_enabled else [])
        self.initialize_memory()

    @torch.no_grad()
    def initialize_memory(self):
        """Transfer exact projection tensors without claiming attention equivalence."""
        if self.cfg.memory_init == 'backbone':
            for block, memory in zip(self.blocks, self.memories):
                query, key, value = block.qkv.weight.chunk(3, dim=0)
                if (query.shape != memory.query.weight.shape or key.shape != memory.key.weight.shape
                        or value.shape != memory.value.weight.shape or block.out.weight.shape != memory.output.weight.shape):
                    raise ValueError('Backbone projection shapes cannot be copied exactly')
            for block, memory in zip(self.blocks, self.memories):
                query, key, value = block.qkv.weight.chunk(3, dim=0)
                memory.query.weight.copy_(query)
                memory.key.weight.copy_(key)
                memory.value.weight.copy_(value)
                memory.output.weight.copy_(block.out.weight)
        self.memory_initialization = dict(mode=self.cfg.memory_init,
                                         layers=len(self.memories),
                                         exact_head_geometry=True,
                                         extra_input_normalization='identity',
                                         runtime='unit Q/K, tanh V, optional phase, gated bounded read')
        return self

    def initial_state(self, batch):
        if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
            raise ValueError('batch must be a positive integer')
        matrices = [memory.initial_state(batch) for memory in self.memories]
        fields = [memory.field.initial_state(batch, self.embedding.weight.device, memory.state_dtype)
                  for memory in self.memories if memory.field is not None]
        summaries = [memory.field.empty_summary(batch, self.embedding.weight.device, memory.state_dtype)
                     for memory in self.memories if memory.field is not None]
        return dict(kv=[None]*len(self.blocks), valid=None,
                    position=torch.zeros(batch, dtype=torch.long, device=self.embedding.weight.device),
                    phase=torch.zeros(batch, dtype=torch.long, device=self.embedding.weight.device),
                    memory=matrices, pending=[matrix.clone() for matrix in matrices],
                    field=fields, field_summary=summaries)

    def freeze_backbone(self, freeze=True):
        if not isinstance(freeze, bool):
            raise ValueError('freeze must be boolean')
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith('memories.') or not freeze)
        return self

    @torch.no_grad()
    def consolidate(self, state, optimizer=None, *, allow_multilane=False):
        """Commit supported experience at a detached training/segment boundary."""
        if not self.training:
            raise RuntimeError('Parameter consolidation is an explicit training transaction, not an evaluation operation')
        self._validate_state(state, len(state['position']))
        if bool(state['phase'].any()):
            raise ValueError('Finish the current memory segment before consolidating')

        def attached(value):
            if isinstance(value, torch.Tensor):
                return value.requires_grad or value.grad_fn is not None
            if isinstance(value, dict):
                return any(attached(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return any(attached(item) for item in value)
            return False

        if attached(state):
            raise ValueError('Finish backward and detach runtime state before consolidating')
        fields = [memory.field for memory in self.memories if memory.field is not None]
        proposals = [field.consolidate_proposal(value, allow_multilane=allow_multilane)
                     for field, value in zip(fields, state['field'])]
        for field, proposal in zip(fields, proposals):
            if (int(proposal['version']) != int(field.consolidation_version)
                    or proposal['delta'].shape != field.node_shape.shape
                    or not bool(proposal['delta'].isfinite().all())):
                raise ValueError('Invalid or stale consolidation proposal; no parameters were changed')
        changed, magnitude, eligible = 0, 0., 0
        refreshed = []
        for field, value, proposal in zip(fields, state['field'], proposals):
            diagnostics = field.apply_consolidation(proposal, optimizer=optimizer)
            magnitude += float(diagnostics['update_norm'])
            eligible += int(diagnostics['eligible'])
            changed += float(diagnostics['update_norm']) > 0
            refreshed.append(field.invalidate_credit(value))
        updated = state | dict(field=refreshed,
                               field_summary=[field.empty_summary(len(state['position']), field.node_shape.device,
                                                                 value['m'].dtype)
                                              for field, value in zip(fields, refreshed)])
        return updated, dict(consolidation_layers_changed=changed, consolidation_update_norm=magnitude,
                             consolidation_eligible=eligible,
                             consolidation_sessions=len(state['position']))

    def load_backbone(self, payload_or_state_dict, freeze=True):
        """Copy only a shape-compatible local Transformer backbone; reject partial loads."""
        source = payload_or_state_dict
        if not isinstance(source, dict):
            raise TypeError('backbone must be a checkpoint payload or state_dict')
        if isinstance(source.get('model'), dict):
            source = source['model']
        elif isinstance(source.get('state_dict'), dict):
            source = source['state_dict']
        expected = {name: value for name, value in self.state_dict().items()
                    if name.startswith(('embedding.', 'blocks.', 'norm.'))}
        actual = {name: value for name, value in source.items()
                  if name.startswith(('embedding.', 'blocks.', 'norm.'))}
        if actual.keys() != expected.keys():
            raise ValueError('Backbone parameter/buffer keys do not match the requested Transformer')
        if any(not isinstance(actual[name], torch.Tensor) or actual[name].shape != value.shape
               for name, value in expected.items()):
            raise ValueError('Backbone tensor shapes do not match the requested Transformer')
        self.load_state_dict(actual, strict=False)
        self.initialize_memory()
        return self.freeze_backbone(freeze)

    def _validate_state(self, state, batch):
        c, device = self.cfg, self.embedding.weight.device
        if not isinstance(state, dict) or not {'kv', 'valid', 'position', 'phase', 'memory', 'pending', 'field', 'field_summary'} <= state.keys():
            raise ValueError('Invalid v1.1 streaming state')
        for name in ('position', 'phase'):
            value = state[name]
            if value.shape != (batch,) or value.dtype != torch.long or value.device != device:
                raise ValueError(f'{name} must be long [batch] on the model device')
        if bool((state['position'] < 0).any()) or not torch.equal(state['phase'], state['position'] % c.segment_tokens):
            raise ValueError('Invalid segment clock')
        if len(state['kv']) != c.layers or len(state['memory']) != len(self.memories) or len(state['pending']) != len(self.memories):
            raise ValueError('Streaming state layer count mismatch')
        expected_fields = sum(memory.field is not None for memory in self.memories)
        if len(state['field']) != expected_fields or len(state['field_summary']) != expected_fields:
            raise ValueError('BioField streaming state layer count mismatch')
        for name in ('memory', 'pending'):
            for value, module in zip(state[name], self.memories):
                expected = (batch, c.bands, c.memory_heads, c.memory_dim, c.memory_dim)
                if value.shape != expected or value.device != device or value.dtype != module.state_dtype:
                    raise ValueError('Memory state shape, dtype or device mismatch')
        valid = state['valid']
        if valid is not None and (valid.ndim != 2 or valid.shape[0] != batch or valid.dtype != torch.bool or valid.device != device):
            raise ValueError('valid must be boolean [batch, cached_tokens]')
        for cache in state['kv']:
            if cache is not None:
                if (not isinstance(cache, (tuple, list)) or len(cache) != 2
                        or any(not isinstance(value, torch.Tensor) or value.ndim != 4 for value in cache)):
                    raise ValueError('KV cache must contain two rank-four tensors')
                expected = (batch, c.heads, cache[0].shape[2], c.width//c.heads)
                if len(cache) != 2 or any(value.shape != expected or value.device != device for value in cache):
                    raise ValueError('KV cache geometry mismatch')
                if valid is None or valid.shape[1] != expected[2] or expected[2] > c.context_limit:
                    raise ValueError('KV validity metadata mismatch')

    def _reset(self, state, reset_mask, batch):
        if reset_mask is None:
            return state
        if reset_mask.dtype != torch.bool or reset_mask.shape != (batch,) or reset_mask.device != self.embedding.weight.device:
            raise ValueError('reset_mask must be boolean [batch] on the model device')
        valid = state['valid']
        fresh = self.initial_state(batch)
        kv = [None if cache is None else tuple(torch.where(reset_mask[:, None, None, None], 0, value)
                                              for value in cache) for cache in state['kv']]
        return state | dict(valid=None if valid is None else valid & ~reset_mask[:, None],
                            kv=kv,
                            position=torch.where(reset_mask, 0, state['position']),
                            phase=torch.where(reset_mask, 0, state['phase']),
                            memory=[torch.where(reset_mask[:, None, None, None, None], 0, value) for value in state['memory']],
                            pending=[torch.where(reset_mask[:, None, None, None, None], 0, value) for value in state['pending']],
                            field=[self._reset_tree(old, new, reset_mask) for old, new in zip(state['field'], fresh['field'])],
                            field_summary=[self._reset_tree(old, new, reset_mask) for old, new in zip(state['field_summary'], fresh['field_summary'])])

    @staticmethod
    def _reset_tree(old, fresh, mask):
        if isinstance(old, dict):
            return {name: Network._reset_tree(value, fresh[name], mask) for name, value in old.items()}
        if isinstance(old, torch.Tensor) and old.ndim and old.shape[0] == len(mask):
            return torch.where(mask.reshape(len(mask), *([1]*(old.ndim-1))), fresh, old)
        return old

    def _attention(self, block, x, past, position, valid):
        batch, length, width = x.shape
        normalized = block.norm1(x)
        q, k, v = block.qkv(normalized).reshape(batch, length, 3, block.heads, block.head_width).permute(2, 0, 3, 1, 4).unbind(0)
        positions = position[:, None]+torch.arange(length, device=x.device)[None]
        q, k = block.rope(q, positions), block.rope(k, positions)
        if past is not None:
            keep = max(0, self.cfg.context_limit-1)
            old_k, old_v = (value[:, :, -keep:] if keep else value[:, :, :0] for value in past)
            k, v = torch.cat((old_k, k), 2), torch.cat((old_v, v), 2)
        offset = k.shape[2]-length
        keys = torch.arange(k.shape[2], device=x.device)[None]
        queries = offset+torch.arange(length, device=x.device)[:, None]
        mask = (keys <= queries) & (keys >= queries-self.cfg.context_limit+1)
        if valid is not None:
            live = torch.cat((valid[:, -offset:] if offset else valid[:, :0],
                              torch.ones(batch, length, dtype=torch.bool, device=x.device)), -1)
            mask = mask[None, None] & live[:, None, None]
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.)
        return x+block.out(attention.transpose(1, 2).reshape(batch, length, width)), (k[:, :, -self.cfg.context_limit:], v[:, :, -self.cfg.context_limit:]), normalized

    def forward(self, tokens, state=None, reset_mask=None, memory_read=True, memory_write=True):
        if tokens.ndim != 2 or tokens.dtype != torch.long or min(tokens.shape) < 1 or tokens.device != self.embedding.weight.device:
            raise ValueError('tokens must be nonempty long [batch,time] on the model device')
        if not isinstance(memory_read, bool) or not isinstance(memory_write, bool):
            raise ValueError('memory_read and memory_write must be boolean')
        batch = len(tokens)
        state = self.initial_state(batch) if state is None else state
        self._validate_state(state, batch)
        state = self._reset(state, reset_mask, batch)
        offset, outputs, radius_errors, field_diagnostics = 0, [], [], []
        while offset < tokens.shape[1]:
            remaining = tokens.shape[1]-offset
            length = min(remaining, int((self.cfg.segment_tokens-state['phase']).min())) if self.memories else min(remaining, self.cfg.context_limit)
            values = tokens[:, offset:offset+length]
            hidden = self.embedding(values)
            position, phase = state['position'], state['phase']
            positions = position[:, None]+torch.arange(length, device=tokens.device)[None]
            boundary = phase+length == self.cfg.segment_tokens
            new_kv, snapshots, pending, fields, summaries = [], [], [], [], []
            for layer, (block, cache) in enumerate(zip(self.blocks, state['kv'])):
                if not self.memories:
                    hidden, cache = block(hidden, cache, position, self.cfg.context_limit, state['valid'])
                else:
                    hidden, cache, normalized = self._attention(block, hidden, cache, position, state['valid'])
                    memory = self.memories[layer]
                    snapshot, writer = state['memory'][layer], state['pending'][layer]
                    controls = memory.prepare(normalized, snapshot, positions)
                    radius_errors.append(controls['phase_radius_error'])
                    if memory.field is not None:
                        field = state['field'][layer]
                        summary = state['field_summary'][layer]
                        if memory_read:
                            query = (controls['query']*controls['band_weights'][..., None]).sum(2)
                            query = F.normalize(query, dim=-1, eps=1e-6)
                            field_read, diag = memory.field.read(normalized, query, field)
                            controls['field_read'] = field_read
                            field_diagnostics.append(diag)
                    if memory_read:
                        hidden = hidden+memory.read(controls, snapshot)
                    hidden = hidden+block.ff(block.norm2(hidden))
                    if memory_write:
                        writer = torch.where((phase == 0)[:, None, None, None, None], snapshot, writer)
                        writer = memory.write(controls, writer)
                        snapshot = torch.where(boundary[:, None, None, None, None], writer, snapshot)
                        if memory.field is not None:
                            summary = memory.field.accumulate(summary, controls, hidden=normalized)
                            if bool(boundary.any()):
                                field = memory.field.commit(summary, field, commit_mask=boundary)
                                empty = memory.field.empty_summary(batch, tokens.device, memory.state_dtype)
                                summary = self._reset_tree(summary, empty, boundary)
                    if memory.field is not None:
                        fields.append(field)
                        summaries.append(summary)
                    snapshots.append(snapshot)
                    pending.append(writer)
                new_kv.append(cache)
            outputs.append(F.linear(self.norm(hidden), self.embedding.weight))
            keep = max(0, self.cfg.context_limit-1)
            valid = state['valid']
            old_valid = valid[:, -keep:] if valid is not None and keep else torch.empty(batch, 0, dtype=torch.bool, device=tokens.device)
            valid = torch.cat((old_valid, torch.ones_like(values, dtype=torch.bool)), -1)[:, -self.cfg.context_limit:]
            state = dict(kv=new_kv, valid=valid, position=position+length,
                         phase=(phase+length) % self.cfg.segment_tokens, memory=snapshots, pending=pending,
                         field=fields, field_summary=summaries)
            offset += length
        diagnostics = {}
        if self.memories:
            diagnostics = dict(memory_rms=torch.stack([value.square().mean().sqrt() for value in state['memory']]).mean(),
                               phase_radius_error=torch.stack(radius_errors).amax())
            for name in set().union(*(diag.keys() for diag in field_diagnostics)):
                values = [diag[name].to(self.memories[0].state_dtype).mean() for diag in field_diagnostics
                          if name in diag and isinstance(diag[name], torch.Tensor)]
                if values:
                    diagnostics['field_'+name] = torch.stack(values).mean()
        return torch.cat(outputs, 1), state, diagnostics
