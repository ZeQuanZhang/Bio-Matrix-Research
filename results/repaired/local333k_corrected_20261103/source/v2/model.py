"""One shared snapshot per block; batched node writes and phase-address updates."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from v07.model import AttentionBlock
from v11.model import Network as StreamingAttention


@dataclass
class Config:
    vocab_size: int = 4096
    width: int = 56
    layers: int = 2
    heads: int = 4
    ff: int = 272
    context_limit: int = 128
    segment_tokens: int = 64
    nodes: int = 32
    phase_bands: int = 8
    memory_enabled: bool = True
    dynamic_addresses: bool = True

    def __post_init__(self):
        for name in ('vocab_size', 'width', 'layers', 'heads', 'ff', 'context_limit',
                     'segment_tokens', 'nodes', 'phase_bands'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.width % self.heads or (self.width//self.heads) % 2:
            raise ValueError('Need even per-head width for RoPE')
        if self.segment_tokens > self.context_limit:
            raise ValueError('The local attention window must cover a memory block')
        if type(self.memory_enabled) is not bool or type(self.dynamic_addresses) is not bool:
            raise ValueError('Memory flags must be boolean')


def select(mask, new, old):
    return torch.where(mask.reshape(len(mask), *([1]*(new.ndim-1))), new, old)


def bounded(matrix):
    return matrix/torch.linalg.vector_norm(matrix, dim=(-1, -2), keepdim=True).div(4).clamp_min(1)


class PhaseMemory(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        h, d, n, r = cfg.heads, cfg.width//cfg.heads, cfg.nodes, cfg.phase_bands
        self.norm = nn.LayerNorm(cfg.width)
        self.address = nn.Linear(cfg.width, cfg.width, bias=False)
        self.value = nn.Linear(cfg.width, cfg.width, bias=False)
        self.output = nn.Linear(cfg.width, cfg.width, bias=False)
        self.output_gate = nn.Linear(cfg.width, cfg.width)
        self.phase_projection = nn.Parameter(torch.randn(h, d, r)/math.sqrt(d))
        self.anchor_angles = nn.Parameter(torch.rand(n, h, r)*2*math.pi)
        self.node_shape = nn.Parameter(torch.zeros(n, h, d, d))
        self.raw_write_rate = nn.Parameter(torch.zeros(h))
        self.raw_address_rate = nn.Parameter(torch.zeros(h))
        self.raw_temperature = nn.Parameter(torch.zeros(h))
        self.log_tau = nn.Parameter(torch.full((h,), math.log(4096.)))
        self.raw_gain = nn.Parameter(torch.tensor(math.log(.125/.875)))
        nn.init.zeros_(self.output_gate.bias)

    def anchor(self):
        return torch.stack((self.anchor_angles.cos(), self.anchor_angles.sin()), -1)

    def initial_state(self, batch):
        c, w = self.cfg, self.node_shape
        d = c.width//c.heads
        return dict(content=bounded(w)[None].expand(batch, -1, -1, -1, -1).clone(),
                    addresses=self.anchor()[None].expand(batch, -1, -1, -1, -1).clone(),
                    free=torch.eye(d, device=w.device, dtype=w.dtype).expand(batch, c.nodes, c.heads, d, d).clone(),
                    blocks=torch.zeros(batch, device=w.device, dtype=torch.long))

    def empty(self, batch):
        c, w = self.cfg, self.node_shape
        d = c.width//c.heads
        return dict(cross=w.new_zeros(batch, c.nodes, c.heads, d, d),
                    gram=w.new_zeros(batch, c.nodes, c.heads, d, d),
                    phase=w.new_zeros(batch, c.nodes, c.heads, c.phase_bands, 2),
                    mass=w.new_zeros(batch, c.nodes, c.heads),
                    count=w.new_zeros(batch))

    def controls(self, hidden, state):
        c = self.cfg
        x = self.norm(hidden)
        key = F.normalize(self.address(x).reshape(*x.shape[:2], c.heads, -1), dim=-1, eps=1e-6)
        value = self.value(x).reshape_as(key).tanh()
        angle = math.pi*torch.tanh(torch.einsum('bthd,hdr->bthr', key, self.phase_projection))
        phase = torch.stack((angle.cos(), angle.sin()), -1)
        similarity = torch.einsum('bthrc,bnhrc->bthn', phase, state['addresses'])/c.phase_bands
        temperature = 1+15*self.raw_temperature.sigmoid()
        gates = (similarity*temperature[None, None, :, None]).softmax(-1)
        node_values = torch.einsum('bnhde,bthe->btnhd', state['content'], key)
        prediction = torch.einsum('bthn,btnhd->bthd', gates, node_values)
        return dict(key=key, value=value, phase=phase, gates=gates,
                    prediction=prediction, error=value-prediction,
                    output_gate=self.output_gate(x).reshape_as(key).sigmoid())

    def read(self, controls, state):
        value = controls['prediction']
        value = (value*controls['output_gate']).flatten(-2)
        return 2*self.raw_gain.sigmoid()*self.output(value)

    def accumulate(self, pending, controls):
        g, k, error, p = (controls[name] for name in ('gates', 'key', 'error', 'phase'))
        # Sufficient statistics preserve fixed memory size and avoid a token loop.
        cross = torch.einsum('bthn,bthd,bthe->bnhde', g, error, k)
        gram = torch.einsum('bthn,bthd,bthe->bnhde', g, k, k)
        phases = torch.einsum('bthn,bthrc->bnhrc', g, p)
        return dict(cross=pending['cross']+cross, gram=pending['gram']+gram,
                    phase=pending['phase']+phases, mass=pending['mass']+g.sum(1).transpose(1, 2),
                    count=pending['count']+k.shape[1])

    def commit(self, pending, state, mask):
        mask = mask & (pending['count'] > 0)
        mass = pending['mass']
        denominator = mass.clamp_min(1e-8)[..., None, None]
        gram, cross = pending['gram']/denominator, pending['cross']/denominator
        count = pending['count'][:, None, None, None, None]
        tau = self.log_tau.clamp(math.log(64), math.log(1e6)).exp()[None, None, :, None, None]
        alpha = torch.exp(-count/tau)
        prior = alpha*state['content']+(1-alpha)*bounded(self.node_shape)[None]
        rate = .5*self.raw_write_rate.sigmoid()[None, None, :]*mass/(mass+1)
        content = bounded(prior+rate[..., None, None]*(cross@state['free']))
        d = self.cfg.width//self.cfg.heads
        eye = torch.eye(d, device=prior.device, dtype=prior.dtype)
        free = alpha*state['free']+(1-alpha)*eye
        free = free-rate[..., None, None]*(free@gram@free)
        free = .5*(free+free.transpose(-1, -2))
        addresses = state['addresses']
        if self.cfg.dynamic_addresses:
            target = F.normalize(pending['phase']/denominator, dim=-1, eps=1e-6)
            step = .05*self.raw_address_rate.sigmoid()[None, None, :, None, None]
            step = step*mass[..., None, None]/(mass[..., None, None]+1)
            candidate = addresses+step*(target-addresses)+.005*(self.anchor()[None]-addresses)
            addresses = F.normalize(candidate, dim=-1, eps=1e-6)
        candidate = dict(content=content, addresses=addresses, free=free, blocks=state['blocks']+1)
        return {name: select(mask, value, state[name]) for name, value in candidate.items()}


class Network(nn.Module):
    _attention = StreamingAttention._attention
    _reset_tree = staticmethod(StreamingAttention._reset_tree)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.width)
        self.blocks = nn.ModuleList([AttentionBlock(cfg.width, cfg.heads, cfg.ff) for _ in range(cfg.layers)])
        self.norm = nn.LayerNorm(cfg.width)
        nn.init.normal_(self.embedding.weight, std=.02)
        self.memories = nn.ModuleList([PhaseMemory(cfg) for _ in self.blocks] if cfg.memory_enabled else [])

    @torch.no_grad()
    def load_backbone(self, payload):
        source = payload.get('model', payload)
        expected = {n: v for n,v in self.state_dict().items() if not n.startswith('memories.')}
        actual = {n:v for n,v in source.items() if not n.startswith('memories.')}
        if expected.keys() != actual.keys() or any(actual[n].shape != v.shape for n,v in expected.items()):
            raise ValueError('Pretrained backbone keys or geometry differ')
        self.load_state_dict(actual, strict=False)
        for block, memory in zip(self.blocks, self.memories):
            q,k,v = block.qkv.weight.chunk(3, 0)
            memory.address.weight.copy_((q+k)/2)
            memory.value.weight.copy_(v)
            memory.output.weight.copy_(block.out.weight)
            memory.norm.load_state_dict(block.norm2.state_dict())
        return self

    def initial_state(self, batch):
        if type(batch) is not int or batch < 1:
            raise ValueError('batch must be positive')
        device = self.embedding.weight.device
        return dict(kv=[None]*self.cfg.layers, valid=None,
                    position=torch.zeros(batch, device=device, dtype=torch.long),
                    phase=torch.zeros(batch, device=device, dtype=torch.long),
                    memory=[m.initial_state(batch) for m in self.memories],
                    pending=[m.empty(batch) for m in self.memories])

    def validate_state(self, state, batch):
        c, device = self.cfg, self.embedding.weight.device
        if not isinstance(state, dict) or not {'kv','valid','position','phase','memory','pending'} <= state.keys():
            raise ValueError('Invalid v2 state')
        for name in ('position','phase'):
            t = state[name]
            if t.shape != (batch,) or t.dtype != torch.long or t.device != device:
                raise ValueError('Clock shape/dtype/device mismatch')
        if bool((state['position'] < 0).any()) or not torch.equal(state['phase'],state['position'] % c.segment_tokens):
            raise ValueError('Invalid persistent block clock')
        if len(state['kv']) != c.layers or len(state['memory']) != len(self.memories) or len(state['pending']) != len(self.memories):
            raise ValueError('State layer count mismatch')
        for memory, pending, module in zip(state['memory'], state['pending'], self.memories):
            shapes = {'content':(batch,c.nodes,c.heads,c.width//c.heads,c.width//c.heads),
                      'free':(batch,c.nodes,c.heads,c.width//c.heads,c.width//c.heads),
                      'addresses':(batch,c.nodes,c.heads,c.phase_bands,2), 'blocks':(batch,)}
            for name, shape in shapes.items():
                dtype = torch.long if name=='blocks' else self.embedding.weight.dtype
                if (name not in memory or memory[name].shape != shape or memory[name].device != device
                        or memory[name].dtype != dtype):
                    raise ValueError('Memory geometry/device mismatch')
            pending_shapes={'cross':shapes['content'],'gram':shapes['content'],
                            'phase':shapes['addresses'],'mass':(batch,c.nodes,c.heads),'count':(batch,)}
            for name,shape in pending_shapes.items():
                if (name not in pending or pending[name].shape!=shape or pending[name].device!=device
                        or pending[name].dtype!=self.embedding.weight.dtype):
                    raise ValueError('Pending statistic geometry/dtype/device mismatch')
            if not torch.equal(pending['count'],state['phase'].to(pending['count'].dtype)):
                raise ValueError('Pending statistics and block clock differ')
        valid = state['valid']
        if valid is not None and (valid.dtype != torch.bool or valid.device != device or valid.shape[0] != batch):
            raise ValueError('KV validity mismatch')
        for cache in state['kv']:
            if cache is not None:
                shape=(batch,c.heads,cache[0].shape[2],c.width//c.heads)
                if len(cache)!=2 or any(v.shape != shape or v.device!=device for v in cache):
                    raise ValueError('Invalid KV cache')
                if valid is None or valid.shape[1] != shape[2] or shape[2] > c.context_limit:
                    raise ValueError('KV length mismatch')

    def reset(self, state, mask):
        if mask is None:
            return state
        b = len(state['position'])
        if mask.dtype != torch.bool or mask.shape != (b,) or mask.device != self.embedding.weight.device:
            raise ValueError('reset_mask must be boolean [batch] on model device')
        fresh = self.initial_state(b)
        return state | dict(
            kv=[None if x is None else tuple(select(mask,torch.zeros_like(v),v) for v in x) for x in state['kv']],
            valid=None if state['valid'] is None else state['valid'] & ~mask[:,None],
            position=select(mask,fresh['position'],state['position']),
            phase=select(mask,fresh['phase'],state['phase']),
            memory=[self._reset_tree(x,y,mask) for x,y in zip(state['memory'],fresh['memory'])],
            pending=[self._reset_tree(x,y,mask) for x,y in zip(state['pending'],fresh['pending'])])

    def forward(self, tokens, state=None, reset_mask=None, memory_read=True, memory_write=True):
        if (tokens.dtype != torch.long or tokens.ndim != 2 or min(tokens.shape)<1
                or tokens.device != self.embedding.weight.device):
            raise ValueError('Expected nonempty long [batch,time] tokens on model device')
        if type(memory_read) is not bool or type(memory_write) is not bool:
            raise ValueError('Memory controls must be boolean')
        state = self.initial_state(len(tokens)) if state is None else state
        self.validate_state(state,len(tokens))
        state = self.reset(state,reset_mask)
        outputs, routing_statistics, offset = [], [], 0
        while offset < tokens.shape[1]:
            remaining=tokens.shape[1]-offset
            length=min(remaining,int((self.cfg.segment_tokens-state['phase']).min())) if self.memories else remaining
            values=tokens[:,offset:offset+length]
            hidden=self.embedding(values)
            boundary=state['phase']+length == self.cfg.segment_tokens
            caches, memories, pendings = [], [], []
            for layer,block in enumerate(self.blocks):
                hidden,cache,_=self._attention(block,hidden,state['kv'][layer],state['position'],state['valid'])
                if self.memories:
                    module=self.memories[layer]
                    old,pending=state['memory'][layer],state['pending'][layer]
                    controls=module.controls(hidden,old)
                    routing_statistics.append(controls['gates'].square().sum(-1).reciprocal().mean())
                    if memory_read:
                        hidden=hidden+module.read(controls,old)
                    if memory_write:
                        pending=module.accumulate(pending,controls)
                    else:
                        pending=pending | {'count':pending['count']+length}
                    if bool(boundary.any()):
                        if memory_write:
                            old=module.commit(pending,old,boundary)
                        pending=self._reset_tree(pending,module.empty(len(tokens)),boundary)
                    memories.append(old);pendings.append(pending)
                hidden=hidden+block.ff(block.norm2(hidden))
                caches.append(cache)
            outputs.append(F.linear(self.norm(hidden),self.embedding.weight))
            keep=max(0,self.cfg.context_limit-1)
            oldvalid=state['valid']
            oldvalid=oldvalid[:,-keep:] if oldvalid is not None and keep else torch.empty(len(tokens),0,dtype=torch.bool,device=tokens.device)
            valid=torch.cat((oldvalid,torch.ones_like(values,dtype=torch.bool)),-1)[:,-self.cfg.context_limit:]
            state=dict(kv=caches,valid=valid,position=state['position']+length,
                       phase=(state['phase']+length)%self.cfg.segment_tokens,memory=memories,pending=pendings)
            offset+=length
        diagnostics={}
        if self.memories:
            diagnostics=dict(content_rms=torch.stack([x['content'].square().mean().sqrt() for x in state['memory']]).mean(),
                             effective_nodes=torch.stack(routing_statistics).mean(),
                             address_change=torch.stack([(x['addresses']-m.anchor()[None]).square().mean().sqrt() for x,m in zip(state['memory'],self.memories)]).mean(),
                             free_trace=torch.stack([x['free'].diagonal(dim1=-2,dim2=-1).mean() for x in state['memory']]).mean(),
                             phase_error=torch.stack([(x['addresses'].square().sum(-1)-1).abs().max() for x in state['memory']]).max())
        return torch.cat(outputs,1),state,diagnostics
