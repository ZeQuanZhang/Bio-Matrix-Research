"""Vector nodes: phase-routed rank-one readout and block-snapshot causal writes."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int = 4096
    width: int = 56
    layers: int = 2
    heads: int = 4
    ff: int = 272
    nodes: int = 32
    phase_bands: int = 8
    segment_tokens: int = 64
    context_limit: int = 128  # Reference Transformer window, not a Bio KV cache.
    dynamic_addresses: bool = True

    def __post_init__(self):
        for name, value in vars(self).items():
            if name == 'dynamic_addresses':
                if type(value) is not bool: raise ValueError('dynamic_addresses must be boolean')
            elif type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.width % self.heads: raise ValueError('width must be divisible by heads')
        if self.segment_tokens > 1024: raise ValueError('Use blocks of at most 1024 tokens')


def select(mask, new, old):
    return torch.where(mask.reshape(len(mask), *([1]*(new.ndim-1))), new, old)


def unit(x):
    return F.normalize(x, dim=-1, eps=1e-6)


def bounded(x):
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).div(4).clamp_min(1)



class VectorNodes(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        h, n, d, r = cfg.heads, cfg.nodes, cfg.width//cfg.heads, cfg.phase_bands
        self.norm = nn.LayerNorm(cfg.width)
        self.qkv = nn.Linear(cfg.width, 3*cfg.width, bias=False)
        self.output = nn.Linear(cfg.width, cfg.width, bias=False)
        self.output_gate = nn.Linear(cfg.width, cfg.width)
        self.key_anchor = nn.Parameter(torch.randn(h,n,d)/math.sqrt(d))
        self.value_anchor = nn.Parameter(torch.randn(h,n,d)*.02)
        self.anchor_angles = nn.Parameter(torch.rand(h,n,r)*2*math.pi)
        self.phase_projection = nn.Parameter(torch.randn(h,d,r)/math.sqrt(d))
        self.raw_temperature = nn.Parameter(torch.zeros(h))
        self.raw_write_rate = nn.Parameter(torch.zeros(h))
        self.raw_address_rate = nn.Parameter(torch.zeros(h))
        self.log_tau = nn.Parameter(torch.full((h,), math.log(4096.)))
        nn.init.zeros_(self.output_gate.bias)

    def anchors(self, batch):
        phase = torch.stack((self.anchor_angles.cos(),self.anchor_angles.sin()),-1).flatten(-2)
        return dict(key=unit(self.key_anchor)[None].expand(batch,-1,-1,-1),
                    value=bounded(self.value_anchor)[None].expand(batch,-1,-1,-1),
                    phase=phase[None].expand(batch,-1,-1,-1))

    def initial(self, batch):
        return {name: value.clone() for name,value in self.anchors(batch).items()}

    def phase(self, x):
        angle = math.pi*torch.tanh(torch.einsum('bthd,hdr->bthr',x,self.phase_projection))
        return torch.stack((angle.cos(),angle.sin()),-1)

    def route(self, query, phase, state):
        keys = unit(state['key'])
        addresses = unit(state['phase'].reshape(*state['phase'].shape[:-1],self.cfg.phase_bands,2))
        match = (query[...,None,:]*keys).sum(-1)
        similarity = (phase[...,None,:,:]*addresses).sum((-1,-2))/self.cfg.phase_bands
        temperature = 1+15*self.raw_temperature.sigmoid()[None,None,:,None]
        gates = (match+temperature*similarity).softmax(-1)
        prediction = torch.einsum('bthn,bthn,bthnd->bthd',gates,match,state['value'])
        return prediction,gates,match,addresses,temperature

    def write_terms(self, key, value, phase, snapshot):
        old = {name:x[:,None] for name,x in snapshot.items()}
        prediction,g,match,p,temperature = self.route(key,phase,old)
        error = value-prediction
        agreement = torch.einsum('bthd,bthnd->bthn',error,old['value'])
        benefit = match*agreement-(error*prediction).sum(-1,keepdim=True)
        return dict(error=error,key=key,phase=phase,value_factor=g*match,
                    key_factor=g*(agreement+benefit),phase_factor=g*benefit*temperature/self.cfg.phase_bands)

    def joint_gradients(self, terms, snapshot, discount):
        """Summed negative joint gradients; the routing snapshot is fixed in a block."""
        def summed(factor,values):
            return torch.einsum('bthn,bthd->bhnd',factor*discount,values)
        key = summed(terms['key_factor'],terms['key'])
        oldkey = unit(snapshot['key'])
        key = key-(key*oldkey).sum(-1,keepdim=True)*oldkey
        phase = summed(terms['phase_factor'],terms['phase'].flatten(-2))
        phase = phase.reshape(*phase.shape[:-1],self.cfg.phase_bands,2)
        oldphase = unit(snapshot['phase'].reshape_as(phase))
        phase = phase-(phase*oldphase).sum(-1,keepdim=True)*oldphase
        return dict(key=key,phase=phase.flatten(-2),
                    value=summed(terms['value_factor'],terms['error']))

    def forward(self, hidden, snapshot, current, read=True, write=True):
        c = self.cfg
        x = self.norm(hidden)
        qkv = self.qkv(x).reshape(*x.shape[:2],3,c.heads,-1)
        q,k = unit(qkv[:,:,:2]).unbind(2)
        v = qkv[:,:,2].tanh()
        qphase,kphase = self.phase(torch.cat((q,k),1)).chunk(2,1)
        old = {name:value[:,None] for name,value in snapshot.items()}
        _,g,match,_,_ = self.route(q,qphase,old)
        query = (g*match).transpose(1,2)
        terms = self.write_terms(k,v,kphase,snapshot)
        tau = self.log_tau.clamp(math.log(64),math.log(1e6)).exp()
        alpha = torch.exp(-1/tau)
        if not write: alpha = torch.ones_like(alpha)
        length = hidden.shape[1]
        times = torch.arange(length,device=x.device,dtype=x.dtype)
        powers = alpha[:,None].pow(times[None,:])[None,:,:,None]
        anchors = self.anchors(len(hidden))
        prediction = torch.einsum('bhtn,bhnd->bhtd',query,current['value'])*powers
        prediction = prediction+torch.einsum('bhtn,bhnd->bhtd',query,anchors['value'])*(1-powers)
        rate = .25*self.raw_write_rate.sigmoid()
        write_factor = terms['value_factor'].transpose(1,2)*rate[None,:,None,None]
        if not write: write_factor = torch.zeros_like(write_factor)
        # Dual causal read avoids materializing [batch,time,head,node,value] states.
        distance = times[:,None]-times[None,:]-1
        causal = (distance>=0).to(x.dtype)
        decay = alpha[:,None,None].pow(distance.clamp_min(0)[None])*causal
        interaction = (query@write_factor.transpose(-1,-2))*decay[None]
        prediction = prediction+interaction@terms['error'].transpose(1,2)
        discount = alpha[:,None].pow((length-1-times)[None])[None].transpose(1,2)[...,None]
        gradients = self.joint_gradients(terms,snapshot,discount)
        end = dict(current)
        if write:
            for name in ('key','value','phase'):
                if name!='value' and not c.dynamic_addresses:continue
                step = rate if name!='phase' else .05*self.raw_address_rate.sigmoid()
                if name!='value': step = step/c.segment_tokens
                end[name] = (anchors[name]+alpha[None,:,None,None].pow(length)*(current[name]-anchors[name])
                             +step[None,:,None,None]*gradients[name])
        output = self.output((prediction.transpose(1,2)*self.output_gate(x).reshape_as(q).sigmoid()).flatten(-2))
        if not read: output = torch.zeros_like(output)
        return hidden+output,end,g.square().sum(-1).reciprocal().mean()

    def commit(self, current):
        return dict(key=unit(current['key']),value=bounded(current['value']),
                    phase=unit(current['phase'].reshape(*current['phase'].shape[:-1],
                              self.cfg.phase_bands,2)).flatten(-2))


class BioLayer(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.nodes = VectorNodes(cfg)
        self.norm = nn.LayerNorm(cfg.width)
        self.ff = nn.Sequential(nn.Linear(cfg.width,cfg.ff),nn.GELU(),nn.Linear(cfg.ff,cfg.width))


class Network(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size,cfg.width)
        self.layers = nn.ModuleList([BioLayer(cfg) for _ in range(cfg.layers)])
        self.norm = nn.LayerNorm(cfg.width)
        nn.init.normal_(self.embedding.weight,std=.02)

    def initial_state(self,batch):
        if type(batch) is not int or batch<1: raise ValueError('batch must be a positive integer')
        return dict(position=torch.zeros(batch,dtype=torch.long,device=self.embedding.weight.device),
                    memory=[layer.nodes.initial(batch) for layer in self.layers],
                    pending=[layer.nodes.initial(batch) for layer in self.layers])

    def validate_state(self,state,batch):
        c,weight = self.cfg,self.embedding.weight
        if not isinstance(state,dict) or set(state) != {'position','memory','pending'}:
            raise ValueError('Invalid V3 state fields')
        position=state['position']
        if (position.shape != (batch,) or position.dtype != torch.long or position.device != weight.device
                or bool((position<0).any())): raise ValueError('Invalid state clock')
        for field in ('memory','pending'):
            if len(state[field]) != c.layers: raise ValueError('Invalid state layer count')
            for row in state[field]:
                if set(row) != {'key','value','phase'}: raise ValueError('Invalid node state fields')
                for name in row:
                    size=2*c.phase_bands if name=='phase' else c.width//c.heads
                    if (row[name].shape != (batch,c.heads,c.nodes,size) or row[name].device != weight.device
                            or row[name].dtype != weight.dtype): raise ValueError('Invalid vector state geometry')

    def forward(self,tokens,state=None,reset_mask=None,memory_read=True,memory_write=True):
        if (tokens.dtype != torch.long or tokens.ndim != 2 or min(tokens.shape)<1
                or tokens.device != self.embedding.weight.device): raise ValueError('Expected long [batch,time] tokens')
        if type(memory_read) is not bool or type(memory_write) is not bool:
            raise ValueError('Memory controls must be boolean')
        state = self.initial_state(len(tokens)) if state is None else state
        self.validate_state(state,len(tokens))
        if reset_mask is not None:
            if reset_mask.shape != (len(tokens),) or reset_mask.dtype != torch.bool or reset_mask.device != tokens.device:
                raise ValueError('Invalid reset mask')
            fresh=self.initial_state(len(tokens))
            state=dict(position=select(reset_mask,fresh['position'],state['position']),
                       **{name:[{key:select(reset_mask,new[key],old[key]) for key in old}
                                 for old,new in zip(state[name],fresh[name])] for name in ('memory','pending')})
        outputs,effective,offset=[],[],0
        while offset<tokens.shape[1]:
            clocks=state['position']%self.cfg.segment_tokens
            length=min(tokens.shape[1]-offset,int((self.cfg.segment_tokens-clocks).min()))
            boundary=clocks+length==self.cfg.segment_tokens
            hidden=self.embedding(tokens[:,offset:offset+length]);memories=[];pendings=[]
            for index,layer in enumerate(self.layers):
                old,current=state['memory'][index],state['pending'][index]
                hidden,current,usage=layer.nodes(hidden,old,current,memory_read,memory_write)
                if bool(boundary.any()):
                    committed=layer.nodes.commit(current)
                    old={key:select(boundary,committed[key],old[key]) for key in old}
                    current={key:select(boundary,old[key],current[key]) for key in current}
                hidden=hidden+layer.ff(layer.norm(hidden))
                memories.append(old);pendings.append(current);effective.append(usage)
            outputs.append(F.linear(self.norm(hidden),self.embedding.weight))
            state=dict(position=state['position']+length,memory=memories,pending=pendings)
            offset+=length
        phase_error=torch.stack([(unit(s['phase'].reshape(*s['phase'].shape[:-1],self.cfg.phase_bands,2))
                                  .square().sum(-1)-1).abs().max() for s in state['pending']]).max()
        diagnostics=dict(effective_nodes=torch.stack(effective).mean(),phase_error=phase_error,
                         value_rms=torch.stack([s['value'].square().mean().sqrt() for s in state['pending']]).mean())
        return torch.cat(outputs,1),state,diagnostics
