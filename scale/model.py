"""V3 operators with equivalent contractions and an ordinary full causal baseline."""
from dataclasses import asdict, replace
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from v3.model import Config, Network, VectorNodes, unit
from v07.model import AttentionBlock

CONDITIONS=('transformer','bio_static','bio_dynamic')


class CompactNodes(VectorNodes):
    def route(self, query, phase, state):
        keys=unit(state['key'])
        addresses=unit(state['phase'].reshape(*state['phase'].shape[:-1],self.cfg.phase_bands,2))
        # Contract directly instead of materializing [batch,time,head,node,dimension].
        match=torch.einsum('bthd,bthnd->bthn',query,keys)
        similarity=torch.einsum('bthrc,bthnrc->bthn',phase,addresses)/self.cfg.phase_bands
        temperature=1+15*self.raw_temperature.sigmoid()[None,None,:,None]
        gates=(match+temperature*similarity).softmax(-1)
        prediction=torch.einsum('bthn,bthn,bthnd->bthd',gates,match,state['value'])
        return prediction,gates,match,addresses,temperature


class Bio(Network):
    def __init__(self,cfg,activation_checkpoint=True):
        super().__init__(cfg)
        for layer in self.layers:
            compact=CompactNodes(cfg);compact.load_state_dict(layer.nodes.state_dict());layer.nodes=compact
            if activation_checkpoint:
                layer.nodes.forward=self._checked(layer.nodes.forward)
                layer.ff.forward=self._checked(layer.ff.forward)

    @staticmethod
    def _checked(original):
        def forward(*args,**kwargs):
            if torch.is_grad_enabled():return checkpoint(original,*args,**kwargs,use_reentrant=False)
            return original(*args,**kwargs)
        return forward

    def forward(self,tokens,state=None,reset_mask=None,memory_read=True,memory_write=True):
        if state is not None or reset_mask is not None or not memory_read or not memory_write:
            return super().forward(tokens,state,reset_mask,memory_read,memory_write)
        if tokens.dtype!=torch.long or tokens.ndim!=2 or not tokens.numel() or tokens.device!=self.embedding.weight.device:
            raise ValueError('Expected nonempty long [batch,time] on model device')
        # A freshly reset batch has identical known clocks; block boundaries need no device-to-host decisions.
        state=self.initial_state(len(tokens));outputs=[];effective=[]
        for start in range(0,tokens.shape[1],self.cfg.segment_tokens):
            length=min(self.cfg.segment_tokens,tokens.shape[1]-start)
            hidden=self.embedding(tokens[:,start:start+length]);memories=[];pendings=[]
            for index,layer in enumerate(self.layers):
                old,current=state['memory'][index],state['pending'][index]
                hidden,current,usage=layer.nodes(hidden,old,current)
                if length==self.cfg.segment_tokens:old=current=layer.nodes.commit(current)
                hidden=hidden+layer.ff(layer.norm(hidden));memories.append(old);pendings.append(current);effective.append(usage)
            outputs.append(F.linear(self.norm(hidden),self.embedding.weight))
            state=dict(position=state['position']+length,memory=memories,pending=pendings)
        phase_error=torch.stack([(unit(s['phase'].reshape(*s['phase'].shape[:-1],self.cfg.phase_bands,2)).square().sum(-1)-1).abs().max() for s in state['pending']]).max()
        diagnostics=dict(effective_nodes=torch.stack(effective).mean(),phase_error=phase_error,
            value_rms=torch.stack([s['value'].square().mean().sqrt() for s in state['pending']]).mean())
        return torch.cat(outputs,1),state,diagnostics


class Transformer(nn.Module):
    def __init__(self,cfg,activation_checkpoint=True):
        super().__init__();self.cfg=cfg;self.activation_checkpoint=activation_checkpoint
        self.embedding=nn.Embedding(cfg.vocab_size,cfg.width)
        self.blocks=nn.ModuleList([AttentionBlock(cfg.width,cfg.heads,cfg.ff) for _ in range(cfg.layers)])
        self.norm=nn.LayerNorm(cfg.width)

    def full_block(self,block,hidden):
        b,t,w=hidden.shape
        q,k,v=block.qkv(block.norm1(hidden)).reshape(b,t,3,block.heads,block.head_width).permute(2,0,3,1,4).unbind(0)
        positions=torch.arange(t,device=hidden.device)[None].expand(b,-1)
        q,k=block.rope(q,positions),block.rope(k,positions)
        attention=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.)
        hidden=hidden+block.out(attention.transpose(1,2).reshape(b,t,w))
        return hidden+block.ff(block.norm2(hidden)),(k,v)

    def forward(self,tokens,state=None):
        if tokens.dtype!=torch.long or tokens.ndim!=2 or not tokens.numel():raise ValueError('Expected nonempty long [batch,time]')
        if tokens.shape[1]>self.cfg.context_limit:raise ValueError('Input exceeds declared Transformer context')
        hidden=self.embedding(tokens);cache=[]
        if state is not None and len(state['kv'])!=len(self.blocks):raise ValueError('Wrong KV layer count')
        position=torch.zeros(len(tokens),device=tokens.device,dtype=torch.long) if state is None else state['position']
        for index,block in enumerate(self.blocks):
            if state is None:
                if self.activation_checkpoint and torch.is_grad_enabled():
                    hidden,kv=checkpoint(self.full_block,block,hidden,use_reentrant=False)
                else:hidden,kv=self.full_block(block,hidden)
            else:hidden,kv=block(hidden,state['kv'][index],position,self.cfg.context_limit)
            cache.append(kv)
        return F.linear(self.norm(hidden),self.embedding.weight),dict(kv=cache,position=position+tokens.shape[1]),{}


def initialize(model):
    for module in model.modules():
        if isinstance(module,(nn.Linear,nn.Embedding)):
            nn.init.normal_(module.weight,std=.02)
            if getattr(module,'bias',None) is not None:nn.init.zeros_(module.bias)
    with torch.no_grad():
        blocks=model.layers if isinstance(model,Bio) else model.blocks
        for block in blocks:
            output=block.nodes.output if isinstance(model,Bio) else block.out
            nn.init.normal_(output.weight,std=.02/math.sqrt(2*len(blocks)))
            nn.init.normal_(block.ff[2].weight,std=.02/math.sqrt(2*len(blocks)))


def configs(spec):
    cfg=Config(**spec)
    if (cfg.width//cfg.heads)%2:raise ValueError('Even head dimension required')
    with torch.device('meta'):
        target=sum(p.numel() for p in Network(cfg).parameters())
        baseline=sum(p.numel() for p in Transformer(cfg).parameters())
    ff=cfg.ff+round((target-baseline)/(cfg.layers*(2*cfg.width+1)))
    return cfg,replace(cfg,ff=ff)


def build(condition,spec,seed,activation_checkpoint=True):
    if condition not in CONDITIONS:raise ValueError(condition)
    cfg,tf_cfg=configs(spec);torch.manual_seed(seed)
    reference=Bio(cfg,activation_checkpoint);initialize(reference)
    if condition!='transformer':
        reference.cfg=replace(cfg,dynamic_addresses=condition=='bio_dynamic')
        for layer in reference.layers:layer.nodes.cfg=reference.cfg
        return reference
    torch.manual_seed(seed+1000);model=Transformer(tf_cfg,activation_checkpoint);initialize(model)
    with torch.no_grad():
        model.embedding.weight.copy_(reference.embedding.weight);model.norm.load_state_dict(reference.norm.state_dict())
        for src,dst in zip(reference.layers,model.blocks):
            dst.norm2.load_state_dict(src.norm.state_dict());shared=min(cfg.ff,tf_cfg.ff)
            dst.ff[0].weight[:shared].copy_(src.ff[0].weight[:shared]);dst.ff[0].bias[:shared].copy_(src.ff[0].bias[:shared])
            dst.ff[2].weight[:,:shared].copy_(src.ff[2].weight[:,:shared]);dst.ff[2].bias.copy_(src.ff[2].bias)
            if tf_cfg.ff>shared:dst.ff[2].weight[:,shared:].zero_()
    return model


def inspect(spec):
    cfg,tf=configs(spec)
    with torch.device('meta'):
        counts={name:sum(p.numel() for p in (Transformer(tf) if name=='transformer' else Network(cfg)).parameters()) for name in CONDITIONS}
    return dict(config=asdict(cfg),transformer_ff=tf.ff,parameters=counts,
                parameter_difference=counts['bio_dynamic']-counts['transformer'],
                fp32_parameters_gradients_adam_bytes={k:v*16 for k,v in counts.items()},
                note='Activation, temporary tensor, CUDA allocator and checkpoint staging memory require device profiling.')
