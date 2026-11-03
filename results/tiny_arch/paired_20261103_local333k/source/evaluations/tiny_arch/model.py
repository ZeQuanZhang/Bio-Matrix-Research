"""Public Transformers implementations at the original 333K total budget."""
from types import SimpleNamespace

import torch
from torch import nn
from transformers import Qwen3_5TextConfig,Qwen3_5ForCausalLM,DeepseekV4Config,DeepseekV4ForCausalLM

from v3.train import build_models

CONDITIONS=('qwen35_tiny','deepseek4_tiny')


def configuration(name):
    common=dict(vocab_size=4096,hidden_size=56,num_hidden_layers=4,max_position_embeddings=512,
                tie_word_embeddings=True,pad_token_id=0,bos_token_id=1,eos_token_id=2)
    if name=='qwen35_tiny':
        cfg=Qwen3_5TextConfig(**common,intermediate_size=81,num_attention_heads=4,num_key_value_heads=2,head_dim=14,
            layer_types=['linear_attention']*3+['full_attention'],linear_conv_kernel_dim=4,
            linear_key_head_dim=8,linear_value_head_dim=14,linear_num_key_heads=2,linear_num_value_heads=4,
            rope_parameters=dict(rope_type='default',rope_theta=10000.,partial_rotary_factor=1.,
                                 mrope_section=[2,2,3],mrope_interleaved=True))
    elif name=='deepseek4_tiny':
        cfg=DeepseekV4Config(**common,moe_intermediate_size=8,num_attention_heads=4,num_key_value_heads=1,
            head_dim=16,q_lora_rank=15,o_groups=2,o_lora_rank=15,partial_rotary_factor=.25,
            num_experts_per_tok=2,n_routed_experts=4,n_shared_experts=1,num_hash_layers=1,
            compress_ratios=[0,0,4,128],hc_mult=4,hc_sinkhorn_iters=20,sliding_window=128,
            index_n_heads=2,index_head_dim=8,index_topk=512,num_nextn_predict_layers=0,
            output_router_logits=True,router_aux_loss_coef=.001)
    else:raise ValueError(name)
    cfg._attn_implementation='eager';return cfg


class Model(nn.Module):
    def __init__(self,name,seed,training):
        super().__init__();self.name=name;cfg=configuration(name)
        torch.manual_seed(seed+2000)
        self.model=(Qwen3_5ForCausalLM if name=='qwen35_tiny' else DeepseekV4ForCausalLM)(cfg)
        old,_,_=build_models(4096,training,seed)
        with torch.no_grad():self.model.get_input_embeddings().weight.copy_(old['v3_dynamic'].embedding.weight)
        del old
        if name=='deepseek4_tiny':
            generator=torch.Generator().manual_seed(seed+19000)
            with torch.no_grad():
                for layer in self.model.model.layers:
                    gate=layer.mlp.gate
                    if hasattr(gate,'tid2eid'):
                        values=torch.rand((4096,cfg.n_routed_experts),generator=generator).argsort(-1)[:,:cfg.num_experts_per_tok]
                        gate.tid2eid.copy_(values)
        self.cfg=SimpleNamespace(vocab_size=4096,context_limit=512,layers=4)

    def forward(self,tokens,state=None):
        # Recompute one complete prefix for validation; no mutable HF cache crosses questions.
        values=tokens if state is None else torch.cat((state['tokens'],tokens),1)
        result=self.model(values,use_cache=False,attention_mask=values.ne(0))
        auxiliary=getattr(result,'aux_loss',None)
        diag={} if auxiliary is None else dict(native_auxiliary=auxiliary)
        return result.logits[:,-tokens.shape[1]:],dict(tokens=values,position=torch.full((len(values),),values.shape[1],dtype=torch.long,device=values.device)),diag

    def auxiliary_weight(self):return .001 if self.name=='deepseek4_tiny' else 0.


def metadata(model):
    total=sum(p.numel() for p in model.parameters());embedding=model.model.get_input_embeddings().weight.numel()
    result=dict(total_parameters=total,embedding_parameters=embedding,nonembedding_parameters=total-embedding,
                configuration=model.model.config.to_dict(),implementation=type(model.model).__name__,
                initialization='Public implementation defaults; same initial embedding as frozen V3 seed. No pretrained weights.',
                common_overrides='4096-token original tokenizer and tied embedding/output, four reduced-width layers, FP32 eager PyTorch.')
    if model.name=='deepseek4_tiny':
        routed=sum(p.numel() for layer in model.model.model.layers for p in layer.mlp.experts.parameters())
        result.update(routed_expert_parameters=routed,token_active_parameter_estimate=total-routed*(1-.5),
            routing='Four routed experts, top2, one shared; deterministic distinct-expert hash initialization; native .001 balance auxiliary.',
            indexer='At 512 tokens, topk512 retains every available CSA entry; no sparse selection benefit or indexer distillation is tested.',
            mtp='Not instantiated by this public implementation; next-token objective shared with V3.')
    else:result['token_active_parameter_estimate']=total
    return result
