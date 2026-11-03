"""Frozen Bio/Transformer and public micro-configurations, with a balanced DS control."""
from dataclasses import asdict
import torch
from torch import nn
from transformers import Qwen3_5ForCausalLM, DeepseekV4ForCausalLM

from evaluations.tiny_arch.model import configuration
from v3.train import build_models, DEFAULTS

NAMES = ('trans_local', 'trans_full', 'v3_static', 'v3_dynamic', 'qwen35_tiny',
         'deepseek4_tiny', 'deepseek4_balanced')


class PublicModel(nn.Module):
    def __init__(self, name, seed, embedding):
        super().__init__()
        self.name = name
        cfg = configuration('deepseek4_tiny' if name == 'deepseek4_balanced' else name)
        if name == 'deepseek4_balanced':
            # Align expert rows to the native grouped-MM 16-byte stride requirement.
            cfg.hc_mult, cfg.moe_intermediate_size, cfg.q_lora_rank = 2, 20, 8
        torch.manual_seed(seed+2000)
        cls = Qwen3_5ForCausalLM if name == 'qwen35_tiny' else DeepseekV4ForCausalLM
        self.model = cls(cfg)
        with torch.no_grad():
            self.model.get_input_embeddings().weight.copy_(embedding)
            generator = torch.Generator().manual_seed(seed+19000)
            if name.startswith('deepseek'):
                for layer in self.model.model.layers:
                    gate = layer.mlp.gate
                    if hasattr(gate, 'tid2eid'):
                        gate.tid2eid.copy_(torch.rand((4096, 4), generator=generator).argsort(-1)[:, :2])

    def forward(self, tokens, state=None):
        values = tokens if state is None else torch.cat((state['tokens'], tokens), 1)
        out = self.model(values, use_cache=False, attention_mask=values.ne(0))
        aux = getattr(out, 'aux_loss', None)
        return out.logits[:, -tokens.shape[1]:], {'tokens': values}, {} if aux is None else {'native_auxiliary': aux}


def build(name, seed):
    if name not in NAMES:
        raise ValueError(name)
    models, _, _ = build_models(4096, DEFAULTS, seed)
    if name in models:
        return models[name]
    return PublicModel(name, seed, models['v3_dynamic'].embedding.weight.detach())


def metadata(name, model):
    native = getattr(model, 'model', model)
    cfg = native.config.to_dict() if hasattr(native, 'config') else asdict(model.cfg)
    total = sum(p.numel() for p in model.parameters())
    if abs(total-333136)/333136 > .001:
        raise ValueError('Total parameter match exceeds 0.1%')
    return dict(total_parameters=total, embedding_parameters=229376, configuration=cfg,
                initialization='Same random embedding per seed; native nonembedding initialization; no pretrained weights',
                auxiliary_weight=.001 if name.startswith('deepseek') else 0.,
                hyperconnection_streams=cfg.get('hc_mult'), expert_intermediate=cfg.get('moe_intermediate_size'))
