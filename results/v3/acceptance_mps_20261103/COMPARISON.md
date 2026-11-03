# Bio-Matrix v3 配对实验

状态：**completed**。最终测试集未评分。

四组全部从零训练；Bio由向量节点层组成，无token注意力或KV。相同种子读取相同样本、mask与优化预算。

| 条件 | 完成种子 | QA NLL均值↓ | 首答案NLL↓ | 首token准确率↑ | 文本PPL↓ | 训练秒数均值 |
|---|---:|---:|---:|---:|---:|---:|
| trans_local | 2 | 8.32996 | 8.35329 | 0.00% | 4074.4159 | 0.10 |
| trans_full | 2 | 8.34892 | 8.35999 | 0.00% | 4081.1418 | 0.06 |
| v3_static | 2 | 8.33969 | 8.33181 | 0.00% | 4091.0264 | 0.57 |
| v3_dynamic | 2 | 8.33967 | 8.33171 | 0.00% | 4090.8845 | 0.61 |

`comparison.json`保留每个种子的原值、种子标准差与配对差。

Bio没有原文KV；局部Transformer窗口仅用于定义相同的远史替换范围。完整窗口Transformer保留整个采样窗口。参数接近，状态容量与FLOPs分别报告。

两Bio组具有同样的向量状态容量；静态组固定运行态key和相位，动态组联合修正key、value和相位。两组都更新value，并通过因果前缀读取块内较早写入。

主指标：答案和首答案NLL、首token准确率、替换远史后的损失变化，同时查看文本损失与计算成本。

训练预算：{"width": 56, "layers": 2, "heads": 4, "ff": 272, "data_root": "<PROJECT>/data/v06/wikitext2", "qa_root": "<PROJECT>/data/v11/squad_remote256", "seeds": "11,23", "cpt_tokens": 2048, "pt_updates": 2, "batch": 4, "segments": 8, "segment_tokens": 64, "context_limit": 128, "nodes": 32, "phase_bands": 8, "data_seed": 22001, "evaluation_seed": 22091, "validation_windows": 8, "pt_eval_every": 256, "checkpoint_every": 16, "learning_rate": 0.0003, "weight_decay": 0.01, "gradient_clip": 1.0, "threads": 4}

状态表：{"trans_local_s11": "completed", "trans_full_s11": "completed", "v3_static_s11": "completed", "v3_dynamic_s11": "completed", "trans_local_s23": "completed", "trans_full_s23": "completed", "v3_static_s23": "completed", "v3_dynamic_s23": "completed"}