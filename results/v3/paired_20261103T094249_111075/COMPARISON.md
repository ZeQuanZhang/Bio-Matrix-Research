# Bio-Matrix v3 配对实验

状态：**completed**。最终测试集未评分。

四组全部从零训练；Bio由向量节点层组成，无token注意力或KV。相同种子读取相同样本、mask与优化预算。

| 条件 | 完成种子 | QA NLL均值↓ | 首答案NLL↓ | 首token准确率↑ | 文本PPL↓ | 训练秒数均值 |
|---|---:|---:|---:|---:|---:|---:|
| trans_local | 3 | 6.92541 | 7.48887 | 0.00% | 1078.3447 | 22.11 |
| trans_full | 3 | 6.91676 | 7.48215 | 0.00% | 1056.4121 | 22.30 |
| v3_static | 3 | 6.90139 | 7.29091 | 0.00% | 916.3142 | 206.28 |
| v3_dynamic | 3 | 6.90148 | 7.28374 | 0.00% | 902.8262 | 240.37 |

`comparison.json`保留每个种子的原值、种子标准差与配对差。

Bio没有原文KV；局部Transformer窗口仅用于定义相同的远史替换范围。完整窗口Transformer保留整个采样窗口。参数接近，状态容量与FLOPs分别报告。

两Bio组具有同样的向量状态容量；静态组固定运行态key和相位，动态组联合修正key、value和相位。两组都更新value，并通过因果前缀读取块内较早写入。

主指标：答案和首答案NLL、首token准确率、替换远史后的损失变化，同时查看文本损失与计算成本。

训练预算：{"width": 56, "layers": 2, "heads": 4, "ff": 272, "data_root": "<PROJECT>/data/v06/wikitext2", "qa_root": "<PROJECT>/data/v11/squad_remote256", "seeds": "11,23,37", "cpt_tokens": 1048576, "pt_updates": 512, "batch": 4, "segments": 8, "segment_tokens": 64, "context_limit": 128, "nodes": 32, "phase_bands": 8, "data_seed": 22001, "evaluation_seed": 22091, "validation_windows": 64, "pt_eval_every": 256, "checkpoint_every": 16, "learning_rate": 0.0003, "weight_decay": 0.01, "gradient_clip": 1.0, "threads": 4}

状态表：{"trans_local_s11": "completed", "trans_full_s11": "completed", "v3_static_s11": "completed", "v3_dynamic_s11": "completed", "trans_local_s23": "completed", "trans_full_s23": "completed", "v3_static_s23": "completed", "v3_dynamic_s23": "completed", "trans_local_s37": "completed", "trans_full_s37": "completed", "v3_static_s37": "completed", "v3_dynamic_s37": "completed"}