# Bio-Matrix v2 配对实验

状态：**completed**。最终测试集未评分。

全部主干与新增参数参与训练；同种子读取相同样本、mask与优化预算。

| 条件 | 完成种子 | QA NLL均值↓ | 首答案NLL↓ | 首token准确率↑ | 文本PPL↓ | 训练秒数均值 |
|---|---:|---:|---:|---:|---:|---:|
| trans_local | 3 | 5.65849 | 6.24343 | 1.04% | 161.4957 | 11.28 |
| trans_full | 3 | 5.65573 | 6.21033 | 1.56% | 158.5254 | 11.80 |
| v2_static | 3 | 5.67663 | 6.24815 | 3.12% | 152.5316 | 100.30 |
| v2_dynamic | 3 | 5.67659 | 6.24808 | 3.12% | 152.5297 | 105.75 |

`comparison.json`保留每个种子的原值、种子标准差与配对差。

局部Transformer窗口与v2相同；完整窗口Transformer保留整个采样窗口，是显式历史正对照，其KV预算更大。

只有v2静态地址/动态地址两组使用相同完整记忆容量；差别为块边界是否更新运行时地址。

主指标：答案和首答案NLL、首token准确率、替换远史后的损失变化，同时查看文本损失与计算成本。

训练预算：{"backbone": "<PROJECT>/results/v09/comparison_no_v08/transformer_n32_s11/last.pt", "data_root": "<PROJECT>/data/v06/wikitext2", "qa_root": "<PROJECT>/data/v11/squad_remote256", "seeds": "11,23,37", "cpt_tokens": 131072, "pt_updates": 512, "batch": 4, "segments": 8, "segment_tokens": 64, "context_limit": 128, "nodes": 32, "phase_bands": 8, "data_seed": 22001, "evaluation_seed": 22091, "validation_windows": 64, "pt_eval_every": 256, "checkpoint_every": 16, "learning_rate": 0.0003, "weight_decay": 0.01, "gradient_clip": 1.0, "threads": 4}

状态表：{"trans_local_s11": "completed", "trans_full_s11": "completed", "v2_static_s11": "completed", "v2_dynamic_s11": "completed", "trans_local_s23": "completed", "trans_full_s23": "completed", "v2_static_s23": "completed", "v2_dynamic_s23": "completed", "trans_local_s37": "completed", "trans_full_s37": "completed", "v2_static_s37": "completed", "v2_dynamic_s37": "completed"}