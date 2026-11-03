# 实验记录索引

更新时点：2026-11-03。保留完成、失败、中止和未执行条件。下表优先引用各阶段最后保存的结果；工程检查与能力训练分开解读。

| 阶段 | 主要记录 | 执行口径 |
|---|---|---|
| 首轮中文模型 | [报告](../results/report.md)、[汇总](../results/summary.json) | 9次训练；三种子配对 |
| 独立状态实验 | [FINDINGS](../matrix_lab/docs/FINDINGS.md)、[总汇总](../results/matrix_all_summary.json) | 54次状态/频率实验；另列优化器实验 |
| v0.3 | [发现](V03_FINDINGS.md)、[数值](../results/v03/summary.json)、[条件登记](../results/v03/conditions.json) | 81个已登记网络条件完成；追加工作未全执行 |
| v0.4 | [报告](../results/v04/continuous/FINDINGS.md)、[状态](../results/v04/continuous/summary.json) | 7完成、1中止、82未执行 |
| v0.5 | [数据审计](V05_DATA_AUDIT.md)、[有效正式队列](../results/v05/verified_streams/summary.json) | 正式144条件未执行；保留旧草案 |
| v0.6 | [报告](../results/v06/public_lm/FINDINGS.md)、[汇总](../results/v06/public_lm/summary.json) | 8完成、1中止、27未执行 |
| v0.7 | [取回后分析](../results/v07/analysis_20261024/REASSESSMENT.md)、[43组结果](../results/v07/analysis_20261024/full_result_summary.json) | 远程45标签43完成、2中止；本机空计划不作正式结果 |
| v0.8 | [修复后的工程测速](../results/v08/mps_20261024_arrival_gate/REPORT.md)、[设计](V08_IMPROVEMENT_PLAN.md) | 首轮失败与修复轮分开；长训部分中止 |
| v0.9 | [四模型比较](../results/v09/final_checks/20261025_n32_all_models/REPORT.md)、[数值](../results/v09/final_checks/20261025_n32_all_models/analysis.json) | N32/seed11固定12M last |
| v0.9停止 | [队列登记](../results/v09/comparison_no_v08/registry.json)、[停止决定](../results/v09/comparison_no_v08/decisions/20261025_stop_v09/README.md) | 4完成、1中止、25未执行 |

## 数值记录的层级

v0.9的各完成运行保留`result.json`、`protocol.json`、逐次验证曲线、暖评估汇总和按文档统计的JSONL。训练曲线是同一训练的连续观察，不能当成许多独立种子。文档bootstrap区间只反映验证样本变化。

首轮中文模型、合成任务和WikiText使用不同任务、分词器、预算与评分方法；跨阶段PPL或准确率不能直接排名。v0.7多用验证选点，v0.9主比较使用固定预算末尾last，比较前应查看对应协议。

## 当前未完成内容

N128 dense的3,099,250目标检查点属于中止条件，没有完成12M或最终暖评估。N512和更多v0.9训练种子未执行。大batch、融合核、混合精度、无历史Bio重训、新query读取与写地址学习规则等，仍是方案或诊断建议。

## 原始与派生材料

本公开副本保留选定数值证据和原始结果文件的SHA256，未包含权重、语料、token序列及整棵机器备份。JSON中过时的机器路径已净化，验证JSONL中的完整sample元数据收缩为其哈希。详见[发布范围](../PUBLICATION_NOTES.md)及`provenance/EXPORT_MANIFEST.json`。


## 后续版本

| 版本 | 记录 | 执行状态 |
|---|---|---|
| V1.0 | [分析](../results/v1/pilot_20261026T172837_376079/analysis/review_20261026/REPORT.md) | 本机pilot归档 |
| V1.1 | [分析](../results/v11/pilot_20261026T194141_722158/analysis/review_completed/REPORT.md) | 本机pilot归档 |
| V2 | [三种子完整分析](../results/v2/paired_20261103T003337_839264/analysis/review_20261103/REPORT.md) | 12条运行全部完成 |
| V3 原型 | [首次工程测速](../results/v3/engineering_20261103/REPORT.md) | 显式前缀原型工程完成，约20倍耗时 |
| V3 双重计算 | [工程测速](../results/v3/engineering_20261103_dual/REPORT.md) | 原生MPS工程完成，约11倍耗时 |
| V3 有限数据验收 | [短程配对](../results/v3/acceptance_mps_20261103/COMPARISON.md) | 四条件×两种子，每条3次更新，完成 |
| V3 完整默认训练 | [完整分析](../results/v3/paired_20261103T094249_111075/analysis/review_20261103/REPORT.md) | completed：四条件×三种子12条，每条1024步；最终测试未执行 |
| V3 LoCoMo | [完整分析](../results/locomo/v3_20261103_seed11/analysis/REPORT.md)、[统计](../results/locomo/v3_20261103_seed11/analysis/statistics.json) | completed：seed11，10/10条件；1540题主F1及1527题证据直给F1全0，生成退化；未追加训练 |
| V3规模验证工程 | [参数配置](../results/scale/capacity_20261103/capacity.json)、[14项验收](../results/scale/acceptance_0133b_final_20261103/verification.json)、[原生MPS有限更新](../results/scale/mps_0133b_delivery_20261103/profile.json) | completed：0.133B三个模型各3次合成更新，长度32；10B正式训练、完整2048 CUDA测速、LoCoMo均not_executed |
| V3本地Qwen/DeepSeek对照 | [分析](../results/tiny_arch/paired_20261103_local333k/analysis/REPORT.md)、[数值](../results/tiny_arch/paired_20261103_local333k/analysis/statistics.json)、[实际生成](../results/tiny_arch/paired_20261103_local333k/analysis/generation_summary.json) | completed：约33万参数，新增2架构×3种子；6144步逐样本匹配；六组seed11生成EM/F1全0 |
| V3修正版微型架构对照 | [完整结果与图](../results/repaired/local333k_corrected_20261103/REPORT.md)、[协议](../evaluations/repaired/PROTOCOL.md)、[配对统计](../results/repaired/local333k_corrected_20261103/summary.json)、[执行状态](../results/repaired/local333k_corrected_20261103/registry.json) | completed：91训练阶段、21保留测试；Bio动态/静态CPT PPL334.44/335.04、生成F1均0.40%；动态增量弱，问答坍缩；三项补充诊断与图完成 |
| V3 公开标准评测 | [冻结协议](../evaluations/standard/protocol.json)、[验收结果](../results/standard_eval/verification_20261104.json)、[PyCharm入口](../run_biomatrix_standard_eval.py) | 15项MPS短验收和3项数值自检通过；正式全量评分和31M评分均not_executed |

历史V3短程验收用于检查执行、配对和保存。旧V3默认队列未评分最终测试；随后独立修正版已完成保留测试，两轮口径分别记录。

V3完整结果：相同目标预算下动态Bio文本PPL降低约16.3%，答案NLL小幅降低；四组首答案token命中均为0%，正确远史替换的影响很小，训练耗时约10.9倍。详见完整分析，保留工程短测与正式训练两种口径。

V3标准评测固定使用 [EleutherAI lm-evaluation-harness v0.4.13](https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.13)。内置任务为 WikiText、LAMBADA OpenAI、PIQA、ARC-Easy、HellaSwag；沿用上游模板、任务默认划分和指标，冻结任务代码哈希及数据修订。统一0-shot、512 token上下文、不加聊天模板，正式评测覆盖默认划分的全部样本。原问答和历史替换结果继续作为机制诊断单独报告。31M新语料还需执行跨基准重合检查。

在本机 PyCharm 选择 **Bio-Matrix 标准评测**，以原生MPS运行，每秒原位显示任务进度。当前入口读取已完成的33万参数修正版预训练检查点，可用 `--reference` 指定复跑目录。公开副本保留代码与数值验收记录；使用入口前需先复现本地权重和分词数据。模型包装只转交因果 logits，似然、滚动窗口和聚合均由上游评分器执行。

```bash
uv venv --python 3.12 results/standard_eval/runtime
uv pip install --python results/standard_eval/runtime/bin/python -r evaluations/standard/requirements.txt
results/standard_eval/runtime/bin/python run_biomatrix_standard_eval.py --smoke
```

`--smoke` 每任务取两条真实样本，仅验收程序。去掉该参数后评测完整默认划分，结果写入独立目录；`--seed 11/23/37` 分别选择一个旧实验种子。首次需下载公开评测数据，已缓存的固定版本可以离线读取，程序会核对实际缓存修订。报告保留各任务原生指标和样本数，不汇成自定义总分。
