# 实验记录索引

发布时点：2026-10-26。保留完成、失败、中止和未执行条件。下表优先引用各阶段最后保存的结果；工程检查与能力训练分开解读。

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
