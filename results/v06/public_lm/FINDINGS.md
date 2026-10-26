# v0.6 实验状态

> 归档说明：本文保留写作时点的设计或状态。项目已停止v0.9后续训练；最新执行状态见[实验索引](../../../docs/RESULTS_INDEX.md)。公开副本不含权重与语料。

状态登记包括未执行、运行中、已完成、失败、中止。

| 条件 | 状态 | test BPE PPL | 历史收益 ΔNLL | 无关历史代价 ΔNLL |
|---|---|---:|---:|---:|
| core_n32_message_s11 | completed | 412.470 | 0.0020 | -0.0014 |
| core_n32_independent_s11 | completed | 412.911 | -0.0039 | 0.0033 |
| core_n32_full_s11 | completed | 412.641 | -0.0035 | 0.0029 |
| core_n32_transformer_s11 | completed | 529.708 | -0.0090 | 0.0059 |
| core_n128_message_s11 | completed | 413.094 | -0.0002 | 0.0005 |
| core_n128_independent_s11 | completed | 411.667 | -0.0004 | -0.0015 |
| core_n128_full_s11 | completed | 411.541 | -0.0002 | -0.0015 |
| core_n128_transformer_s11 | completed | 529.708 | -0.0090 | 0.0059 |
| core_n512_message_s11 | aborted | — | — | — |
| core_n512_independent_s11 | not_executed | — | — | — |
| core_n512_full_s11 | not_executed | — | — | — |
| core_n512_transformer_s11 | not_executed | — | — | — |
| core_n32_message_s23 | not_executed | — | — | — |
| core_n32_independent_s23 | not_executed | — | — | — |
| core_n32_full_s23 | not_executed | — | — | — |
| core_n32_transformer_s23 | not_executed | — | — | — |
| core_n128_message_s23 | not_executed | — | — | — |
| core_n128_independent_s23 | not_executed | — | — | — |
| core_n128_full_s23 | not_executed | — | — | — |
| core_n128_transformer_s23 | not_executed | — | — | — |
| core_n512_message_s23 | not_executed | — | — | — |
| core_n512_independent_s23 | not_executed | — | — | — |
| core_n512_full_s23 | not_executed | — | — | — |
| core_n512_transformer_s23 | not_executed | — | — | — |
| core_n32_message_s37 | not_executed | — | — | — |
| core_n32_independent_s37 | not_executed | — | — | — |
| core_n32_full_s37 | not_executed | — | — | — |
| core_n32_transformer_s37 | not_executed | — | — | — |
| core_n128_message_s37 | not_executed | — | — | — |
| core_n128_independent_s37 | not_executed | — | — | — |
| core_n128_full_s37 | not_executed | — | — | — |
| core_n128_transformer_s37 | not_executed | — | — | — |
| core_n512_message_s37 | not_executed | — | — | — |
| core_n512_independent_s37 | not_executed | — | — | — |
| core_n512_full_s37 | not_executed | — | — | — |
| core_n512_transformer_s37 | not_executed | — | — | — |

负的 full_minus_baseline_nll 表示 full 较好。原始窗口差值见 paired_differences.json；重复种子单独计数。
PPL 为自训练 byte-BPE 的窗口指标；计算预算和状态容量见每组 ledger。
同步程度只作诊断，不作为成功标准。短程 smoke/preflight 不支持能力结论。
