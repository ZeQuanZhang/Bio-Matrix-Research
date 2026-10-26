# v0.8 本机工程实测

> 归档说明：本文保留写作时点的设计或状态。项目已停止v0.9后续训练；最新执行状态见[实验索引](../../../docs/RESULTS_INDEX.md)。公开副本不含权重与语料。

真实设备计算；无远程任务。本报告检验正确性和吞吐，不评价长期语言能力。

| 条件 | 状态 | 中位耗时 ms | 合计有效目标/秒 | 参数 | 状态字节 |
|---|---|---:|---:|---:|---:|
| kernel_n32_t16_sequential | completed | 4.17 | — | — | — |
| kernel_n32_t16_parallel | completed | 4.22 | — | — | — |
| kernel_n32_t16_triangular | completed | 6.81 | — | — | — |
| kernel_n32_t32_sequential | completed | 7.37 | — | — | — |
| kernel_n32_t32_parallel | completed | 9.00 | — | — | — |
| kernel_n32_t32_triangular | completed | 18.07 | — | — | — |
| v07_n32_sequential_c1 | completed | 1442.16 | 84.5 | 309265 | 238144 |
| v08_n32_sequential_c16 | failed | — | — | — | — |
| v08_n32_parallel_c16 | failed | — | — | — | — |
| v08_n32_triangular_c16 | failed | — | — | — | — |
| v08_n32_triangular_c32 | completed | 266.01 | 478.8 | 314281 | 238144 |
| transformer_n32_batched_c1 | completed | 8.81 | 2153.5 | 314800 | 459280 |
| kernel_n128_t16_sequential | completed | 5.70 | — | — | — |
| kernel_n128_t16_parallel | completed | 13.82 | — | — | — |
| kernel_n128_t16_triangular | completed | 28.83 | — | — | — |
| kernel_n128_t32_sequential | completed | 12.45 | — | — | — |
| kernel_n128_t32_parallel | completed | 33.24 | — | — | — |
| kernel_n128_t32_triangular | completed | 72.42 | — | — | — |
| v07_n128_sequential_c1 | completed | 1756.88 | 73.0 | 315409 | 853312 |
| v08_n128_sequential_c16 | completed | 544.70 | 234.7 | 320425 | 853312 |
| v08_n128_parallel_c16 | completed | 575.21 | 221.1 | 320425 | 853312 |
| v08_n128_triangular_c16 | completed | 642.66 | 198.7 | 320425 | 853312 |
| v08_n128_triangular_c32 | completed | 608.49 | 210.4 | 320425 | 853312 |
| transformer_n128_batched_c1 | completed | 6.42 | 12081.1 | 320224 | 459280 |
| kernel_n512_t16_sequential | completed | 15.27 | — | — | — |
| kernel_n512_t16_parallel | completed | 59.91 | — | — | — |
| kernel_n512_t16_triangular | completed | 114.20 | — | — | — |
| kernel_n512_t32_sequential | completed | 32.02 | — | — | — |
| kernel_n512_t32_parallel | completed | 139.79 | — | — | — |
| kernel_n512_t32_triangular | completed | 295.19 | — | — | — |
| v07_n512_sequential_c1 | completed | 3169.40 | 39.1 | 339985 | 3313984 |
| v08_n512_sequential_c16 | completed | 1600.43 | 80.4 | 345001 | 3313984 |
| v08_n512_parallel_c16 | completed | 1634.83 | 76.4 | 345001 | 3313984 |
| v08_n512_triangular_c16 | completed | 2216.03 | 57.2 | 345001 | 3313984 |
| v08_n512_triangular_c32 | completed | 2181.38 | 59.1 | 345001 | 3313984 |
| transformer_n512_batched_c1 | completed | 8.01 | 5369.5 | 345216 | 524816 |

## 可比较范围

v0.8 同 C 的 sequential/parallel/triangular 是同一动力学的计算实现对照。
C16 与 C32 的反馈时序不同；v0.7 与 v0.8 也属于不同模型，吞吐比不能充当保函数提速或质量结论。
各模型使用相同数据、batch、序列长度、FP32、优化器和完整训练更新计时。参数及状态容量照实报告，未声称状态相等。
预热独立保存；phase_seconds 是额外同步的单独剖析，不混入主计时。MPS 内存是快照，不是独占峰值。
设备驻留输入的计时包括前向、CE、反向、裁剪、AdamW、状态 detach/信用失效，不含下载、数据采样、保存和进度显示。
v0.7 原有诊断保留，故另提供纯递推核对照。未关闭旧诊断的整网比不能精确归因于架构本身。

本次 v0.7 对照使用 H32 信用槽，与旧正式 H8 不同；每个条件的完整配置见 result.json。
尚未实现保持同一块边界的逐 token 生成 API；不足 C 的 forward 会提前提交边界。
三个计时样本是探索实测，固定执行顺序可能受 Mac 热状态影响。

完成 33/36；未执行、失败、中止见 registry.json。
