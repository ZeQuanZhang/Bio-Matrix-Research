# v0.8 本机工程实测

> 归档说明：本文保留写作时点的设计或状态。项目已停止v0.9后续训练；最新执行状态见[实验索引](../../../docs/RESULTS_INDEX.md)。公开副本不含权重与语料。

真实设备计算；无远程任务。本报告检验正确性和吞吐，不评价长期语言能力。

| 条件 | 状态 | 中位耗时 ms | 合计有效目标/秒 | 参数 | 状态字节 |
|---|---|---:|---:|---:|---:|
| kernel_n32_t16_sequential | completed | 4.09 | — | — | — |
| kernel_n32_t16_parallel | completed | 4.13 | — | — | — |
| kernel_n32_t16_triangular | completed | 7.41 | — | — | — |
| kernel_n32_t32_sequential | completed | 7.52 | — | — | — |
| kernel_n32_t32_parallel | completed | 7.99 | — | — | — |
| kernel_n32_t32_triangular | completed | 15.96 | — | — | — |
| v07_n32_sequential_c1 | completed | 1365.24 | 93.0 | 309265 | 238144 |
| v08_n32_sequential_c16 | completed | 293.21 | 434.8 | 314281 | 238144 |
| v08_n32_parallel_c16 | completed | 293.14 | 435.0 | 314281 | 238144 |
| v08_n32_triangular_c16 | completed | 328.71 | 377.7 | 314281 | 238144 |
| v08_n32_triangular_c32 | completed | 256.04 | 497.6 | 314281 | 238144 |
| transformer_n32_batched_c1 | completed | 6.26 | 20789.4 | 314800 | 459280 |
| kernel_n128_t16_sequential | completed | 6.34 | — | — | — |
| kernel_n128_t16_parallel | completed | 13.69 | — | — | — |
| kernel_n128_t16_triangular | completed | 27.47 | — | — | — |
| kernel_n128_t32_sequential | completed | 15.13 | — | — | — |
| kernel_n128_t32_parallel | completed | 31.26 | — | — | — |
| kernel_n128_t32_triangular | completed | 69.11 | — | — | — |
| v07_n128_sequential_c1 | completed | 1756.32 | 73.1 | 315409 | 853312 |
| v08_n128_sequential_c16 | completed | 553.01 | 223.0 | 320425 | 853312 |
| v08_n128_parallel_c16 | completed | 581.14 | 223.4 | 320425 | 853312 |
| v08_n128_triangular_c16 | completed | 616.36 | 207.7 | 320425 | 853312 |
| v08_n128_triangular_c32 | completed | 583.07 | 218.0 | 320425 | 853312 |
| transformer_n128_batched_c1 | completed | 5.92 | 21421.8 | 320224 | 459280 |
| kernel_n512_t16_sequential | completed | 13.65 | — | — | — |
| kernel_n512_t16_parallel | completed | 53.56 | — | — | — |
| kernel_n512_t16_triangular | completed | 109.80 | — | — | — |
| kernel_n512_t32_sequential | completed | 29.82 | — | — | — |
| kernel_n512_t32_parallel | completed | 137.12 | — | — | — |
| kernel_n512_t32_triangular | completed | 272.41 | — | — | — |
| v07_n512_sequential_c1 | completed | 3145.54 | 40.8 | 339985 | 3313984 |
| v08_n512_sequential_c16 | completed | 1495.05 | 85.2 | 345001 | 3313984 |
| v08_n512_parallel_c16 | completed | 1717.50 | 74.1 | 345001 | 3313984 |
| v08_n512_triangular_c16 | completed | 2155.83 | 58.3 | 345001 | 3313984 |
| v08_n512_triangular_c32 | completed | 2362.60 | 54.2 | 345001 | 3313984 |
| transformer_n512_batched_c1 | completed | 7.22 | 17736.7 | 345216 | 524816 |

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

完成 36/36；未执行、失败、中止见 registry.json。

## 本轮结果解读

初次运行保留3个N32/C16失败。到达一致性修复后，本目录36/36条件完成，26项回归检查通过。
| 节点 | v0.7 H32 目标/秒 | v0.8 顺序 C16 | v0.8 三角 C16 | TF | 新顺序/旧模型 | 三角/新顺序 |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 93.0 | 434.8 | 377.7 | 20789.4 | 4.68× | 0.87× |
| 128 | 73.1 | 223.0 | 207.7 | 21421.8 | 3.05× | 0.93× |
| 512 | 40.8 | 85.2 | 58.3 | 17736.7 | 2.09× | 0.68× |

本次矩阵注意力/分块候选相对v0.7的吞吐改善包含新动力学和执行方式变化。对同一C16新动力学，三角实现没有超过顺序实现。
当前d8小矩阵下，更短时间依赖没有抵消三角展开的额外计算和访存；不能用矩阵调用次数推导加速。
修复后短更新梯度有限，但内容干预显示输入到读口的作用仍弱；尚未验证长期学习质量、历史收益或规模效应。
源码、全部原始计时、初始化配置、数据内容哈希和18个工程训练快照均已保存；各规模数据哈希一致。
