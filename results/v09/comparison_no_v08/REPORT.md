# Bio-Matrix v0.9 本机对比实验

> 归档说明：本文保留写作时点的设计或状态。项目已停止v0.9后续训练；最新执行状态见[实验索引](../../../docs/RESULTS_INDEX.md)。公开副本不含权重与语料。

状态：aborted；独立运行 4/30；配对格 6/36。

独立训练有效目标：51,099,250/360,000,000。

content_only 每种子只训练一次；N128/N512 的对应格引用 N32，不算独立重复。

| seed | N | 模型 | 状态 | 有效目标 | 训练 | 评估 | 结果目录 |
|---|---|---|---|---:|---|---|---|
| 11 | 32 | content_only | completed | 12,000,000 | completed | completed | [content_only_n32_s11](content_only_n32_s11) |
| 11 | 32 | v09_dense | completed | 12,000,000 | completed | completed | [v09_dense_n32_s11](v09_dense_n32_s11) |
| 11 | 32 | v09_sparse50 | completed | 12,000,000 | completed | completed | [v09_sparse50_n32_s11](v09_sparse50_n32_s11) |
| 11 | 32 | transformer | completed | 12,000,000 | completed | completed | [transformer_n32_s11](transformer_n32_s11) |
| 11 | 128 | content_only（引用 N32） | completed | 12,000,000 | completed | completed | [content_only_n32_s11](content_only_n32_s11) |
| 11 | 128 | v09_dense | aborted | 3,099,250 | aborted | not_executed | [v09_dense_n128_s11](v09_dense_n128_s11) |
| 11 | 128 | v09_sparse50 | not_executed | 0 | not_executed | not_executed | v09_sparse50_n128_s11（本地留存：`v09_sparse50_n128_s11/`） |
| 11 | 128 | transformer | not_executed | 0 | not_executed | not_executed | transformer_n128_s11（本地留存：`transformer_n128_s11/`） |
| 11 | 512 | content_only（引用 N32） | completed | 12,000,000 | completed | completed | [content_only_n32_s11](content_only_n32_s11) |
| 11 | 512 | v09_dense | not_executed | 0 | not_executed | not_executed | v09_dense_n512_s11（本地留存：`v09_dense_n512_s11/`） |
| 11 | 512 | v09_sparse50 | not_executed | 0 | not_executed | not_executed | v09_sparse50_n512_s11（本地留存：`v09_sparse50_n512_s11/`） |
| 11 | 512 | transformer | not_executed | 0 | not_executed | not_executed | transformer_n512_s11（本地留存：`transformer_n512_s11/`） |
| 23 | 32 | content_only | not_executed | 0 | not_executed | not_executed | content_only_n32_s23（本地留存：`content_only_n32_s23/`） |
| 23 | 32 | v09_dense | not_executed | 0 | not_executed | not_executed | v09_dense_n32_s23（本地留存：`v09_dense_n32_s23/`） |
| 23 | 32 | v09_sparse50 | not_executed | 0 | not_executed | not_executed | v09_sparse50_n32_s23（本地留存：`v09_sparse50_n32_s23/`） |
| 23 | 32 | transformer | not_executed | 0 | not_executed | not_executed | transformer_n32_s23（本地留存：`transformer_n32_s23/`） |
| 23 | 128 | content_only（引用 N32） | not_executed | 0 | not_executed | not_executed | content_only_n32_s23（本地留存：`content_only_n32_s23/`） |
| 23 | 128 | v09_dense | not_executed | 0 | not_executed | not_executed | v09_dense_n128_s23（本地留存：`v09_dense_n128_s23/`） |
| 23 | 128 | v09_sparse50 | not_executed | 0 | not_executed | not_executed | v09_sparse50_n128_s23（本地留存：`v09_sparse50_n128_s23/`） |
| 23 | 128 | transformer | not_executed | 0 | not_executed | not_executed | transformer_n128_s23（本地留存：`transformer_n128_s23/`） |
| 23 | 512 | content_only（引用 N32） | not_executed | 0 | not_executed | not_executed | content_only_n32_s23（本地留存：`content_only_n32_s23/`） |
| 23 | 512 | v09_dense | not_executed | 0 | not_executed | not_executed | v09_dense_n512_s23（本地留存：`v09_dense_n512_s23/`） |
| 23 | 512 | v09_sparse50 | not_executed | 0 | not_executed | not_executed | v09_sparse50_n512_s23（本地留存：`v09_sparse50_n512_s23/`） |
| 23 | 512 | transformer | not_executed | 0 | not_executed | not_executed | transformer_n512_s23（本地留存：`transformer_n512_s23/`） |
| 37 | 32 | content_only | not_executed | 0 | not_executed | not_executed | content_only_n32_s37（本地留存：`content_only_n32_s37/`） |
| 37 | 32 | v09_dense | not_executed | 0 | not_executed | not_executed | v09_dense_n32_s37（本地留存：`v09_dense_n32_s37/`） |
| 37 | 32 | v09_sparse50 | not_executed | 0 | not_executed | not_executed | v09_sparse50_n32_s37（本地留存：`v09_sparse50_n32_s37/`） |
| 37 | 32 | transformer | not_executed | 0 | not_executed | not_executed | transformer_n32_s37（本地留存：`transformer_n32_s37/`） |
| 37 | 128 | content_only（引用 N32） | not_executed | 0 | not_executed | not_executed | content_only_n32_s37（本地留存：`content_only_n32_s37/`） |
| 37 | 128 | v09_dense | not_executed | 0 | not_executed | not_executed | v09_dense_n128_s37（本地留存：`v09_dense_n128_s37/`） |
| 37 | 128 | v09_sparse50 | not_executed | 0 | not_executed | not_executed | v09_sparse50_n128_s37（本地留存：`v09_sparse50_n128_s37/`） |
| 37 | 128 | transformer | not_executed | 0 | not_executed | not_executed | transformer_n128_s37（本地留存：`transformer_n128_s37/`） |
| 37 | 512 | content_only（引用 N32） | not_executed | 0 | not_executed | not_executed | content_only_n32_s37（本地留存：`content_only_n32_s37/`） |
| 37 | 512 | v09_dense | not_executed | 0 | not_executed | not_executed | v09_dense_n512_s37（本地留存：`v09_dense_n512_s37/`） |
| 37 | 512 | v09_sparse50 | not_executed | 0 | not_executed | not_executed | v09_sparse50_n512_s37（本地留存：`v09_sparse50_n512_s37/`） |
| 37 | 512 | transformer | not_executed | 0 | not_executed | not_executed | transformer_n512_s37（本地留存：`transformer_n512_s37/`） |

全部训练/评估日志、检查点、失败和中止记录保留。失败不自动重试；使用 --retry-failed 显式重试。
训练与评估均为本机 MPS 顺序执行；本入口不连接远端。
WikiText-2 结果属于已使用语料的探索性比较；此队列不读取 test。
