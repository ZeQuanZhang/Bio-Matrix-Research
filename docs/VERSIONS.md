# Bio-Matrix 版本编号与入口

2026-11-03 起统一使用以下展示编号。历史目录、Python 包、结果路径和冻结协议中的标识保留，以维持源码哈希、导入路径和检查点的可复现性。

| 展示版本 | 兼容路径 | 核心探索 | 入口／记录 |
|---|---|---|---|
| V0.1 原始构想 | `docs/sources/V01_ORIGINAL_RESEARCH_IDEA.md` | 有限复杂动态节点 | 原始构想稿，未补造对应正式训练版本 |
| 首轮中文 Bio-LLM | 根目录 | Transformer 中的八状态动力单元 | `run_experiment.py` |
| matrix_lab | `matrix_lab/` | 独立矩阵状态实验 | `matrix_lab/README.md`；未补命名为 V0.2 |
| V0.3 | `v03/` | 网络规模、振荡与耦合对照 | `docs/V03_FINDINGS.md` |
| V0.4 | `v04/` | 潜在情景与动态矩阵 | `run_biomatrix_v04.py` |
| V0.5 | `v05/` | 保持、遗忘与数据审计 | `run_biomatrix_v05.py` |
| V0.6 | `v06/` | 球壳、脉冲与公开语言数据 | `run_biomatrix_v06.py` |
| V0.7 | `v07/` | 连续状态与局部反馈修复 | `run_biomatrix_v07.py`；历史 A800 已过期 |
| V0.8 | `v08/` | 分块与矩阵注意力 | `run_biomatrix_v08.py`；历史训练已中止 |
| V0.9 | `v09/` | 内容通路、节点组和配对比较 | `run_biomatrix_v09_compare.py`；队列已停止 |
| V1.0 | `v1/` | Transformer 内集成完整 Bio 机制 | `run_biomatrix_v1.py` |
| V1.1 | `v11/` | 远史问答、信用与训练修正 | `run_biomatrix_v11.py` |
| V2 | `v2/` | Transformer 加块并行矩阵记忆 | `run_biomatrix_v2_console.py`；四条件×三种子已完成 |
| **V3** | `v3/` | **独立向量节点、相位路由、因果双重计算** | **`run_biomatrix_v3.py`**；12条训练完成；`run_biomatrix_v3_locomo.py`的10条件冻结评估完成，生成F1全0 |

`v11` 表示 V1.1，`v09` 表示 V0.9。路径属于实现兼容标识，README、设计文档和新发布说明使用上表展示编号。

后续每个版本交付包含代码、实现边界、实际检查、执行状态与 GitHub 同步。历史数据和二进制保留在原档案；公开副本遵循 `PUBLICATION_NOTES.md`。
