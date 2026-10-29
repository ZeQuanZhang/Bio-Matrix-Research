# Bio-Matrix Research

Bio-Matrix Research 记录了把神经网络节点扩展为有限矩阵状态系统的构想、实现与实验。代码探索多时间尺度状态、节点间传播和局部反馈，研究这些机制能否保存并读取对预测有用的历史信息。

这份档案面向研究状态模型、记忆机制和实验归因的读者。仓库提供各版源码、配置、研究笔记与选定数值证据，可以沿着同一项假设查看实现、对照方法和结果。

v0.9 后续训练已于 **2026-10-25 停止**。已完成的 N32/seed11 四模型配对中，参数接近的 Transformer 在预测质量、远史利用和训练速度上均领先 Bio-Matrix。失败、修复、中止和未执行条件一并保留。

研究者：[ZeQuanZhang](https://github.com/ZeQuanZhang)。构想、实现和分析在与 AI 助手的协作中逐步迭代；公开整理日期为 2026-10-26。

## 阅读与使用

- 理解研究目标：从[构想原文](docs/sources/BIOMATRIX_FULL_IDEA.md)进入，再读[研究过程与取舍](docs/RESEARCH_JOURNAL.md)。
- 复核实验结论：通过[结果索引](docs/RESULTS_INDEX.md)定位报告、数值文件和执行状态。
- 研究实现：根目录保留首轮中文模型，`matrix_lab/` 提供独立状态实验，`v03/`—`v09/` 保存后续实现。各版代码与当时的任务、配置和对照一起阅读。
- 设计新的对照：参考[v0.9 比较协议](docs/V09_COMPARISON_PROTOCOL.md)和[机制诊断](docs/V09_MECHANISM_DIAGNOSIS.md)，为数据、预算、运行态和干预条件建立明确口径。

公开副本包含选定报告、曲线、数值记录和状态登记。训练语料、编码后数据、二进制权重与完整运行检查点保留在原实验档案中，新的模型实验需要另行准备这些条件。

## 快速查看已保存的结果

首次阅读只需 Git 和 Python 3。以下命令读取仓库内的 JSON，不安装训练依赖，也不启动模型：

```bash
git clone https://github.com/ZeQuanZhang/Bio-Matrix-Research.git
cd Bio-Matrix-Research
python3 - <<'PY'
import json
from pathlib import Path

root = Path("results/v09")
status = json.loads((root / "comparison_no_v08/summary.json").read_text(encoding="utf-8"))
analysis = json.loads(
    (root / "final_checks/20261025_n32_all_models/analysis.json").read_text(encoding="utf-8")
)
print("queue:", status["status"])
print("completed independent jobs:", status["completed_jobs"], "/", status["independent_jobs"])
for name, row in analysis["records"].items():
    print(f"{name}: warm PPL={row['warm_ppl']:.3f}, core minutes={row['train_core_seconds'] / 60:.2f}")
PY
```

输出应包含队列状态 `aborted`、完成独立训练 `4 / 30`，以及四个模型的暖历史 PPL 和核心更新分钟数。命令输出后退出，可再次运行核对记录。Windows PowerShell 可先执行相同的 `git clone` 和 `cd` 命令，再将 Python 部分保存为 `read_results.py`，运行 `python read_results.py`。

结果的解释和对照图见[四模型最终比较](results/v09/final_checks/20261025_n32_all_models/REPORT.md)。该报告保留分析时点的 N128 状态；最后的执行状态以[队列汇总](results/v09/comparison_no_v08/summary.json)和[停止决定](results/v09/comparison_no_v08/decisions/20261025_stop_v09/README.md)为准。

## v0.9 四模型结果

主比较采用 N32、训练种子 11、同一训练流、12M 有效预测目标、FP32 和本机 MPS，取固定预算末尾的 `last` 检查点。数据为 WikiText-2 官方切分，使用仅在训练集拟合的 4096 词表 BPE。暖评估共享 64 个窗口、30 篇文档和 16,384 个不重复评分目标。

| 模型 | 参数 | 冷验证 NLL ↓ | 暖历史 NLL ↓ | 暖历史 PPL ↓ | 核心更新分钟 |
|---|---:|---:|---:|---:|---:|
| content_only | 303,680 | 5.036041 | 5.051925 | 156.323 | 4.91 |
| Bio dense N32 | 315,817 | 4.939364 | 4.913514 | 136.117 | 398.80 |
| Bio sparse50 N32 | 315,817 | 4.937495 | 4.912728 | 136.010 | 427.61 |
| Transformer | 316,608 | 4.799269 | 4.727059 | 112.963 | 16.45 |

Transformer 的暖 PPL 比 dense 低约 **17.01%**，dense 的核心更新耗时约为其 **24.25 倍**。计时覆盖前向、反向、梯度裁剪、优化器和状态收尾，数据采样、验证、日志与检查点保存另计。实验匹配了相近参数量与数据预算；FLOPs 和运行态存储预算各不相同，Transformer 保存显式 KV，Bio 使用有限矩阵状态。

相关远史收益 `U0R−A0R` 的 dense 结果为 **0.000228 NLL**，文档重采样 95% 区间为 **[−0.000112, 0.000617]**，跨过零；Transformer 为 **0.005643**，区间为 **[0.003620, 0.007519]**。这些区间来自同一次训练的评分文档 bootstrap，反映评估样本变化。独立训练种子的稳定性需要额外配对实验。

N32 sparse50 在必选组与探索规则下全程执行 **32/32 节点**，核心更新耗时为 427.61 分钟。这项记录没有展示稀疏执行收益。

目前完成的全配对只有 N32/seed11，WikiText-2 已用于多轮探索。当前结论适用于这项实现、规模、数据流与评分协议；更大节点规模和独立种子的结果仍缺失。

### 队列与停止状态

v0.9 登记了 30 条独立训练、36 个配对格；content_only 在不同节点标签间复用同一训练。

| 状态 | 独立训练数 | 保存的记录 |
|---|---:|---|
| 完成 | 4 | N32/seed11 的 content_only、dense、sparse50、Transformer；训练与暖历史评估均完成 |
| 中止 | 1 | N128 dense 停在第 24,388 次更新、3,099,250 / 12M 有效目标；最终暖评估未执行 |
| 未执行 | 25 | 其余节点规模与训练种子保留登记 |

原训练队列保持停止。历史 PyCharm 入口会尝试恢复各自的旧队列，新的实验需先准备数据、检查点与独立输出目录，操作条件见[环境与复现范围](docs/REPRODUCIBILITY.md)。

## 实现与研究边界

v0.9 节点使用完整矩阵表示内容与运行态，包含多时间尺度快态、慢态、动态映射、概率状态、相位和局部反馈。当前输入沿已有合法径向父边到达读口；快态在块内递推，部分控制与记忆在块边界更新。候选球图保留同层和双向边，时间因果由输入可见性和状态更新顺序约束。实现细节见[v0.9 说明](v09/README.md)。

独立合成实验已经通过状态干预观察到可读取的递推信息；完整语言网络的稳定历史收益仍未建立。原构想中的运行态经验自动固化到节点参数，尚未形成独立实现算法。当前证据也未证明自演化、通用长时记忆或 N32 稀疏计算优势。

代码诊断将 slow 写入部分任务梯度的截断、写读关系、有效写入尺度及后续覆盖列为待检验因素。这些是带实现依据的原因假说，具体因果贡献仍需独立干预确认。

## 迭代记录

| 阶段 | 主要探索 | 已保存的结果 |
|---|---|---|
| 首轮中文 Bio-LLM | Transformer 中的八状态动力单元、普通模型与单独训练的无状态对照 | 9 次训练完成；平均损失差很小，Bio 三种子一胜两负，训练更慢 |
| 独立状态实验 matrix_lab | 递推矩阵状态、动态读出、二阶反应、概率和频率 | 合成任务读取了状态；复杂机制缺少稳定组合收益 |
| [v0.3](docs/V03_FINDINGS.md) | C1/C2/C3、32/128/512 节点、容量、迭代与图切分 | 81 个已登记条件完成；答案与问题先验主导，未显示可靠规模收益 |
| [v0.4](docs/V04_DESIGN.md) | 动态矩阵、Joseph、Fourier、情景槽与结构化设备流 | 7 完成、1 中止；部分 shared 模型零训练即可解题 |
| [v0.5](docs/V05_DATA_AUDIT.md) | 保持/遗忘、局部慢更新、资格迹与数据捷径修复 | 完成工程和数据审计；144 个有效正式条件未执行 |
| [v0.6](docs/V06_DESIGN.md) | 单份状态、实心球壳、脉冲、概率与公开 WikiText-2 | 8 完成、1 中止；历史作用弱，随后发现训练与信号标定问题 |
| [v0.7](results/v07/analysis_20261024/REASSESSMENT.md) | 连续状态、局部反馈与信用修复，A800 编译和并发 | 45 标签中 43 完成、2 中止；输出趋同、梯度放大、记忆收益不稳定 |
| [v0.8](docs/V08_IMPROVEMENT_PLAN.md) | 分块 delta、共旋坐标、矩阵注意力与并行实现 | 修复归一化放大；唯一长训在 2.54M 目标中止 |
| [v0.9](docs/V09_COMPARISON_PROTOCOL.md) | 内容载体、计算复用、批量反馈与实际节点组执行 | N32 四组完成；Transformer 胜出；N128 部分训练中止 |

早期实现位于根目录，matrix_lab 没有可靠的“v0.2”正式命名记录。版本、来源与执行细节见[完整迭代时间线](docs/ITERATION_HISTORY.md)。

## 开发与资料入口

历史本机环境主要为 Python 3.12 与 PyTorch，v0.9 已完成训练使用 PyTorch 2.14.1、FP32、MPS；A800 阶段使用独立 CUDA 环境。依赖、命令入口、平台条件与新实验目录要求见[环境与复现范围](docs/REPRODUCIBILITY.md)。

- [研究过程与取舍](docs/RESEARCH_JOURNAL.md)
- [实验记录索引](docs/RESULTS_INDEX.md)
- [v0.9 记忆机制诊断](docs/V09_MECHANISM_DIAGNOSIS.md)
- [公开范围与证据处理](PUBLICATION_NOTES.md)
- [数据和第三方材料](NOTICE.md)

自有代码和文档当前**未指定统一开源许可证**。第三方数据、论文与依赖遵循各自的许可，具体来源保留在 [NOTICE.md](NOTICE.md)。
