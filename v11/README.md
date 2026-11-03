# Bio-Matrix v1.1

本版继续在已训练 Transformer 中加入完整 Bio 记忆分支，修复 v1.0 的传播覆盖和投影初始化，并用公开标注问答提供明确的历史训练信号。原 v1.0 源码、检查点和负结果保留。

## PyCharm 启动

选择 **Bio-Matrix v1.1**，或用项目 `.venv/bin/python` 运行根目录 `run_biomatrix_v11.py`。配置启用 **Emulate terminal**。默认 `DEVICE='auto'` 在 Apple Silicon 使用原生 MPS，不连接过期 A800，不开启 CPU 算子回退。

显示总进度、四个条件、当前阶段和 CPU/峰值 RSS/Metal 分配量，后台每秒原位刷新。每个训练窗口结束后更新损失和答案目标计数；刷新界面不会凭空增加训练步数。首次准备公开问答数据时另有下载/整理进度。

入口顶部可修改：

```python
MODE = 'train'           # verify 只自检
DEVICE = 'auto'
CPT_TARGETS = 65_536
PT_UPDATES = 128
RESUME = None            # 已有 results/v11/pilot_... 目录
```

CLI 也可覆盖，例如 `--cpt-tokens 1024 --pt-updates 1 --validation-windows 2` 是短程工程验收。默认正式队列由用户启动；本轮开发只执行自检、校准和短程验收。

## 四组与预算

| 条件 | 作用 | 可训练部分 |
|---|---|---|
| `transformer` | 同一已训练主干、128 token 局部上下文参考评分 | 0；不计为训练速度基线 |
| `adapter` | 参数量相近的静态 FFN 适配器 | 适配器 |
| `bio_summary` | 完整 Bio，场内关联使用摘要写入 | Bio 分支 |
| `bio_ordered` | 完整 Bio，场内关联和 fast 按原 token 顺序更新 | Bio 分支 |

两种 Bio 的初始权重逐项相同，图、读出、状态矩阵、固定控制缓存和数据采样顺序相同。两组都逐 token 批量计算 fresh 图传递；写入更新次数、计算成本和结果单独记录。摘要组与保序组比较的是整套写入规则，其中包含聚合和有效步长差异。

主干沿用本地 v0.9 已训练检查点：词表4096、宽56、2层、4头。记忆几何自动匹配为4头×14维，Q/K/V和输出矩阵从每层主干精确复制，保留独立参数存储。输入沿用 block.norm1 的坐标，不再额外做一次 LayerNorm。运行时仍有单位 Q/K、tanh V、相位旋转、有限矩阵读写和门控。

默认每个更新 B2×512 输入，64 token 一段，8段保留完整反向图。

1. **CPT：65,536个有效语言建模目标，通常64次更新。** 冻结 WikiText-2，主目标为下一 token CE，默认不做 KD。
2. **PT：128次更新。** 官方 SQuAD v1.1，历史原文后接问题与标注答案；仅答案位置计算 CE，另加0.1倍全上下文教师 top-64 KL。教师是同一小型 WikiText 主干，`teacher_quality.json` 会报告其问答损失和首答案 token 命中率。

每个适配条件默认192次更新。学习率3e-4，两个阶段分别预热/衰减，AdamW、梯度裁剪1。四组同样本不等于同 FLOPs，耗时和吞吐独立记录。

进度条工作量定义为 **CPT有效目标 + PT可见输入 token**；PT答案监督目标另报 `answer_targets`，不能把补齐位置算成答案训练量。

## 数据合同

WikiText-2、4096词表 tokenizer 和预训练主干身份沿用冻结 manifest。SQuAD 使用官方 `rajpurkar/SQuAD-explorer` 固定提交，验证 Git blob 和本地 SHA-256；最终准备记录及筛选计数写入 `data/v11/squad_remote256`。不重新拟合 tokenizer。开发中的旧128距离缓存保留在 `data/v11/squad`，默认入口使用新版。

官方 train 按文章划分训练/验证；官方 dev 留为最终测试。历史是官方文章中的连续文本，问答与答案沿用人工标注。候选需要满足标注 span 与原文一致，完整证据早于 `prefix − layers×context_limit`；两层128窗口的默认设置要求答案证据完整落在384 token历史的最早128之内，与回答距离大于256。局部重复答案筛查也按256范围执行，避免高层近期KV间接保留原始证据。提示和答案整体按字符offset进行规范BPE编码，避免边界拆词改变首答案token。

这是公开SQuAD派生的远史子任务，筛选使用标注位置；模型只接收文本token。其答案NLL和首token命中率有独立定义，不对应原始完整SQuAD榜单的EM/F1。

最终缓存包含 train **51,416题/420篇**、validation **2,553题/22篇**、reserved test **7,565题/48篇**；正式记录以manifest及其哈希为准。

训练器只读取 train/validation，最终 test 不参与调参或本轮工程验收。准备阶段可编码 test，尚未计算其模型分数。

右侧 PAD 位不计损失；因果模型的前面答案分数不受后面 PAD 影响。PT 的含 PAD 末态不用于参数巩固，且下一窗口重置运行态。

## 保留的 Bio 机制与本次修复

相位、Fourier边调制、多时间带 fast、m/P Joseph更新、slow与资格迹、内外动态映射、软协作群、合法球壳图、脉冲阈值、有限8段信用、局部反馈和显式 node_shape 巩固全部保留。完整对应关系沿用 `docs/V1_IDEA_IMPLEMENTATION_MAP.md`；本版变化见 `docs/V11_PROTOCOL.md`。

默认内部通信3轮，构造时根据实际最外壳深度检查可达性。相位频率与耦合均按真实 token 时间 `count/J` 归一。脉冲阈值使用经自然训练输入校准的幅值单位，保存每壳实际 fresh mass、m/P变化和关联记忆幅度；覆盖不足会在正式更新前终止。

新 association 矩阵与固定64槽控制缓存同时存在于两组，容量不随会话长度增长。保序组的 association 和 fast 按 token 顺序做真实 delta 更新；m/P、相位、slow反馈和信用时钟每个真实段提交一次。读出的新状态只供之后的段使用。

CPT每16次完整监督更新及CPT结束时尝试参数巩固，记录尝试次数和实际改动。默认CPT通常有4次机会；局部支持、创新门槛仍可能使某次提议不改参数。显式启用多窗口提案均值，不能据此声称解决了未知来源经验独立性。

## 结果与恢复

结果独立保存在 `results/v11/pilot_日期时间/`：

- `protocol.json` 与 `source/`：代码、数据、主干哈希、预算、设备与完整协议。
- `calibration.json`：自然训练输入的逐壳覆盖、状态容量和初段耗时。
- `teacher_quality.json`、`validation_sample.json`：教师质量与固定评分样本身份。
- 各组 `initial_validation.json`、`cpt_validation.json`、`validation.jsonl`：训练前、CPT后、PT期间与完成后的 LM/QA 分数。
- `metrics.jsonl`、`consolidation.jsonl`、`last.pt`、`result.json`：原始训练数值、显式参数事务、模型/优化器/RNG检查点。
- `registry.json`、`summary.json`：completed / failed / aborted / not_executed 条件状态。

直接干预保持完全相同的当前KV、位置和后缀文本，比较正常历史、清空Bio、后缀关闭读取、替换Bio旧历史、打乱旧历史。另有从历史起点全程禁读、仅替换最旧原文并重新编码两种整条路径控制；这两项允许Bio造成的KV差异。这样可以检查先前Bio读出已写入近期KV的间接影响。

QA同时报告按有效token加权的答案NLL、每题分数与**首答案token** NLL/准确率；后者尚未计入teacher-forced答案前缀，适合观察更直接的历史依赖。首token命中率与整段答案EM是不同指标。

停止信号在安全优化边界保存为 aborted。恢复需完全相同的代码、数据、设备和预算；已有进度但安全检查点缺失时明确拒绝，防止悄悄重跑并混合日志。PyCharm 强制杀进程只能恢复到最后持久检查点，未保存的窗口不算已恢复。

## 如何判断结果

首先看QA首答案与答案NLL相对t=0、CPT后和参数匹配adapter的变化，再看清空/替换历史是否稳定损害正确答案。LM损失、错误历史干扰和训练耗时一起报告。保序组有收益时还需多个训练种子复现。

v1.0试验没有可靠收益，v1.1的改动为机制提供更明确的训练路径和可检验条件。默认单种子、小型主干与短预算仍属于机制试验；当前QA主要检验远史内容利用，近期证据足够且应忽略旧经验的严格独立后验尚未单列。与前沿LLM的能力或训练效率比较需要另外的预算与实验。

本轮开发验收见 `results/v11/engineering_20261026/REPORT.md`：67项CPU检查、原生MPS两阶段入口短测、真实检查点继续更新、每秒原位进度刷新均已完成。默认65,536 CPT目标+128 PT更新的正式队列尚未启动。
