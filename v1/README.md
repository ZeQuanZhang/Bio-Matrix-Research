# Bio-Matrix v1.0：Transformer 内部的完整 Bio 记忆分支

v1.0 使用已训练的因果 Transformer 承担语言计算。每层在 attention residual 与 FFN 之间加入可训练记忆分支：多时间尺度矩阵、相位地址、状态调节映射，以及完整节点协作场。旧版本代码、实验与检查点保持原样。

## 在 PyCharm 启动

打开项目根目录的 `run_biomatrix_v1.py`，使用项目 `.venv/bin/python`，点击 Run。已提供共享配置 **Bio-Matrix v1.0**，开启终端模拟，显示总进度、四个条件、当前阶段及资源行；每秒原位刷新。

入口文件顶部可调整：

```python
MODE = 'train'                 # train / verify
DEVICE = 'auto'                # Apple Silicon 默认 MPS
TOKENS_PER_CONDITION = 65_536   # 每个适配条件的有效预测目标
RESUME = None                  # 续跑填写 results/v1/pilot_... 目录
```

默认顺序是代码自检 → 数据身份校验 → 加载本地主干 → 固定主干评分 → 普通 adapter → 无相位 Bio → 完整 Bio。新输出位于 `results/v1/pilot_日期时间/`。每次新启动创建新目录，已有非空目录拒绝覆盖。

只做 CPU 自检：

```bash
.venv/bin/python run_biomatrix_v1.py --mode verify
```

命令行启动和续跑：

```bash
.venv/bin/python run_biomatrix_v1.py --device auto
.venv/bin/python run_biomatrix_v1.py --resume results/v1/pilot_实际目录
```

`auto` 在本机选择原生 MPS，禁用静默 CPU fallback。没有可用 Metal 时会明确报错；需要 CPU 时设置 `DEVICE='cpu'`。不连接远端、不下载数据、不启动 v0.8/v0.9 队列。

## 默认实际运行范围

| 项目 | 默认设置 |
|---|---|
| 主干 | 本地 v0.9 已完成的 Transformer `last.pt`，316,608 参数，56 宽、2 层、4 头 |
| 完整模型 | 365,590 总参数；48,982 可训练参数，主干冻结 |
| 普通 adapter | 49,056 可训练参数，比完整记忆分支多 74；同一主干起点 |
| 数据 | 已校验的官方 WikiText-2、原始切分、train-only 4096 BPE |
| 优化窗口 | B2 × 8 段 × 64 token = 1,024 个预测目标；段间保持计算图 |
| 局部注意力 | 128 token；教师看完整的 512-token 采样窗口 |
| 目标 | CE + Top-64 教师分布 KL；两个目标使用相同有效位置 mask |
| 节点场 | 每层 32 节点、2 轮通信、完整概率/相位/动态映射/局部修订 |
| 时间带 | 3 套完整矩阵；每头矩阵 8×8，4 个记忆头 |
| 预算 | adapter、bio_no_phase、bio_full 各 65,536 有效目标；参考主干只评分 |
| 巩固 | 每100步或完整预算末步显式提交；不完整 mask 尾窗跳过 |
| 评估 | 同样的384-token历史与128-token评分后缀，验证集固定8窗 |

当前默认是本地主干上的工程 pilot。正式效果结论需要更长预算、多种子和未参与选择的公开评测。主干及 WikiText 验证集已经用于旧版本研究。

## 完整机制与时序

1. `model.py` 的核心矩阵按 token 执行稳定 delta 写入，片段内查询同一历史快照。边界由持久时钟决定；整段、分块及逐token调用具有相同语义。
2. `dynamics.py` 的 BioField 保存 m/P、fast、slow、trace、支持/创新、相位与8段信用。球壳上保留同层和回落连接；向外传播受脉冲强度门控制。Fourier相位差、耦合及临时软群真实影响消息与查询。
3. BioField 在段末用有限摘要提交一次；段内可微统计按固定token顺序累积。局部反馈是下一段潜在观察修订上一段预测的代理任务。
4. 未来片段损失通过普通 autograd 训练早期写入、查询、门控与协作参数。模型内部没有整体 detach，训练窗口结束后才截断状态。
5. `Network.consolidate` 在已完成优化且运行态detach之后，把达到支持/创新门槛的经验写入真实 `node_shape` 参数，清对应优化器矩和旧信用。评估不做参数固化。

完整逐项对应与边界见 [构想实现表](../docs/V1_IDEA_IMPLEMENTATION_MAP.md)。原构想中的 JEPA、无限路径携带与 Taylor 逆恢复继续排除。默认动态群为全节点软协作；可用 `--active-fraction 0.5` 运行实际gather子图的选择计算。必选路径与探索节点计入真实活跃比例，闲置矩阵冻结；硬路由、gather/scatter的额外成本需要实测。

32/128/512节点、重复种子和容量/内部轮数控制设计见 [实验计划](../configs/v1/experiment_plan.json)。可以通过 `--nodes 128 --seed 23` 指定新条件；各节点规模的配对独立进行，扩大规模没有小节点先胜出的前置条件。默认入口先完成N32/seed11工程pilot。

## 怎样判断记忆有用

`evaluate.py` 在相同评分文本、相同初始注意力 KV、位置和段时钟下，对照 warm、清空Bio状态、替换错配记忆、打乱历史所得记忆，以及评分阶段关闭记忆读取。另报冷启动同后缀NLL。错配文档仅作为无关历史的内容代理。

保存每窗原始NLL与汇总，不把loss下降、相位同步、单次分数或一个种子当作能力验收。当前状态、梯度检查、参数巩固成功只确认相应计算已执行。

## 结果与恢复

- `protocol.json`：预算、数据/分词器身份、主干SHA、源码SHA、参数和设备。
- `registry.json`、`summary.json`：条件的 `not_executed/running/completed/aborted/failed` 状态。
- 各条件的 `metrics.jsonl`、`validation.jsonl`、`consolidation.jsonl`：原始更新、验证和固化记录。
- `last.pt`：模型、优化器、RNG、采样器、计数器和协议，保存完整优化边界。
- `source/`：本次代码及直接数值/运行依赖快照。

按一次停止会请求在优化边界保存；强制结束仍可从最近的 `last.pt` 恢复。恢复校验预算、源码、数据、主干、设备与Torch身份；预检失败单独记录，不改原实验状态。每个训练窗口从新运行态开始，续跑恢复下一个窗口及共享参数。

## 相关研究

矩阵快权重与 delta 更新参考 [Fast Weight Transformers](https://proceedings.mlr.press/v139/schlag21a.html)；长期神经记忆与注意力结合可参见 [Titans](https://arxiv.org/abs/2501.00663)；分段写读、完整上下文教师与内部记忆的训练组织参照 [MemT-1.0](https://arxiv.org/abs/2610.26825)。本版为本项目的组合实验实现，未复制对方模型权重或推理软件。
