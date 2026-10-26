# 中文 Bio-LLM 对照实验

> 归档说明：本文保留写作时点的设计或状态。项目已停止v0.9后续训练；最新执行状态见[实验索引](RESULTS_INDEX.md)。公开副本不含权重与语料。

最新实现为 **Bio-Matrix v0.9**，说明见 [v09/README.md](../v09/README.md)。PyCharm 选择 **Bio-Matrix v0.9 验证与测速**，或直接运行 [run_biomatrix_v09.py](../run_biomatrix_v09.py)；正式训练选择 **Bio-Matrix v0.9 训练**。默认使用本机 Apple GPU，每秒原位更新详细进度。v0.9 保留完整矩阵状态，加入合法图内容通路、计算复用、反馈批量化和真实节点组执行；旧版本与结果保留。

**本机配对对比入口：** [run_biomatrix_v09_compare.py](../run_biomatrix_v09_compare.py)，PyCharm 配置 **Bio-Matrix v0.9 对比实验**。按用户要求取消 v0.8，四模型配对、三种子，纯内容基线复用后为 30 条独立训练；新结果存入 `results/v09/comparison_no_v08`，旧队列与中止检查点保留。显示每秒原位更新的详细进度，自动保存并支持续跑。协议见 [v0.9 配对训练协议](V09_COMPARISON_PROTOCOL.md)。

以下为历史版本入口。A800 已取回归档，v0.9 不连接旧远程机器。

历史 A800 完整训练入口为 [run_biomatrix_v07_a800.py](../run_biomatrix_v07_a800.py)，原 PyCharm 配置为 **Bio-Matrix v0.7 A800**。操作记录见 [A800 启动说明](A800_PYCHARM.md)，硬件与并发验收见 [执行记录](V07_A800_EXECUTION.md)。

**Bio-Matrix v0.7** 历史入口为 [run_biomatrix_v07.py](../run_biomatrix_v07.py)，
在 PyCharm 使用项目 `.venv` 解释器运行，Run Configuration 已包含动态终端进度条设置。
该版本修复连续状态训练、局部矩阵反馈和脉冲标定，保留多维动态矩阵、球壳传播、Fourier/相位、概率更新及有限状态。
使用冻结的公开 WikiText-2，比较三种图网络、参数接近的 Transformer 与同宽无状态基线，原生支持 Mac MPS。
见 [v0.7 启动说明](../v07/README.md)和[修复设计与实验协议](V07_DESIGN_AND_EXPERIMENT.md)。
v0.6 代码、结果及运行队列保留，旧入口为 [run_biomatrix_v06.py](../run_biomatrix_v06.py)，[原说明](../v06/README.md)仍可使用。
下文是首轮中文 Bio-LLM 的原始说明。

后续的 **Bio-Matrix 非线性反馈与动态矩阵实验** 位于 [matrix_lab](../matrix_lab/README.md)，
完整首轮结果见 [FINDINGS](../matrix_lab/docs/FINDINGS.md)。该实验直接测试递推状态记忆，
不包含 Transformer。下文保留上一轮八状态语言模型的原始说明，两轮结果分别存放。

本项目把分享聊天记录中的构想实现为一个可训练模型：在因果 Transformer 的每层加入有限维、具有持续内部状态的生物启发动力系统。输入驱动状态变化，状态影响当前输出和后续计算。反应速率与读出参数通过语言建模损失学习；**没有 JEPA，没有调用任何外部大模型，也没有付费 API**。

这是完整实现且可复现的计算实验，不是完整真实神经元的分子仿真。每个通道的八类状态都有明确更新规则；不把未实现的细胞机制算作已经实现。

## 已固定的对照协议

| 项目 | 设置 |
|---|---|
| 公开中文数据 | `RobinChen2001/TinyStories-Zh-2M` 固定版本的第一份 Parquet 分片 |
| 原始版本 | `7601b862e5b2fbf60e83d7b27b8bd8017f105e70` |
| 使用范围 | 前 80,000 篇符合中文比例和长度要求的不同文档 |
| 划分 | 规范化全文 SHA-256 取模，训练 98%、验证 1%、测试 1%；全文完全去重 |
| 分词器 | 仅用训练文档训练的 8,192 词表 Byte-level BPE，三组共享 |
| 主干 | 4 层、256 隐藏维、4 个注意力头、256 Token 上下文、权重绑定语言头 |
| 参数量 | 三组均为 **5,329,408**；标准组 FFN 宽 1,032，Bio 两组宽 1,024 |
| 随机种子 | 42、137、2026，每组各训练三次 |
| 预算 | 每次 5,000 步、批量 16；同种子三组读取完全相同的文档和裁剪位置 |
| 优化 | AdamW，学习率 0.0006，预热加余弦衰减，梯度裁剪 1.0，FP32 |
| 模型选择 | 只按验证集损失选择检查点；测试集不参与选择 |
| 主指标 | 同一中文测试集上按有效 Token 加权的交叉熵和困惑度 |
| 其他指标 | 每文档损失、实际训练速度、采样的 Metal 分配量、分块状态消融、续写样本 |

分组为 `baseline`（标准 Transformer）、`bio`（完整八状态递推）和 `stateless`（相同 Bio 参数与读出，逐 Token 重置全部内部状态，并单独训练）。第三组用于判断改进来自历史状态还是额外的静态非线性。

**参数与数据预算完全匹配；计算量并不相等。** Bio 需要额外计算，报告中单独显示耗时。只匹配 Token 数不能解释为同 FLOPs 的胜利。

## Bio 状态方程

对每层每个通道保留

`s = [c, u, r, b, v, a, e, m]`。

为简洁起见，定义一步精确的一阶松弛算子

`E(target, rate, previous) = exp(-rate)*previous + (1-exp(-rate))*target`。

它对应一个采样区间内目标和速率固定的连续时间方程。所有速率由可训练的对数速率转换为正值，所有状态有明确范围。

| 状态 | 更新的计算含义 | 范围 |
|---|---|---|
| `c` 钙信号 | 外部刺激的积累与消退 | [0,1] |
| `u` 释放利用率 | 同时受当前刺激和钙信号影响的易化 | [0,1] |
| `r` 可释放资源 | 释放导致消耗，静息时恢复；输出使用释放前资源 | [0,1] |
| `b` 受体占用 | 递质依赖的结合与解离 | [0,1] |
| `v` 膜状态 | 输入与受体信号的漏积分 | [-1,1] |
| `a` 适应 | 缓慢跟踪膜活动，调节后续响应 | [0,1] |
| `e` 资格迹 | 积累输入与响应的局部相关性 | [-1,1] |
| `m` 慢速可塑性 | 由输入变化调制的资格迹巩固，改变有效输出增益 | [-1,1] |

令 `p=sigmoid(x)`、`z=tanh(x)`、`q_i=sigmoid(theta_i)`，每一步按以下依赖顺序计算。`E` 的最后一项总是该状态自己的上一时刻值。

```text
c = E(p, k_c, c_previous)
u = E((p+c)/2, k_u, u_previous)
d = k_r * u * c * (1+q_0)
r = E(k_r/(k_r+d), k_r+d, r_previous)
transmitter = u * c * r_previous
binding = k_b * transmitter * (1+q_1)
unbinding = k_b * (0.05+q_2)
b = E(binding/(binding+unbinding), binding+unbinding, b_previous)
v = E(tanh(z+2*b-1), k_v, v_previous)
a = E(sigmoid(v), k_a, a_previous)
post = tanh(v-q_3*(a-0.5))
e = E(z*post, k_e, e_previous)
modulation = abs(p-c)*q_4+0.01
m = E(e, k_m*modulation, m_previous)
output = x*(1+tanh(w_2)*m) + tanh(w_0)*post + tanh(w_1)*(b-a)
```

初始状态中资源 `r=1`，其他为零。状态由显式函数参数传入和返回，不存放在共享模型对象里，因此不同序列互不串扰。动力学沿通道局部更新，通道间信息由 Transformer 的可学习线性变换和注意力混合。

其中受体部分有结合／解离的动力学结构，其余属于归一化的现象学近似。资格迹、可塑性和稳态适应并未声称是已验证的人脑同一机制。这里固定计算拓扑、学习参数；没有进行反应拓扑的遗传搜索。

CPU 使用并行仿射扫描；Metal 使用每通道一个线程的融合递推内核。两者实现同一状态方程，反向传播均使用伴随递推，减少 Python 调度及中间存储。测试会核对串行参考结果、梯度、CPU/Metal 完整模块数值、因果性、状态边界、分块等价性与序列隔离。

## 运行

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m unittest -v test_bio
.venv/bin/python download_data.py
.venv/bin/python prepare.py --documents 80000
caffeinate -i .venv/bin/python -u run_experiment.py
```

原始分片和 Hugging Face 元数据放在 `data/`；下载器固定版本并核对 SHA-256，支持通过标准 `HTTPS_PROXY` 环境变量使用本机网络代理。数据下载地址、分片校验和、筛选方式、划分文档哈希和编码后 Token 校验和均保存在 `data/manifest.json` 与相邻文件中。数据集说明见 [中文数据卡](https://huggingface.co/datasets/RobinChen2001/TinyStories-Zh-2M)，英文上游见 [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)。中文数据卡标注 MIT；上游为 CDLA-Sharing-1.0，保留两者来源说明。

训练器每 500 步原子写入 `last.pt` 和最优 `best.pt`，同时保存优化器、数据随机数状态及训练协议。同一命令再次运行会恢复未完成训练，并跳过已经有 `result.json` 的完成项。不同协议不能误覆盖旧训练。

`results/` 中保存各次训练的日志、协议、权重、测试结果与续写样本。所有实验结束后 `report.py` 生成聚合报告与学习曲线。

训练完成后，本地续写及保存状态：

```bash
.venv/bin/python generate.py --checkpoint results/bio_seed42/best.pt --prompt '从前，有一只小兔子' --state-out results/demo_state.pt
.venv/bin/python generate.py --checkpoint results/bio_seed42/best.pt --prompt '后来，' --state-in results/demo_state.pt
```

推理每个上下文块保留内部动力学状态，并重算当前块的注意力。保存文件包含块开始时的 Bio 状态和当前块 Token；不同检查点的状态不能误混用。

## 结果的边界

主测试在每篇独立文档的最多 256 Token 前缀上评估，三组都从新状态开始、都能使用同样长的注意力上下文。报告不把这个测试描述成长上下文基准。

另有一个诊断：把同一测试文本分成 64 Token 块，块间清空注意力历史，分别保留、清零或跨样本交换 Bio 状态。第一块不计入该指标。三种条件共享模型权重和输入，用于检查历史状态是否影响后续预测。这是相对训练分布存在变化的诊断，不等同于证明模型已经学会任意事实检索或解决无限上下文。

模型仍然包含因果注意力；没有预先宣称消除了 KV 缓存。推理保存内部状态也仍然需要明确的重置、隔离与持久化。

`probe_memory.py` 另做中文事实回忆诊断：八种人物名称、四种颜色与不同中文干扰文本构成平衡样例，在 0/64/192/512 Token 间隔后续写颜色。四个候选颜色各为一个 Token，机会准确率 25%。每 256 Token 清空注意力，比较保留与清零 Bio 状态。探针不用于训练或挑选检查点，且没有针对该题微调；需先看零间隔是否已经具备读出能力，再解释远距离表现。

该语料为机器翻译的儿童故事，任务范围窄且包含翻译瑕疵。5.33M 参数模型属于架构验证用小语言模型，不是成熟聊天助手。即使测试损失改善，也不能外推到 1B 或更大模型、世界模型、通用推理与真实生物机制。
