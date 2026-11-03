# Bio-Matrix V3

V3 用独立 Bio 节点层处理序列。节点外部保存动态条件向量 K、内容向量 V 和多分量相位 P；输入生成 Q/K/V，查询选择节点组合并联合读出。每节点内部内容方阵与自由方向预条件方阵已移除。网络使用残差、归一化和逐位置 MLP，无 token 自注意力主干或 token KV 缓存。

原 V2 代码、全部检查点与结果保持原样。V3 使用独立检查点格式和结果目录，不加载 V2 或早期 Transformer 训练权重。

## 节点计算

对每个头，节点读取为：

```text
match_tj = dot(q_t, normalize(key_j))
g_t = softmax(match_t + temperature * phase_similarity(q_t, phase_j))
y_t = Σ_j g_tj * match_tj * value_j
```

固定相位权重时，节点群表达 `Σ_j g_tj value_j key_jᵀ`。每节点动态内容具有 rank-one 结构；增加节点可扩大分布式表达范围。该结构限制了每节点自由度，不宣称等价于任意完整矩阵。

写入使用共享 Q/K/V 接口中的 K/V，对块起点快照计算联合残差 `e_t = v_t − prediction_t`。value、key 和圆周相位的调整方向由同一重构误差导出；相位在单位圆切向修正，块末归一化，无自由旋转。最终语言 CE 继续通过写入路径训练投影、路由和更新速度。

## 块内因果并行

块内 key 和相位读取同一块起点快照。value 按 `V_{t+1}=αV_t+(1−α)V_anchor+η(g_t*match_t)e_tᵀ` 更新；token t 读取 t 之前的状态。

批量计算采用双重形式：

```text
R = query routing weights * query matches
W = write routing weights * write matches * write rate
block_output = decayed old-state read + causal(R @ Wᵀ) @ errors
```

因果矩阵包含严格下三角和时间衰减。它与显式逐 token value 递推等价，避免展开每个位置的 `[head,node,value]` 状态。单块时间交互规模为 `block_tokens²`，块长固定；块与块仍递推。流式调用可以任意切分，块时钟由处理位置决定。

key 与相位的加权梯度在块末一起提交，保持锚点牵引；value 在块末作范数限制。单次写入速率有界、记忆具有可学习衰减。静态对照仅冻结序列内 key 和相位，value 仍通过因果更新改变；两组的锚点均可训练。

每层持久保存块快照与待提交向量，大小随 batch、头数、节点数和向量维度固定。没有运行时节点增生、精确容量饱和检测或自动将经历固化为参数。

## PyCharm 启动

打开本项目，选择项目 `.venv/bin/python`，运行配置 **Bio-Matrix V3**，或直接运行根目录 `run_biomatrix_v3.py`。配置启用 **Emulate terminal**。默认原生 MPS；不访问旧 A800。

入口顶部：

```python
MODE='train'              # train / verify
DEVICE='auto'
CPT_TARGETS=1_048_576
PT_UPDATES=512
SEEDS='11,23,37'
RESUME=None              # results/v3/paired_...
```

每秒原位刷新总进度、阶段、资源和12条运行详情。短控制台自动分栏；在极窄窗口中使用 L/F/S/D 与种子简称。完整预算由用户手动启动。

默认宽56、2层4头、每头14维、32节点、8相位分量、块64。B4×512，先512次自然文本更新，再512次答案监督更新；每个条件共1024次。CPT计数为有效预测目标；PT进度为输入槽，实际答案目标数另记。

## 配对对照与数据

| 条件 | 序列结构 | 参数 | 运行态 |
|---|---|---:|---|
| `trans_local` | 普通 Transformer，原文窗口128 | 333,106 | 显式 token KV |
| `trans_full` | 普通 Transformer，原文窗口512 | 333,106 | 显式 token KV |
| `v3_static` | Bio向量节点，序列内key/相位固定 | 333,136 | key/value/phase向量 |
| `v3_dynamic` | Bio向量节点，联合调整key/value/相位 | 333,136 | key/value/phase向量 |

四组全部从零训练；共享初始词嵌入和兼容的逐位置 FFN 张量。Transformer FFN 宽度用于匹配参数量，差30个参数。状态容量、感受野与 FLOPs 各自报告。

复用冻结 WikiText-2、仅训练集拟合的4096 BPE，以及官方 SQuAD v1.1 派生远史问答；默认使用本机已有 `data/v06/wikitext2` 和 `data/v11/squad_remote256`。同种子的四组逐步使用相同样本与答案mask。每个独立窗口重置状态，窗口内保留跨块梯度。

QA证据位于局部Transformer保守感受野之外；对Bio采用相同远史替换范围，便于配对。Bio自身没有原文窗口，报告中的 reference_context_tokens 只定义干预范围。问答记录答案NLL、首答案NLL/命中；不冒充完整SQuAD榜单EM/F1。保留测试集不评分。

首次从公开仓库准备环境与数据：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_biomatrix_v3.py --mode verify
.venv/bin/python -m v07.data --root data/v3/wikitext2
.venv/bin/python run_biomatrix_v3.py --data-root data/v3/wikitext2 --qa-root data/v3/squad_remote256
```

数据准备下载固定公开源，核对原始文件哈希、切分和train-only分词；新目录不会覆盖历史冻结数据。QA在新目录按同一规则派生，协议记录实际分词器和准备代码身份。训练入口对缺失或损坏的既有冻结文件报错，避免静默更换数据。

## 保存、停止与恢复

新运行保存到 `results/v3/paired_时间/`，已有非空目录拒绝覆盖。包含冻结协议/源码/数据身份、初始与阶段验证、原始 `metrics.jsonl`、历史干预、每条件 `last.pt`、注册状态和汇总报告。状态区分 completed / aborted / failed / not_executed。

正常停止在优化边界保存；强制 Stop 可从最后安全检查点恢复，未保存更新不计。将入口 RESUME 设为原目录，或用 `--resume results/v3/paired_...`；源码、设备、预算和数据身份必须一致。保存优化器、数据随机数和训练随机状态；已有进度缺少检查点时拒绝自动重跑。旧版本检查点不兼容。

## 已完成的验收与速度限制

32项CPU自检覆盖因果性、双重读取与串行参考的数值/梯度、任意切分、混合重置、联合误差方向、固定容量、对照配对、优化器恢复与继续更新、历史干预和12条控制台布局。

原生MPS检查中，CPU/MPS最大logit差约 `7.45e-8`，43个梯度张量最大差约 `2.09e-7`。真实公开数据完成四条件×两个种子、每条件1次CPT＋2次PT的有限验收，原始记录在 `results/v3/acceptance_mps_20261103`。这用于验证执行与保存链路，不能判断学习效果。

在B4×512、FP32、两次预热后五次完整更新的工程测量中，双重实现动态版约0.223秒/更新，局部Transformer约0.0203秒，耗时比约10.96。显式展开前缀的首个原型约0.410秒，记录保留在 `results/v3/engineering_20261103`；双重实现见 `results/v3/engineering_20261103_dual`。这些短测尚未展示速度优势，后续完整训练需同时看质量、时间和状态预算。

首个原型在块内读取了更新后的key和相位；最终版把这两类变化延后到块边界。双重形式与最终版的显式value递推已做数值和梯度等价验证，两次原型测速之间包含上述语义调整，不能作为严格同算子的提速比。

```bash
.venv/bin/python -m v3.benchmark --device mps --output results/v3/new_engineering
```

完整三种子训练现已完成，见下列正式运行；最终测试、容量上限、独立新情景干扰和长期参数固化未执行。


## 完整三种子运行结果

`results/v3/paired_20261103T094249_111075` 已完成默认12条运行，每条1024次更新。四组参数约33.3万，全部从零训练，三个种子逐步配对。

| 条件 | 答案NLL↓ | 首答案NLL↓ | 首token命中 | 文本PPL↓ | 训练秒/种子 |
|---|---:|---:|---:|---:|---:|
| 局部Transformer | 6.925408 | 7.488867 | 0% | 1078.345 | 22.11 |
| 完整窗口Transformer | 6.916758 | 7.482149 | 0% | 1056.412 | 22.30 |
| V3静态 | 6.901393 | 7.290911 | 0% | 916.314 | 206.28 |
| V3动态 | 6.901476 | 7.283738 | 0% | 902.826 | 240.37 |

相同训练目标预算下，动态Bio文本PPL比局部Transformer低16.3%，QA均值小幅较低但文章重采样区间跨0。动态减静态的QA差约+0.000083；正确远史替换代价约+0.000341，区间跨0。四组首token命中均为0%，当前远史事实检索仍弱。动态版训练耗时约10.9倍，持久状态比局部KV少21.5%，训练峰值显存未测。

完整分析、源码身份核对与状态回放见[正式报告](../results/v3/paired_20261103T094249_111075/analysis/review_20261103/REPORT.md)。V3从零训练，早期部分版本采用预训练适配，绝对PPL分别解读。
