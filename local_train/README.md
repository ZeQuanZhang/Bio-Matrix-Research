# V3 本地 31M 配对训练

在 PyCharm 顶部运行配置中选择 **Bio-Matrix 本地31M训练**，点击 Run。也可以打开根目录 [run_biomatrix_local.py](../run_biomatrix_local.py) 运行。项目已提供开启 Emulate terminal 的共享运行配置，并复用标准评测的隔离环境。

入口依次执行：固定公开数据准备、三模型配对训练、五项公开标准评测。原生 Mac MPS，FP32；完整队列由用户启动。已有 A800 环境不参与。

## 配置与预算

| 项目 | 默认值 |
|---|---|
| 模型 | 独立 Transformer、Bio 静态地址、Bio 动态地址；两种 Bio 均保留相位 |
| 参数 | 31,468,872 / 31,469,328 / 31,469,328 |
| 网络 | 6 层、384 维、6 头、每头 128 节点、8 相位分量 |
| 上下文 | 512 token；Bio 每 128 token 提交状态，每个样本内四块完整反传 |
| 批量 | microbatch 4，梯度累积 4，有效批量 16 |
| 训练 | 每模型 100,000,000 个有效预测目标，seed 11 |
| 配对里程碑 | 每个模型先完成 10M，再分别完成 50M、100M |
| 优化 | AdamW，峰值 LR 3e-4，warmup 1M，随后余弦衰减；梯度裁剪 1 |
| 定期保存 | 每 100 次更新、里程碑、正常停止时保存断点 |
| 留出验证 | 每 500 步与里程碑使用固定 64 个窗口 |
| 正式评测 | 三模型完成后，零样本、512 上下文、五项任务完整默认划分 |

三种模型使用完全相同的 token 批次、掩码、顺序、学习率和更新预算。每个里程碑校验累计批次哈希一致。参数差 456，约 0.00145%；实际计算量与训练时长单独记录。Bio 是独立节点网络，普通 Transformer 为单独对照。

状态在每个打包样本开始时重置，512 token 内连续传递。当前协议的跨样本记忆与多种子正式实验均未执行。100M 是本机初步预算，后续扩展需要新协议；配置见 [v3_31m.json](../configs/local/v3_31m.json)。

## 数据与评测

公开语料来自固定版本 [FineWeb-Edu sample-100BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)。读取 `scale/data.py` 已锁定的 revision，只通过 HTTPS Range 读取 Parquet 文本列。复用固定 GPT-2 分词器，从零初始化所有模型权重。

准备阶段先载入五个官方评测划分，建立规范化词级 13-gram 索引；过滤命中文档，去除规范化全文重复。按文档哈希分配 1% 验证候选，填充固定验证预算，其余填充训练缓存。train 与 validation 的实际编码缓存均校验 SHA256。部分远程分片仅核对固定版本和字节长度，没有宣称完成整个远程分片 SHA256 校验。13-gram 过滤覆盖长文字重合，短题目重合和语义改写仍有残余风险。

评分复用 [lm-evaluation-harness v0.4.13](https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.13) 的原生提示、似然和聚合；[五任务锁定协议](../evaluations/standard/protocol.json) 保持不变。WikiText、LAMBADA OpenAI、PIQA、ARC-Easy、HellaSwag 的结果分别报告。评分只使用固定预算末尾权重，验证期间不根据公开基准选点。

首次数据准备需要网络与约 10 GiB 检查点余量；100M token 缓存约 200 MB。单次运行会额外保留优化器断点和三档权重。准备过程被打断后，下次重新编码当前未完成的数据缓存；已完成缓存直接验哈希使用。

## 控制台与停止恢复

- 每秒原位刷新一个面板，显示全部 9 条详细进度：数据、TF、BioS、BioD、WT、LAMB、PIQA、ARC-E、HS。
- 阶段行显示当前模型、步骤或评测请求、loss、token/s、学习率和估计剩余时间。资源行显示 CPU、RSS、Metal 分配与驱动占用。
- 80×8 小窗口自动排成多列，零尺寸终端使用 120×24；窗口过小时提示扩大，不隐藏成“hidden more”。
- 底层库的额外输出进入 `runtime.private.log`，避免抢占进度条。数据请求、模型初始化和保存时，界面时钟与资源信息仍刷新。
- PyCharm 点一次 Stop 后，完成当前训练更新再保存。强制终止进程只能恢复到上次已提交断点。再次 Run 使用同一目录自动续跑；重复启动会被进程锁拒绝。
- 协议、源码、数据或环境变化时拒绝续跑。新实验请通过命令参数指定新的 `--output`；调整数据预算时也使用新的 `--data-root`。

默认输出：`results/local_train/v3_31m_100m_s11/`。逐步训练指标、验证指标、阶段状态、参数检查点和官方评测 JSON 均自动保存。原始实验目录保持原样。公开仓库只发布代码、协议、数值与验收记录，语料、token、私有日志和权重留在本地。

## 环境和有限验收

本机已经准备好 `results/standard_eval/runtime/bin/python`。新机器可使用 Python 3.12 建立同名环境并安装 [固定依赖](../evaluations/standard/requirements.txt)：

```bash
uv venv --python 3.12 results/standard_eval/runtime
uv pip install --python results/standard_eval/runtime/bin/python -r evaluations/standard/requirements.txt
results/standard_eval/runtime/bin/python run_biomatrix_local.py --mode smoke
```

`smoke` 使用相同 31M 架构和真实 FineWeb-Edu 文本，每模型 2 次更新、16,384 token；每项公开任务仅取 2 条样本。它写入独立时间目录。默认 `train` 执行完整 100M 配对队列，自动准备数据、续跑和最终评分。

[交付验收记录](../results/local_train/verification_20261104.json) 区分已完成的短验收、失败/中止记录和未执行的完整训练。短验收得分不用于能力比较。
