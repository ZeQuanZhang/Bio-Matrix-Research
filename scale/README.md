# V3 规模验证：0.133B × 10B token

这轮实验检验更充分的训练和更大的网络，能否建立基本问答能力，并让V3的动态条件、内容和相位产生收益。使用约1.33亿参数的独立Bio网络，配对静态地址Bio与普通Transformer，三个模型各自从零学习100亿个有效预测目标，再接受相同的公开问答监督。

当前代码与本机有限验收已完成。**GPU正式训练、2048长度CUDA显存/吞吐测试及LoCoMo评分均未执行。** 历史V3源码、检查点和负结果保持原样；新代码位于`scale/`，沿用V3机制，展示名称为“V3规模验证”。

## 配置与对照

| 项目 | 默认值 |
|---|---|
| Bio静态、动态参数 | 各133,284,672 |
| Transformer参数 | 133,281,168；差3,504，约0.0026% |
| 深度、宽度、头数 | 12层、768维、12头 |
| Bio节点、相位、块长 | 每头128节点、8相位分量、128 token块 |
| 分词 | 固定GPT-2分词器，50,257词表；只复用分词器 |
| 序列 | 2048 token；每个独立序列重置状态，序列内16个块保持完整梯度 |
| 每模型预训练 | 10,000,000,000个有效目标；EOS计入目标 |
| 检查点 | 三模型先配对完成1B，再3B，再10B；学习率按完整10B预算连续变化 |
| 初始批量 | microbatch 2、累积32次，共64序列/更新，最多131,072个目标 |
| 问答监督 | SQuAD train v1.1按文章切分；3000次更新，共192,000次样本抽取，答案和EOS参与loss |
| 精度、优化器 | CUDA BF16 autocast + FP32参数/状态，AdamW；Mac有限验收FP32 |
| 训练设备 | 单CUDA GPU；完整队列由用户在新租用机器启动 |

参数匹配通过调整Transformer FFN宽度至3596实现。三模型共享分词、样本顺序、loss mask、有效批量、预算和学习率；词嵌入与兼容FFN张量共享初始化。Transformer使用完整2048序列上的因果SDPA，可由PyTorch选择兼容的融合核。FLOPs、持久状态与感受野分别解释。

Bio保留V3的key/value/phase更新。路由改用直接张量收缩，省去显式展开的节点×向量中间张量；从零时钟开始的批次采用已知块边界，省去逐块设备同步判断。任意流式状态仍走原时钟路径。节点和FFN使用activation checkpointing，以反向重算换取显存。输出、梯度、完整/部分块状态和分段输入已与冻结实现做等价核对。

## 数据

语料为[FineWeb-Edu `sample-100BT`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)，英文、约100B GPT-2 token；首轮实际训练10B。固定revision为`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`，140个Parquet分片约286.4GB。按固定顺序下载所需分片，检查LFS SHA256和长度，编码后删除可重新下载的原始分片。

GPT-2分词器固定为`openai-community/gpt2@607a30d783dfa663caf39e06633721c8d4cfcd7e`，文件SHA256为`8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`。训练不加载GPT-2或其他模型权重。

编码缓存采用uint16，10B目标约20GB，另加少量边界余量。文档全文哈希固定分配train/validation/test；完全相同的文本属于相同切分。训练按固定置乱的2048块顺序取样，检查点保存块游标，三个条件的输入与mask摘要可逐步核对。准备阶段按完整原始分片提交进度；中断后重做尚未提交的分片，完整`.part`文件可直接验哈希恢复。

SQuAD官方train共87,599题已经在本机完成真实编码：83,484训练、4,115文章隔离验证，跳过0题。训练混合短证据与可容纳的长证据，裁剪保留答案所在区间。原始dev保留。LoCoMo未参与训练，尚未实施跨语料模糊污染检查。

## 第一次运行

在新GPU机器获取本仓库，使用Python 3.12和与GPU驱动匹配的CUDA版PyTorch，然后安装其余依赖：

```bash
git clone https://github.com/ZeQuanZhang/Bio-Matrix-Research.git
cd Bio-Matrix-Research
python3.12 -m venv .venv
# 按云平台或PyTorch官方说明，在此环境安装匹配的CUDA PyTorch。
.venv/bin/python -m pip install -r scale/requirements.txt
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.is_bf16_supported())"
.venv/bin/python run_biomatrix_scale.py --mode inspect
.venv/bin/python run_biomatrix_scale.py --mode verify
```

CUDA和BF16检查应为True。BF16不受支持时入口报错；如确实要用FP32，显式传`--precision fp32`，并重新检查吞吐及显存。

先在数据盘准备缓存，再测完整训练长度：

```bash
.venv/bin/python run_biomatrix_scale.py --mode prepare --data-root /你的数据盘/biomatrix-data
.venv/bin/python run_biomatrix_scale.py --mode profile --device cuda
```

准备程序在下载前检查空间。建议为数据、下载临时文件和检查点预留至少100GB数据盘；当前Mac约14GB空闲空间会被明确拒绝。准备完成后启动：

```bash
.venv/bin/python run_biomatrix_scale.py --mode train --device cuda --data-root /你的数据盘/biomatrix-data
```

首次GPU测速使用完整2048长度，每模型3次合成更新，排除首个预热值。租用硬件尚未确定，实际可用microbatch、吞吐和完整训练时间需要实测。要修改批量，编辑[配置](../configs/scale/v3_0133b.json)，保持`microbatch × accumulation = 64`；完成测速后再冻结正式运行。完整训练期间配置、源码、数据、精度或PyTorch版本变化会阻止恢复。

## PyCharm与进度

使用根目录[run_biomatrix_scale.py](../run_biomatrix_scale.py)，运行配置 **Bio-Matrix V3 Scale** 已启用 **Emulate terminal**，解释器使用当前项目环境。入口默认`MODE='inspect'`，核算配置后退出。新GPU上依次改成`prepare`、`profile`、`train`；训练时设`DEVICE='cuda'`和实际`DATA_ROOT`。

训练控制台每秒原位更新六条详细进度：三模型的预训练和问答阶段。阶段行显示当前条件、有效目标数、CE、目标/s与学习率；资源行显示CPU、RSS与CUDA allocated/reserved显存。长任务会持续刷新；不连接过期A800。

新结果保存到`results/scale/train_时间/`，不会覆盖历史运行。正常停止完成当前优化器更新，再保存并登记`aborted`；强制杀进程时保留上一次原子检查点。恢复时传入原运行目录：

```bash
.venv/bin/python run_biomatrix_scale.py --mode train --device cuda --data-root /你的数据盘/biomatrix-data --resume results/scale/train_原时间
```

PyCharm可设置`RESUME`并保持`MODE='train'`。恢复前校验协议、源文件与数据。未提交的日志尾部另存归档，恢复后的逐步轨迹使用已提交的检查点游标。

## 结果与判断

各阶段保存参数/协议身份、原子`last.pt`、1B/3B/10B权重、有效目标与问答监督计数、每步loss、梯度、学习率、时延、CUDA峰值显存，以及固定验证NLL/PPL。问答末尾逐题独立贪心生成，最多64 token或EOS结束，保存EM/F1、预测多样性和触顶率。含原文预测的文件标为`.private.json`。

SQuAD文章隔离验证F1至少20%是本轮预设的工程门槛，用于决定是否进入后续LoCoMo评估。该门槛属于实验选择。基本证据问答通过后，还需要历史清空、无关历史、距离、干扰和独立训练种子，才能评价记忆机制。当前入口不自动跑LoCoMo。

本机已完成[14项验收](../results/scale/acceptance_0133b_final_20261103/verification.json)和[完整参数规模、32长度MPS合成更新](../results/scale/mps_0133b_delivery_20261103/profile.json)。它们检验实现与有限设备执行；完整2048 CUDA显存、10B训练收敛和下游能力仍待执行。
