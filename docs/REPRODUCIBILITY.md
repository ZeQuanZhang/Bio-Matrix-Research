# 环境与复现范围

本仓库是停止后的研究快照，包含代码、配置和选定证据，没有训练数据与二进制检查点。重新执行会产生新的运行身份与结果，不能把公开记录目录当作可直接续跑的完整备份。

## 历史环境

- 首轮与本机后续主要使用Python 3.12、PyTorch、Mac Apple GPU的MPS路径。
- v0.9完成训练时为PyTorch 2.14.1、FP32。
- A800阶段使用PyTorch 2.8.0+cu128及独立编译适配，环境和执行记录见[历史A800说明](V07_A800_EXECUTION.md)。远端已经到期，凭据没有发布。
- `requirements.txt`与锁定清单保留原始依赖记录；不同平台的Metal/CUDA实现需要对应硬件。

## 安装与安全检查

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m v09.train --help
python -m v09.experiment --help
```

首轮中文数据准备见[历史根目录说明](HISTORICAL_README.md)。WikiText-2下载、冻结revision、train-only tokenizer与数据检查见[v0.7说明](../v07/README.md)及[v0.9协议](V09_COMPARISON_PROTOCOL.md)。不提供重新划分后的测试集或第三方语料镜像。

每个版本的测试与配置保留在对应目录。部分测试会使用MPS、访问数据或执行短数值运算；请先查看测试定义。公开准备过程只检查了Python语法、JSON、发布链接、证据一致性与公开范围，没有重新训练模型或执行整套硬件测试。

## 新实验使用新目录

历史PyCharm入口默认指向当时的结果目录。若复现v0.9，应先准备冻结语料，并通过`python -m v09.experiment --help`确认参数，指定新的`--output`，保持预算和日程从一开始固定。不要用本仓库中缺少权重的归档目录尝试恢复训练。

跨设备、源码、数据或预算变化会触发检查点身份校验；请保留这些校验。历史结果中的哈希描述的是原始实验，公开副本净化或可移植性调整后的字节另见导出清单。

## 字体

生成中文图表需要可用的中文字体。公开副本把原机器的字体配置绝对路径改为用户目录下可选配置，并尊重`FONTCONFIG_FILE`；没有打包个人字体或专有字体。图表外观可能随平台变化，数值证据独立保存。
