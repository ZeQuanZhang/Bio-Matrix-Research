# 环境与复现范围

本仓库保留停止后的研究源码、配置和选定证据。首次阅读可按 [README 的结果查看步骤](../README.md#快速查看已保存的结果)，使用 Python 标准库读取数值与队列状态。训练数据和二进制检查点需另行准备，公开结果目录用于查阅证据。

## 历史环境

- 首轮与后续本机实验主要使用 Python 3.12、PyTorch 和 Apple GPU 的 MPS 路径。
- v0.9 完成训练时为 PyTorch 2.14.1、FP32。
- A800 阶段使用 PyTorch 2.8.0+cu128 及独立编译适配；环境与执行记录见[历史 A800 说明](V07_A800_EXECUTION.md)。原远端环境已到期。
- [requirements.txt](../requirements.txt) 与[锁定清单](../requirements.lock.txt)保留原依赖记录；Metal/CUDA 路径需要相应硬件与运行环境。

## 查看代码入口

以下命令在仓库根目录执行。先检查目标平台能提供所记录的依赖版本，再创建环境：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m v09.train --help
python -m v09.experiment --help
```

PowerShell 的虚拟环境激活命令为 `.\.venv\Scripts\Activate.ps1`。两个 `--help` 命令显示单条件训练与比较队列的参数后退出。

v0.9 的单条件训练入口接受 `auto/cpu/mps/cuda` 设备参数，具体可运行条件取决于平台与依赖；比较队列 `v09.experiment` 固定使用本机 MPS。各版工程测速、合成检查和模型训练有不同入口，应先查对应目录的 README 与测试定义。部分测试会访问数据、使用设备或执行短数值运算。

首轮中文数据准备见[历史根目录说明](HISTORICAL_README.md)。WikiText-2 的下载、冻结 revision、train-only tokenizer 和数据检查见[v0.7 说明](../v07/README.md)及[v0.9 协议](V09_COMPARISON_PROTOCOL.md)。数据集采用原来源的切分，语料与使用条件见 [NOTICE](../NOTICE.md)。

## 新实验目录与恢复条件

v0.9 原训练队列已停止，执行状态见[停止决定](../results/v09/comparison_no_v08/decisions/20261025_stop_v09/README.md)。历史 PyCharm 入口默认指向当时的结果目录，主动运行会尝试恢复对应队列。

重新设计实验时，先准备冻结语料，按 `python -m v09.experiment --help` 核对参数，再通过 `--output` 指定独立的新目录，固定预算与日程。新运行会建立自己的身份和结果记录。原地恢复还需要完整检查点、数据游标与运行态，本公开副本保留的是选定证据。

跨设备、源码、数据或预算变化会触发检查点身份校验。历史结果中的哈希描述原始实验文件，公开副本的净化与可移植性调整另有[导出清单](../provenance/EXPORT_MANIFEST.json)。

## 中文图表

生成中文图表需要可用的中文字体。脚本使用用户目录下的可选字体配置，并尊重 `FONTCONFIG_FILE`。个人字体与专有字体没有打包，图表外观可能随平台变化；数值证据独立保存在结果文件中。

公开整理时检查了 Python 语法、JSON、文档链接、证据一致性与发布范围，具体记录见[公开说明](../PUBLICATION_NOTES.md)。历史模型训练、硬件测试与新环境依赖安装需要各自验证。
