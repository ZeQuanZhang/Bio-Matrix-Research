# 在 PyCharm 启动 Bio-Matrix v0.6

> 归档说明：本文保留写作时点的设计或状态。项目已停止v0.9后续训练；最新执行状态见[实验索引](../docs/RESULTS_INDEX.md)。公开副本不含权重与语料。

入口是项目根目录的 `run_biomatrix_v06.py`。解释器使用本项目 `.venv/bin/python`。打开项目后，可选择随代码提供的 **Bio-Matrix v0.6** Run Configuration；它已设置工作目录、解释器和 **Emulate terminal in output console**。也可以右键入口 → Run，并在运行配置里勾选该选项。使用 Run 工具窗口运行。

## 默认会做什么

默认 `MODE = 'paired'`：WikiText-2 公开语言建模任务，32 / 128 / 512 节点，各比较 message、independent、full、Transformer，种子 11 / 23 / 37，共 **36 组**。先完成一个规模的四个模型，再进行下一规模；三个规模的首轮完成后才补下一个种子。

```python
MODE = 'paired'                 # paired / single / smoke / preflight / plan / prepare
STAGE = 'core'                  # core / mechanism / capacity / iterations / topology / credit / all
SCALES = [32, 128, 512]
SEEDS = [11, 23, 37]
DEVICE = 'auto'                 # Apple Silicon 自动选择 MPS
TRAIN_STEPS = 4000
OUTPUT_DIRECTORY = 'results/v06/public_lm'
KEEP_AWAKE = True
```

默认每组 4000 次更新、batch=2、每条 64 token，共 512,000 个目标 token。这是配对初筛预算；完整训练集有 3,084,660 个本分词器 token，训练窗口有放回抽样，不按整篇逐遍遍历。增加训练预算应在首次启动前调整 `TRAIN_STEPS`；已产生结果后改预算、模型源码、设备或配置，应使用新的输出目录。

`single` 只训练首个规模与种子的 full；`smoke` 跑四个模型各两步，输出到独立目录；`preflight` 测资源；`plan` 登记未执行条件，不训练；`prepare` 仅下载、校验和编码数据。补充阶段使用新的输出目录，`all` 共 60 组。补充条件是探索性实验，应结合 core 结果决定执行顺序。

## Mac 训练方式

- Apple Silicon 使用原生 PyTorch MPS 和 FP32；MPS 不可用时明确报错，正式入口不会静默退回 CPU。需要 CPU 时手动设置 `DEVICE='cpu'`。
- 使用 16 token 的截断反向传播窗口，状态在一条 64 token 样本内继续传递，每段结束后截断梯度；四段累计梯度后做一次 AdamW 更新。Transformer 的旧 KV 也在相同边界 detach。
- 不依赖 CUDA、FlashAttention 扩展或 CUDA AMP。保留 PyTorch/系统默认 MPS 内存保护，不设置无限内存水位。
- `configs/v06/base.json` 中的 `checkpoint_events` 可以开启激活重计算。启用后要使用新结果目录，并先重新预检；它减少保存的激活，会增加计算。
- `KEEP_AWAKE=True` 使用 macOS 自带 `caffeinate -i -w PID`，进程存活时避免空闲休眠；脚本结束或强制停止后退出。屏幕可以熄灭。

## 控制台与恢复

控制台显示阶段名称、总条件进度、当前模型/节点数/种子。训练只保留一条原地更新的进度条，显示 loss、学习率、token/s、节点到达率、速度和 ETA。验证、测试有各自的原地进度条；阶段切换和条件结束输出简短说明，不逐步打印 JSON。

首次 Ctrl+C/SIGTERM 请求在当前完整更新后保存；验证/测试期间在窗口边界停止。第二次中断可立即退出，保留最近原子检查点。PyCharm 的强制 Stop 只能保证恢复最近已写入的检查点，默认每 50 步保存。相同源码、配置、数据和设备下再次 Run 自动恢复，完成的条件自动跳过。未提交到检查点的日志尾部会另存 `.uncommitted-*`，恢复后的主指标文件不会重复统计这些步。

v0.6 使用系统文件锁，进程退出后自动释放；同时检查旧 v0.4/v0.5 GPU 队列。不要同时启动多个 GPU 队列。

## 数据与结果

数据在 `data/v06/wikitext2/`，官方下载文件、分词器、token 数组、文档 offsets 和含 SHA-256 的 `manifest.json` 均已保留。首次重新准备时只下载缺失/损坏文件；已有编码文件损坏时明确报错。

默认结果在 `results/v06/public_lm/`：

- `conditions.json`：not_executed / running / completed / failed / aborted。
- `events.jsonl`：阶段和验证记录；`FINDINGS.md`、`summary.json`、`paired_differences.json`、`seed_summary.json`：汇总与配对原始差值。
- 每组的 `protocol.json`、`ledger.json`、`metrics.jsonl`、`validation.jsonl`、`learning_curve.png`。
- `initial.pt`、`last.pt`、`best.pt`；测试的逐 token NLL、目标、预测与置信度保存在 `test_predictions.json`。
- `runtime_example.pt` 保存最佳共享权重与一条公开文本运行后的完整有限状态，包含局部慢更新，可用于继续输入。仅保存 `model.state_dict()` 不会保存个体经历。

能力主指标为固定公开文本窗口的 byte-BPE NLL/PPL，并报告相关历史收益、无关历史代价、A→B→A 返回代价和置信度校准。这里的 PPL 不是 WikiText 官方 word-level PPL，不能直接与论文中不同分词器的数字横比。跨模型使用完全相同的数据、训练 token 预算和测试窗口；Transformer 参数量匹配 full，实际状态和计算成本另报。

## 命令行等价方式

```bash
.venv/bin/python -m unittest -v v06.test_v06
.venv/bin/python run_biomatrix_v06.py --mode plan
.venv/bin/python run_biomatrix_v06.py --mode preflight --device mps
.venv/bin/python run_biomatrix_v06.py --mode smoke --device mps
.venv/bin/python run_biomatrix_v06.py
.venv/bin/python -m v06.report results/v06/public_lm
```

完整机制、对照定义与预先规定的评价方式见 `docs/V06_EXPERIMENT.md`。
