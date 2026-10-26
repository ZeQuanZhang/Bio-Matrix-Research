# Bio-Matrix v0.4：在 PyCharm 运行

> 归档说明：本文保留写作时点的设计或状态。项目已停止v0.9后续训练；最新执行状态见[实验索引](../docs/RESULTS_INDEX.md)。公开副本不含权重与语料。

从项目打开根目录 `Self-evo-model`，Python 解释器选择：

```
.venv/bin/python
```

右键根目录 **run_biomatrix_v04.py → Run**。建议在 Run Configuration 中勾选 **Emulate terminal in output console**，使多层进度条在原位置刷新。没有该选项时仍能运行，但控制台可能显示多行进度。不要使用 Python Console 执行整份脚本。

入口文件顶部可编辑：

```python
MODE = 'paired'       # 完整配对；single 只训练 full；smoke 检查；plan 只列计划
SCALES = [32, 128, 512]
SEEDS = [11, 23, 37]
VARIANTS = ['graph', 'full', 'matrix', 'probability', 'fourier']
TRAIN_STEPS = 1200
DEVICE = 'auto'
READ_PATHS = ['shared', 'separated']
OUTPUT_DIRECTORY = 'results/v04/continuous'
```

默认共 90 个训练条件：两种读取路径 × 三个规模 × 五组模型 × 三个种子。同一规模先完成共享端口与分离端口的全部配对，再进入下一规模，最后补重复种子。每组使用相同种子的相同输入序列。训练进度条显示 step/total、loss、acc、dev、速度与 ETA；测试/干预也有进度条。开发集还会评价 step=0；报告同时保存零训练表现，避免把预设更新规则的作用误当成训练收益。

**暂停与恢复**：在终端第一次 Ctrl+C 后等当前更新结束；PyCharm 的 Stop 会由系统决定是否允许清理，最迟恢复到最近 25 步检查点。保持配置和代码不变，再运行同一文件即可恢复。改配置、源码或设备请更换输出目录，避免混淆结果。不要同时启动多个训练器。

没有训练时长限制，但默认完整扫描可能很久。先在 PyCharm 运行 `MODE = 'smoke'` 确认显示 MPS，然后改回 paired；smoke 使用独立目录，不污染正式训练。

命令行也可运行，路径均相对于项目代码定位：

```bash
.venv/bin/python -m unittest -v v04.test_v04
.venv/bin/python run_biomatrix_v04.py --dry-run
.venv/bin/python run_biomatrix_v04.py --smoke --device mps
.venv/bin/python run_biomatrix_v04.py
```

本轮实现与训练协议详见 `docs/V04_DESIGN.md`。数据是中文可读的结构化流，没有自然中文 LM 预训练。标准 Transformer 尚未加入这个机制实验的对照。

每条件目录包含 protocol.json、ledger.json、last.pt、best.pt、metrics.jsonl、test_predictions.json、各干预预测、result.json。总目录生成 conditions.json、summary.csv/json、FINDINGS.md；not_executed/running/completed/aborted/failed 分别记录。结果沿用旧项目惯例不进入 Git，应自行备份 results 目录。

默认已包含 shared 与 separated 两种读取路径。断图控制可复制配置文件，将 read_paths 设为 ["separated"]、topology 改成 cut，并使用独立输出目录。调整内部迭代数或矩阵维数同样需要独立目录；固定总容量与计算匹配应依据 ledger 的实测状态和耗时设计，不把节点数量或参数相同直接当成计算相同。
