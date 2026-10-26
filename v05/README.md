# 在 PyCharm 运行 Bio-Matrix v0.5

> 归档说明：本文保留写作时点的设计或状态。项目已停止v0.9后续训练；最新执行状态见[实验索引](../docs/RESULTS_INDEX.md)。公开副本不含权重与语料。

项目解释器使用 `.venv/bin/python`。

右键根目录 `run_biomatrix_v05.py` → Run。建议在运行配置勾选 **Emulate terminal in output console**。总条件、训练步数、验证和干预都有进度条、速度与 ETA；不要通过 Python Console 执行整个脚本。

入口顶部设置：

```python
MODE = 'paired'       # paired / single / smoke / plan / preflight
STAGE = 'core'        # core / capacity / iterations / topology / credit / compute / all
SCALES = [32, 128, 512]
SEEDS = [11, 23, 37]
TRAIN_STEPS = 1200
DEVICE = 'auto'
OUTPUT_DIRECTORY = 'results/v05/verified_streams'
```

`preflight` 运行数据检查、各规模的实际反向/单次更新及完整生命周期更新，写入独立检查目录；`smoke` 用两组两步检查训练器，结果不进入正式目录；`plan` 只打印计划；`paired` 运行全部 core 配对。默认144条件。`single` 只运行首个规模/种子/路径的 full。

v0.4 训练仍在运行时，请等待它结束再启动 v0.5 GPU 正式队列。入口会检测旧训练锁，避免两个 GPU 队列同时争用本机资源。不要删除仍存活进程的锁。

首次 Ctrl+C/SIGTERM 在完整更新结束后原子保存；第二次中断保留最近安全检查点。PyCharm 强制 Stop 可能只能恢复每25步的最近检查点。配置和源码不变时重新 Run 即可恢复，已完成条件自动跳过。修改代码/配置/设备请更换输出目录。

补充实验可将 STAGE 改为 capacity/iterations/topology/credit/compute，并使用新的 OUTPUT_DIRECTORY；all 包含270条件。固定容量组每个节点的带数较多，先看 ledger 和预检资源，具体限制见设计文档。

命令行等价入口：

```bash
.venv/bin/python -m unittest -v v05.test_v05
.venv/bin/python -m v05.audit_data --root results/v05/data_audit_v2
.venv/bin/python -m v05.profile --device cpu --root results/v05/preflight_cpu
.venv/bin/python run_biomatrix_v05.py --dry-run
.venv/bin/python run_biomatrix_v05.py --smoke --device mps
.venv/bin/python run_biomatrix_v05.py
```

每个完成条件有 `result.json`、`test_predictions.json`、各干预概率、`last.pt`、`best.pt`、`initial.pt`、`metrics.jsonl`、`ledger.json`，以及第一组测试生命周期的 `lifetime_example.pt`。后者包含共享模型和 runtime_state，可在相同协议下继续输入新事件。在线慢权重存在 runtime_state 中，彼此独立；保存模型 state_dict 单独一份不会保存个体经历。

总目录保留条件状态、summary.json/csv、FINDINGS.md。结果目录未进入 Git，应与源码一起另行备份。

本实验使用中文可读的结构化设备事件，检验记忆、通信、长期巩固和纠错，不是中文自然语言模型预训练。模型 full 的详细机制、对照和未覆盖项见 `docs/V05_DESIGN.md`。

数据v2修复了固定“真值+1”声明造成的捷径。不可信声明在四个反事实轨迹中保持相同，缺少证据的早期请求不计入主评分。dev使用2/4/8步反馈延迟，test使用16/32步，并独立保留long任务。默认dev每任务8组、test每任务32组独立基础轨迹；每组四种颜色变化不算四份独立证据。主要报告各任务末次查询的宏平均。
