# Bio-Matrix v0.7

> 归档说明：本文保留写作时点的设计或状态。项目已停止v0.9后续训练；最新执行状态见[实验索引](../docs/RESULTS_INDEX.md)。公开副本不含权重与语料。

在 PyCharm 打开项目，选择 **Bio-Matrix v0.7** Run Configuration，解释器为项目 `.venv/bin/python`。已启用 **Emulate terminal in output console**，进度条在同一行更新，阶段和验证结果另起一行。也可右键项目根目录的 `run_biomatrix_v07.py` → Run，并在运行配置中启用该终端选项。

入口顶部可调整：

```python
MODE = 'paired'       # paired / single / smoke / preflight / plan / prepare
STAGE = 'core'        # core / mechanism / write_scale / credit / capacity / iterations / topology / all
SCALES = [32, 128, 512]
SEEDS = [11, 23, 37]
DEVICE = 'auto'       # Apple Silicon使用MPS；Linux有CUDA时使用CUDA；FP32
TRAIN_STEPS = 24000
```

默认 core 是 3 个节点规模 × 5 个模型 × 3 个种子，共 **45 组**。每个规模先完成 message、independent、full、Transformer、stateless 五个配对，再扩大规模，之后补种子。`single` 仅跑第一个规模/种子的 full；`all` 才包含后续机制、梯度范围和容量等阶段。

命令行示例：

```bash
.venv/bin/python run_biomatrix_v07.py --mode plan
.venv/bin/python run_biomatrix_v07.py --mode preflight --device auto
.venv/bin/python run_biomatrix_v07.py --mode paired --device auto
```

CPU 工程检查可在 v0.6 使用 MPS 时进行：

```bash
.venv/bin/python scripts/check_v07_conformance.py
.venv/bin/python run_biomatrix_v07.py --mode smoke --device cpu
```

`smoke` 只做各模型两次优化更新及保存/评估流程检查，不用于判断能力。`preflight` 使用所选正式序列/BPTT长度，测两次更新、状态字节及设备分配。MPS/CUDA 启动会同时检查 v0.7 自有锁和 v0.6 共用 GPU 锁；旧队列占用时直接退出，保持旧队列继续运行。勿删除锁文件。

数据默认直接校验并复用 `data/v06/wikitext2` 的官方切分、缓存和分词器；缺失或损坏时明确报错，不覆盖旧数据。自建新目录须显式指定 `--data-root`，`prepare` 才会创建新缓存。

正式训练使用4096-token连续会话：A顺序读取，中途插入64-token B，再返回未见的A后文；独立文档切换也保留状态。接缝一半有EOS/BOS，一半无标记；未标记条件连缓存自带的BOS/EOS也去除。仅会话边界重置。默认batch2、每次128 token、BPTT128，24,000步输入6,144,000个token；seed11实际6,100,155个有效目标，其余接缝只mask CE，不改变模型反馈。document-reset另作显式消融。

当前输出分别是 `results/v07/aligned_public_lm`、`aligned_preflight` 和 `aligned_smoke`。旧public_lm/audit2/validation目录保留原协议和结果。`conditions.json`区分未执行、运行中、已完成、失败、中止；每组保留训练JSONL、逐token测试、干预诊断、权重和曲线。

`last.pt` 保存当前共享权重、优化器、全部运行状态、文档排列/游标、RNG 和已有最佳权重。第一次 Ctrl+C/SIGTERM 在当前完整更新后保存；再次中断保留最近原子检查点。相同代码/配置/数据/设备下再次运行会恢复并跳过已完成条件。更改源码或训练预算须选择新输出目录，不能与旧协议混用。`runtime_example.pt` 是最佳共享权重下一条独立文本的继续输入示例，和用于精确训练恢复的 `last.pt` 用途不同。

完整构想对照见[设计一致性清单](../docs/V07_CONFORMANCE.md)，可重复CPU验收入口为`scripts/check_v07_conformance.py`。完整修复说明、局部反馈的精确范围与检验标准见 [v0.7 设计与协议](../docs/V07_DESIGN_AND_EXPERIMENT.md)。v0.7 需要重新训练，不加载 v0.6 的权重或运行状态。


A800单卡并行验证使用 `scripts/v07_a800_parallel.py`：先用 `scripts/benchmark_v07_cuda.py` 比较1/2/4进程总吞吐，再选worker数。各worker持有独立项目副本和锁，共用同一份经SHA校验的冻结数据；不改变每组batch、步数和随机种子。父调度器按种子/节点规模设置配对屏障，并保留每组原始输出。远端环境和当前运行位置见 `results/v07/remote_a800/` 下的连接/部署记录。
