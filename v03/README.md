# Bio-Matrix v0.3

> 归档说明：本文保留写作时点的设计或状态。项目已停止v0.9后续训练；最新执行状态见[实验索引](../docs/RESULTS_INDEX.md)。公开副本不含权重与语料。

独立的网络级实验。主实现、所有配置和结果均与首轮分开，原始首轮文件不变。

使用项目已有的`.venv`和固定依赖。公开任务需先从[CLEVR官方页面](https://cs.stanford.edu/people/jcjohns/clevr/)下载无图像包，解压到`data/clevr/CLEVR_v1.0/`并保留zip；本工作区已准备好该数据和校验和。`v03.public`只准备固定子集，不会自动下载数据。

```sh
.venv/bin/python -m v03.profile
.venv/bin/python -m v03.public
.venv/bin/python -m v03.run --stage all --dry-run
.venv/bin/python -u -m v03.run --stage all
```

必须在MPS可用的进程运行；沙箱可能隐藏Metal设备，需要允许该训练进程访问GPU。没有CPU训练回退。训练串行；不要同时启动两个训练器。

训练完成后串行运行以下控制器；它依次训练输入独立基线，执行干预、全部单元检查、安全检查点验证并生成报告：

```sh
.venv/bin/python -m v03.postprocess
.venv/bin/python -m v03.repeat
```

`v03.repeat`对P2/P3全部条件补齐23/37种子。该驱动等待初始postprocess完成，先保存首批报告、图和汇总，再串行训练并重新诊断、验证和生成最终报告。补充预算见 `docs/V03_REPEAT_AMENDMENT.md` 和 `results/v03/repeat_plan.json`；`--dry-run`列出90个追加条件。可用 `caffeinate -i` 在这个本地工作流存活期间防止空闲休眠。原始模块`v03.baselines`、`v03.diagnose`、`v03.verify`、`v03.report`也可分别运行，但完整排队流程使用上述控制器。

`run --stage`支持pilot/main/capacity/iterations/topology/public；`--only`指定完整条件ID；`--max-seconds`在完整更新边界显式中止，保留last.pt供同一命令恢复。恢复校验配置、模型、源文件哈希，拒绝覆盖另一协议的结果。KeyboardInterrupt只保留已有安全检查点，不保存半次优化更新。

配置：`configs/v03/base.json`；预登记：`docs/V03_PREREGISTRATION.md`；代码审阅：`docs/V03_IMPLEMENTATION_NOTES.md`；新报告：`docs/V03_FINDINGS.md`。完整条件、历史状态和原始结果在`results/v03/`。结果目录按原项目惯例不进入Git，不能用只复制源码代替备份原始结果。

本轮共171个网络条件：9个学习率试跑、27个主扫描、135个三种子机制/公开任务补充；另有6个三种子输入独立诊断。先完成配对，再补齐重复。三种子初筛不作五种子确认结论。固定/动态专家、流式变体和其他生物模块不混入主扫描。
