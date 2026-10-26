# v0.7 A800 完整训练与执行记录

> 归档说明：本文保留写作时点的设计或状态。项目已停止v0.9后续训练；最新执行状态见[实验索引](RESULTS_INDEX.md)。公开副本不含权重与语料。

本轮使用 NVIDIA A800 80GB PCIe，服务器已有 PyTorch 2.8.0+cu128、CUDA 12.8、Python 3.12。独立虚拟环境复用 CUDA PyTorch，补齐与冻结数据兼容的 tokenizers 0.23.3、pyarrow 26.0.0。Mac 的 v0.6 队列和源文件保留。

## 固定训练协议

45 条件：32/128/512 节点 × message/independent/full/Transformer/stateless × seed11/23/37。各条件从零训练24,000步，B2、T128、BPTT128、FP32，复用同一份冻结 WikiText-2 官方切分和 train-only byte-BPE。会话、接缝、输入及目标与对照一致。各条件独立，配对关系由冻结配置和输入指纹保持；2026-10-23 的调度升级允许空闲 worker 提前执行后续规模与种子的条件。

每条seed11训练流共有6,144,000个输入、6,100,155个有效目标。完整训练指执行本协议的全部优化、验证、模型选择和15类适用历史干预。能力结论由实际结果形成；24,000步是固定预算，收敛程度保留在原始曲线中。

## 并发与编译

当前采用14个独立worker PROJECT，分别持有自己的锁、优化器、运行状态和检查点。父调度器持有全局锁，按原45条件顺序补齐空闲worker，并检查活动任务的峰值显存预留、GPU实际空闲显存和12 GiB余量。数据共享只读冻结文件。各条件记录原始stdout、源码/数据/配置指纹、每次尝试及完成/失败/中止状态。初始5 worker与seed×scale屏障在两个快基线完成后只剩3条训练，先迁移到12路，后在保留训练子进程的情况下扩到14路。

本机容器的实际CPU配额为14核、主存上限120 GiB。保留原3条生产训练，并分别增加3/5/7/9条独立N128 full工程任务后，6/8/10/12并发的混合有效吞吐为980/1522/1808/2104目标每秒，GPU平均利用率为78%/87%/92%/97%。每个探针先完成编译及warmup，再计时8次更新；24个探针全部完成，计时内没有新图编译。该短测用于选择12 worker；正式训练速率另采样，完整数据与局限见`results/v07/remote_a800/utilization_audit_20261023T042807Z/TUNING_REPORT.md`。

CUDA MPS 使用本实验独立的pipe和log目录。下表保留早期单独N128 full工程探针的合计有效目标吞吐；指标为同步开始到全部worker进程退出的墙钟，排除各worker导入、编译和warmup。额外的compute-span和完整启动耗时均保存。

| 执行方式 | 并发进程 | 合计有效目标/秒 |
|---|---:|---:|
| eager + MPS | 1 | 29.31 |
| eager + MPS | 2 | 63.16 |
| eager + MPS | 4 | 126.17 |
| eager + MPS | 5 | 149.92 |
| compiled + MPS | 1 | 82.18 |
| compiled + MPS | 2 | 220.66 |
| compiled + MPS | 4 | 446.49 |
| compiled + MPS | 5 | 513.62 |

每组只计时2–3次更新，数据用于选择调度方式，正式速率以训练日志为准。512节点的4个eager full进程同时反向亦完成，没有OOM。单进程正式形状的CUDA峰值分配约为N32:0.66GB、N128:2.48GB、N512:9.77GB；完整状态、激活和框架资源分项保留在preflight。

独立适配器仅编译`Network.step`，设置为Inductor/default、fullgraph=True、dynamic=False、FP32/禁TF32。门控warmup保持原400步规则，动态blend通过固定0维Tensor送入编译图；forward返回时恢复公开float以兼容日志。Transformer和stateless保持原执行路径。适配器不添加模型参数、运行态键或训练标签，编译失败明确失败，禁止静默换回eager。完整15干预验收定位默认8个专化缓存不足后，独立c32版明确设置recompile_limit=32、accumulated_recompile_limit=256并纳入运行身份；未更改模型数学。旧8版流程失败独立归档。

同输入三路径检查比较原eager、Tensor化eager、compiled的logits、loss、全部原始梯度、状态与优化器轨迹。N32三种图模型及N512 full的初期warmup检查通过；full单进程稳态约5.6–6.1倍加速。首次编译约100秒，另计，不作训练步骤。

硬门随机初始化的额外检查保留为`numerical_mismatch`：原始梯度逐元素验收有少数坐标失败，两个eager路径之间也有3个坐标失败。该例原始全局梯度范数约4.66e7；core与compiled的全局相对L2差约3.14e-6，分别归一裁剪至1后的相对L2差约2.37e-6、最大坐标差约1.98e-6，cosine约0.999999999997。logits差低于4e-7，原容差下的短程优化器轨迹检查通过。原始失败不改标；离线FP64裁剪几何分析也不冒充CUDA逐指令复现。基于实际更新方向和轨迹的对齐采用编译，同时保留每步原始gradient_norm、clip及所有失败记录。

## PyCharm 入口和持久队列

本地运行`run_biomatrix_v07_a800.py`或选择`Bio-Matrix v0.7 A800`配置。默认`MODE='follow'`，共享配置明确为`--mode follow`，每秒用本地入口同款tqdm原位更新总进度、资源行和每个worker的详细条；连接和条件状态变化才增加事件行。动态尺寸检测保护PyCharm暂时为0的PTY宽高。总进度为全部条件训练步数之和，每条显示模型、规模、种子、step、loss、速度和阶段；原始step不用于估算本地查看期间的虚假it/s或ETA。显式`--mode status`保持单次查看。

`MODE='pause'`请求远端在完整更新后保存；`MODE='resume'`恢复中止条件；`MODE='retry_failed'`显式重试失败条件；`MODE='fetch'`取回原始归档。关闭本地查看不会停止远端进程。连接配置保存在本机`results/v07/remote_a800/connection.local.json`，权限600，该目录已在Git忽略范围。

远端冻结代码为`<REMOTE_WORKSPACE>/biomatrix/releases/v07-aligned-15e7f5f3d9fd-c32`；完整队列位于`<REMOTE_WORKSPACE>/biomatrix/runs/v07-full-fp32-c32`。`status.json`登记45条件，`raw/<id>/<id>/`保存每组结果，`attempts/<id>/`保存每次进程日志。源码、编译设置、数据或训练预算变化必须使用新的workspace。

12 worker调度器位于独立`operations/backfill-51d4bcd660be/v07_a800_parallel.py`，未覆盖冻结release。迁移先安全暂停全部父子进程，验证训练源码、数据、配置、wrapper、adapter和已完成结果哈希，再归档旧manifest/status与receipt；原3条分别从2665、2214、2080步恢复，完成的2条未重跑。调度策略迁移与现场验收保存在`results/v07/remote_a800/backfill_upgrade/`，`fetch`包含迁移前后身份与凭据。

后续12→14→16→12隔离测速显示14路总吞吐约2714目标/秒，比12路约2383目标/秒高13.9%；16路降至约2513目标/秒，CPU已碰14核配额。依用户要求正式扩为14路，使用独立`operations/live14-fa7ecb0a030e/v07_a800_live.py`加载原冻结基础调度器。交接仅停止旧父监督进程，原12个训练PID、启动时间、attempt数和运行状态全部保留，并验证步数继续增加；新增slot12／13启动seed23、N32的Transformer／stateless，后续仍自动回填原45条件。原manifest字节及240个静态／已完成文件哈希保持一致，训练配置和预算不变。接管器身份与前后状态在`live_adoptions/`，本地验收在`results/v07/remote_a800/worker14_upgrade/`；控制器fetch已包含此审计目录。

本地原始工程验收与远端快照保存在`results/v07/remote_a800/`。完整构想对应关系见[V07_CONFORMANCE.md](V07_CONFORMANCE.md)，操作细节见[A800_PYCHARM.md](A800_PYCHARM.md)。
