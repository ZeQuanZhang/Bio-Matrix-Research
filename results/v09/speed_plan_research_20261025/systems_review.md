# Bio-Matrix v0.9 训练速度：GPU 与系统执行路线审查

> 归档说明：本文保留写作时点的设计或状态。项目已停止v0.9后续训练；最新执行状态见[实验索引](../../../docs/RESULTS_INDEX.md)。公开副本不含权重与语料。

日期：2026-10-25。本文为优化研究笔记，没有修改生产源码、启动训练、执行 GPU benchmark 或连接远端。仅使用已保存的实测结果、当前源码、本机 Python API 的 inspect/hasattr 与官方文档。所有预计收益均为待验证假说。

## 先处理哪里

本机近期应先减少频繁的小算子与状态搬运，再对确定的热点做编译和融合；未来 CUDA 路线可以进一步使用 Triton 和 CUDA Graphs。两条路线共享“固定张量形状、设备内路由、融合前向与反向”的接口设计，执行工具不同。当前不需要租赁设备或恢复旧 A800。

已有工程数据指向两个主热点：系数/空间计算，以及 slow/Joseph/信用的顺序状态提交；反向计算本身占完整更新的大半。仅更换 fast recurrence 核，无法解释或消除主要差距。

## 1. 已有证据与计时边界

来源为项目内 results/v09/mps_20261025_v09_final。统一 FP32、MPS、B=2、T=64，6 次 warmup、3 次正式计时，另有 1 次同步 profile。完整训练更新包含前向、CE、反向、clip、optimizer、detach 和信用失效；采样、保存、路由转 CPU 的记录工作与控制台不在主计时内。模型内部的路由决策、收集和回填在计时内。

| 节点数 | dense 有效目标/s | sparse50 有效目标/s | 参数接近的 Transformer 有效目标/s | TF/dense 速度比 |
|---:|---:|---:|---:|---:|
| 32 | 508.77 | 483.00 | 20,652.80 | 40.59 |
| 128 | 298.12 | 232.54 | 20,866.02 | 69.99 |
| 512 | 105.16 | 129.29 | 20,134.93 | 191.47 |

这是短工程测速，不是完成 12M 目标的长训统计。测速 Transformer 的 context_limit=256；当前正式训练队列为 4096，不能把上表直接标成当前 4096 上下文的速度差距。参数接近也没有固定 FLOPs、动态状态字节或信息表示方式。

| dense 节点数 | 前向 ms | 反向 ms | coefficients ms | serial_slow_commit ms | fast_recurrence ms |
|---:|---:|---:|---:|---:|---:|
| 32 | 104.29 | 144.92 | 42.10 | 42.00 | 5.40 |
| 128 | 176.24 | 254.10 | 100.93 | 48.30 | 6.60 |
| 512 | 485.55 | 715.86 | 350.51 | 73.37 | 14.14 |

后面三个字段是前向内的区域时间，不能相加当成反向的原因分解。反向约占本次 profile 总更新的 57–59%。N512 的 coefficients 已比 fast recurrence 贵约 25 倍；N32 的系数构造和顺序提交均约 42ms。

实际 sparse50 活跃比例为 N32=100%、N128≈58.3%、N512≈50.9%。它在 N32/N128 比 dense 慢约 5.1%/22.0%，N512 才快约 22.95%。这说明少算节点还没有稳定转化为少用时间。

## 2. 本机 PyTorch/MPS 能力已经核对

本机解释器为项目 .venv/bin/python，torch.__version__=2.14.1。只读检查得到：

| API/实现 | 本机结果 | 本笔记对它的使用判断 |
|---|---|---|
| torch.mps.compile_shader(source: str) | 存在，返回底层 Metal shader library | 可以写专用 Metal 核；本轮未编译或调用 |
| torch.mps.profiler.profile(mode, wait_until_completed) | 存在，默认 interval/False | 生成 OS Signposts，交由 Instruments 查看 |
| profiler.start/stop | 存在 | 可控制短观察窗口 |
| profiler.metal_capture(fname) | 存在 | 输出 Metal gputrace；需单独短捕获 |
| profiler.is_metal_capture_enabled | 存在 | 本地源码注明通过 MTL_CAPTURE_ENABLED 启用 |
| torch.mps.Event(enable_timing=False) | 存在 | 可研究事件计时；不能把主机异步入队时间当完成时间 |
| torch.library.custom_op/register_autograd/register_fake | 均存在 | 自定义核需要明确训练导数与编译元信息 |
| torch.library.triton_op/wrap_triton | 均存在 | API 可见不意味着 Triton 能在当前 MPS 上执行 |
| 本地 Inductor MetalScheduling/MetalKernel | 存在且注册 mps backend | 可以评估局部 torch.compile，不应笼统说 MPS 没有编译后端 |

PyTorch 2.14 官方 MPS 页面也列出了 shader、Signpost、capture 与事件接口。本机 profiler 源码说明 wait_until_completed=True 会让每次 GPU 操作等待完成，适合拆解时间线，会明显扰动性能。正式速度结果应来自关闭捕获的独立计时窗口。[PyTorch 2.14 MPS API](https://docs.pytorch.org/docs/2.14/mps.html)

本地源码证据位置：.venv/lib/python3.12/site-packages/torch/mps/__init__.py、torch/mps/profiler.py、torch/_inductor/codegen/mps.py，以及 codegen/common.py 中 mps 到 MetalScheduling 的注册。未创建 MPS/CUDA tensor，未启动 profiler、capture 或编译任务。

Apple 官方页面确认 PyTorch 的 MPS 路线使用 MPS Graph 与经过设备适配的 Metal 核，也允许自定义 Metal。该页面仍写着 PyTorch 2.11 的安装说明，因此本机具体 API 以已安装的 2.14.1 源码为准；不能据此降级或升级正在进行的实验。[Apple：Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/)

## 3. Mac 近期路线：优先保留原数学与训练协议

以下成本是开发工作量估计，不是承诺。每项独立实现、独立测速；不把多项变化一次叠加进当前冻结队列。

| 优先级 | 候选 | 对应热点 | 开发成本估计 | 主要限制与验收 |
|---|---|---|---|---|
| P0 | Instruments + MPS Signposts 短窗口，分别观察 dense/sparse 的前向和反向 | 判断 CPU 发射、GPU 执行、同步和内存访问谁主导 | 半天至 1 天 | 捕获会扰动时间；只用来定位，再用无捕获完整 update 验证 |
| P1 | 固定容量的设备路由张量、邻接映射与按 shape 分桶执行 | sparse 的 CPU 排序、lane 拆批、gather/scatter | 2–4 天 | 不得改变选择规则、并列分数次序、图切分隔离与 idle 冻结；N32 常全激活，不应强行走 compact |
| P1 | 复用常量图张量，合并标量/诊断读取，控制临时缓冲生命周期 | launch、CPU/GPU 往返、分配与拷贝 | 1–3 天 | 不覆盖 autograd 保存的值；保留每步非有限检查、目标计数和原始记录 |
| P1 | coefficients 的纯张量子函数局部 torch.compile | 大 N 最重的空间计算及其反向 | 2–5 天试验 | 记录图断裂/重编译、编译耗时与缓存；编译成功不等于训练更快 |
| P2 | Metal 融合邻居 gather→局部映射/门控→归约，以及对应 backward | coefficients 的多次小 kernel 和大临时张量 | 约 1–2 周起 | 先保留成熟 GEMM，融合外围访存/逐元素运算；手写梯度需对照参考实现 |
| P2 | Metal 内的 C16 顺序 commit 及必要 backward | Joseph、m/p/trace 与 slow/credit 顺序状态更新 | 约 1–3 周起 | 顺序依赖保留在设备循环内；并不变成独立 token，不跨反馈可见边界 |

### 3.1 路由与形状稳定的具体做法

当前 v09/model.py 的稀疏路由会执行 scores.detach().cpu()、route_step.cpu().tolist()，随后用 Python set/sort 为每个 batch lane 选组。不同路线还可能拆成两个更小的 compact 批次。dense 在 active_fraction>=1 时直接返回全图索引，避开上述 CPU 排序，因此这项主要解决稀疏条件，不是 dense 的主要成本解释。

候选实现是每个原静态连通分量使用确定的预算槽，固定 K 个 active ID 和有效 mask；top-k/稳定排序、inverse map、缺失边 mask 尽量留在设备。每个 lane 可以选不同 ID，张量形状保持一致。将 B 个 lane 的固定 K 子图合并批处理，代替“每种 ID 集合一次 Python 执行”。如果分量预算不同，按少量静态 shape 桶处理，而不是缓存所有可能路线组合。

保持以下语义：mandatory 路径和探索槽仍计入原预算；摘要只来自块起点、分量内部历史；无效 padding 不获得边、状态更新、读取质量或梯度；全局 pending 信用在 compact gather 前照原顺序消费；idle 原节点的全部状态保持原规则。不能把所有 lane 强行改为同一条路线，也不能通过读取全图当前 token 摘要换取便宜调度。

### 3.2 编译与缓冲复用的边界

先从固定 N/d/C 的纯张量 coefficients 子函数入手，把 CPU 决策、文件记录、字典状态组装留在区域之外；把 logits、完整末态和参数梯度一起对照。无重编译的热态速度有收益后，才扩大编译区域。编译冷启动、图断裂与实际 eager fallback 都要单独记录。

官方 torch.compile 文档明确 reduce-overhead 的关键收益来自 CUDA Graphs，针对 CUDA 图且仍受输入修改等限制。这个选项不能直接作为 Mac 的“减少 Python 开销开关”。本机有 Metal codegen，只能证明具备候选执行路线，具体算子组合与训练反向仍需检验。[PyTorch 2.14 torch.compile](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html)

缓冲先复用于拓扑、mask、路由元数据与生命周期明确的无梯度 scratch。m/p/x 等训练状态参与跨块梯度，不能仅为省分配就对尚被 backward 引用的 storage 原地覆盖；ping-pong 缓冲也必须证明旧值已经不再被引用。不要在训练内循环反复 empty_cache。

当前 train.update 在计时前后做 synchronize，且 loss 的非有限检查与若干日志会取回 CPU。这是同步点候选，但当前每步安全检查与可恢复状态不能直接删除。优先把诊断标量打包一次传回、把显示节流到 1Hz，必要时另外提供连续窗口的准确计时。减少同步后仍应在计时窗口边界等待 GPU 完成，不能靠漏计异步工作制造加速。

### 3.3 自定义 Metal 的训练要求

compile_shader 是计算核入口，不会因为 Python 能调用 Metal，就自动为任意 shader 推导正确梯度。候选核需声明输入、输出和 mutation；通过自定义算子/明确 autograd 公式接入，并提供编译所需的 fake/meta 行为。只优化 forward 而让 backward 退回昂贵拆解，整体上限仍很低。

coefficients 的邻居归约 backward 涉及梯度回填；融合后的归约顺序可能产生浮点差异。commit 同时有可微的 m/p/trace 演化和原本显式 detach/no_grad 的局部反馈，必须逐项沿用原来的导数边界，不能把整段 commit 意外 detach。Joseph 的正定/有限性、clamp 分支及多块梯度都属于验收内容。

## 4. 未来 CUDA 路线：复用接口，另做后端

### 4.1 Triton 自定义算子与反向

以固定 K 图块和 C16 为接口，优先融合 coefficients 的 gather/点运算/归约与顺序 commit 的小矩阵操作；大投影继续交给成熟矩阵库。目标是减少读写、launch 与 autograd 小节点，而不是把每种算子全部重写。

PyTorch 官方 Triton 教程推荐 triton_op/wrap_triton 接入编译，并通过 register_autograd 显式提供训练导数；若 backward 还调用 Triton 核，该核也需按算子规则封装。它提供的是未来 CUDA 后端开发路径，不是当前 Mac 的可用加速保证。[PyTorch：User-Defined Triton Kernels](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)

成本预估为每个主要融合区域约 1–2 周起，复杂 commit backward 更高。先完成单区域正确性，再测完整前向+反向+optimizer；内核微测速再快，不能替代有效目标/s。

### 4.2 CUDA Graphs

先捕获形状固定、内存地址稳定的数值子区域，路由 ID 的值可变而 shape 固定；输入与状态通过持久缓冲更新。当前的 Python 排序、CPU 取数、状态字典重建和变长 compact 路线不能直接原封不动整体 capture。梯度、优化器状态、RNG 与运行时状态的生命周期也需纳入设计。

PyTorch 的常规 graph capture 要求静态 shape/layout、稳定内存地址，禁止 .item() 等 CPU/GPU 同步；CPU 工作不会在 replay 时自动重做。make_graphed_callables 可用于部分前向与反向区域，适合在全网不可捕获时逐步推进。若使用 CUDA 的 TF32，也必须记录其精度设置；它会改变 FP32 matmul 的内部数值行为。[PyTorch 2.14 CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html)

NVIDIA 文档将图的主要作用解释为把反复提交的工作预先定义、实例化，再低开销重放。它减少提交成本，保留依赖关系；不会把必须等待前态的递推自动变成时间并行。底层虽有图更新和条件节点，不能据此假定 PyTorch wrapper 已支持任意 Python 动态路由。[NVIDIA CUDA Programming Guide：CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)

预估先做可捕获区域需 2–5 天；全更新 capture 取决于状态与优化器封装，可能明显更长。只有 trace 证明 CPU 发射占比较高时，这条路线才有较大上限；GPU 算术/访存本身占主导时应先融合。

## 5. 需要单列为新训练条件的变化

增大 batch 可能改善 GPU 利用率，但会改变同时聚合的样本数、梯度噪声、固定 token 预算内的 optimizer 更新次数，甚至连续数据流的 lane 调度；不能原地当作当前实验的等价提速。梯度累积也不会使“每 B2 更新一次”与“累计后更新一次”变成同一训练过程。BPTT 长度、C、反馈提交频率的变化同样需要另列条件。

FP16/BF16/autocast 是数值方案，不是纯执行重排。API 存在不代表所有 MPS 算子及该 Mac/OS 组合都支持目标 dtype。优先只评估内容投影和输出头的低精度，m/p/slow/credit、归一化、协方差和关键累加保留 FP32；已有极小的局部改变量尤其容易在低精度状态存储中消失。记录溢出、梯度缩放、相位/协方差与长期累计误差，再比较相同有效目标的质量。CUDA TF32 也单列，不与 IEEE FP32 混报。

数据预取、CPU token 数组缓冲和有限的异步写盘可以改善端到端时间，但已有主测速已经排除了数据采样与保存，所以这些优化不会解释表中的 40–191 倍差距。应分别报告训练数值核心时间和包含记录/保存的墙钟时间。

## 6. 显存还有余量，能否一次放入更多训练内容

可以放更多，但首先区分容量与正在执行的计算。显存占用回答“保存了多少参数、激活、状态和缓存”，GPU 利用率还取决于同时有多少工作、矩阵尺寸、访存带宽、发射间隙与串行依赖。分配一个很大的缓存能提高占用数字，却不会增加有效目标/s。当前参数与运行时状态本来就按 device 驻留；缺少的不是简单把模型再搬一次到 GPU。

### 真正增加同一时刻的工作：优先试更大 B

保持 T64、C16、d8 和节点数不变，后续独立工程试验可以依次测 B=2/4/8/16/32。每个 lane 持有独立的 m/p/x/slow/phase/credit/pending，使用同一个模型参数；批量执行相同算子，可以摊薄发射成本，让小矩阵具有更大的并行批次。稀疏条件同时记录不同 route 数、每个 shape 桶中的 lane 数及 padding 浪费，否则 B 增大仍可能被 Python 拆回许多 B1。

这里的“更多内容”主要是更多同时活跃的独立会话，不是每个会话获得更长历史。不要把不同文档的动态状态合并；也不要把 T/BPTT/C 增大当成与增 B 等价。T 增大还会增加顺序工作，C 改变会改变块起点冻结与反馈时序。

建议先从同一 checkpoint/初态分支做短工程对照，测完整 update 的有效目标/s、每目标时间、实际状态/激活字节和稳定性，寻找吞吐开始饱和的 B；不是以耗尽内存为终点。这类试验本笔记没有执行。

增 B 后每个 optimizer update 聚合更多样本，固定 12M 目标时更新次数减少，数据流 lane 调度也会变化，所以训练质量必须另列条件。若坚持保持现有 B2 的逐次 optimizer 更新数学，不能简单把多个原 update 并成大 B；前一次更新后的参数依赖仍然存在。至少报告每次更新的有效目标数、总更新数与学习率按 token 的进度，并观察 loss/验证质量是否变化。

### 多会话驻留与多流

可以维护一池会话状态，选出当前 ready 的 lane 组成大批次，完成后按原会话 ID 回填。inactive 的会话必须保持原 idle 规则；读取路径、pending 信用和 session reset 分开管理。这样的驻留池有价值的前提是后续计算确实批处理，只有更多 Python 对象驻留不会自动提速。

未来 CUDA 可以研究独立 stream 的预取与计算重叠，或图安全区域的并发；先确认 kernel 尚未占满计算/带宽。不同 stream 不能越过同一会话的状态依赖。Mac 近期优先使用 B 维合并与 Metal 的实际时间线，不能假设 torch.mps 具有与 CUDA 完全相同的 stream 控制接口。多进程或多个训练任务竞争同一设备不保证更快，本轮也没有启动任何额外队列。

### 数据常驻与双缓冲的实际规模

当前冻结 train 有 3,084,660 个 token，连续 int32 数组约 12.3MB，全部转 int64 约 24.7MB。只让训练数组驻留设备、按既定流索引 gather，可以减少微小批次的拷贝/分配；不需要把 test 搬入训练进程。常驻内存也不等于模型可读未来内容，模型输入仍严格只给当前因果片段，原始来源元数据仍不进入路由或反馈。

也可用两个固定输入缓冲：CPU/采样端准备下一批，GPU 计算当前批；设备读完前不覆盖该缓冲，队列有界、Stop/checkpoint 时完整落盘。CUDA 的数据拷贝重叠需要正确事件与存储生命周期；MPS 上先测实际收益，不照搬 pinned-memory 配方。由于已有核心测速排除了采样与传输准备，且每批只有 B2×65 个 token，这项预计解决的是端到端零碎开销，不能用来解释 coefficients 与反向的主要耗时。

### Mac 与独立显存设备的差异

Mac MPS 使用统一内存，CPU 与 GPU 共享物理内存预算；CPU tensor、MPS tensor、驱动工作区与操作系统都消耗这个预算，逻辑 tensor/device 边界仍然存在。把所有东西都缓存到 MPS 可能挤压系统内存，而不是无代价使用一块闲置的独立显存。未来独立显存 CUDA 设备可以容纳更大的激活和状态批次，但仍应通过吞吐与时间线判断瓶颈，显存剩余百分比不是扩大 B 的充分条件。

建议可测组合为：先 B 扫描找饱和点，再固定该 B 对比设备路由与 shape 分桶，最后比较输入双缓冲/训练数组常驻。每次只改一个因素，同时报告训练数值核心与完整墙钟；更大 B 或精度改动另做验证质量实验。

## 7. 统一验收与推进顺序

1. 先冻结参考实现、checkpoint、同一输入/初态及训练协议，另起优化实验目录。当前正式实验不混入源码改动。
2. 按参考实现比较逐块 logits、完整末态、所有参与学习的参数梯度与一次 optimizer 更新；多块/会话 reset、cut、inactive、pending 消费和因果前缀都覆盖。保留原有 detach 边界。小规模 CPU float64 可作数值参照，MPS 对照 FP32；不能要求 MPS 分配 float64。
3. 对自定义核检查导数，再检查完整 BPTT；归约换序允许事先声明的浮点误差，但必须单独量化误差与长序列累积。FP16/BF16/TF32和更大 batch 另走训练质量对照。
4. 无捕获/无 profile 的完整训练更新测速，覆盖 N32/128/512、dense/sparse、相同有效目标数；充分 warmup，建议至少 100 次热态更新并重复独立窗口。报告 sum(targets)/sum(seconds)、中位与尾部延迟、内存、编译冷启动、实际 active ratio 与完整墙钟时间。
5. Transformer 使用与当前训练相同的 4096 上下文机会，按会话年龄覆盖实际 KV 长度；不要继续用 context256 的旧数字宣称新完整倍率。
6. 若候选仅改善局部微核而完整 update 无稳定收益，停止扩大该候选；若改变学习或路由语义，重新归类为架构/训练方案，不记为等价系统优化。

建议实施顺序：短 trace 定位 → 设备路由与固定 shape 的 sparse 批处理 → coefficients 局部编译/融合并包含 backward → 顺序 commit 的设备循环与导数 → 再决定 CUDA Triton/Graph 后端和独立精度方案。近期不预先承诺能达到 Transformer 的某个百分比。
