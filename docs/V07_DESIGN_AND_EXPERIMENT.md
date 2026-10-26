# Bio-Matrix v0.7：训练生命周期与局部反馈修复

> 归档说明：本文保留写作时点的设计或状态。项目已停止v0.9后续训练；最新执行状态见[实验索引](RESULTS_INDEX.md)。公开副本不含权重与语料。

v0.7 单独实现于 `v07/`，入口 `run_biomatrix_v07.py`，输出 `results/v07/`。v0.6 源码、配置、数据和结果保留；两版权重和运行状态不兼容。此次更改解决已定位的实现与实验协议缺口，性能收益通过新训练检验。

## 修复范围

| 项目 | v0.6 | v0.7 |
|---|---|---|
| 状态生命周期 | 每个随机64-token窗口从零开始 | 每条lane保留4096-token会话；独立切换、B插入、回A后文均保留状态，只有会话边界重置 |
| 时间信用 | 16-token TBPTT | 默认128-token BPTT；另列16/64/128配对 |
| 训练年龄与评估 | 训练状态年龄1–64，carry评分65–128 | 连续会话及未标记A→B→A训练覆盖这些年龄；评分加入相同年龄对照 |
| 输出直通 | 当前embedding直接加到输出 | 主图模型全部经过网络；直通作为bypass消融，同宽stateless单独从零训练 |
| 局部更新 | 历史key平均乘矩阵误差，缺失输出头Jacobian | 使用实际关联读出的query与权重，解析计算完整输出头VJP |
| 慢记忆信用 | 标量response被用于近似梯度方向 | response定义沿图的分布式慢记忆读出，并提供非负步长预条件；方向来自实际读出VJP |
| 脉冲强度 | unit key与mass=1也贡献能量，中心首跳必开 | 强度只统计语义value和载波；零语义可关门，使用train信号校准阈值 |
| 时间带 | 指数衰减外还乘每轮(1−mixing) | 单步指数松弛，mixing只调目标幅度，移除额外旧态强收缩 |
| 恢复 | 参数/优化器/RNG | 加完整运行状态、会话与源文档游标、待呈现token和真实目标计数 |
| 诊断 | 部分末token统计与prefix干预 | 所有有效目标位置诊断，加清短期、清慢状态、只清信用缓存等冻结权重干预 |

保留的架构包括：动态多维内部矩阵、对角加秩一外部映射、结构化行列旋转、多频Fourier调制、显式局部相位耦合、有限时间带、Joseph协方差更新、有限资格迹、支持度、慢状态和固定度数的实心球体图。没有JEPA、情境状态副本或随历史增长的路径记录。

## 实际读出与局部修订

对于节点n，设慢矩阵为Sₙ，当前到达key与有限trace形成的实际单位查询为qₙ。其关联读出为：

\[
R_n=(S_n q_n)q_n^T,\qquad R=\sum_n a_nR_n.
\]

非负权重aₙ由局部兼容性及图返回响应确定。返回响应由外层read ports沿现有flow有界回传，与read ports相加后归一，定义分布式慢记忆的实际汇聚算子。其目的是让可达内部节点的slow也能影响读出、收到对应反馈；切断拓扑后不能跨断面读取。fast载波和m仍只从最外层读取。三种图对照共用此算子及其计算预算。

这里的标量返回响应是所定义读取算子的权重，没有把它宣称为整个非线性传播网络的精确伴随。算子无需把每个慢矩阵再次沿路径逐轮搬运，额外复杂度O(BNd²)，状态容量不随历史增长。

R进入read_projection的memory半支，后续经过完整的残差FFN及两层LayerNorm，最后与共享embedding作线性词表输出。下一token到达时，`embedding(token)−expected_embedding`给出上一输出hidden空间的负CE梯度。对LayerNorm、GELU、FFN和read_projection进行解析VJP。令G为到R的负梯度，则：

\[
\Delta S_n\propto a_n(Gq_n)q_n^T.
\]

实际步长再乘曝光支持度/创新一致性、非负响应预条件及实际读权重参与数 E=(Σa)²/Σa² 相对32节点的校正。共同降低置信度仍减弱更新。单次增量限幅0.05；容量投影只裁当前query子空间，保留正交经验并维持慢矩阵Frobenius范数≤4。none校正另列配对。支持度在此是曝光量近似，不冒充方向覆盖度。历史trace参与实际q的构建，不再任意逐列乘到另一种读取结构的梯度上。

缓存保存最近8个token的head输入、query、读取权重、期望embedding及有效位，容量固定；不复制完整节点状态。自然语言训练采用下一可见token的一步反馈；显式延迟API接受已知lag并一次消费缓存，尚无自动语义归因器。缓存的query与权重视为局部修订时的常量，采用一阶局部学习。这个VJP精确对应上述**显式直接读取分支**；slow影响控制、旋转、路由和未来活动的间接信用继续依赖BPTT。整网长期学习仍是待检验问题。

每次optimizer更新后，仅把反馈缓存有效位置为false，避免用新head参数解释旧head缓存；slow、m、x等经历保留。下一可见token重新建立一致缓存。这会跳过每个优化边界的一次旧反馈，不会清空运行记忆。

## 脉冲和时间带

脉冲强度是载波RMS与事件包中value RMS的合成，unit key及mass仅用于路由和证据归一，不提供语义能量。门槛为 `0.001+0.009·sigmoid(control)`，分数为 `energy−threshold·(1−0.25·cosine)`，代理温度0.01。零能量时即使相似度最高仍严格关门。训练前400步由软门逐步切到硬前向；验证和测试始终硬门。

`results/v07/calibration/check_gate.py` 使用冻结train窗口与初始化模型检查各壳到达率和硬门率；该校准不使用测试集、不运行优化器。保留原先过高阈值的诊断和修正后原始统计，避免只测试0/1极端输入。

时间带采用：

\[
x_{j+1}=e^{-1/(\tau J)}x_j+(1-e^{-1/(\tau J)})u_j.
\]

u包含受控消息和输入；旋转保范。τ描述孤立旧态项的衰减尺度，消息回流、控制变化和边界限幅仍会影响全网实际记忆。测试通过隔离目标项验证衰减，不把τ直接当作整网保证的记忆长度。

## 数据、连续训练与对照

复用v0.6已冻结的WikiText-2 raw官方train/validation/test及仅train拟合的4096 byte-BPE。默认路径 `data/v06/wikitext2` 只校验，缺失/损坏时报错，禁止静默重建。原始revision、SHA及文档切分见其manifest。每个种子遍历同一打乱文档排列，所有模型接收相同的有效输入、目标、reset与padding掩码。

每条lane保持4096个输入token的会话。基底A按原文顺序读取，在随机256–768个A token后插入不同文档的64-token B片段，再回到尚未见过的下一A位置；A结束时换到另一独立文档。50%接缝显示EOS/BOS，50%完全无特殊标记。缓存自带首BOS/尾EOS先按真实ID剥离，防止未标记条件泄露边界；标记真实文档接缝保留合法的原文EOS/BOS监督。不同源/不连续位置及人工标记的伪next-token目标只做CE mask，mask从不抑制模型或局部反馈。document-reset是同输入、同目标的单独消融。每次更新保存lookahead与源offset，以便精确恢复。

节点动态输入映射同时作用于fast注入和Joseph观测；概率矩阵使用平稳先验回归，未观测方向也受限。公式与局部容量投影的范围见[V07_CONFORMANCE.md](V07_CONFORMANCE.md)。

默认AdamW、FP32、batch2、sequence128、BPTT128，3e-4学习率、200步warmup、cosine、weight decay0.01、梯度裁剪1。固定24,000步的满条目标上限为6,144,000，实际目标数、接缝mask、状态年龄、会话重置及训练集等价曝光写入日志和ledger；不把它直接称作收敛预算。新流seed11完整data-only计数见 `results/v07/conformance/aligned_stream_budget/result.json`：6,100,155有效目标、43,845接缝mask；原文档流计数保留作历史记录。

Transformer为两层pre-LN、RoPE和因果SDPA，参数量接近full；context固定256。逐query滑动mask使chunk长度改变时上下文一致，逐lane文档重置清除旧KV有效位并重置RoPE位置。KV存储有固定上限。

stateless使用与Bio相同width、embedding、FFN和输出头，从零独立训练，无运行记忆；它控制当前token直接预测的能力，参数总量照实报告。bypass在Bio读出额外加当前embedding，仅作为机制消融。

默认core：32/128/512节点 × message/independent/full/Transformer/stateless × seeds11/23/37，共45组。每个规模先完成五组配对，再扩大规模，之后补种子。图网络共享图、状态形状和读写算子；训练token配对，FLOPs与墙钟时间另报。stateless在不同节点规模栏位重复作为配对参考，本身没有节点数。

额外阶段仅显式选择时执行：

- mechanism：128节点，bypass、no_local、no_slow、no_probability、no_dynamic、single_timescale、no_threshold，以及full的document-reset消融。
- credit：32节点，full/no_local × BPTT16/64/128，会话生命周期相同。
- write_scale：32/128/512的full无参与数校正，与core的participation条件配对。
- iterations：128节点，independent/full × J2/8，J4由core提供。
- topology：128节点三种图模型，切断中间壳边界，固定原候选图和端口。
- capacity：三种图模型，(N,d)=(32,32)/(128,16)/(512,8)，固定Nd²=32768；总状态和真实成本另报。

## 评分与复现

验证16窗口、测试64窗口，每个窗口prefix64、target64。所有模型/干预使用相同目标，保存窗口、评分输入/目标和评估协议SHA；模型选择仅使用验证carry NLL。PPL是本byte-BPE窗口指标，不与不同分词器的论文数字横比。

干预包括carry、reset-prefix、shuffle-prefix、unrelated-prefix、A→B→A、带EOS/BOS的无关前缀及reset-each-token；Bio另有no_local/no_slow，bypass另有head_only。不适用项目明确标记。每个目标位置记录NLL、KL、预测置信度、门控/到达/慢状态/局部更新等诊断，汇总前16和全段损失。

连续自然文本并不保证每个目标必须使用历史，故同步曲线和单次PPL提高不能验收机制。比较多个种子的full优势、历史消融损失、无关干扰与返回代价，并一起审视计算成本。A→B→A加入B→B→A、shuffle(A)→B→A、B→A→A对照，匹配年龄、近期内容和远史词频。远期重复/稀有远期重复子集只用train频率预注册，固定测试64窗口的稀有远期重复子集含114个目标位置；它是评分代理。

`last.pt`在同一完整优化边界保存参数、optimizer、所有runtime state、流游标、排列、RNG和最佳权重快照。参数更新后反馈缓存已失效；恢复后继续原始文档下一位置。验证使用独立state，不清空训练state。源码/数据/配置/设备变更拒绝混用输出目录。

CPU单测、短程smoke和preflight属于工程验收。正式训练、失败和未执行状态分别登记；v0.6结果保留为旧协议结果。GPU队列使用v0.7自有锁及旧版共享kernel锁，旧锁只加锁、不重写内容。

## 历史验收：初版v0.7（2026-10-23；已保留）

- 26项CPU检查通过：输出头解析VJP、关联读出autograd/有限差分、内部慢矩阵实际更新、断图隔离、时间带、Transformer跨上下文分块等价、逐lane重置、padding隔离、全家族精确恢复、干预一致性与冻结权重。
- 五类core模型各完成2次CPU工程优化更新，走通验证、全部适用历史干预、权重/运行状态保存、曲线和聚合报告。原始目录 `results/v07/validation/cpu_smoke/`；已核对五组训练批次与评分目标相同，控制台包含原位更新的回车进度条。
- 正式B2/T128/BPTT128配置下，N32 full完成2次CPU资源更新，第二次约1.89秒/256有效token，运行状态126,466字节。进程累计峰值RSS约1.07GiB；这是该CPU进程的快照范围，不能当MPS显存或N512资源预测。原始 `results/v07/validation/cpu_profile/preflight.json`。
- 仅train、冻结共享权重校准中，N32/128/512外壳到达率分别约75.74%/51.46%/40.74%，外层向外门开放率约29.73%/20.21%/17.72%。原始 `results/v07/calibration/calibrated_gate.json`。这检查信号能到达且门有选择性，不用于宣称能力收益。
- seed11的24,000次data-only流更新产生6,064,569有效目标和79,431 padding，约1.9664次train目标等价曝光；最大有效年龄25,651，年龄超过192的目标5,833,579个。没有构造模型或运行优化器。
- v0.6全部10个Python源文件哈希保持不变。实际被旧队列持有的kernel锁已检验拒绝v0.7进入模型构建，旧锁内容保持不变；该锁检查模拟MPS设备选择，未分配GPU。

汇总与完整source hashes见 `results/v07/validation/release_checks.json`，单测记录见同目录 `unit_tests.log`。当前工具进程中的MPS可用性为false，且旧训练队列仍持有GPU锁，因此MPS实跑尚未执行。v0.7正式core的45组均登记为not_executed；完整训练和长期收益需要后续实跑。

## 历史验收：第二轮交叉审查（2026-10-23；旧源码快照）

本轮最新源码验收位于 `results/v07/audit2/release_checks.json`，30项CPU检查全部通过，五类模型再次完成短程完整流程，T128/BPTT128 CPU资源预检通过。上一轮源码及验收快照保留在原目录和 `audit2/before/`，未混写旧检查点。

已修复的可复现程序问题：

1. 首次检查点尚未保存时中断，再启动会追加重复训练步。现在按提交水位0归档未提交日志及best；回归测试注入故障验证恢复后的step唯一。
2. 最终验证日志写完、best保存前崩溃时，再启动可能跳过验证并找不到best。现在last.pt记录独立validation_step提交水位，验证日志与best按该事务状态恢复。
3. ECE的十分位边界重复进入相邻两箱。现在互斥整数分箱；p=0.5且全部错误的ECE从错误的1.0恢复为0.5，p=1归最后一箱。
4. 载波每内部tick的旋转未除J，增加内部轮数同时放大总旋转量。现在按每token时间归一，隔离相位/输入后的J2/4/8结果相同；原始失败与修复后回归均保留。

第二轮当时记录的结构问题如下；这些实现差异已在本次完整对照中补齐，最新验收以conformance目录及下方说明为准：

- **广播观测**：fresh drive只作标量图混合，满足drive_i=α_i[value,key,1]。到达质量充分时，Joseph得到相同key/value；动态外部映射作用fast注入，但未用于Joseph观测。节点可因初始形状、可靠度及到达历史而不同，但节点数增加并未自动增加观测变换多样性。完整实现“各节点先过滤再写内部矩阵”需要单独定义一致的观测坐标与协方差更新。
- **局部写入随N稀释**：在均匀归一读权重的隔离条件下，整体慢读出的一步变化按1/N缩小。N32/128/512复现为0.00375559/0.000938898/0.000234724。相同步数下不同规模的有效局部写入速度并不相同，需标定对照；直接将学习率乘N会同时改变稳定性和预算，尚未这样改。
- **协方差幅度**：p固定大小但无幅度上界。固定低秩观测时未观测方向为1+0.002001t，25,651token约52.33；PSD未失效，但长期增长可能使controls饱和。状态大小固定不保证每个数值都有统一上界。
- **情境切换与年龄控制**：训练在真实文档边界重置，未训练无重置的A→B→A或EOS→BOS。ABA还比carry多64次活动，return_cost含干扰及年龄变化。“不同文档”也未保证语义独立，尚无预先固定的真实长依赖子集。
- **证据与预算**：test64窗口来自38文档、4096个不同评分位置；重复窗口不充当独立训练种子。固定Nd²不等于固定总参数/状态。同seed的三个stateless及部分TF配对存在可复用训练，不能作为额外独立证据。训练diag取最后处理token，文档尾padding时需结合mask解释；正式评估按每个有效目标采样。

冻结CPU数学复现见 `audit2/model_audit.json`；采样与容量核对见 `audit2/evaluation_sample_and_budgets.json`、`audit2/fixed_capacity_budget.json`。未运行GPU或正式长训练，也未将上述结构选择临时改成未经配对的新算法。


## 当前对照版与A800验证

当前协议指纹为 `v07-aligned-content-context-lm-4`；输出使用新的 `results/v07/aligned_public_lm`，不续接旧v0.7检查点。完整构想原文归档在 `docs/sources/BIOMATRIX_FULL_IDEA.md`；设计—代码—回归检验逐项关系在 `configs/v07/conformance.json`。运行 `python scripts/check_v07_conformance.py` 会重新检查实现要求、记录源码指纹并保存独立报告。

CPU检查与短程CUDA优化用于验收程序和机制；正式配对训练另存状态、原始loss与逐token干预。A800采用服务器已有CUDA PyTorch和FP32，精确版本与硬件写入远端环境清单；Mac使用MPS。不同设备的检查点不混用。并行调度保持每组原定数据/种子/batch/步数，使用独立worker目录、进程和检查点，先测1/2/4进程总吞吐再选择并发。相同规模的五个模型完成后才扩大规模；三个种子分别报告。

自主功能分工、规模收益、自然语言规则反转以及超过固定信用范围的归因仍由公开任务结果检验。有限动态状态、局部耦合和资格元信息已实现，实际能力结论等待配对结果。
