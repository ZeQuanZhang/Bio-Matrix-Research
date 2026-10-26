# Bio-Matrix v0.9 训练提速：数学与算法审查

> 归档说明：本文保留写作时点的设计或状态。项目已停止v0.9后续训练；最新执行状态见[实验索引](../../../docs/RESULTS_INDEX.md)。公开副本不含权重与语料。

日期：2026-10-25。范围：当前代码只读、既有实测复核、一手论文和官方实现核对。本文是研究计划，没有修改 v07/v08/v09 源码、训练配置和运行进程，没有新增 GPU、训练或远程任务。

**判断：当前最值得做的是保持完整矩阵的因式分解、时轴收缩和融合执行。** 当前图本来就是固定度数稀疏图，快态也已有三种同方程递推；把整个模型再包装成一个大矩阵，或单纯换成并行 scan，解决不了主要瓶颈。完整矩阵可以继续保留，同时让“共享内容只传幅度”“冻结历史只读一次”“连续小算子留在同一个内核里”。

## 1. 当前已经实现什么

核对源：`v09/core.py`、`v09/feedback.py`、`v09/model.py`、`v08/recurrence.py`、继承的 `v07/model.py`；对照 `docs/V08_IMPROVEMENT_PLAN.md` 和 `results/v07/math_analysis_20261024/MATH_OPTIONS.md`。

| 项目 | 当前状态 | 后续不能重复记作新成果 |
|---|---|---|
| 完整 x/slow/m/P | 已保留；x 含多个时间带 | 低维地址不等于低秩内容状态 |
| 动态内外映射 | 已是 diagonal + rank-one 的 O(d²) 实现 | 没有需要替换的通用 O(d³) 映射乘法 |
| Q/K 地址 | 默认从 d² 投影到 8 维；Value 仍完整 d² | 再次宣称“首次将注意力地址降维”不成立 |
| 控制器 | 块起点统计、固定首层投影和 old 矩阵先按 B 计算再扩展 | 剩余仍有可避免的扩展和逐轮重复项 |
| Fourier | 已拆为接收/发送节点投影，沿边相加；bias 一次 | 后续优化空间是固定部分跨轮复用，不是首次端点拆分 |
| F1/反馈 | F1 方向 stop-gradient；期望 embedding、方向和 proposal 已整块批处理；信用每块整理 | Joseph 和 query 容量投影仍按顺序；F1 当前先建局部计算图再 detach |
| Joseph | 已用标量测量的 rank-one 展开，O(d²) | 仍可合并三项外积，复杂度阶数不再下降 |
| 快递推 | sequential、Hillis–Steele affine scan、triangular 三核均存在 | triangular 已是分块三角形式，不能把 WY 原理算作全新功能 |
| 节点组 | 真正 compact 执行、空闲冻结、同路由合批、全选短路已有 | 不同 lane 路由仍可能拆小 batch |
| 流式接口 | 已有保持块边界的前缀重放参考 | 不是增量高吞吐实现；训练提速和生成提速分开评价 |

当前公开短测速 `results/v09/mps_20261025_v09_final/` 为 B2/T64/d8/C16、FP32、6 次预热和 3 次完整更新计时，另有一次同步前向区域剖析。dense N32/128/512 的完整更新吞吐为 508.8/298.1/105.2 targets/s。前向 coefficients 为约 42.1/100.9/350.5 ms，serial slow commit 为 42.0/48.3/73.4 ms，fast recurrence 为 5.4/6.6/14.1 ms。反向约占完整更新的 57%–59%。这些数来自现成工程快照；当前长训上下文协议为 4096，不能直接拿该短测速的 Transformer 比值充当长训比值。

由此确定顺序：**先优化空间系数/消息和慢提交，再优化快核；每项包含反向。** 单核前向更快不足以说明训练更新更快。

## 2. 记号和计算账本

B 是独立序列数，C 是生物块边界长度，N 是实际活跃节点数，K 是每个目标的候选邻居槽，J 是同步空间轮数，d 是矩阵边长，L 是快时间带数，a 是匹配地址维数。当前典型值 C16/K8/J4/d8/L3/a8。总输入长度 T 可以包含多个 C 块。

每轮主要消息写成：

`transport_i = Σ_j w_ij sender_j`，`received_i = Σ_j w_ij drive_j`。

`w` 包含低维矩阵匹配、Fourier、含 null 的 softmax、跃迁门和绝对强度门，随 token、节点、空间轮变化。通信主项已经是 O(B C J N K d²)，没有隐藏的全图 N² 注意力。

当前重要临时量：

| 内容 | 元素量 | B2/C16/N512/K8/d8/L3 的 FP32 大小 |
|---|---:|---:|
| 一轮 sender 邻居完整展开 | B C N K d² | 32 MiB |
| 四轮若各保留一份上述激活 | B C J N K d² | 128 MiB；实际还有其他保存量 |
| raw 快态或 physical 的完整时间轨迹，各一份 | B C N L d² | 12 MiB |
| 一个冻结 m/slow 的重复时间副本 | B C N d² | 4 MiB |
| 持久 x，自身不含时间轨迹 | B N L d² | 0.75 MiB |

以上只是单个逻辑张量的账本，不是已测峰值。Autograd、重复中间量、allocator、优化器和词表头另计。`expand` 可以是 view，但随后将 B/T 合并的 `flatten(0,1)` 在 B>1 时未必仍可用相同步幅表达，可能产生物化。把显存留给更多独立样本，比把同一快照复制 C 遍更有训练价值。

## 3. 不改模型方程的候选

### E1：fresh 脉冲包只传播幅度，且复用它的线性控制投影

**状态：旧数学计划已推导并做局部 CPU 检查；当前 v09 尚未实现。** 完整内容消息仍为 d×d。

对每个 token，原始包 `p=[tanh(v), normalize(k), 1]` 在所有节点共享。初始 `drive_i=write_i p`；各通道乘相同标量 w，传播后入口恢复同一个 p；observe 的过滤结果没有写回 drive。因此逐轮严格有：

`drive_i^r = a_i^r p`，

`a_i^(r+1)=.25 a_i^r+.75 Σ_j w_ij^r a_j^r`，入口置 1。

边上只传一个 a，代替 2d+1 个通道。当前 `drive_controls` 明确 `bias=False`，故还可用 `W_drive(a_i p)=a_i(W_drive p)`：每 token 投影一次包，逐节点乘幅度，代替每轮每节点再次做完整线性投影。权重 w 仍由实际 a 和原包恢复的输入生成，梯度也通过 a 传播。

保留：mass floor `.05`、value clamp、两次 normalize 及其 eps、到达阈值 `1e-4`、硬门 STE、入口复位、所有历史 transport。不能将 `normalize(a k)` 无条件替换为 `normalize(k)`，近零幅度/零向量的数值和导数不同。

复杂度：脉冲边通信从 O(B C J N K(2d+1)) 变为 O(B C J N K)；drive 控制投影主项由 O(B C J N(2d+1)h) 变为 O(B C(2d+1)h+B C J N h)，h 为控制输出宽度。完整矩阵通信主项不变。

风险：重排可能改变阈值附近的浮点判定；多入口若未来输入不同包，单 a 表示失效。验收：FP64 前向/所有输入和参数梯度，FP32 a=0、极小 a、mass floor、跃迁/到达阈值两侧；整网状态、所有参数梯度、cut 与恢复一致。依据是当前代码归纳及旧 `MATH_OPTIONS.md`，不借论文的全图线性注意力结论。

### E2：基础内容载体保留合法到达，用标量质量乘共享完整内容表示

**状态：本次代码推导；当前仍显式搬运 base_content。** 当前单入口、父边恒等拷贝且 write_ports 是 0/1；没有对基础载体应用节点动态映射，所以每轮都有 `base_content_i^r=b_i^r F_t`，其中 F_t 是完整 fresh d×d 矩阵，b 是合法父边传播的 0/1 可达质量。

保留每轮 `b_parent * valid` 和入口复位，只存 F_t 与 b；当 communication 使用 fresh 时再广播乘入。b 不依赖 token/训练参数，可按实际 compact 图缓存逐轮质量。最终读取依旧沿实际 read_ports 汇聚，只在真实到达质量非零时有基础项；切断图严格为零。

这减少每轮完整基础矩阵的 gather/写回，持久矩阵不变；**不能把 F_t 直接无条件接 head**。缓存必须由实际 write/read_ports、edge_mask、active IDs 和 J 决定。当前常见 connected 配置中基础池化确实等于 F_t，优化前后都应保留这一归因事实。

复杂度：质量传播 O(B C J N) 可进一步在固定路由图上预计算为 O(JN)；完整 F_t 仅 O(B C d²)，使用时产生的 communication 仍需完整矩阵。风险：将来若基础路径加入不同映射、多个不同内容入口、非二值质量或额外噪声，须重推，不能沿用该特化。验收：逐轮矩阵/质量、read/logits/所有梯度严格对照；connected/cut/缺父边/不足 J/不同路由；不只验最终 NLL。来源：当前 `base_parent/advance_base` 方程，非外部论文能力主张。

### E3：用同一稀疏算子合并消息，并融合边匹配与归约

**状态：尚未实现。** 每轮同一个 W 作用于完整 sender 和 drive，可精确写成 `W [vec(sender), drive]`，完成一次邻居遍历/归约后拆分。如果结合 E1，则附加通道只有 a，节点 Value 为 d²+1 维。

先做 PyTorch 级联合聚合，检查 cat/布局开销；再考虑固定 K 的专用邻居内核：在寄存器/片上存储中算 score、null softmax、跃迁门、绝对幅度与完整 Value 累加，避免全量 `sender[:,source]` 的 B C N K d² 临时张量。完整外部矩阵仍计算，后向必须包含 Value 的散加与 w 的内积梯度；只融合前向收益有限。

可附带减少 receiver 中间张量：receiver 仅被 Q 投影读取，可融合 `P vec(M D+.15(Mv)uᵀ)`，不落盘完整 receiver；sender 仍必须保留完整矩阵供 Value。当前外部映射已经 O(d²)，融合主要降低 IO，不保证算术减少。

复杂度阶数仍 O(B C J N K d²)，重点减少边展开和重复读写。K8 的行归约很短，使用面向巨大 dense softmax 的内核未必合适。不能把 null 或绝对强度门吸收到重新归一化分母里，否则低置信度会被放大。

风险：回传原子加法顺序、门边界、mask 全关、稀疏索引错位；`response_filter` 后面仍需要实际 w，不能为了少存 w 直接删掉反向流。验收：输出、完整状态、所有梯度和 cut；记录前/反向 kernel 数、邻居临时峰值及完整更新 TPS。FlashAttention 的可迁移经验是精确融合、分块和反向重算以减 IO；它不直接实现本模型 K8 图核，也不提供本机倍速保证。[FlashAttention 原论文](https://arxiv.org/html/2205.14135v2)

### E4：块快照按时间做矩阵乘，避免重复复制 m/slow/P

**状态：固定特征已复用；read_block 的这部分尚未实施。** 块内 m/slow 冻结，query 随 token 变化。将 `S:[B,N,d,d]` 与 `Q:[B,N,d,C]` 直接相乘，得到所有 `S q_t`，再按 `(weight_t * value_t) q_tᵀ` 汇聚；不要先把 S 复制成 `[B*C,N,d,d]`。

`prior_m` 预测也用同样布局；固定读口的 `Σ_i port_i m_i` 只按 B 计算一次再广播到 C，不需要每个 token 重复做该完整矩阵归约。保持 query、confidence、read_scale、trust 和 F1 使用的真实读取信息。

复杂度仍 O(B C N d²)，固定 m 读口求和从重复 C 次降为一次，减少时间展开副本；布局优先 `[B,N,d,C]` 或显式 `einsum`，不能假设 einsum 自动最优，要检查实际 materialize。风险：转置/展开引入额外拷贝、误把随 token 的 weight/query 也冻结。验收：warm/cold、动态权重、未来扰动、全部梯度，记录内存分配与读区耗时。依据为矩阵乘法结合与块快照规则，无外部质量主张。

### E5：慢写始终用 rank-one 向量中间量，最终 slow 仍完整

**状态：旧数学计划已有推导；当前仍物化 direction/proposal/update/orthogonal 完整矩阵。** 单 query 的 F1 提议形如 `Δ=A qᵀ`，Frobenius 限幅为 `||A|| ||q||`，只需缩放 A。所有实际 gain、eta、γ 与 feedback_reachable mask 均须保留。

令 `u=normalize(q,eps=1e-12)`，`s=S u`，`h=Δu=A(qᵀu)`。原 query 容量投影可重写为：

`E_orth=||S||_F²-(2-||u||²)||s||²`，

`radius=sqrt(max(16-E_orth,0))`（沿用原零 slack 的有限次梯度），

`c=min(1,radius/max(||s+h||,1e-8))`，

`S_new=S+[c(s+h)-s]uᵀ`，零 update 时仍原样返回 S。

这里保留 `||u||²`，不能只写单位向量版本：q 极小或为零时 eps 语义仍要一致。最终 S 是满维 d×d，历史不会积成越来越长的因子列表。

复杂度仍 O(d²)，减少多个 d² 临时量和外积；方向/提议缓存可从 B C N d² 缩到 B C N d。风险：能量差在容量边界存在浮点相消，rank-one 范数的下溢/零 update 判定也可能不同。必要时保留边界参考分支，不能跳过容量约束。验收：随机及近半径 4、零/极小 query、零/极小 update、持续满容量、η/旧 S 梯度、正交内容保护；长会话轨迹。依据是原投影代数与旧本地推导，未测端到端性能。

### E6：Joseph 三项外积合成一项，并缓存真正静态量

**状态：当前已 O(d²)；外积进一步合并尚未实施。** 令 `r=P_prior k`，`s=kᵀr+R`，`z=weight/max(s,1e-6)`。当前展开可精确重写为：

`P_new=sym(P_prior+(-2z+s z²) r rᵀ)+1e-6 I`。

须保留 s 与 clamp 后分母的区别；不能将系数直接简化成 `-weight(2-weight)/s` 而遗漏 floor 分支。先验 `rho P+(1-rho)I` 与 process 对角项可融合；eye 可按 d/device/dtype 缓存；有梯度的 bounded node_anchor 可在同一块前向内算一次，不能跨 optimizer 复用旧图。

复杂度继续 O(d²)，减少外积和完整矩阵中间项。mean 的 `bounded(m)`、arrived 选择、P 对称化、jitter 全部保留。风险：抵消误差/PSD 边界；不要用直接减去一次 rank-one 的常规 Kalman 公式取代当前带 reliability 的 Joseph。验收：所有梯度、极小 R/denominator、weight=0/1/中间值、PSD/对称性与 covariance 上界、未到达冻结。此条属于本地精确代数候选。

### E7：F1 方向计算直接不建图，保留 eta 与旧 slow 的梯度

**状态：已做单个 CPU FP32 全网等价探针；生产源未改。** 当前 v08 的 `super().local_direction(...).detach()` 先执行带 autograd 的 head VJP/矩阵方向，再丢弃它的计算图。仅给这一段方向计算加 no_grad，与既定 F1 规则一致；不能将 gain、eta、容量投影或整个 slow 放入 no_grad。

已有原始记录：`results/v09/speed_audit_20261025/f1_no_grad_cpu/{probe.py,result.json}`。N32/d8/T32/B1，同权重、非零旧 slow、无 optimizer 更新：loss 都 8.3451623917；logits、完整 state、所有参数梯度最大绝对差均为 0；None 状态一致。eta 梯度范数 1.27535e-8、旧 slow 输入梯度范数 0.00721735，均保留。

复杂度不改变数值计算，减少 autograd 图创建/保存和释放；尚无速度测量。风险是包裹范围过大、将 F0 或其他训练规则也误用 no_grad。验收下一步仅需扩至现成 FP64/FP32 回归及原生设备，再测完整更新；不能用此一项保证训练不变或声称某倍速度。

### E8：保留逐观测顺序，融合慢提交；把无反馈依赖的状态整理挪出 token 循环

**状态：proposal/expected embedding 已批量；顺序循环仍在。** 将一个节点的 m/P/slow 留在同一融合执行单元中，内部按 C 个时间步依次处理，跨 B×N 并行。相同顺序、相同 arrival、相同 query 方向投影、相同 pending 规则，消除的是 C 次 Python/设备调度和中间张量往返。

可以预计算 reliability、variance、常量 decay、anchor 等；trace 不影响块内已生成的读取，可用固定衰减加权和求末状态；evidence_count 是非负整数加法加上限，最终一次 sum/clamp 可精确保持。response/phase 只取最后帧，信用已整块整理。innovation 的驱动取决于顺序 m，可在求出每步 innovation 后再整理最终递推。

support 不能简单只在最后 clamp：`s_t=min(100,.999 s_(t-1)+q_t² arrived_t)` 的早期截断会影响后来衰减。如真要并行，可研究标量 `min(a x+b,c)` 的封闭复合，而非丢掉中间 cap；对 C16 先保留融合循环通常更直接。slow 容量投影和 bounded(m) 也不能改成总和后只投影一次。

复杂度仍 O(B C N d²)，序列依赖仍 C；融合降低调度/IO而非宣称 O(log C)。风险：自定义 backward 的 F1、eta、旧 slow、概率和硬门代理梯度；不同 lane 的 reset/cut/pending。验收必须包含全状态/全部梯度/信用消费顺序/恢复，不以快核等价代替整个 commit 的等价。

### E9：继续复用逐轮不变的 Fourier 部分，以及重复归一化

**状态：Fourier 端点拆分已实现，固定 Fourier 部分跨 J 轮尚未缓存。** features 的 12 维中，当前块空间轮内只有 phase 的 2 维变化；可像 controls[0] 一样，将 Fourier 两端的固定 10 维先按 B×N 投影，逐轮只补 2 维 phase 投影，再沿边组合，后面的 harmonic/tanh 保持原式。

同一轮的 observe 与 receiver external 使用同一输入 u/v，可共享它们的 normalize 结果；当前最后 observation 的 value/key 来自传播后的新 drive，不能错误复用传播前的 value/key。receiver/source 相位 gather 和静态 source/parent/view 索引也可复用。

复杂度：Fourier 节点投影的重复固定部分由 B C J 次缩到 B 次，动态项和逐边非线性仍 B C J 次；d8/J4 下不是整个 coefficients 的阶数降低。风险：缓存了随 drive/phase 变化的部分、bias 加两次、跨块沿用旧状态。验收复用开/关整网前向/状态/所有梯度和硬门判定，再测收益；现有 controls 复用测试可作为模式，不能将未实现项写成已完成。

## 4. 快递推、scan 与 WY：已有能力和真正下一步

当前快态方程为：

`H_t=α_t H_(t-1)(I-β_t k_t k_tᵀ)+β_t v_t k_tᵀ`。

空间 J 轮生成本块所有 k/v/α/β；它们不读取本块尚未求出的 H，因此 raw 快态可重排。physical read 对 raw 状态旋转和 Frobenius bound；最后 physical 写入块末 x。这个 read-view bound **不会破坏 raw 递推的 affine scan**，但会妨碍直接使用只有 `H q` 的纯线性读出捷径。若把 bound 写回每个 raw 时间步，则又是另一方程。

| 核 | 当前状态 | 每个节点/时间带的主要 work | 主要依赖与风险 |
|---|---|---|---|
| rank-one sequential | 已实现，当前默认 | O(C d²) | C 步；d8/C16 实测比当前三角核快 |
| Hillis–Steele affine prefix | 已实现 | O(C d³ log C) | log C 轮；中间转移是 d×d 稠密，work 增加 |
| triangular | 已实现 | 构造/求解约 O(C²d)，完整轨迹重建 O(C²d²) | 保存 C² 系统和 C 个 d² 状态；浮点顺序不同 |
| work-efficient affine tree scan | 未实现候选 | O(C d³) work，O(log C) depth | 可减少 Hillis–Steele 的重复合成；仍可能不及 rank-one |
| 融合 rank-one + physical read + backward | 未实现，高于再换 scan 的优先级 | O(Cd²)，额外重算按 tile 设计 | 避免轨迹反复落盘；不改 C、bound、反馈边界 |
| WY/三角分块的 IO 优化 | 原理已有，专用 fused backward/tiling 未完成 | 由内层块长与读取形式决定 | 不能把更大生物 C 当成纯 kernel 调参 |

DeltaNet 论文的关键相关性是将 rank-one delta 转移的连续乘积写成紧凑 WY/三角形式；Gated DeltaNet 进一步包含 retention。它们说明完整矩阵记忆可训练并可并行执行，没有证明任意复杂反馈都能一起 scan。[DeltaNet 论文](https://arxiv.org/html/2406.06484v3)、[Gated DeltaNet 论文](https://arxiv.org/html/2412.06464v3)

当前三角核的独立推导：设 `u_t=β_t(v_t-α_t H_(t-1)k_t)`、`D[t,j]=∏_(l=j+1)^t α_l`、`g_t=∏_(l=1)^t α_l`。则

`H_t=g_t H_0+Σ_(j≤t) D[t,j] u_j k_jᵀ`，

`u_t+Σ_(j<t) β_t Dt,j（本地留存：`k_jᵀk_t`）u_j=β_t(v_t-g_t H_0 k_t)`。

所以 unit-lower 系统的 strict-lower 系数是**加号**；本地实现正确。论文 HTML 某些展示公式与递推/官方实现存在记号或印刷不一致，实施时应按递推、官方 naive 和本地参考交叉验证。带门控输出还必须保留相对 retention，不能从论文简式里漏掉 D。[FLA 官方 delta_rule 与 gated_delta_rule](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops)

具体交叉核对入口为 [delta_rule/README.md](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/delta_rule/README.md)、[delta_rule/wy_fast.py](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/delta_rule/wy_fast.py#L191)、[gated_delta_rule/naive.py](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/naive.py#L117)、[common/chunk_o.py](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_o.py#L128)。仓库 main 会变化，这些为本次阅读的直接文件入口，不代表已冻结到本地可执行依赖。复用内核前必须锁 commit 并保存源码；当前没有安装或导入 FLA。

**E10：融合完整轨迹计算、读出和反向。** 先做专用 sequential 内核：每个节点/时间带在寄存器中保留 H，按时间更新并计算当前旋转/bound/读口贡献，只有必要的块边界矩阵、读出和稀疏 checkpoint 落盘。反向从 checkpoint 重算局部 raw 状态，保持原梯度。不要通过反向求逆恢复 H：低 retention 或 `1-β||k||²` 接近零时会放大误差。

本模型的 head 使用完整矩阵池化，不只读取一个 `Hq`；因此不能照搬 FLA 的 q 向量 output-only 公式并宣称已经完全避免 d² 内容计算。可以融合计算和归约，不能取消完整矩阵的参与。norm 若仍在寄存器按 d² 求就保持原方程，且不要求复杂 Gram 范数捷径。

**E11：利用实际多时间带共享，而不是合并状态。** 当前各 band 共享 k/v/β，α_ell 由每个时间带的固定 τ 生成，块内不随 token 变化。三角特化可共享 `K Kᵀ`、初始 key 投影的可共享部分及 β；`D_ell[t,j]=α_ell^(t-j)` 可只按 band/C 构造，再广播 B/N，避免给 B×N 重复构造 retention。H_0、L_ell 和解仍因 band 不同而独立。默认 sequential 也可融合共享 key/value 加载。所有完整 band 状态继续存在，不能只递推均值。

实现条件：这是当前模型系数合同的特化，不是通用 recurrence API 的无条件优化；τ 的梯度、各种 α 的 generic 回退要保留。验收共享/不共享时整模型全部参数梯度，以及修改 band 数/τ 边界。相关 WY/retention 处理参照官方 FLA 实现，Bio 的专用 IO/多带收益仍待实测。

**算法 tile 与模型 C 必须分开。** 在固定 C16 里面用 tile4/tile8 或 fusion 是实现优化；将 C16 改成 C32/64，会改变 slow/m/P 冻结时间和监督反馈可用时刻。FLA 内核常用的 chunk 长度不能直接作为新 Bio 边界而不声明模型变化。跨 C 块的系数依赖上一个块提交后的状态，不能提前把整段 T 所有块系数并行算完。

验收：现成 sequential 为 oracle，FP64/FP32 原生设备验证 logits、全状态、所有参数梯度，包含 α=0、近 1、非二次幂、极小 key、硬门、容量边界和未来因果性；专用共享版本再检查 τ 梯度。性能以完整训练更新和反向占比计，不能拿 FLA 的 CUDA 测速外推当前 MPS。

## 5. Joseph 能不能也 scan

线性高斯滤波确有把过滤/平滑写成 associative scan 的一手研究，减少的是时间依赖深度；合成元素包含矩阵求解，不能只报 O(log C) 而不算 work 和状态维度。[Temporal Parallelization of Bayesian Smoothers](https://arxiv.org/html/1905.13002)

当前 Joseph 不是可直接替换成该标准滤波器的实现：

- gain 乘了 learned reliability w；当前 covariance 为 `P - w(2-w)/s rrᵀ`（无 floor 时），不是简单调整 R 后的标准 `P-w/s rrᵀ`。
- mean 每次到达后 bounded，未到达时 m/P 冻结；还混合 bounded node_anchor。
- query 慢容量投影是非线性、按顺序作用；慢反馈不属于线性高斯观测乘积。

因此优先 E6/E8。另建“标准 Bayesian 概率分支”才可以研究该论文的闭合 scan，但要作为新方程对照。不要将 smoothing 的未来观测结果注入已生成 token 的预测；训练可离线计算等价 filtering prefix，因果性仍逐位置保持。新分支须比较概率校准、历史污染和收敛，而非只比算子耗时。

## 6. 会改变模型规则的研究路线

下表保留完整 d×d 内容状态，但已经不属于当前方程的免费重排。均未新增实现或训练。

| 路线 | 改什么方程/复杂度条件 | 风险与验收 | 已有状态/依据 |
|---|---|---|---|
| 局部边 + 固定秩公共模式 | `W=A_local+ΦΨᵀ`，`WH=A_local H+Φ(ΨᵀH)`；约 O(N(k+2r)d²)，取代原 O(NKd²) | K8 本来线性；k2/r2 约6N，k2/r4约10N，不一定更省。模式必须按 cut/方向分区，罕见信息会混合。验收质量、污染、合法可达、真实 FLOPs/IO/TPS | 旧 M 支线；未实现。结合律来自核特征线性注意力，但任意 softmax/硬门不是精确低秩。[原论文](https://proceedings.mlr.press/v119/katharopoulos20a.html) |
| 可分相位矩 | 可分权重下用少量 sin/cos 群矩代替逐边耦合 | 当前 w 有成对匹配、Fourier 非线性和跃迁门，不能直接全局可分。固定 w 后局部三角恒等式仍成立，但不消除 w 的生成。验收相位/内容/传播，不只同步曲线 | 旧数学备选；未实现 |
| 多时间率更新 | 某些置信/经验/相位只每 M token 或每若干块更新，完整状态保留 | 减少事件数但改变反馈/适应延迟。固定指数衰减可解析跳步；状态依赖控制不能假装解析补算等价。验收 token 对齐的修订/ABA恢复、稀有线索和独立后验 | 当前 C 内快照已经是一种多时间率；进一步加 M 是新规则 |
| 径向前馈或削减同层/返回边 | 降 K/J，按球壳前推，保留矩阵 | 减少协作/路径，传播深度与 J 耦合；原图 read 在 shells-1。验收同一合法端口、路径覆盖、恢复/新情景质量及训练可比预算 | 旧 H 支线，不能写成已测等价提速 |
| 分组稀疏更激进 | 更少组或更慢路由，空闲律/访问概率变化 | 已有 sparse50 在 N128 反而更慢；先修路由/小 batch，再少算。验收实际活跃率、覆盖/探索、空闲冻结、全局信用和 cut | G 已实现初版；降低频率/改共享选择另计模型变化 |
| 低秩/对角/块对角永久状态 | `H≈UVᵀ` 或只留 block/diag，使 O(d²) 下降 | 改了完整矩阵容量和方向交互；连续 rank-one 写入一般涨秩，必须截断。用户当前要求完整矩阵，因此不列首轮实施 | 新模型、非当前授权主线；不能把“低维地址”当成它已完成 |
| 弱化慢投影/近似梯度 | 将多次约束合一、冻结更多梯度、固定置信或更长 detach | 容易用训练规则变化换速度，破坏保持/遗忘目标。仅已有 F1 的 no_grad 执行 E7 保持现有规则 | 单列新变体，完整状态与梯度贡献必须归因 |
| FFT/球谐/一次全局求解 | 要有共同谱/卷积、低秩或明确线性隐式方程 | 当前动态稀疏球图没有共同特征基；J 轮有相位/门/softmax非线性。直接换 FFT/求逆会换模型，且可能 O(N³) | 旧备选，无当前可用的免费等价核 |

## 7. 缓存与失效：让更多显存承载有效样本

| 可复用内容 | 正确生命周期/失效条件 | 是否可 detach |
|---|---|---|
| neighbors、边槽、radial parent、图转置索引、基础逐轮质量 | 图/edge_mask/write/read/J/active IDs 不变可长期；不同 compact 路由按键缓存 | 静态图可无梯度 |
| eye、静态几何、时间索引、三角 mask | d/C/device/dtype 等匹配；模型改变即失效 | 常量可 |
| packet 的 W_drive p、fresh F_t | 同一 token 块及参数版本；不能跨 optimizer 或不同样本复用 | 训练时不可 detach |
| frozen m/slow/old、fixed features/control/Fourier 投影 | 仅同一块/同一路由/同一输入状态 | 块内 BPTT 所需路径不可 detach |
| normalized u/v、phase 的 node/edge 视图 | 同一空间轮控制值；phase更新后失效 | 不可为缓存而删梯度 |
| band decay、bounded anchor | 同一前向参数版本可复用；optimizer 后失效；τ/anchor 仍有梯度 | 不可跨训练步存旧 autograd 图 |
| 输出头权重的方向缓存 | 当前参数版本及真实信用年龄；optimizer 后按现规则 invalidate | F1 方向已定义 detach；eta/旧 slow 不随之 detach |

固定图/layout 缓存节约的是重复调度，激活因式分解节约的是无效副本。增加 B 或梯度累积由主计划的资源实测决定；本数学方案不以增加节点或矩阵尺寸来填显存。梯度累积提升的是有效 batch，不保证单步设备利用率；要区分“同一前向更大 B”和“顺序跑多个小 B”。

## 8. 建议验收顺序与最小开发矩阵

这是一份待执行计划，现有训练继续使用冻结源；不在正在运行的源码上直接打补丁。

1. **低风险等价执行包**：E7 no_grad、E9 固定 Fourier/normalize 复用、E4 时轴收缩；每项单独开关，先 CPU/现成回归，再原生设备完整更新。
2. **有直接代数节省的包**：E1 scalar drive + 投影复用、E2 合法基础载体因式分解、E6 Joseph 外积合并、E5 慢写向量化。分别验边界，再逐项合并，保留原参考。
3. **针对主要瓶颈的融合包**：E3 图边完整 Value 聚合 + backward，E8 保序慢提交；先定位内存/调度/实际占用，再选择 MPS 原生可用的执行路径，不能只移植 CUDA 包名。
4. **快核后续**：E10 fused rank-one/read/backward，E11 多带共享；保留 C16 和现有慢时序。三角/树 scan 与 sequential 配对，不强制认为并行核更快。
5. **研究分支**：公共模式通信、多时间率/新拓扑只在等价执行方案稳定后按相同 token/质量协议另做；当前没有依据承诺它们保持质量。

等价验收按：FP64 单算子与全模型 → FP32 硬门/近零/容量边界 → 多块 full state/梯度 → 恢复/路由/信用 → 少量相同起点更新。未声明改变的项目要求所有状态和所有参数梯度保持，None 与 0 单列。阈值附近的浮点分支变化要显式记录，不能只报均值误差。

速度验收固定数据、参数/状态起点、B/T/C、精度、线程、日志/保存频率；包含 forward、CE、backward、clip、optimizer、状态处理；编译/预热分开。报告中位数和分布、峰值资源、实际活跃节点/边、内核/设备同步次数；不要把同步 profile 的区域和当作正常无同步运行的归因。

本文没有给任何未经实测的倍速。当前证据足够支持优先级：**先减少主要路径的重复内容搬运和小算子，再讨论新模型规则；快态本身已经有合适方程，后续优势取决于实现与硬件。**

## 9. 本轮实际阅读的一手来源（6 个主来源）

1. [Parallelizing Linear Transformers with the Delta Rule over Sequence Length — DeltaNet](https://arxiv.org/html/2406.06484v3)：delta/WY/三角形式；注意按递推与实现复核展示公式。
2. [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/html/2412.06464v3)：retention 与 delta，块计算的适用条件。
3. [Flash Linear Attention 官方实现](https://github.com/fla-org/flash-linear-attention)：阅读 delta_rule/gated_delta_rule 的 naive/chunk/wy 与 common chunk 输出；官方代码作为公式交叉核对，不能把 CUDA/Triton 实现的条件隐藏。
4. [FlashAttention 原论文](https://arxiv.org/html/2205.14135v2)：精确融合、IO、前/反向重算；这里使用方法原则，没有转用其性能数字。
5. [Temporal Parallelization of Bayesian Smoothers](https://arxiv.org/html/1905.13002)：线性高斯 filter/smoother 的 associative scan；当前 reliability Joseph 与非线性投影需另推，不能直接套用。
6. [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://proceedings.mlr.press/v119/katharopoulos20a.html)：可分核的结合律；用于限定低秩公共通信的新结构条件。

本地依据另含 `results/v09/speed_audit_20261025/profile_summary.json`、`results/v09/mps_20261025_v09_final/REPORT.md`、F1 no_grad CPU 原始记录、旧数学/改进计划。论文结果、当前实测、本文独立代数推导与尚待测试的实现候选已分别标注。
