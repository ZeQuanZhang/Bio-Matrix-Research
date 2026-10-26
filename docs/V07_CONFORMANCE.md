# Bio-Matrix v0.7：构想、实现与验收对照

> 归档说明：本文保留写作时点的设计或状态。项目已停止v0.9后续训练；最新执行状态见[实验索引](RESULTS_INDEX.md)。公开副本不含权重与语料。

本文对应 `configs/v07/base.json` 的 `v07-aligned-content-context-lm-4`，记录本轮设计对齐后的实现。依据为原始《BioMatrix_完整构想描述》、`docs/V06_DESIGN.md` 与历次 v0.7 审查。`docs/V07_DESIGN_AND_EXPERIMENT.md` 中早期的“每文档重置”“仅上一 token 信用缓存”“广播观测尚未修复”等内容属于当时的版本记录；当前行为以本文及源码协议哈希为准。旧结果保持原配置和源码标识。

实现已经覆盖动态内部矩阵、状态调节的输入／输出映射、概率状态、多时间带、局部相位耦合、球体图传播、有限容量经验修订及持续情境训练。数学与工程回归检查这些机制是否实际进入计算；能力、规模收益及成本优势由后续公开任务的配对训练结果检验。

## 1. 构想到代码的映射

| 构想要求 | 当前实现位置 | 验收方式与边界 |
|---|---|---|
| 节点是一套可保留经历的有限动力系统 | `v07/model.py: Network.initial_state/step/tick` | m、P、x、S、trace、response、support、innovation、相位及信用元信息均固定形状；跨训练更新保留内容状态 |
| 当前刺激先经节点自身外部映射，再影响内部矩阵 | `observe`、`tick`、`step` | 同一最终 tick 的过滤后 value/key 同时用于 fast 注入、创新、Joseph 更新和 trace；验证修改输入映射会实际改变观测与写入 |
| 内部状态改变对外计算和路径 | `features`、`controls`、`external`、`weights` | 历史矩阵、不确定性、创新、相位和几何进入控制；输出映射、旋转、脉冲门及 Fourier 权重进入真实消息计算 |
| 多时间尺度保持与衰减 | `raw_tau`、`raw_trace_tau`、`tick` | 时间带指数松弛；每个 token 的旋转按内部轮数 J 归一；隔离输入与回流后核对解析衰减 |
| 概率判断参与经验使用 | `stationary_prior`、`joseph`、兼容性与 trust | P 调节更新增益并进入控制；创新影响读出与局部更新；P 是节点矩阵回归的不确定性近似 |
| 局部协作、球心到表层与允许回落 | `v07/graph.py: solid_ball`、`weights/tick` | 固定 4 层同心球壳、每节点 8 个候选入邻居；向外强度门，向内／同壳无同类门槛；断图对照检查可达性 |
| 有方向的经验修订 | `head_vjp`、`local_direction`、`local_proposal` | 对实际关联读出和完整输出头计算条件 VJP；与 autograd、有限差分比较 |
| 多维经验避免全局连带遗忘 | `project_slow_update` | 更新限定在实际 query 的右子空间，保留正交经验；在容量饱和时同样验收 |
| 有限延迟信用且消息不随历史膨胀 | `cache_feedback/feedback_frame/apply_feedback` | 默认 8 步固定元信息缓存；核对旧 query、一次消费、越界拒绝与参数版本失效；不保存旧网络状态副本 |
| 持续情境中利用旧经验并接受新证据 | `v07/data.py: ContinuousStream` | 默认 session 内跨文档、插入 B 再返回 A，不按情境切换清状态；marked/unmarked 混合，无情境 ID 输入 |
| 网络规模与协作收益 | `v07/run.py: cases` | 32／128／512 节点、多种子、普通消息／独立振荡／局部耦合及标准基线；另有容量、J、断图、写入归一和信用范围控制 |

原始构想允许多种数学实现。当前选择固定容量矩阵、低秩映射和规则候选图，并未把这些选择当作构想的唯一形式。显式新建专家、额外群体聚类、优化器自演化、JEPA、逐节点完整大模型和全网情境副本仍在本版主干之外。局部有效子图与相位关系可以形成选择性协作；当前执行仍为规则张量上的稠密计算，没有宣称获得 MoE 的稀疏计算节省。

## 2. 节点内外矩阵使用同一观测

节点输入映射在固定的节点 latent 轴上提取特征。对当前节点控制向量形成

\[
A^{in}=\operatorname{diag}(1+0.1\tanh D_{in})+0.15\hat u\hat v^T,
\qquad
\tilde v=\tanh(A^{in}v),\quad
\tilde k=\operatorname{normalize}(A^{in}k).
\]

fresh drive 的图混合仍携带固定宽度 `[value,key,mass]`，不会追加路径记录。节点取得到达的 value/key 后，通过自己的状态控制执行上述过滤。这样即使 raw 刺激相同，不同节点状态也能形成不同的实际观测。

每个内部 tick 用该过滤观测构成 \(\tilde v\tilde k^T\)，影响 fast 载波；最后一个 tick 返回的同一观测用于本次新证据的 Joseph 更新、创新、trace 与后续 query。多个内部轮次只有一次 fresh 证据计数。滤波改变的是输入特征，P 始终位于该节点固定 latent 坐标内；这里没有把 raw 空间协方差随动态映射作任意坐标搬运。

对外内容映射使用结构化右乘

\[
M\left[\operatorname{diag}(1+0.1\tanh D_{out})+0.15\hat v_{out}\hat u_{out}^T\right],
\]

再配合有界结构旋转和实际边权形成邻居消息。输入过滤、输出映射、相位及概率控制路径的可微性与实际输出影响均有单独回归；只有参数存在或梯度非空不足以完成验收。

## 3. 实际读取、参与数与局部写入尺度

对节点 n 的慢矩阵 \(S_n\)，当前到达 key 与有限 trace 构成单位 query \(q_n\)。实际慢读出为

\[
R=\sum_n a_n(S_nq_n)q_n^T.
\]

其中 \(a_n=c_n b_n\)：\(c_n\) 是创新与可靠度形成的兼容性，\(b_n\) 是 `read_ports` 与有界沿图返回响应相加后的归一分布。fast 和 m 仍由表层端口读取；慢矩阵通过这个明确的沿图分布算子读取。断开图边界后，返回响应不能跨断面引入不可达内部慢内容。

局部学习使用**实际读取权重 a** 的参与数：

\[
E=\frac{(\sum_n a_n)^2}{\sum_n a_n^2},\qquad
s=\frac{E}{N_{ref}},\qquad N_{ref}=32.
\]

所有 a 为零时定义 \(E=s=0\)。实现先除以最大权重再计算比值以保持数值分辨率，所得数学比值不变。该 E 衡量读取分布的参与广度，不是独立样本数，也没有假设参与节点相互统计独立。

在均匀读取下，E 等于参与节点数，消除仅因池化节点增多导致的一步慢读出变化按 1/N 稀释。缩放不会除掉共同置信幅度：若全部 \(a_n\) 同乘 c，E 不变，而直接读取与局部方向中的 c 仍保留。默认 `local_write_normalization=participation`；`write_scale` 阶段的 `none` 对照保留原池化稀释方式，记录有效读取节点数、scale、裁剪比例和实际步长比。

R 进入 `read_projection` 的 memory 半支，经过残差 FFN、两层 LayerNorm 和 tied embedding 输出。下一可见 token 提供上一输出的负 CE 方向

\[
e=E_{token}-\sum_j p_j E_j.
\]

对完整输出头求解析 VJP，令回到 R 的负梯度为 G，则实际读取分支的局部方向为

\[
D_n=a_n(Gq_n)q_n^T,
\qquad
\Delta_n=\eta\,s\,g_n D_n.
\]

\(g_n\ge0\) 由 cached 返回响应、曝光支持度和创新信任构成。先把每节点 proposal 的 Frobenius 范数限制到 0.05，再执行下节容量投影。query、读取权重、head 输入和增益按生成该预测时的值缓存，局部修订将其作为常量。

条件 VJP 精确对应这个显式直接读取分支。S 对控制、后续路由、旋转和未来状态的间接影响由 BPTT 学习；标量返回响应用于定义读取与非负预条件，不充当整网非线性伴随。全局训练损失仍为公开文本 next-token CE，默认 128-token BPTT；更早写入的参数信用不会因为保留数值状态就自动变成无限时距反向传播。

## 4. 容量投影保留正交经验

慢矩阵每节点的 Frobenius 预算为 C=4。对非零单位 query q，分解

\[
S_{\parallel}=(Sq)q^T,\qquad S_{\perp}=S-S_{\parallel},
\qquad u=Sq+\Delta q,
\qquad r=\sqrt{\max(0,C^2-\|S_{\perp}\|_F^2)}.
\]

更新为

\[
S'=S_{\perp}+\operatorname{Proj}_{\|\cdot\|_2\le r}(u)q^T.
\]

因此对任意满足 \(q^Tv=0\) 的方向，\(S'v=Sv\)。在初始 S 满足预算的条件下，\(\|S'\|_F\le4\)。即时反馈和延迟反馈共用该算子，避免先全矩阵归一化而连带缩小与当前 query 正交的经验。零 proposal 保持 S 原样。

有限预算的代价明确可观测：如果 \(S_{\perp}\) 已占满容量，新的 query 分量无法写入。`capacity_clip_fraction` 和 `local_effective_step_ratio` 记录这类受限更新。非正交经验仍会共享容量，旧内容之间的近似解耦程度取决于学到的 query 表示；这里没有承诺任意语义概念天然正交或无限保留。

## 5. 固定 8 步信用元信息

默认 `credit_horizon=8`。当前帧及 7 个历史 slot 只保存：期望 embedding、head 输入、节点 query、读取权重、反馈增益、读取 scale 与有效位。每条消息和每个 slot 的大小固定，不保存旧 m、P、x、S、整网情境或原始文本副本。

普通 LM 始终自动消费 lag=1：新 token 到达后修订上一预测，再处理当前输入。显式 API 为 `apply_feedback(state, token, lag, feedback_mask)`。手动延迟机制回归先关闭自动消费，再使用 lag 指定旧 slot；掩码只用于调用方明确提供的反馈通道。实现核对旧 query、单次消费、逐 lane 掩码、lag>8 拒绝及 reset 后失效。optimizer 改变参数后，所有旧 slot 的有效位置零，内容状态继续保留。

这个 API 处理有明确 lag 的反馈，不会自动从自然语言中解析“这句话是在纠正哪次旧判断”。自然公开 LM 没有隐藏延迟标签。等待期间 S 若已被其他反馈修改，旧帧给出的是滞后的局部更新方向，不能等同于重放旧整网或完整长时 BPTT。超过 8 步的精确反馈关联、开放语言中的反馈对象识别仍是后续研究问题。

## 6. 平稳概率先验与数值范围

令 \(\rho=e^{-1/\tau_P}\)，默认 \(\tau_P=4096\)。每个 fresh 到达事件使用

\[
\bar m=\sqrt\rho\,m+(1-\sqrt\rho)\operatorname{bounded}(node\_shape),
\qquad
\bar P=\rho P+(1-\rho)I.
\]

随后对 \(\bar P+QI\) 执行一次有可靠度的 Joseph 更新，默认 \(Q=0.002\)，对称化并加 \(\epsilon I\)，\(\epsilon=10^{-6}\)。可靠度在 [0,1] 时 Joseph 的观测项不放大 PSD 序，因此

\[
\lambda_{max}(P)\le 1+\frac{Q+\epsilon}{1-\rho}\approx9.1971
\]

在初始 P=I、当前参数与精度容差下给出解析上界。未到达节点保持原 m/P，不把内部反复计算算作新观测。连续到达时 P 的回归时间常数为 4096，m 的均值回归时间常数为 8192；它们按到达事件演化。配置将 tau 限于 [1,65536]，确保 FP32 能分辨更新系数。

m、S 及各时间带载波有显式范数预算；相位以单位圆二维坐标保存并归一，避免长时浮点漂移；support 有上限，证据计数饱和而不整型回绕。不确定性与创新摘要先作有界压缩再进入 controls。head 输入和期望 embedding 的幅度仍依赖当前学习参数的有限范数，不能把所有缓存都宣称为与参数无关的统一数值上界。PSD、局部范数预算及梯度裁剪也不单独构成整个状态依赖网络的全局收敛定理。

P 表达节点矩阵回归的 key 方向不确定性近似；它不保存每个权重的完整后验。token ECE 检查输出置信度校准的算术与表现，不能替代对内部 P、观测噪声或相关信息源的概率校准。

## 7. 真正包含无标记切换的持续训练

数据复用冻结的 WikiText-2 官方 train／validation／test 与仅 train 拟合的 4096 byte-BPE。每个 seed 的 stream 调度与模型参数无关，所有对照收到同一输入、真实目标与监督掩码。

默认一条 lane 的 session 含 4096 个输入 token，只有 session 起点全状态重置。A 的真实正文连续推进，默认约每 512 个 A token 插入一段来自异文档的 B，长度 64，再回到 A 尚未见过的后文；插入间隔在固定 RNG 下随机化。A 文档结束后换下一个真实文档，仍保留状态。`reset_policy=document` 只作为显式对照。

每次独立切换、B 插入或返回都按 `marked_probability=0.5` 决定是否给出可见 EOS/BOS。stream 从冻结 token 缓存中剥离每个文档已有的边缘 BOS/EOS，仅在 marked 分支显式加入。因而 unmarked 接缝两侧都是原文正文，真正没有缓存遗留的 BOS/EOS；模型只观察文字变化。

source 文档号、offset、切换种类及自然变化 cue 只进入日志和真实邻接 CE 掩码，不输入模型。跨拼接边不伪造 next-token 监督；所有输入仍正常推动 forward、Joseph、局部反馈与信用缓存，隐藏 CE 掩码不禁用自然反馈，也不为情境切换失效缓存。marked 的原文文档尾 EOS／下一文档 BOS 邻接仅在 source/offset 真实相邻时保留监督。

全文数据中的 `formerly/later/renamed` 等固定 cue 用于可审核的自然变化代理清单和接触计数，不改变 stream 抽样、模型输入或 reset。WikiText 没有经过核验的时变规则标签；同 query 规则修订、正交后验隔离与有限延迟反馈由平衡的可解释机制回归检查，不冒充公开规则推理 benchmark。

## 8. 历史利用、隔离与同龄 ABA 评估

设 A 是真实连续前缀，T 是 A 的未见后文评分段，B 来自另一文档。每个条件都 teacher-force 相同的当前输入和相同目标。默认 A、B、T 各 64 token；所有 target 位置记录 NLL、KL、置信度和动态诊断。

| 条件 | 评分前历史 | 解释 |
|---|---|---|
| carry | A | 正常连续历史参考 |
| reset | 无 | 丢掉 prefix；T 内继续正常建立状态 |
| reset_each_token | 每个 T token 都 fresh | 移除所有跨 token 状态 |
| unrelated／shuffled | B／shuffle(A) | 无关文档代理与顺序破坏控制 |
| A_B_A | A→B | 远处相关 A、近期干扰 B，再评分 T |
| B_B_A | B→B | 与 A_B_A 同龄且近期 B 相同；远处 A 改为重复 B |
| shuffled_A_B_A | shuffle(A)→B | 与 A_B_A 同龄、近期 B 相同且远史 token 频次完全相同 |
| B_A_A | B→A | 同龄、最近可见相关 A 的参考 |
| unrelated_boundary | B→EOS→BOS | 可见边界的历史隔离探针 |

主要 matched contrasts 为：

\[
\text{remote retention}=L(B\!\to\!B,T)-L(A\!\to\!B,T),
\]
\[
\text{ordered remote gain}=L(shuffle(A)\!\to\!B,T)-L(A\!\to\!B,T),
\]
\[
\text{age-matched interference}=L(A\!\to\!B,T)-L(B\!\to\!A,T).
\]

前两者为正分别表示远史相关线索有益、有序历史超出词频统计带来收益；第三者为正表示近期 B 相对近期 A 有代价。B_B_A 的远史是重复 B，不能单独当作完全独立的第三文档；三种控制联合解释。旧 `return_cost=A_B_A−carry` 保留，但明确混合了干扰与额外状态年龄。

Bio 还有 no_local、no_slow，以及 `reset_fast_after_prefix`、`clear_slow_after_prefix`、`credit_only_reset_after_prefix` 三个 prefix 后内容探针。后面三者使用相同 A 与 T；credit-only 只失效信用有效位，作为 reset_fast 和 clear_slow 的共同基准，避免把一次反馈失效误记成内容记忆收益。bypass 才提供 head_only；默认 Bio 无当前 embedding 直接输出通路，stateless 是从零训练的同宽 FFN 基线。

公开文本另有模型无关评分 mask：至少 32 token 远的重复线索、train 词频处于正频次 75 分位以下的稀有重复线索、A/B 均无目标 token 精确匹配。重复子集排除最近 16 个可见 token 及 B 中同 token。统计 mask 不进入 forward、局部反馈、抽样或优化。

冻结 test 的 64 窗口、4096 个不同评分位置中，稀有远处重复子集有 114 token、41 窗口，前 16 位置共 33 token；一般远处重复有 614 token，A/B 均无精确匹配有 2560 token。它们提供可审核自然线索，尚不构成“每个标签必需长历史”或“语义独立”的真值。词袋 fixture 在远史乱序控制下必须严格零顺序收益，防止把词频先验当作有序经验调用。

窗口、当前输入／目标、训练词频统计及评估协议均记录 SHA；不适用的干预明确标记。模型选择只用 validation carry NLL，测试比较保留按文档／窗口及独立训练 seed 的层级；窗口不能充当额外训练种子。

## 9. 计算、状态与证据范围

图主干复杂度为 \(O(BTJNk d^2)\)，Joseph 为 \(O(BTN d^2)\)，状态输入过滤为 \(O(BTJNd)\)。固定 H=8 的信用元信息为 \(O(BH(Nd+2N+2w+1))\)，与 session 长度不增长。完整 LM 还包括 embedding、FFN、词表输出及期望 embedding 的计算，实际成本以对应设备 preflight 与训练 ledger 为准。

默认 d=8、3 时间带、3 频率模态、2 trace 带、w=64、H=8、FP32 下，每条 stream 的运行态张量字节为

\[
2244N+4136.
\]

| 节点数 | 单 stream 运行态字节 |
|---:|---:|
| 32 | 75,944 |
| 128 | 291,368 |
| 512 | 1,153,064 |

batch2 对这部分加倍；表格不含参数、优化器、激活、CUDA/MPS 临时分配或框架开销。状态固定容量并不意味着训练显存只等于这些字节，BPTT 激活还依赖梯度范围。固定 Nd² 的容量阶段只固定矩阵条目，总参数、总元信息和实际计算另行报告。

core 保留 45 个配对标签，图模型同尺度的状态形状、图、输入和读取算子匹配；Transformer 参数近似匹配，stateless 控制同宽当前输入预测能力。Transformer 使用两层 pre-LN、RoPE、因果 SDPA 与固定 256-token 滑动 KV；逐 query mask 保持分块等价，session／显式 reset 按 lane 清除历史有效位和位置。Bio 内容状态可覆盖整个 4096-token session，这与 KV 的历史容量不同，不能称为相同上下文容量的比较。FLOPs 和墙钟没有强行匹配。同 seed 的多个 stateless 标签及部分同形 Transformer 会重复同一结构／数据训练，不能作为额外独立证据。

数学回归覆盖：条件 VJP／有限差分、实际过滤观测一致性、实际读取参与数和共同置信幅度、容量饱和时正交内容保留、lag8／越界／消费与失效、平稳 P 的 PSD 和界、单位相位、固定消息／状态形状及断图隔离。训练协议回归覆盖：真正无标记接缝、返回未见 A 后文、source 注释不改变输入、隐藏 CE 掩码不控制反馈、有效 token 诊断，以及全家族原子恢复。评估回归覆盖：相同 targets、同龄与词频控制、子集无输入泄漏和内容／信用消融。

这些检查的原始记录位于 `results/v07/conformance/`；需求与方法映射为 `evaluation_requirements.json`，模型数学与字节记录为 `model_conformance.json`。`public_evaluation_*` 保存初始化冻结权重的真实公共文本接口验收，文件内记录对应源码哈希，不能当作训练后的能力结果。早期快照独立保留，GPU 工程记录与正式训练结果按各自 protocol、preflight、ledger 和 result 登记。

后续需要用实际配对结果回答：完整机制能否优于同规模对照、是否真正利用有序历史、相关新证据是否能修订旧经验且抑制无关串扰、旧路径能否在干扰后重新激活、局部相位协作是否带来规模收益，以及这些收益是否值得训练／推理成本。更长语义反馈对象识别、跨 session 的自主知识巩固、内部概率校准和显式稀疏专家计算仍没有从本轮数学回归中获得能力结论。
