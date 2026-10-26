Bio-Matrix 数学计算优化候选备忘录（2026-10-24）

本次只读分析与小规模CPU数学校验；未修改v0.7、连接远端或新增训练。以下把保持现有数学函数的代数重排与改变模型/时序的候选设计分开。所有MAC或复杂度变化均不是已测GPU加速。

**一、保持当前数学函数的候选重排**

1. 当前刺激包的精确秩一表示。

每token只有一个包 p_t=[v_t,k_t,1]。drive初值是write_ports*p_t，各通道共享同一标量边权W，并在线性混合后将write ports恢复同一个p_t；observe的局部过滤不写回drive。因此归纳有 D_i^r=a_i^r p_t，且 a_i^{r+1}=.25a_i^r+.75∑_j W_ij^r a_j^r，入口a=1。

可以仅传播a，替代d8时17维drive。节点完整8×8内容消息不压缩。value仍是clamp(a_i v/max(a_i,.05),−2,2)，key仍须normalize(a_i k,eps)，不能在近零a/k时直接用全局k。强度平方可用mean(X_i²)+a_i² mean(v²)复用状态摘要，但零点须保留vector_norm的有限零次梯度语义。

CPU FP64四轮、边权依赖强度的前向最大误差4.44e−16、普通输入梯度最大差1.42e−14，近零key案例梯度绝对差4.66e−10。FP32硬门阈值边界的轨迹/梯度仍需单独核查。

2. 节点投影与静态项复用。

Fourier的单层仿射 W[f_i;f_j]+b=A f_i+B f_j+b，可将每边24→11改为每节点12→22后gather/add。K8时这项投影MAC变为1/8。N512/J4控制器与Fourier合计8,650,752→4,866,048 MAC/token，另需边上加法、访存，其他算子不在该计数内。

full版本12维特征中9维在一个token四轮内不变，3维随x/z变化。控制器第一层可缓存静态投影，再加动态投影后保留原SiLU；Fourier投影也可做相同分拆。归一化内容、节点差模复相位、载波sin/cos可先按节点计算再广播复用。编译器可能已消除部分冗余，需检查生成内核再计收益。

3. Joseph更新合并外积。

令 q=P_prior k，D=kᵀq+variance，r=weight/max(D,1e−6)，则
P_new=sym(P_prior−(2r−D r²)qqᵀ)+1e−6I。

三项外积合成一项，保留现有clamp分支。CPU FP64普通PSD案例前向差4.44e−16，梯度差1.60e−14。依旧O(d²)，主要减少中间张量和算子次数。

4. 慢记忆写入按rank-one中间表示。

proposal=Aqᵀ，Frobenius范数为||A||||q||；限幅仅缩放A。令u=normalize(q)，a=Su，则最终投影可写为 S_new=S+[clip_radius(a+A(qᵀu))−a]uᵀ。
正交部分能量可由 ||S−auᵀ||²=||S||²−(2−||u||²)||a||²求出，不必物化orthogonal矩阵。近容量边界有浮点相消风险，原零slack/零update分支需保持。最终S仍为完整矩阵，不存在跨历史无限增加rank-one列表。

**二、最值得研究的结构化时间递推：振荡坐标系＋定向记忆修正**

设累积二维分块正交旋转 Q_t=L_tQ_{t−1}，P_t=R_tP_{t−1}，物理矩阵S_t=Q_tH_tP_tᵀ。将key/value换到旋转坐标：k̃=P_tᵀk，ṽ=Q_tᵀv。

在旋转系中使用完整矩阵delta更新：
H_t=α_t H_{t−1}(I−β_t k̃_t k̃_tᵀ)+β_t ṽ_t k̃_tᵀ。

它等价于物理坐标的“先旋转、阻尼旧记忆，再按当前key定向纠错”，保留完整d×d状态、相位和经验方向。若||k̃||=1，0≤α,β≤1，且这些系数/键不依赖本块尚未生成的快状态，则快状态通道的条件线性Jacobian算子范数≤α。它不保证全部控制器与参数梯度的全局稳定。

二维块旋转可以累计角度，不必每token做特征分解。Gated Delta的WY/分块三角算法可用于旋转系中的更新。独立8步、d8 FP64前向与全部输入/角度/门系数梯度校验见rotating_frame_check.json：误差4.44e−16 / 2.66e−15。

这是候选动力学的坐标等价校验，相对于当前v0.7整套非线性规则仍是新模型。当前每步bounded、状态依赖门/频率/边权、Joseph和逐token完整输出头反馈不能原样宣称可scan。

可行执行安排：块起点用慢矩阵、置信度和历史摘要生成控制上下文；块内系数由当前及过去可得输入/前层输出生成，使用可组合快递推；块尾处理复杂耦合、全局误差反馈、慢记忆和下一块路由。训练和推理采用相同的延迟规则。块长度16/32只能作为待试参数，不是已测最佳值。

特别注意：只冻结控制器仍不够。任意跨节点状态耦合的稀疏转移连续相乘会变稠密；不能构造Nd²维大矩阵做dense scan。复杂跨节点耦合要分层前馈、限制在小块或放到较慢边界。

参考：
- Gated Delta Networks: https://arxiv.org/html/2412.06464v3
- SSD / Mamba-2: https://arxiv.org/abs/2405.21060
- LRU: https://arxiv.org/abs/2303.06349

**三、保留同层协作，改变通信的代数结构**

固定少量精细局部边＋固定秩公共模式：W=A_local+ΦΨᵀ，节点完整内容H∈R^{N×64}，则 WH=A_local H+Φ(ΨᵀH)。每轮公共通道只有r×64，覆盖刷新，不携带无限路径。可按壳和方向分组，避免跨切分边界泄漏。

正特征时，分母也可结合律重排：m_i=φ_iᵀ(∑ψ_jH_j)/max(φ_iᵀ∑ψ_j,eps)。成本约N(k+2r)64，现有K=8本来是线性复杂度，不能宣称现有N²→N。k2/r2的主通信乘加约6N对原8N；k2/r4则约10N、更多乘加。潜在收益还在于少算逐边超网络与使用规整GEMM，需实际计时。

相位耦合的更小摘要：∑b_j sin(θ_j−θ_i)=cosθ_i·∑b_j sinθ_j−sinθ_i·∑b_j cosθ_j。均匀/可分耦合时每群只需两个相位矩；多个频率是固定数量的相位矩。当前pairwise相似度sigmoid、硬门及Fourier整体sigmoid破坏这种可分性，因此改用公共模式属于新通信核。

保留局部边保护稀有线索和独立经验；公共模式秩太低会混合互不相关内容，不能只追求小r。

参考：
- Linear attention: https://proceedings.mlr.press/v119/katharopoulos20a.html
- Multipole Graph Neural Operator: https://arxiv.org/abs/2006.09535

**四、备选而非首选**

固定蝶形/Givens混合：每层大量独立2×2正交块，log₂N层获得广域混合，保留满秩并控制固定混合范数。N512需要9个依赖阶段，不保证快于当前K8/J4。https://proceedings.mlr.press/v97/dao19a.html

线性低秩隐式反馈：若新模型显式定义H=B+κUVᵀH，则可用 H=B+κU(I−κVᵀU)^{-1}VᵀB，在小r系统求解中包含闭环反馈。必须控制谱半径/条件数，只适用于规定的线性低秩耦合；不能等价替代当前有限四轮非线性传播，也不是应该对大图做一次求逆。尚未数值或训练验证。

一般非线性时间并行：ParaRNN/DEER用Newton类方法求轨迹。高效现成路径主要是对角/分块对角Jacobian；当前Bio跨节点耦合和全局head反馈不满足条件。可作为“小块非线性节点＋慢块间耦合”的远期方向。
https://machinelearning.apple.com/research/pararnn
https://github.com/apple-aiml-research/ml-pararnn
https://arxiv.org/html/2407.19115v1

**不采用的免费午餐假设**

不同衰减率的多时间带不能只保留均值；多次rank-one写入后的状态一般会满秩；diag+rank-one转移连续相乘会秩增长；动态边权通常不共用特征向量；普通FFT依赖规则平移/循环结构，球面不自动满足。直接复用输出head激活作反馈VJP会改变hidden/参数导数，不能简单声称训练等价。

**建议顺序**

先独立验证精确重排，再研究旋转系结构化快记忆与慢控制；通信公共模式作为单独变体。保留v0.7全轮结果，不将这些推导当成已训练的性能收益。
