# v1.1 版本差异与实验合同

## 目的

v1.0的Bio与冻结Transformer几乎持平，清空/替换历史影响很小；最外壳18个节点没有收到新观测，记忆投影随机，默认64步训练仅在最后巩固一次。v1.1修复工程覆盖，并通过公开长历史问答直接训练“之前写入的信息帮助之后回答”的路径。

## 与v1.0的具体差异

| 项目 | v1.0 | v1.1 |
|---|---|---|
| 记忆投影 | 随机4×8；额外归一化 | 继承主干Q/K/V与输出，4×14；归一化输入坐标对齐 |
| fresh传播 | 默认J2，实际最外深度3 | 默认J3，深度下界拒绝；真实train输入逐壳校准 |
| 向外阈值 | 与衰减后信号尺度不匹配 | 明确幅值单位，记录mass/m/P/关联实际变化 |
| 相位时间 | coupling未统一除J | 频率与coupling都按count/J |
| 场内记忆 | 段均值 | 同预算summary/ordered双组；新增association及有界token缓存 |
| 顺序写入 | core逐token，field摘要 | core继续逐token；ordered的association/fast逐token；其他场动力学按段 |
| 训练目标 | 同段LM CE+KD | 自然文本CPT CE → 人工标注历史QA答案CE+小权重KD |
| 关键比较 | frozen/adapter/no-phase/full | frozen/adapter/summary/ordered，Bio默认全部含phase |
| 参数巩固 | 每100步或末尾 | 仅完整CPT窗，每16步与CPT末尾；PT PAD末态跳过 |
| 验证 | 训练中与完成后LM | t=0、CPT后与PT过程，LM+历史QA与配对干预 |

QKV复制改善初始坐标，但运行时unit-QK、tanh-V、相位和门控保留，因此没有完整softmax attention等价承诺。4×8改4×14增加矩阵容量，v1.0→v1.1的跨版变化不能归因给一个机制；v1.1内部两Bio组才匹配容量与初始化。

## 时间与因果性

每64 token提交一次场。场内token信号的合法图扩散以B×T合批，全部使用进入该段时的旧field与相位。ordered在段末按真实token顺序把这些观测写入association/fast，summary合并后写一次。m/P只测量一次，段级latent预测只消费一次上一段信用。

本段的新状态进入下一段读取；没有把未来token的新场状态读回过去。不同Python调用的切分保持同一段时钟与缓冲顺序。固定缓冲和全矩阵预算相同，summary也计算逐token图扩散，写入次数与耗时分别报告。

固化事务在backward和optimizer结束、运行态detach后执行。验证不固化、不开梯度、不更新任何模型参数。PT中标签mask只计分，字段不会传入Network.forward；右PAD末态也不会进入参数巩固。

## 数据与测量

完整协议、上游固定提交、字节hash、筛选统计、采样记录和每题分数保存在各自manifest及结果目录。WikiText使用已冻结split，SQuAD文章级训练/验证切分，官方dev留作测试。数据准备仅转换官方文本与标注，不生成问题或答案。

历史QA的完整答案证据早于 `prefix−layers×context_limit`，默认两层128窗口时span end<128、距回答>256，避开多层局部传播的间接原文感受野；重复答案筛查采用相同256范围。提示和答案整体规范BPE编码，字符offset决定答案target mask。只计答案及首答案token。

直接干预保持相同当前KV与位置后分别清空Bio、后缀禁读、替换和打乱Bio旧历史；另做全程禁读、最旧原文替换后重新编码控制，检查历史Bio读出进入近期KV的间接贡献。整条路径控制不承诺KV相同，其后缀输入及位置保持相同。若记忆提供内容帮助，正常历史应在重复种子中系统优于远史错配；t=0与adapter同时帮助判断静态新增参数的作用。

当前公开远史QA子任务侧重内容记忆；近期证据充分、应忽略独立旧经验的严格后验控制尚未单列，LM错配仅提供污染代理。

teacher_quality记录小型教师的真实QA分数，CE仍占主导；教师低质量不会被隐去。样本较少或单种子的不确定性通过每题原始值暴露，不将单次准确率或同步曲线作为成功判据。

## 研究依据

- [MemT](https://arxiv.org/html/2610.26825v2)：采用主干投影初始化与显式历史上下文到后续回答的训练思路。本地规模、教师和数据量都单独定义。
- [Linear Transformers Are Secretly Fast Weight Programmers](https://proceedings.mlr.press/v139/schlag21a.html)：有限关联矩阵与delta纠错写入。
- [Titans](https://arxiv.org/abs/2501.00663)：输入相关的记忆更新/遗忘与跨历史训练信用。

v1.1保留球壳、相位/Fourier、概率、内外映射、有限信用、局部修订和长期参数固化这组独立假说。更强训练信号是否带来实用收益由保存的原始结果判断。
