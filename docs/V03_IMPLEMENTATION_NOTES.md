# v0.3 代码审计与实现说明

> 归档说明：本文保留写作时点的设计或状态。项目已停止v0.9后续训练；最新执行状态见[实验索引](RESULTS_INDEX.md)。公开副本不含权重与语料。

已读取首轮 `matrix_lab/models.py`、`extensions.py`、`data.py`、`run.py`、当前FINDINGS及原始 `reaction_dynamic_seed42/result.json`、`matrix_verification.json`。首轮MemoryModel没有节点维度和邻居通信；Oscillator是独立坐标的旋转与写入，不是Kuramoto网络。旧报告支持状态干预与短期周期拟合的有限结论，不回答v0.3网络规模问题。

新代码位于 `v03/`，复用首轮的原子检查点保存和MPS同步函数，未修改首轮文件。入口 `python -m v03.profile`、`python -m v03.run --stage all --dry-run`、`python -m v03.run --stage pilot`。计划数量和依赖由实际CLI输出，不依赖任务书里的拟议脚本。配置使用JSON，避免增加YAML依赖。

输入：每条物体记录独立编码，问题由GRU编码；没有全证据Transformer、KV缓存或符号关系执行器。每个内部轮次由节点状态、固定随机角色和当前问题产生查询，读取原始快照证据，再沿独立内部稀疏图交流。这里允许共享证据读取与全局读出，因此切分内部图不等于完全隔绝所有间接信息通道。

内容状态为逐坐标保留门与tanh候选。局部消息内容得分softmax归一化；C2/C3再乘相位相似门。C3签名耦合使用同一活跃内容得分的tanh，按绝对边权和归一化，频率/驱动/步长/κ有界。C1两个非周期辅助坐标也参与局部消息门与读出，并非闲置补数。C1的非周期门使用sigmoid内积，C2/C3使用(1+圆周内积)/2；这是一种强非周期通信对照，不是所有函数类完全等价的证明。

当前状态N(d+2)之外另记固定角色、图索引与编码证据缓存；自动微分保存激活不计为持久状态，但在真实采样内存与时间中体现。MAC估计明确遗漏编码器GRU、非线性、读出及自动微分，不能当成精确FLOPs；实际完整步/推理时间为成本依据。

P0使用训练范围中最重的160条记录批次剖析，开发/测试另有M64。未使用PYTORCH_ENABLE_MPS_FALLBACK。CPU只做小规模确定性测试。第一次实现检查发现四个8节点分量不可能保持每节点8个邻居，现统一采用两个隔离分量；随后七项CPU测试与MPS正反向校验通过。

外部数据源：[CLEVR官方页面](https://cs.stanford.edu/people/jcjohns/clevr/)；只下载官方no-images包，CC BY 4.0。CLEVR-SG获得真实物体分组，不是标准图像VQA，也不能独立证明从无分组感知中产生物体绑定。
