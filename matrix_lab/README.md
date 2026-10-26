# Bio-Matrix：仅依靠递推状态的关联记忆实验

> 归档说明：本文保留写作时点的设计或状态。项目已停止v0.9后续训练；最新执行状态见[实验索引](../docs/RESULTS_INDEX.md)。公开副本不含权重与语料。

这是新的矩阵反馈与动态读出实验；根目录原有八状态中文语言模型与结果保持独立。
模型不含 Attention、KV Cache 或隐式共享状态。每条序列仅携带一个32维状态向量。

## 运行

在项目根目录使用已有虚拟环境，不需新增依赖：

新机器先安装根目录的固定依赖；本机已有 `.venv`，无需重建：

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

实验命令：

```bash
.venv/bin/python -m unittest -v matrix_lab.test_core
.venv/bin/python -m matrix_lab.run --variant gru_add --seed 777 --root results/matrix_smoke --smoke
caffeinate -i .venv/bin/python -u -m matrix_lab.run
.venv/bin/python -m matrix_lab.report
.venv/bin/python -m matrix_lab.diagnose
.venv/bin/python -m unittest -v matrix_lab.test_extensions
caffeinate -i .venv/bin/python -u -m matrix_lab.extension_run
caffeinate -i .venv/bin/python -u -m matrix_lab.signal_run
.venv/bin/python -u -m matrix_lab.optimizer_run
.venv/bin/python -m matrix_lab.findings
.venv/bin/python -m matrix_lab.verify
```

正式配置在 `matrix_lab/config.json`，完整运行21次，每次4000步、批量64。
已完成项会核验协议后跳过；中断项恢复模型、优化器和数据随机数。
`results/matrix/` 保存原始逐事件预算、数据日程摘要、逐样本测试指标、权重与报告。

## 研究问题

- 相同查询输入下，保留、清零或交换状态是否改变正确率？
- 输入条件仿射更新、门控非线性反馈、二阶状态反应有什么差别？
- 同一读出下，Bio状态是否优于GRU？同一状态下，乘性动态读出是否优于加性读出？

对比7种模型和3个种子，固定数据日程、持久状态容量、优化器和步数。
参数量和计算量不完全相同，报告实际值；不把等数据结果说成等FLOPs结果。

这是中文符号构成的合成记忆任务，不是自然中文语料上的语言模型。
完整方程见 `docs/THEORY.md`，评测协议见 `docs/EXPERIMENTS.md`。

概率/频率扩展在 `results/matrix_extensions/`；周期预测在 `results/matrix_signals/`；
独立优化器在 `results/matrix_optimizer/`。三者使用单独的预设预算，不能与核心4000步实验直接混排。
高斯状态是有明确先验、观测和方差更新的对角EKF近似，未经过不确定性校准。

已经完成的本轮解释见 [FINDINGS](docs/FINDINGS.md)。核验器检查54次MPS训练的预算、
源文件摘要、配对数据日程、验证集选点和权重加载，以及120次解析函数与12次MLP优化器试验。
已完成结果与源版本不匹配时会明确报错，不会静默覆盖为另一套实验。
