# 本地33万参数：Qwen与DeepSeek架构对照

本轮回放已经完成的V3本地实验规模：4096词表、约33万总参数，每模型1,048,576个文本预测目标，再做512次、batch4的原问答更新。使用原WikiText-2与SQuAD派生数据，逐步核对旧运行的输入和loss mask哈希，比较相同低预算下不同结构能学到什么。

它与准备中的`scale/` 0.133B、10B目标队列分别保存。当前执行状态以[registry](../../results/tiny_arch/paired_20261103_local333k/registry.json)为准；统计见[analysis](../../results/tiny_arch/paired_20261103_local333k/analysis/)。

## 具体原型

| 条件 | 总参数 | 保留的结构 |
|---|---:|---|
| 原局部Transformer | 333,106 | 两层、56维、128局部窗口 |
| 原完整窗口Transformer | 333,106 | 两层、56维、512窗口 |
| 原静态/动态Bio | 333,136 | 两层、56维，原节点条件/内容/相位 |
| `qwen35_tiny` | 332,950 | 四层、56维；3层Gated DeltaNet加1层完整注意力，RMSNorm、SwiGLU、GQA与门控 |
| `deepseek4_tiny` | 333,065 | 四层、56维；局部、CSA/HCA压缩注意力，低秩分组输出、4路mHC、Hash-MoE与Top2路由、共享专家 |

新原型直接实例化Transformers 5.22.1的[Qwen3.5](https://huggingface.co/docs/transformers/model_doc/qwen3_5)与[DeepSeek-V4](https://huggingface.co/docs/transformers/model_doc/deepseek_v4)类，从零训练，不加载品牌模型权重。具体维度保存于协议和`model.py`；它们是本项目选择的缩小配置。

总参数误差均小于0.06%。共同使用旧4096分词器、绑定输入输出词嵌入，并复制原Bio同种子的初始词嵌入。新模型其余参数沿用公共实现初始化；FFN种类、深度、运行态和FLOPs分别解释。新结构使用四层以容纳混合机制，原Bio保持两层。

DeepSeek原型有4个路由专家、每token选择2个、另加1个共享专家。估计激活参数约322,313；总参数以333,065匹配。Hash表采用可复现的独立随机选择，每token的两个专家不同。使用公共实现的0.001负载均衡辅助项，并分别记录CE和总目标；官方训练的无辅助损失路由偏置调度未实现。

在512输入下，CSA的topk512容纳全部可用压缩块，索引器不会筛掉可见条目，索引蒸馏未测试。公共实现未实例化MTP。缩小后的Qwen注意力采用14维全头RoPE；位置方案、专家数、长上下文、量化、后训练和系统优化分别以保存配置为准。原型未复现旗舰模型的完整训练流程。

## 训练与评分

按种子11、23、37逐对运行Qwen和DeepSeek，共6条新增训练；每条1024次更新。学习率、权重衰减、裁剪、阶段热身和答案mask沿用旧协议。旧问答监督只覆盖答案，EOS缺口也保留，方便回放同一训练条件。

每一步计算输入/mask SHA256，必须与同种子原`trans_full`第对应步完全相等，才允许更新。问答监督目标数另行计数。初始、文本结束、问答中段和最终评分保存原样。

语言评分仍为64个窗口、28篇文档、8192个目标；问答为原64次样本抽取、20篇文章、475个答案目标。指标为原正常历史NLL/PPL、首答案NLL及命中率。预先指定的附加诊断在seed11的同64道验证问答上进行独立自由生成，最多50 token、遇EOS结束，以SQuAD规范化EM/F1评分。每题生成额度为`min(50, 513−前缀长度)`，六模型共用同一额度，确保最后一次模型输入不超过原512长度。

新模型的验证适配器重算完整前缀，避免HF可变cache跨题污染。计时包括训练前向、反向、裁剪与优化器，使用本机FP32参考算子；原Bio/Transformer计时来自旧运行时段。Mac缺少CUDA版融合GDN和MoE核，不能据此估计旗舰系统的速度。适配器保存的输入token缓冲区也不能用于原生KV/递推状态的显存排名。

## 本机入口

使用独立环境，保留原`.venv`的依赖：

```bash
uv venv --python .venv/bin/python --system-site-packages results/tiny_arch/runtime
.venv/bin/python - <<'PY'
from pathlib import Path
import sysconfig
site = Path('results/tiny_arch/runtime/lib/python3.12/site-packages')
(site/'biomatrix_existing_runtime.pth').write_text(sysconfig.get_paths()['purelib']+'\n')
PY
uv pip install --python results/tiny_arch/runtime/bin/python -r evaluations/tiny_arch/requirements.txt
results/tiny_arch/runtime/bin/python run_biomatrix_tiny_arch.py --device mps
```

入口每秒原位显示六条详细进度和阶段信息。恢复时使用原输出目录：

```bash
results/tiny_arch/runtime/bin/python run_biomatrix_tiny_arch.py --device mps --resume results/tiny_arch/paired_20261103_local333k
```

代码拒绝覆盖已有目录，保存协议、源码快照、版本、原子`last.pt`、数值轨迹及完成/失败/中止/未执行状态。回放需要本地冻结语料、分词器、SQuAD编码数据和旧运行的1024步哈希。公开仓库保留净化协议和数值，原始数据与二进制权重保留本机；恢复原材料与路径后才能复跑。

原V3的58份计算源码、已训练检查点、LoCoMo结果和0.133B队列保持原样。新增模型的完整LoCoMo评估、其他优化器和专门超参数搜索未执行。
