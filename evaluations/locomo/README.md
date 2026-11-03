# V3 LoCoMo 检查点评估

使用V3已训练检查点，在官方LoCoMo全部10段对话上进行一次冻结参数评估。网络、训练日志和源文件不改动；评估器位于独立目录，保留原V3恢复协议。

固定数据提交为 `snap-research/locomo@3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`。下载数据、评分原文件与许可证后，核对Git blob SHA-1、文件长度并记录SHA-256。数据缓存仅保存在本机 `data/locomo`，不随GitHub发布上传。

## 范围与输入

- 固定训练种子11，使用同一完整V3运行的四个1024步检查点，不选择最好种子。
- 官方1986题全部进入评估。主分数是第1–4类1540题的平均词级F1；446道对抗题单列。
- 历史按会话编号的时间顺序输入，保留日期、说话人、话轮ID，以及数据已有的图像caption；不加载图像，不输入事件标注、生成观察或摘要。
- 本项目4096 BPE编码的历史长度为20,643—39,046 token；保持完整历史，无裁剪。Bio逐块更新有限状态，Transformer仅保留各自128/512窗口。
- 问题格式沿用训练的 `Question: ... / Answer:`。每题从相同历史快照独立分支，其他问题和答案不会进入后续历史。
- 采用贪心解码、最多50个本项目BPE token，遇EOS结束。原文答案仅用于评分和额外teacher-forced NLL；F1基于自由生成答案。

## 十个条件

| 名称 | 检查点与处理 |
|---|---|
| D-main | 动态Bio，完整历史形成的内部状态 |
| D-clear | 动态Bio，清空历史状态但保留相同时钟 |
| D-donor | 动态Bio，其他对话的token流重复/截断到相同长度，时钟与主组一致 |
| D-oracle | 动态Bio，仅输入标注证据中的原始话轮 |
| S-main | 静态地址Bio，完整历史形成的内部状态 |
| S-oracle | 静态地址Bio，标注证据直给 |
| L128 | 原128窗口Transformer，完整历史流式处理后保留近期KV |
| L-oracle | 相同Transformer，标注证据直给 |
| F512 | 原512窗口Transformer，完整历史流式处理后保留近期KV |
| F-oracle | 相同Transformer，标注证据直给 |

Oracle为使用证据标注的诊断，主组不使用这些标注。4题缺失证据、9题引用无效话轮，保留主组评分，在oracle中标为not_applicable；对抗题也不进入oracle。Oracle共1527题，比较时同时提供主组的证据有效子集。

为让oracle中的完整证据可见，两个Transformer仅在oracle组将评测窗口扩展到1772 token；这是超出训练窗口的诊断。主组窗口不变。F512的名称只表示训练的512窗口，完整LoCoMo原文可见基线尚未运行。

## 评分与解释

`scoring.py`保留官方 `normalize_answer`、`f1_score` 和多答案 `f1` 的原始计算。仅移除不使用的BERT等依赖，增加分类包装。保留来源、提交、Git blob和CC-BY-NC-4.0许可证，见 [LICENSE.locomo.txt](LICENSE.locomo.txt)。

第1类使用逗号拆分子答案的F1；第2/3/4类使用官方词干F1，第3类按官方规则取参考答案分号前部分。第5类使用官方“No information available / not mentioned”短语规则；采用原始问题直接生成，未使用部分官方脚本的随机两选项提示，因此作为单列拒答诊断。

F1不是LLM裁判准确率；与MemT等使用LLM裁判的数字分开。答案NLL与首token命中为附加诊断，主生成不输入金标准答案。单种子结果只描述当前检查点；统计区间按10段对话聚类，未覆盖训练种子方差。

## 运行与保存

先安装项目主依赖和本评估的两个评分依赖：

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r evaluations/locomo/requirements.txt
.venv/bin/python run_biomatrix_v3_locomo.py --run-root results/v3/你的完整运行目录
```

本机默认运行目录为 `results/v3/paired_20261103T094249_111075`；需要原始 `last.pt`、冻结分词器及一致的模型源码。这些二进制与数据不随公开仓库发布。

PyCharm选择 **Bio-Matrix V3 LoCoMo**，使用项目`.venv`并启用 **Emulate terminal**。入口为根目录 `run_biomatrix_v3_locomo.py`，默认原生MPS、batch16，每秒原位显示10条条件进度、阶段和资源。不连接过期A800，不调用外部回答器或付费裁判。

新运行默认保存在 `results/locomo/v3_时间/`，拒绝覆盖已有输出。保存冻结协议、评分/模型源码身份、检查点和分词器哈希、源快照、数据审计、独立逐题预测、时延、状态登记及最终汇总。

逐题结果以固定批次原子保存到各条件的 `cXX_bXXX.json`。设置入口RESUME或使用 `--resume`恢复，已完成批次复用，协议变化则拒绝恢复。正常停止标记aborted，失败保留failed及traceback，未执行条件保留not_executed；原始训练检查点保持只读。

批量生成支持各题不同的块时钟，oracle的Transformer KV使用左填充及有效性mask。测试核对CPU/MPS、逐题与批量输出、状态不变、答案标签隔离、官方评分一致性与坏数据拒绝。
