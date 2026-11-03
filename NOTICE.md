# 数据和第三方材料

本仓库发布研究代码、构想、报告与数值测量，不提供训练语料镜像、编码语料或训练权重。

- 首轮中文任务使用[RobinChen2001/TinyStories-Zh-2M](https://huggingface.co/datasets/RobinChen2001/TinyStories-Zh-2M)。原数据卡标注MIT，英文上游[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)有其自身的数据许可。使用前应查看各数据来源的原始说明；代码中保留固定revision和校验流程。
- 后续语言任务使用[Salesforce WikiText](https://huggingface.co/datasets/Salesforce/wikitext)的官方切分；历史协议记录CC-BY-SA-3.0/GFDL来源。公开结果为本项目自定义BPE下的NLL/PPL，不能直接当作官方word-level排行榜指标。
- v0.3的CLEVR-SG派生验证基于[CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/)的真值场景和问题，历史记录标注CC BY 4.0。仓库未重新发布下载数据；该任务不是标准图像VQA。
- 相关论文通过链接引用，没有重新发布论文PDF或作者的完整文章。PyTorch及其他依赖保留各自许可。

构想文本中的生物、概率与动力系统术语表示计算研究设想。实现与实际证据的对应关系见迭代记录和各版conformance文档。

自有代码和文档当前未附加统一开源许可证。仓库公开可读不等于另行授予第三方数据或未明确许可材料的使用权。


V1.1—V3还使用[官方SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/)派生的远史问答。源文件固定提交与校验位于 `v11/data.py`，准备协议记录切分、答案位置、分词器与文件身份。公开仓库保留数值与代码，原文、输入token、编码缓存和权重不随发布上传。

V3长对话评估使用[LoCoMo官方仓库](https://github.com/snap-research/locomo/tree/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376)的`data/locomo10.json`。官方项目采用CC-BY-NC-4.0；本仓库只发布数值投影与评分代码适配，原始数据不上传。`evaluations/locomo/scoring.py`改编自官方`task_eval/evaluation.py`，保留归一化、词干与多答案F1计算，去除无关BERT依赖并增加分类包装。出处与固定提交见该文件头部，完整许可证见[LICENSE.locomo.txt](evaluations/locomo/LICENSE.locomo.txt)。

V3规模验证使用[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)的`sample-100BT`，数据卡采用ODC-By-1.0并说明Common Crawl使用条件。固定提交、分片LFS SHA256和tokenizer身份保存在`scale/data.py`及准备清单。只复用[openai-community/gpt2](https://huggingface.co/openai-community/gpt2)的MIT分词器，不加载其权重。实际训练预算为每模型10B目标，语料原文、编码缓存、SQuAD原文与权重不随仓库发布。

本地33万参数架构对照使用[Transformers 5.22.1](https://github.com/huggingface/transformers/tree/v5.22.1)的Qwen3.5和DeepSeek-V4模型类。该软件包采用Apache-2.0许可证，通过独立运行环境安装；仓库新增自有缩小配置与适配器，全部从零训练。第三方代码文件SHA与版本保存在实验协议；公开副本保存数值、源码身份与文档链接，不包含品牌模型权重。
