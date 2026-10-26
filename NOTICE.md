# 数据和第三方材料

本仓库发布研究代码、构想、报告与数值测量，不提供训练语料镜像、编码语料或训练权重。

- 首轮中文任务使用[RobinChen2001/TinyStories-Zh-2M](https://huggingface.co/datasets/RobinChen2001/TinyStories-Zh-2M)。原数据卡标注MIT，英文上游[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)有其自身的数据许可。使用前应查看各数据来源的原始说明；代码中保留固定revision和校验流程。
- 后续语言任务使用[Salesforce WikiText](https://huggingface.co/datasets/Salesforce/wikitext)的官方切分；历史协议记录CC-BY-SA-3.0/GFDL来源。公开结果为本项目自定义BPE下的NLL/PPL，不能直接当作官方word-level排行榜指标。
- v0.3的CLEVR-SG派生验证基于[CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/)的真值场景和问题，历史记录标注CC BY 4.0。仓库未重新发布下载数据；该任务不是标准图像VQA。
- 相关论文通过链接引用，没有重新发布论文PDF或作者的完整文章。PyTorch及其他依赖保留各自许可。

构想文本中的生物、概率与动力系统术语表示计算研究设想。实现与实际证据的对应关系见迭代记录和各版conformance文档。

自有代码和文档当前未附加统一开源许可证。仓库公开可读不等于另行授予第三方数据或未明确许可材料的使用权。
