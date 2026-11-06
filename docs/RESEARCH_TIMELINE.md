# Bio-Matrix 研究时间线

实验日期采用北京时间，依据各阶段报告。下方另列 GitHub 的公开提交日期。早期成果在 10 月 26 日集中归档，后续实现、评测和修复分别提交。

| 研究阶段 | 研究与执行记录 |
| --- | --- |
| 2026-10-21 · 首轮中文模型与独立状态实验 | 先训练八状态中文语言模型并完成三种子对照，随后独立验证状态递推、读出与频率机制。见 [实验迭代历程](ITERATION_HISTORY.md)。 |
| 2026-10-21 至 22 · V0.3–V0.5 | V0.3 筛查节点规模和局部耦合，V0.4 引入快慢矩阵状态和相位通信，V0.5 加入巩固与独立数据审计。正式训练、短验收和未执行条件分别记录。见 [结果索引](RESULTS_INDEX.md)。 |
| 2026-10-22 至 25 · V0.6–V0.9 | V0.6–V0.7 验证连续状态与球层路由，V0.8 检查块计算和数值稳定，V0.9 对照内容、动态历史和稀疏路由。见 [结果索引](RESULTS_INDEX.md)。 |
| 2026-10-26 起 · V1.0、V1.1、V2 | 先在 Transformer 中集成并诊断状态机制，随后改为 V2 块并行内容与相位地址。见 [实验迭代历程](ITERATION_HISTORY.md)。 |
| 2026-11-03 · V3 | 独立 Bio 向量节点与参数接近的 Transformer 配对，完成默认队列、LoCoMo 冻结评估和修正版微型架构对照。见 [结果索引](RESULTS_INDEX.md)。 |
| 2026-11-04 至 07 · 31M 本地验证 | 三个参数接近的模型各完成 1 亿 token 训练和五项公开评测。Transformer 在留出集和 WikiText 的语言建模指标领先。见 [完整结果](V3_LOCAL_31M_RESULTS.md)。 |

## 公开提交（北京时间，从早到晚）

| 公开日期 | 提交 |
| --- | --- |
| 2026-10-26 | 建立公开研究档案 · [289801e69](https://github.com/ZeQuanZhang/Bio-Matrix-Research/commit/289801e694b22c568d165d16a7a1ebc616bc7c35) |
| 2026-10-26 | 集中归档早期构想、实验与负结果 · [105462d8f](https://github.com/ZeQuanZhang/Bio-Matrix-Research/commit/105462d8f856c44ee59ffa16db591e2f3662f3d9) |
| 2026-10-29 | docs:重写研究说明与结果复核流程 · [a2ef9d16f](https://github.com/ZeQuanZhang/Bio-Matrix-Research/commit/a2ef9d16f72c68e33cd82628605d4dfce6b3bcca) |
| 2026-11-03 | feat:新增V3向量节点并统一版本编号 · [49723c54f](https://github.com/ZeQuanZhang/Bio-Matrix-Research/commit/49723c54fb3bf2858675bda9ac0784251949798b) |
| 2026-11-03 | docs:归档V3完整实验结果与状态分析 · [40e0fd85d](https://github.com/ZeQuanZhang/Bio-Matrix-Research/commit/40e0fd85dde45e7656a483c3185770cb574803c0) |
| 2026-11-03 | docs:新增V3的LoCoMo冻结评估与完整负结果 · [ad7e1115d](https://github.com/ZeQuanZhang/Bio-Matrix-Research/commit/ad7e1115d796121a2ccfd7d92229fa9475d4340b) |
| 2026-11-03 | feat:新增0.133B规模验证与百亿token训练入口 · [8ada96b3c](https://github.com/ZeQuanZhang/Bio-Matrix-Research/commit/8ada96b3c0d87a4f05f2c9dce3ee4ad823a3b860) |
| 2026-11-03 | feat:新增本地33万参数Qwen与DeepSeek配对评估 · [51043e3df](https://github.com/ZeQuanZhang/Bio-Matrix-Research/commit/51043e3dfcfd3803594958ce4fbe9a7be3d25e01) |
| 2026-11-03 | docs:公开修正版对比协议与冻结实现 · [18e744a4c](https://github.com/ZeQuanZhang/Bio-Matrix-Research/commit/18e744a4c818a9ee2d2b525beccfad703fd8fb76) |
| 2026-11-03 | docs:登记文本历史与答案先验补充控制 · [78ee6095c](https://github.com/ZeQuanZhang/Bio-Matrix-Research/commit/78ee6095c54742da82bca884b9d4aab47f8fdcb5) |
| 2026-11-03 | fix:修复MPS历史诊断的精度累加 · [ce4c6f4a8](https://github.com/ZeQuanZhang/Bio-Matrix-Research/commit/ce4c6f4a8dc460190cca04c97971dc15b60b0aa9) |
| 2026-11-03 | feat:新增修正版微型架构对比与独立保留测试 · [76399e9ba](https://github.com/ZeQuanZhang/Bio-Matrix-Research/commit/76399e9baa4dc93ac5c1c57ca12b9486e8fefe1f) |
| 2026-11-04 | feat:接入公开标准语言模型评测 · [62cfb8140](https://github.com/ZeQuanZhang/Bio-Matrix-Research/commit/62cfb8140ce8e0d730ce1a34b8c53babdcd1616e) |
| 2026-11-04 | feat:新增Mac本地31M配对训练与PyCharm入口 · [d7c7b5df4](https://github.com/ZeQuanZhang/Bio-Matrix-Research/commit/d7c7b5df46403bf885e0a291240cf34bc31c13c5) |
| 2026-11-04 | fix:使用项目解释器启动本地训练 · [e085e0d75](https://github.com/ZeQuanZhang/Bio-Matrix-Research/commit/e085e0d753704267749d869e902003f6098b431f) |
| 2026-11-04 | fix:补全PyCharm本地训练运行配置字段 · [613c64c9a](https://github.com/ZeQuanZhang/Bio-Matrix-Research/commit/613c64c9a6b79688322c5defc9626c6530823631) |
