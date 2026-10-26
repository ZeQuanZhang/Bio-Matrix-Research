"""Aggregate the completed paired runs without hiding unfavorable outcomes."""
import json
import os
from pathlib import Path
import statistics

fontconfig = Path.home()/'.config/codex-fonts/fonts.conf'
if fontconfig.exists():
    os.environ["FONTCONFIG_FILE"] = str(fontconfig)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def summarize(root=Path("results")):
    cfg = json.loads(Path("experiment.json").read_text())
    results = {}
    for variant in cfg["variants"]:
        results[variant] = []
        for seed in cfg["seeds"]:
            path = root / f"{variant}_seed{seed}" / "result.json"
            if path.exists():
                results[variant].append(json.loads(path.read_text()))
    if not any(results.values()):
        raise ValueError("No completed runs yet.")
    training_complete = all(len(rows) == len(cfg["seeds"]) for rows in results.values())
    probes_complete = all((root / f"{v}_seed{s}" / "memory_probe.json").exists()
                          for v in cfg["variants"] for s in cfg["seeds"])
    complete = training_complete and probes_complete
    status = ("全部完成" if complete else "训练完成，事实回忆探针尚未全部完成" if training_complete else
              "部分完成，不能据此作最终比较")
    lines = ["# 中文 Bio-LLM 实测结果", "", f"实验状态：{status}。",
             "", "同规模、同数据、同训练步数；计算成本单独报告。测试交叉熵为 nats/token，越低越好。",
             "", "| 模型 | 完成种子数 | 参数量 | 测试交叉熵（均值 ± 标准差） | 困惑度均值 | 训练速度 Token/s | 训练分钟/次 | 采样 Metal GB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    aggregate = {}
    for variant, rows in results.items():
        if not rows:
            continue
        losses = [row["test"]["loss"] for row in rows]
        std = statistics.stdev(losses) if len(losses) > 1 else 0.
        average = lambda key: statistics.mean(row[key] for row in rows)
        aggregate[variant] = {"runs": len(rows), "test_loss_mean": statistics.mean(losses),
                              "test_loss_std": std, "perplexity_mean": statistics.mean(row["test"]["perplexity"] for row in rows),
                              "tokens_per_second": average("tokens_per_second"), "training_seconds": average("train_seconds")}
        lines.append(f"| {variant} | {len(rows)} | {rows[0]['protocol']['parameters']:,} | {statistics.mean(losses):.4f} ± {std:.4f} | "
                     f"{aggregate[variant]['perplexity_mean']:.2f} | {average('tokens_per_second'):,.0f} | "
                     f"{average('train_seconds')/60:.2f} | {average('peak_sampled_driver_gb'):.2f} |")
    lines += ["", "## 配对结果", "", "| 随机种子 | 相同有效训练 Token 数 | baseline 测试损失 | bio 测试损失 | bio − baseline | stateless 测试损失 | bio − stateless |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    deltas, state_deltas = [], []
    for seed in cfg["seeds"]:
        pair = {v: next((r for r in rows if r["protocol"]["seed"] == seed), None) for v, rows in results.items()}
        if pair["baseline"] is None or pair["bio"] is None:
            continue
        base, bio = pair["baseline"], pair["bio"]
        assert base["tokens_seen"] == bio["tokens_seen"], "Training token budgets differ"
        assert base["protocol"]["parameters"] == bio["protocol"]["parameters"], "Parameter budgets differ"
        delta = bio["test"]["loss"] - base["test"]["loss"]; deltas.append(delta)
        static = f"{pair['stateless']['test']['loss']:.4f}" if pair["stateless"] else "未完成"
        state_delta_text = "未完成"
        if pair["stateless"]:
            control = pair["stateless"]
            assert control["tokens_seen"] == bio["tokens_seen"], "Stateless training token budget differs"
            assert control["protocol"]["parameters"] == bio["protocol"]["parameters"], "Stateless parameter budget differs"
            state_delta = bio["test"]["loss"] - control["test"]["loss"]
            state_deltas.append(state_delta)
            state_delta_text = f"{state_delta:+.4f}"
        lines.append(f"| {seed} | {bio['tokens_seen']:,} | {base['test']['loss']:.4f} | {bio['test']['loss']:.4f} | {delta:+.4f} | {static} | {state_delta_text} |")
    if deltas:
        delta = statistics.mean(deltas)
        lines += ["", f"Bio 相对标准模型的平均交叉熵差值为 **{delta:+.4f} nats/token**；"
                  f"由平均交叉熵换算的困惑度相对变化为 **{(np.exp(delta)-1)*100:+.2f}%**。负值代表 Bio 较好。",
                  "", f"在 {len(deltas)} 个已完成的配对种子中，Bio 更好的有 {sum(d < 0 for d in deltas)} 次，"
                  f"更差的有 {sum(d > 0 for d in deltas)} 次。这是重复实验的结果汇总，不是统计显著性检验。",
                  "", "只有三个随机种子，结果是探索性证据。训练数据、领域和规模均有限，不能解释为通用架构已获证明。"]
        aggregate["paired_delta"] = {"mean": delta, "values": deltas}
    if state_deltas:
        state_delta = statistics.mean(state_deltas)
        aggregate["bio_minus_stateless"] = {"mean": state_delta, "values": state_deltas}
        lines += ["", f"Bio 相对单独训练的无状态对照，平均交叉熵差值为 **{state_delta:+.4f} nats/token**，"
                  f"对应困惑度变化 **{(np.exp(state_delta)-1)*100:+.2f}%**；"
                  f"{len(state_deltas)} 个配对种子中，Bio 更好 {sum(d < 0 for d in state_deltas)} 次，"
                  f"更差 {sum(d > 0 for d in state_deltas)} 次。"]
    lines += ["", "## 状态消融诊断", "", "以下诊断每 64 Token 清空注意力历史，只统计第二块以后的 Token。它与主测试条件不同。",
              "", "| 种子 | 保留状态 | 清零状态 | 交换其他样本状态 |", "|---|---:|---:|---:|"]
    for row in results.get("bio", []):
        m = row["memory_diagnostic"]
        lines.append(f"| {row['protocol']['seed']} | {m['carry']['loss']:.4f} | {m['reset']['loss']:.4f} | {m['swap']['loss']:.4f} |")
    lines += ["", "## 中文事实回忆诊断", "", "随机指定人物球的颜色，插入不同长度的无关中文后要求续写颜色。"
              "四种颜色均衡，机会水平 25%；模型未针对该题微调。每 256 Token 清空注意力，仅 carry 条件保留 Bio 状态。",
              "", "| 模型 | 种子 | 间隔 Token | 保留状态准确率 | 清零状态准确率 |", "|---|---:|---:|---:|---:|"]
    for variant, rows in results.items():
        for row in rows:
            seed = row["protocol"]["seed"]
            path = root / f"{variant}_seed{seed}" / "memory_probe.json"
            if path.exists():
                probe = json.loads(path.read_text())
                for gap, values in probe["scores"].items():
                    lines.append(f"| {variant} | {seed} | {gap} | {values['carry']['accuracy']:.1%} | {values['reset']['accuracy']:.1%} |")
    lines += ["", "该诊断须结合零间隔成绩解读：若零间隔也很低，失败可能首先来自语言理解或读出，不能单独归咎于记忆。"]
    lines += ["", "## 续写样本", "", "以下为固定提示、固定采样随机数的完整样本，不挑选好看的输出。"]
    for variant, rows in results.items():
        if rows:
            lines += ["", f"### {variant}，种子 {rows[0]['protocol']['seed']}"]
            for sample in rows[0]["samples"]:
                lines += ["", f"提示：{sample['prompt']}", "", f"> {sample['completion']}"]
    lines += ["", "## 如何解读", "", "先看 bio 是否稳定优于 baseline，再看是否优于单独训练的 stateless。"
              "若只优于 baseline 而不优于 stateless，额外非线性可能已经解释收益。"
              "若保留状态不优于清零状态，尚无证据表明跨块记忆有帮助。"
              "状态交换会破坏输入分布，只能作为诊断。", "", "内存数值是周期采样的 PyTorch Metal 驱动分配量，"
              "不是真正系统峰值；速度只统计优化步骤，不含验证、保存与数据准备。", "",
              "每次训练约 1,000 万有效 Token，是固定预算的首轮架构对照，不代表充分预训练或已经收敛。", "",
              "![训练与验证曲线](learning_curves.png)"]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    (root / "summary.json").write_text(json.dumps({"complete": complete, "aggregate": aggregate}, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    colors = {"baseline": "#64748b", "bio": "#0d9488", "stateless": "#d97706"}
    for variant in cfg["variants"]:
        for number, seed in enumerate(cfg["seeds"]):
            path = root / f"{variant}_seed{seed}" / "metrics.jsonl"
            if not path.exists():
                continue
            entries = [json.loads(line) for line in path.read_text().splitlines()]
            train = [e for e in entries if "train_loss" in e]
            valid = [e for e in entries if "validation" in e]
            label = variant if number == 0 else None
            axes[0].plot([e["step"] for e in train], [e["train_loss"] for e in train], color=colors[variant], alpha=.55, label=label)
            axes[1].plot([e["step"] for e in valid], [e["validation"]["loss"] for e in valid], color=colors[variant], alpha=.65, label=label)
    for axis, title in zip(axes, ("Chinese training loss", "Chinese validation loss")):
        axis.set(title=title, xlabel="Optimizer step", ylabel="Cross entropy (nats/token)")
        axis.grid(alpha=.2); axis.legend()
    fig.tight_layout(); fig.savefig(root / "learning_curves.png", dpi=180); plt.close(fig)
    print(json.dumps({"complete": complete, "report": str(root / "report.md"), "aggregate": aggregate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    summarize()
