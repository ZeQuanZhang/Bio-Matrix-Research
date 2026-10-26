"""Balanced Chinese fact-recall probe; never used for training or checkpoint selection."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from bio_lm import Config, LanguageModel, detach_states, load_checkpoint


def cases(tokenizer, gap):
    names = ["小明", "莉莉", "汤姆", "小兔子", "小猫", "小狗", "小鸟", "小熊"]
    colors = ["红色", "蓝色", "黄色", "绿色"]
    fillers = ["院子里有一棵大树，微风轻轻地吹过树叶。", "远处传来一阵歌声，大家开心地拍起手来。",
               "厨房里飘出面包的香气，爸爸正在准备晚饭。", "太阳慢慢落山，星星出现在安静的夜空中。"]
    candidate_ids = [tokenizer.encode(color).ids for color in colors]
    if any(len(ids) != 1 for ids in candidate_ids):
        raise ValueError("This balanced probe requires single-token color candidates.")
    rows, seen = [], set()
    for variant in range(4):
        for name in names:
            for label, color in enumerate(colors):
                noise = []
                while len(noise) < gap:
                    noise.extend(tokenizer.encode(fillers[(variant + len(noise)) % len(fillers)]).ids)
                prefix = tokenizer.encode(f"{name}有一个{color}的球。").ids
                query = tokenizer.encode(f"现在，{name}的球是").ids
                tokens = [1, *prefix, *noise[:gap], *query]
                key = tuple(tokens)
                if key not in seen:
                    seen.add(key)
                    rows.append((tokens, label))
    return rows, [ids[0] for ids in candidate_ids]


@torch.no_grad()
def evaluate_checkpoint(checkpoint_path, tokenizer, device):
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = LanguageModel(Config(**checkpoint["protocol"]["config"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    static = model.cfg.variant == "stateless"
    mode = "stateless" if static else "bio"
    scores = {}
    for gap in [0, 64, 192, 512]:
        rows, candidates = cases(tokenizer, gap)
        scores[str(gap)] = {}
        for condition in ("carry", "reset"):
            correct, nll, count = 0, 0., 0
            for start in range(0, len(rows), 16):
                selected = rows[start:start + 16]
                last = torch.tensor([len(ids) - 1 for ids, _ in selected], device=device)
                labels = torch.tensor([label for _, label in selected], device=device)
                tokens = torch.zeros(len(selected), int(last.max()) + 1, dtype=torch.long, device=device)
                for i, (ids, _) in enumerate(selected):
                    tokens[i, :len(ids)] = torch.tensor(ids, device=device)
                states, predictions = None, torch.zeros(len(selected), 4, device=device)
                for position in range(0, tokens.shape[1], model.cfg.context):
                    prior = states if condition == "carry" and not static else None
                    logits, states = model(tokens[:, position:position + model.cfg.context], states=prior, mode=mode)
                    states = detach_states(states)
                    valid = (last >= position) & (last < position + logits.shape[1])
                    indices = torch.where(valid)[0]
                    predictions[indices] = logits[indices, last[indices] - position][:, candidates]
                correct += int((predictions.argmax(-1) == labels).sum())
                nll += float(torch.nn.functional.cross_entropy(predictions, labels, reduction="sum"))
                count += len(selected)
            scores[str(gap)][condition] = {"accuracy": correct / count, "candidate_nll": nll / count, "cases": count}
    return {"chance_accuracy": .25, "condition": "four single-token colors; balanced labels; attention cleared every 256 tokens",
            "gap_unit": "BPE tokens between fact and query; final query is additional",
            "limitation": "A synthetic diagnostic without task-specific fine-tuning; failure may reflect language/readout limitations.",
            "scores": scores}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = Tokenizer.from_file("data/tokenizer.json")
    config = json.loads(Path("experiment.json").read_text())
    for variant in config["variants"]:
        for seed in config["seeds"]:
            folder = Path(args.results) / f"{variant}_seed{seed}"
            if not (folder / "result.json").exists() or (folder / "memory_probe.json").exists():
                continue
            result = evaluate_checkpoint(folder / "best.pt", tokenizer, device)
            (folder / "memory_probe.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print(variant, seed, result["scores"], flush=True)
