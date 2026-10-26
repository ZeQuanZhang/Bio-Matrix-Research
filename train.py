"""Matched Chinese language-model training, checkpointing, and held-out evaluation."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from bio_lm import Config, LanguageModel, detach_states, load_checkpoint


class Corpus:
    def __init__(self, root, split):
        self.tokens = np.load(root / f"{split}_tokens.npy", mmap_mode="r")
        self.offsets = np.load(root / f"{split}_offsets.npy")

    def document(self, index):
        return self.tokens[self.offsets[index]:self.offsets[index + 1]]

    def batch(self, ids, length, device, rng=None):
        x = np.zeros((len(ids), length), np.int64)
        y = np.full((len(ids), length), -100, np.int64)
        for i, index in enumerate(ids):
            doc = self.document(index)
            start = int(rng.integers(max(1, len(doc) - length))) if rng is not None else 0
            segment = doc[start:start + length + 1]
            count = len(segment) - 1
            x[i, :count], y[i, :count] = segment[:-1], segment[1:]
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def evaluate(model, corpus, batch_size, device, mode="bio", max_docs=None):
    model.eval()
    total, count, document_means = 0., 0, []
    n = len(corpus.offsets) - 1 if max_docs is None else min(max_docs, len(corpus.offsets) - 1)
    for start in range(0, n, batch_size):
        ids = list(range(start, min(start + batch_size, n)))
        x, y = corpus.batch(ids, model.cfg.context, device)
        logits, _ = model(x, mode=mode)
        losses = F.cross_entropy(logits.transpose(1, 2), y, ignore_index=-100, reduction="none")
        mask = y != -100
        sums = losses.sum(1).cpu().numpy()
        counts = mask.sum(1).cpu().numpy()
        total += float(sums.sum()); count += int(counts.sum())
        document_means.extend((sums / counts).tolist())
    model.train()
    return {"loss": total / count, "perplexity": math.exp(total / count),
            "tokens": count, "documents": n, "document_losses": document_means}


@torch.no_grad()
def memory_evaluate(model, corpus, batch_size, device, chunk=64, limit=256):
    """Clear attention between chunks; independently keep, reset, or swap Bio state."""
    model.eval()
    totals = {name: [0., 0] for name in ("carry", "reset", "swap")}
    for start in range(0, min(limit, len(corpus.offsets) - 1), batch_size):
        ids = list(range(start, min(start + batch_size, limit, len(corpus.offsets) - 1)))
        x, y = corpus.batch(ids, model.cfg.context, device)
        states = {name: None for name in totals}
        for position in range(0, model.cfg.context, chunk):
            for name in totals:
                prior = None if name == "reset" else states[name]
                if name == "swap" and prior is not None:
                    prior = [None if s is None else s.roll(1, 0) for s in prior]
                logits, updated = model(x[:, position:position + chunk], states=prior)
                states[name] = detach_states(updated)
                # The first chunk has no preceding state, so it cannot test memory transfer.
                if position:
                    target = y[:, position:position + chunk]
                    loss = F.cross_entropy(logits.transpose(1, 2), target, ignore_index=-100, reduction="sum")
                    totals[name][0] += float(loss); totals[name][1] += int((target != -100).sum())
    model.train()
    return {name: {"loss": value / count, "tokens": count} for name, (value, count) in totals.items()}


def atomic_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def run(args):
    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    manifest = json.loads((root / "manifest.json").read_text())
    cfg = Config(vocab_size=manifest["vocab_size"], width=args.width, layers=args.layers,
                 context=args.context, variant=args.variant,
                 ff_width=4 * args.width + (8 if args.variant == "baseline" else 0))
    model = LanguageModel(cfg).to(device)
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim >= 2 and ".bio." not in name else no_decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": .1},
                                   {"params": no_decay, "weight_decay": 0.}],
                                  lr=args.lr, betas=(.9, .95))
    train, valid, test = [Corpus(root, split) for split in ("train", "valid", "test")]
    protocol = {"config": asdict(cfg), "seed": args.seed, "steps": args.steps,
                "batch_size": args.batch_size, "lr": args.lr,
                "data_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
                "precision": "float32", "objective": "causal next-token cross entropy; no JEPA",
                "weight_decay": "0.1 on backbone matrices; 0 on normalization and Bio kinetics",
                "evaluation": "independent document prefixes, at most context tokens; no state carry",
                "parameters": model.parameter_count(), "torch": str(torch.__version__),
                "device": str(device), "platform": platform.platform()}
    checkpoint_path = out / "last.pt"
    step, tokens_seen, train_seconds, best = 0, 0, 0., float("inf")
    if checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path, device)
        if checkpoint["protocol"] != protocol:
            raise ValueError("Existing run has a different protocol. Choose a new output directory.")
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        rng.bit_generator.state = checkpoint["data_rng"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        step, tokens_seen = checkpoint["step"], checkpoint["tokens_seen"]
        train_seconds, best = checkpoint["train_seconds"], checkpoint["best"]
    (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    print(json.dumps({"start": protocol, "resume_step": step}), flush=True)
    mode = "stateless" if args.variant == "stateless" else "bio"
    log = (out / "metrics.jsonl").open("a")
    model.train()
    peak_driver = 0
    rolling_losses = []
    for current in range(step, args.steps):
        ids = rng.integers(len(train.offsets) - 1, size=args.batch_size)
        x, y = train.batch(ids, args.context, device, rng)
        lr_factor = min(1., (current + 1) / min(100, max(1, args.steps // 10)))
        cosine = .1 + .9 * .5 * (1 + math.cos(math.pi * current / args.steps))
        for group in optimizer.param_groups:
            group["lr"] = args.lr * lr_factor * cosine
        synchronize(device); started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x, mode=mode)
        loss = F.cross_entropy(logits.transpose(1, 2), y, ignore_index=-100)
        if not bool(loss.isfinite()):
            raise FloatingPointError(f"Non-finite training loss at step {current + 1}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not bool(grad_norm.isfinite()):
            raise FloatingPointError("Non-finite gradient")
        optimizer.step(); synchronize(device)
        train_seconds += time.perf_counter() - started
        tokens_seen += int((y != -100).sum())
        rolling_losses.append(float(loss.detach()))
        if device.type == "mps":
            peak_driver = max(peak_driver, torch.mps.driver_allocated_memory())
        number = current + 1
        if number % 25 == 0 or number == 1:
            entry = {"step": number, "train_loss": float(np.mean(rolling_losses)),
                     "tokens_seen": tokens_seen, "train_seconds": train_seconds,
                     "tokens_per_second": tokens_seen / train_seconds,
                     "peak_sampled_driver_gb": peak_driver / 1e9}
            rolling_losses.clear(); print(json.dumps(entry), flush=True)
            log.write(json.dumps(entry) + "\n"); log.flush()
        if number % args.eval_every == 0 or number == args.steps:
            evaluation = evaluate(model, valid, args.batch_size, device, mode)
            entry = {"step": number, "validation": {k: v for k, v in evaluation.items() if k != "document_losses"}}
            print(json.dumps(entry), flush=True); log.write(json.dumps(entry) + "\n"); log.flush()
            improved = evaluation["loss"] < best
            best = min(best, evaluation["loss"])
            payload = {"protocol": protocol, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "step": number, "tokens_seen": tokens_seen, "train_seconds": train_seconds, "best": best,
                       "data_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state()}
            atomic_checkpoint(checkpoint_path, payload)
            if improved:
                atomic_checkpoint(out / "best.pt", {"protocol": protocol, "model": model.state_dict(), "step": number})
    log.close()
    checkpoint = load_checkpoint(out / "best.pt", device)
    model.load_state_dict(checkpoint["model"])
    test_result = evaluate(model, test, args.batch_size, device, mode)
    # Report memory diagnostics only for the recurrent model; the primary score uses identical context.
    memory = memory_evaluate(model, test, args.batch_size, device) if args.variant == "bio" else None
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    samples = []
    model.eval()
    for prompt in ("从前，有一只小兔子，它最喜欢", "小明把红色的球放进盒子里。第二天，"):
        generated = [1, *tokenizer.encode(prompt).ids]
        generator = torch.Generator(device="cpu").manual_seed(123)
        for _ in range(80):
            context = torch.tensor([generated[-cfg.context:]], device=device)
            logits, _ = model(context, mode=mode)
            probabilities = (logits[0, -1].float().cpu() / .8).softmax(-1)
            top, indices = probabilities.topk(min(40, len(probabilities)))
            token = int(indices[torch.multinomial(top, 1, generator=generator)])
            if token == 2:
                break
            generated.append(token)
        samples.append({"prompt": prompt, "completion": tokenizer.decode(generated)})
    result = {"protocol": protocol, "best_step": checkpoint["step"], "tokens_seen": tokens_seen,
              "train_seconds": train_seconds, "tokens_per_second": tokens_seen / train_seconds,
              "peak_sampled_driver_gb": peak_driver / 1e9, "test": test_result,
              "memory_diagnostic": memory, "samples": samples}
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"complete": str(out), "test_loss": test_result["loss"],
                      "memory": memory, "samples": samples}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["baseline", "bio", "stateless"], required=True)
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=6e-4)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.context, args.eval_every, args.width, args.layers) <= 0:
        parser.error("Training dimensions, steps and evaluation frequency must be positive.")
    run(args)
