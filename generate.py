"""Local Chinese continuation, optionally carrying Bio state across context blocks."""
import argparse
import hashlib
from pathlib import Path

import torch
from tokenizers import Tokenizer

from bio_lm import Config, LanguageModel, detach_states, load_checkpoint


@torch.no_grad()
def generate(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    path = Path(args.checkpoint)
    # One open inode keeps the identity consistent if a trainer replaces best.pt atomically.
    with path.open("rb") as stream:
        signature = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        checkpoint = load_checkpoint(stream, device)
    model = LanguageModel(Config(**checkpoint["protocol"]["config"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    manifest = Path(args.data) / "manifest.json"
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != checkpoint["protocol"]["data_sha256"]:
        raise ValueError("The data/tokenizer manifest does not belong to this checkpoint.")
    tokenizer = Tokenizer.from_file(str(Path(args.data) / "tokenizer.json"))
    mode = "stateless" if model.cfg.variant == "stateless" else "bio"
    states, buffer = None, [1]
    if args.state_in:
        memory = load_checkpoint(args.state_in, device)
        if memory["checkpoint_sha256"] != signature:
            raise ValueError("Saved state belongs to a different model checkpoint.")
        states, buffer = memory["states"], memory["tokens"]
    buffer = buffer + tokenizer.encode(args.prompt).ids
    # shortcut: positions restart per block; train with streamed blocks before claiming long-context quality.
    while len(buffer) > model.cfg.context:
        _, states = model(torch.tensor([buffer[:model.cfg.context]], device=device),
                          states=None if mode == "stateless" else states, mode=mode)
        states = detach_states(states)
        buffer = buffer[model.cfg.context:]
    generated = []
    rng = torch.Generator().manual_seed(args.seed)
    for _ in range(args.tokens):
        logits, updated = model(torch.tensor([buffer], device=device),
                                states=None if mode == "stateless" else states, mode=mode)
        probabilities = (logits[0, -1].float().cpu() / args.temperature).softmax(-1)
        top, indices = probabilities.topk(min(40, len(probabilities)))
        token = int(indices[torch.multinomial(top, 1, generator=rng)])
        generated.append(token)
        if len(buffer) == model.cfg.context:
            states, buffer = detach_states(updated), [token]
        else:
            buffer.append(token)
        if token == 2:
            break
    print(args.prompt + tokenizer.decode(generated))
    if args.state_out:
        torch.save({"checkpoint_sha256": signature, "states": states, "tokens": buffer}, args.state_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data")
    parser.add_argument("--prompt", default="从前，有一只小兔子")
    parser.add_argument("--tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=.8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--state-in")
    parser.add_argument("--state-out")
    args = parser.parse_args()
    if args.tokens < 1 or args.temperature <= 0:
        parser.error("tokens and temperature must be positive")
    generate(args)
