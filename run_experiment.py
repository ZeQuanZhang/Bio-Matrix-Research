"""Run the registered experiment sequentially on one GPU; completed runs are reused."""
import json
from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    cfg = json.loads(Path("experiment.json").read_text())
    # Complete each matched pair before the separately trained state-removal control.
    runs = [(variant, seed) for seed in cfg["seeds"] for variant in ("baseline", "bio")]
    runs += [("stateless", seed) for seed in cfg["seeds"]]
    for variant, seed in runs:
        out = Path("results") / f"{variant}_seed{seed}"
        if (out / "result.json").exists():
            print(f"Already complete: {out}", flush=True)
            continue
        command = [sys.executable, "-u", "train.py", "--variant", variant, "--seed", str(seed), "--out", str(out)]
        for key in ("width", "layers", "context", "batch_size", "steps", "eval_every", "lr"):
            command += ["--" + key.replace("_", "-"), str(cfg[key])]
        out.mkdir(parents=True, exist_ok=True)
        print(f"Starting {out}", flush=True)
        with (out / "console.log").open("a") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"Completed {out}", flush=True)
    subprocess.run([sys.executable, "probe_memory.py"], check=True)
    subprocess.run([sys.executable, "report.py"], check=True)
