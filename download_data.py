"""Download the pinned public Chinese shard, checking its exact content hash."""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request


DATASET = "RobinChen2001/TinyStories-Zh-2M"
REVISION = "7601b862e5b2fbf60e83d7b27b8bd8017f105e70"
SHA256 = "9276478203557d0f6d5ee6547343e5a9286a2f72057115b9c24bb6da0b4e230c"


if __name__ == "__main__":
    root = Path("data"); root.mkdir(exist_ok=True)
    target = root / "train-00000-of-00003.parquet"
    valid = target.exists() and hashlib.file_digest(target.open("rb"), "sha256").hexdigest() == SHA256
    if not valid:
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/{target.name}"
        temporary = target.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if hashlib.file_digest(temporary.open("rb"), "sha256").hexdigest() != SHA256:
            raise ValueError("Downloaded shard failed SHA-256 verification; .part retained for inspection.")
        temporary.replace(target)
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{DATASET}/revision/{REVISION}", timeout=30) as response:
        metadata = json.load(response)
    if metadata["sha"] != REVISION:
        raise ValueError("Dataset revision mismatch.")
    (root / "source.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Verified {target}: {target.stat().st_size:,} bytes")
