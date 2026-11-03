"""Atomic records, device selection and a single updating console progress line."""
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import torch
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def source_hashes():
    paths = [p for p in sorted((PROJECT/'v06').glob('*.py')) if not p.name.startswith('test')]
    paths += [PROJECT/'v04/model.py', PROJECT/'v05/model.py', PROJECT/'v05/run.py']
    return {str(p.relative_to(PROJECT)): file_hash(p) for p in paths}


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary); temporary.replace(path)


def detach(value):
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, dict):
        return {k: detach(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(detach(v) for v in value)
    if isinstance(value, list):
        return [detach(v) for v in value]
    return value


def state_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel()*value.element_size()
    if isinstance(value, dict):
        return sum(state_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(map(state_bytes, value))
    return 0


def device_for(name):
    if name == 'auto':
        if platform.system() == 'Darwin' and platform.machine() == 'arm64':
            name = 'mps'
        else:
            name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS 不可用：请用项目 .venv 的 Apple Silicon Python 启动。需要 CPU 时请明确设置 DEVICE="cpu"。')
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if name not in ('cpu', 'mps', 'cuda'):
        raise ValueError(f'Unknown device {name}')
    return torch.device(name)


def sync(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize(device)


class Console:
    """One carriage-return bar; only stage transitions and condition endings add lines."""
    def __init__(self, log=None, enabled=True):
        self.log = Path(log) if log else None
        self.enabled = enabled
        self.bar = None
        self.stage_name = None

    def record(self, event):
        if self.log:
            self.log.parent.mkdir(parents=True, exist_ok=True)
            with self.log.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(dict(time=time.time(), **event), ensure_ascii=False, allow_nan=False)+'\n')

    def stage(self, name, detail='', total=None, initial=0, unit='步', announce=True):
        self.close()
        self.stage_name = name
        self.record(dict(stage=name, detail=detail))
        if self.enabled and announce:
            tqdm.write(f'[{time.strftime("%H:%M:%S")}] {name}  {detail}')
        if total is not None:
            self.bar = tqdm(total=total, initial=initial, desc=name, unit=unit,
                            leave=False, dynamic_ncols=True, mininterval=1., maxinterval=5.,
                            disable=not self.enabled, file=__import__('sys').stdout)

    def update(self, amount=1, **metrics):
        if self.bar:
            if metrics:
                self.bar.set_postfix(metrics, refresh=False)
            self.bar.update(amount)

    def close(self):
        if self.bar:
            self.bar.close(); self.bar = None


class RunLock:
    """Kernel lock is released even after PyCharm force-stop; no stale-PID deletion."""
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open('a+')
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.close()
            raise RuntimeError('已有 v0.6 队列在运行；请勿重复启动。') from None
        self.handle.seek(0); self.handle.truncate(); self.handle.write(str(os.getpid())); self.handle.flush()
        return self

    def __exit__(self, *_):
        self.handle.close()
