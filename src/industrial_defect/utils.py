"""Shared utilities: seeding, environment capture, JSON helpers and timing."""

from __future__ import annotations

import json
import os
import platform
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_threads(num_threads: int) -> None:
    if num_threads > 0:
        torch.set_num_threads(num_threads)
        try:
            torch.set_num_interop_threads(max(1, num_threads // 2))
        except RuntimeError:
            # Inter-op pool can only be set once per process.
            pass


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def environment_info() -> dict[str, Any]:
    """Capture the actual hardware/software environment used for a run."""
    vm = psutil.virtual_memory()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "ram_total_gb": round(vm.total / 1e9, 2),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "numpy": np.__version__,
        "scikit_learn": _package_version("scikit-learn"),
        "scipy": _package_version("scipy"),
        "pillow": _package_version("pillow"),
    }


def jsonable(value: Any) -> Any:
    """Convert numpy/path/torch scalars into JSON-serialisable Python objects."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def save_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, ensure_ascii=False)
    return path


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """Usage: ``with timer() as t: ...`` then ``t["seconds"]``."""
    state: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield state
    finally:
        state["seconds"] = round(time.perf_counter() - start, 3)


def human_bytes(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"
