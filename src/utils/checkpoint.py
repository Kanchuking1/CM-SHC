"""Training checkpoint discovery."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch


def find_latest_training_checkpoint(run_dir: Path) -> Path | None:
    """Pick ``epoch_XXXX.pt`` with largest epoch number."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return None
    best: Path | None = None
    best_n = -1
    for p in run_dir.glob("epoch_*.pt"):
        m = re.match(r"epoch_(\d+)\.pt$", p.name)
        if not m:
            continue
        n = int(m.group(1))
        if n > best_n:
            best_n = n
            best = p
    return best


def load_model_weights_only(model: torch.nn.Module, path: Path | str, device: torch.device | str) -> tuple[int, dict[str, Any]]:
    """Load only ``model_state_dict`` from a training checkpoint."""
    path = Path(path)
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    meta = ckpt.get("meta") or {}
    return int(ckpt.get("epoch", 0)), meta
