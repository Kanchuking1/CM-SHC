"""Label-based pairwise similarity (multi-hot OR class indices)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def calc_neighbor(label_1: torch.Tensor, label_2: torch.Tensor) -> torch.Tensor:
    """
    S_ij = 1 if samples i and j share at least one label, else 0.
    label_* : (n, C) multi-hot or (n,) long class indices.
    """
    if label_1.dim() == 1:
        label_1 = F.one_hot(label_1.long(), num_classes=int(label_1.max().item()) + 1).float()
    if label_2.dim() == 1:
        label_2 = F.one_hot(label_2.long(), num_classes=int(label_2.max().item()) + 1).float()
    inner = torch.mm(label_1.float(), label_2.float().t())
    return (inner > 0).float()
