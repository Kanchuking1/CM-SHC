"""Hamming distance / bit utilities for binary hash codes."""

from __future__ import annotations

import torch


def hamming_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Pairwise Hamming distances between rows of a (n, d) and b (m, d) in {0,1} or {-1,1}.
    Returns (n, m) long tensor.
    """
    if a.dtype != torch.float32 and a.dtype != torch.float64:
        a = a.float()
    if b.dtype != torch.float32 and b.dtype != torch.float64:
        b = b.float()
    an = a.size(0)
    bn = b.size(0)
    d = a.size(1)
    return ((a.unsqueeze(1) - b.unsqueeze(0)).abs() > 1e-6).sum(dim=-1).long()
