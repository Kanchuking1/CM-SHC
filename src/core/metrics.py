"""Retrieval metrics for cross-modal hashing (paired queries: index i matches index i)."""

from __future__ import annotations

import torch


def recall_at_k_hamming(
    dist: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[int, float]:
    """
    Mean Recall@K when the single relevant item for query i is database index i.

    ``dist`` shape (N, N): dist[i, j] = Hamming distance from query i to item j (lower is better).
    """
    n = dist.size(0)
    if n == 0:
        return {k: 0.0 for k in ks}
    out: dict[int, float] = {}
    for k in ks:
        kk = min(k, n)
        hits = 0
        for i in range(n):
            topk = dist[i].argsort()[:kk]
            hits += int((topk == i).any().item())
        out[k] = hits / n
    return out


def mean_reciprocal_rank_hamming(dist: torch.Tensor) -> float:
    """MRR when relevant index for query i is i."""
    n = dist.size(0)
    if n == 0:
        return 0.0
    rr_sum = 0.0
    for i in range(n):
        order = dist[i].argsort()
        rank = (order == i).nonzero(as_tuple=True)[0]
        if rank.numel() == 0:
            continue
        rr_sum += 1.0 / float(rank[0].item() + 1)
    return rr_sum / n
