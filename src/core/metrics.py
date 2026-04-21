"""Retrieval metrics for cross-modal hashing."""

from __future__ import annotations

import torch
from tqdm import tqdm


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


def mean_average_precision_hamming(
    dist: torch.Tensor,
    query_labels: torch.Tensor,
    db_labels: torch.Tensor,
) -> float:
    """MAP with label-based relevance (DCMH paper protocol).

    Two items are relevant iff they share at least one semantic label:
    ``relevant(i, j) = (query_labels[i] @ db_labels[j] > 0)``.

    Parameters
    ----------
    dist : (Nq, Nd) Hamming distance matrix (lower = more similar).
    query_labels : (Nq, C) multi-hot label matrix for queries.
    db_labels : (Nd, C) multi-hot label matrix for database items.

    Returns
    -------
    float -- Mean Average Precision averaged over all queries.
    """
    nq = dist.size(0)
    if nq == 0:
        return 0.0

    ap_sum = 0.0
    for i in tqdm(range(nq), desc="MAP", leave=False):
        order = dist[i].argsort()
        rel = (query_labels[i] @ db_labels[order].t() > 0).float()
        n_rel = rel.sum().item()
        if n_rel == 0:
            continue
        cumrel = rel.cumsum(dim=0)
        precision_at_k = cumrel * rel / torch.arange(1, rel.numel() + 1, dtype=torch.float32)
        ap_sum += float(precision_at_k.sum().item() / n_rel)

    return ap_sum / nq
