"""Deterministic dataset splits for cross-modal hashing experiments."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SplitSubset(Dataset):
    """Wraps a dataset with a subset of indices, re-indexing to ``0..len-1``.

    Unlike ``torch.utils.data.Subset``, the ``"index"`` field in each returned
    dict is overwritten with the *local* position so that trainer buffers
    (F, G, B) are sized and indexed correctly.  Also exposes ``get_label``
    for fast label extraction without loading images.
    """

    def __init__(self, dataset: Dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, local_idx: int) -> dict:
        item = self.dataset[self.indices[local_idx]]
        item["index"] = local_idx
        return item

    def __len__(self) -> int:
        return len(self.indices)

    def get_label(self, local_idx: int) -> torch.Tensor:
        return self.dataset.get_label(self.indices[local_idx])


def make_mirflickr_split(
    n_total: int,
    query_size: int = 2000,
    train_size: int = 10000,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Deterministic 3-way split matching the DCMH paper (Section 4.1).

    Returns
    -------
    query_idx : list[int]
        Held-out query set (never trained on).
    train_idx : list[int]
        Training subset, sampled from the database (not overlapping query).
    db_idx : list[int]
        Full retrieval database = everything except query. Superset of train.
    """
    if query_size + train_size > n_total:
        raise ValueError(
            f"query_size ({query_size}) + train_size ({train_size}) "
            f"exceeds dataset size ({n_total})"
        )
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=gen)

    query_idx = sorted(perm[:query_size].tolist())
    rest = perm[query_size:]
    train_idx = sorted(rest[:train_size].tolist())
    db_idx = sorted(rest.tolist())
    return query_idx, train_idx, db_idx
