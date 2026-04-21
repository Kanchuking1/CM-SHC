from __future__ import annotations

import torch

from src.hashing.similarity import calc_neighbor


def test_calc_neighbor_multi_hot():
    L = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.float)
    S = calc_neighbor(L, L)
    assert S[0, 1].item() == 1.0
    assert S[0, 2].item() == 0.0
