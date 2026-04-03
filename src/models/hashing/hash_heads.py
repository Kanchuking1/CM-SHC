"""
Linear / MLP heads mapping backbone features to hash codes.
DCMH folds heads into backbones; reuse this module for newer architectures.
"""

from __future__ import annotations

import torch.nn as nn


def linear_hash_head(in_dim: int, bit_dim: int) -> nn.Module:
    return nn.Linear(in_dim, bit_dim)
