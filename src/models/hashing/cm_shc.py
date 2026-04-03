"""
CM-SHC (semantic center cross-modal hashing) — integrate with ``src.hashing.centers``.
"""

from __future__ import annotations

import torch.nn as nn


class CMSHC(nn.Module):
    """Placeholder; replace with full model when stages 1–3 are implemented."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("CM-SHC model not yet implemented; use model: dcmh in config.")
