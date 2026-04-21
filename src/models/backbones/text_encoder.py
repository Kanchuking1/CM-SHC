"""Text encoders used by DCMH / CM-SHC.

Two paths:

* :class:`MLPBowTextEncoder` -- two-layer MLP on fixed-length feature vectors
  (MIR-Flickr-25k BOW, 1386-dim).
* :class:`HFTransformerTextEncoder` -- Hugging Face ``AutoModel`` with
  masked mean pooling + linear projection.

Both expose the same ``.encoder`` / ``.proj`` attribute convention so
hashing models can decompose them uniformly:

* ``.encoder`` is the (possibly frozen) encoder body -- may be ``None``
  for the MLP path, in which case all trainable parameters live in
  ``.proj``.
* ``.proj`` is the final nn.Module that maps encoder output to the
  ``out_dim``-dim hash space.
"""

from __future__ import annotations

import torch.nn as nn


class MLPBowTextEncoder(nn.Module):
    """Two-layer MLP on fixed-length BOW text features.

    Matches the DCMH paper's text pathway: ``(B, in_dim) -> 4096 -> out_dim``
    with a ReLU in the middle and no output activation.

    All trainable parameters live in ``self.proj``; ``self.encoder`` is
    ``None`` so downstream trainers can probe the encoder/proj split
    uniformly with the transformer path.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.encoder = None
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_dim),
        )

    def forward(self, text_features):
        return self.proj(text_features)


class HFTransformerTextEncoder(nn.Module):
    """Hugging Face ``AutoModel`` with masked mean pooling + linear projection to hash dim."""

    def __init__(
        self,
        model_name: str,
        out_dim: int,
        freeze: bool = False,
        local_files_only: bool = False,
    ):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        h = self.encoder.config.hidden_size
        self.proj = nn.Linear(h, out_dim)
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        h = out.last_hidden_state
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return self.proj(pooled)
