"""Text encoders (Transformer backbones)."""

from __future__ import annotations

import torch.nn as nn


class HFTransformerTextEncoder(nn.Module):
    """
    Hugging Face ``AutoModel`` with masked mean pooling + linear projection to hash dim.
    """

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
