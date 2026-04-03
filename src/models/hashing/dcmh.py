"""
Deep Cross-Modal Hashing (Jiang et al., CVPR 2017).
https://openaccess.thecvf.com/content_cvpr_2017/papers/Jiang_Deep_Cross-Modal_Hashing_CVPR_2017_paper.pdf
"""

from __future__ import annotations

import torch.nn as nn

from ..backbones.cnn import ResNet50ImageEncoder
from ..backbones.text_encoder import HFTransformerTextEncoder


class DCMH(nn.Module):
    """
    Shared K-bit space for image and text; binary codes B = sign(F + G) in the trainer.
    Set ``text_feature_dim`` to an int to use a shallow MLP on fixed vectors instead of HF.
    """

    def __init__(
        self,
        bit_dim: int,
        text_model_name: str,
        text_feature_dim: int | None = None,
        freeze_text_encoder: bool = False,
    ):
        super().__init__()
        self.bit_dim = bit_dim
        self.text_backend = "mlp" if text_feature_dim is not None else "transformer"
        self.image_net = ResNet50ImageEncoder(bit_dim)

        if text_feature_dim is not None:
            self.text_encoder = None
            self.text_proj = nn.Sequential(
                nn.Linear(text_feature_dim, max(bit_dim, 256)),
                nn.ReLU(inplace=True),
                nn.Linear(max(bit_dim, 256), bit_dim),
            )
        else:
            enc = HFTransformerTextEncoder(
                text_model_name,
                bit_dim,
                freeze=freeze_text_encoder,
            )
            self.text_encoder = enc.encoder
            self.text_proj = enc.proj

    def encode_image(self, x):
        return self.image_net(x)

    def encode_text(
        self,
        input_ids=None,
        attention_mask=None,
        text_features=None,
    ):
        if self.text_backend == "mlp":
            if text_features is None:
                raise ValueError("MLP text backend expects text_features= (B, text_feature_dim).")
            return self.text_proj(text_features)

        assert self.text_encoder is not None and attention_mask is not None
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        h = out.last_hidden_state
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return self.text_proj(pooled)

    def forward(self, image=None, input_ids=None, attention_mask=None, text_features=None):
        if image is not None and (input_ids is not None or text_features is not None):
            return self.encode_image(image), self.encode_text(
                input_ids, attention_mask, text_features
            )
        if image is not None:
            return self.encode_image(image)
        if input_ids is not None or text_features is not None:
            return self.encode_text(input_ids, attention_mask, text_features)
        raise ValueError("Provide image and/or text inputs.")
