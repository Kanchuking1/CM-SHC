"""
Cross-modal Semantic Hash Centers (CM-SHC).

A cross-modal extension of the Semantic Hash Centers idea (Chen et al.,
TOIS 2025) on top of DCMH-style backbones. Architecturally identical to
``DCMH`` (AlexNet / ResNet-50 image encoder, MLP-on-BOW or HF-transformer
text encoder, both projecting to a shared ``q``-bit space). The *training
objective* is what differs -- see ``src.models.losses.semantic_center_loss``
and ``src.core.trainer.CMSHCTrainer``.
"""

from __future__ import annotations

import torch.nn as nn

from ..backbones.cnn import AlexNetImageEncoder, ResNet50ImageEncoder
from ..backbones.text_encoder import HFTransformerTextEncoder


_IMAGE_BACKBONES = {
    "alexnet": AlexNetImageEncoder,
    "resnet50": ResNet50ImageEncoder,
}


class CMSHC(nn.Module):
    """Shared ``q``-bit space for image and text.

    Parameters
    ----------
    bit_dim : int
        Hash code length ``q``.
    text_model_name : str
        Hugging Face repo id for the text encoder; ignored when
        ``text_feature_dim`` is set.
    image_backbone : str
        Either ``"alexnet"`` or ``"resnet50"``.
    text_feature_dim : int or None
        If ``int``, a 2-layer MLP is used on fixed-length feature vectors
        (paper default: 1386-dim BOW). If ``None``, a HF transformer is
        used.
    freeze_text_encoder : bool
        Whether to freeze the HF encoder body when the transformer text
        path is active.
    local_files_only : bool
        Offline flag forwarded to the HF ``AutoModel``.
    """

    def __init__(
        self,
        bit_dim: int,
        text_model_name: str,
        image_backbone: str = "alexnet",
        text_feature_dim: int | None = None,
        freeze_text_encoder: bool = False,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.bit_dim = bit_dim
        self.text_backend = "mlp" if text_feature_dim is not None else "transformer"

        img_cls = _IMAGE_BACKBONES.get(image_backbone)
        if img_cls is None:
            raise ValueError(
                f"Unknown image backbone {image_backbone!r}. "
                f"Choose from {list(_IMAGE_BACKBONES)}"
            )
        self.image_net = img_cls(bit_dim)

        if text_feature_dim is not None:
            self.text_encoder = None
            self.text_proj = nn.Sequential(
                nn.Linear(text_feature_dim, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, bit_dim),
            )
        else:
            enc = HFTransformerTextEncoder(
                text_model_name,
                bit_dim,
                freeze=freeze_text_encoder,
                local_files_only=local_files_only,
            )
            self.text_encoder = enc.encoder
            self.text_proj = enc.proj

    # ------------------------------------------------------------------
    # Encoding helpers (signature matches DCMH so evaluate/retrieve work)
    # ------------------------------------------------------------------
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
