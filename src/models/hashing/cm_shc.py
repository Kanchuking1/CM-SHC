"""Cross-modal Semantic Hash Centers (CM-SHC).

A cross-modal extension of the Semantic Hash Centers idea (Chen et al.,
TOIS 2025) on top of DCMH-style backbones. Architecturally identical to
``DCMH`` (AlexNet / ResNet-50 / CLIP image encoder, MLP-on-BOW or
HF-transformer / CLIP text encoder, both projecting to a shared
``q``-bit space). The *training objective* is what differs -- see
``src.models.losses.semantic_center_loss`` and
``src.core.trainer.CMSHCTrainer``.

Backbones are resolved through the registry in
:mod:`src.models.backbones`; legacy kwargs are accepted and translated
internally so existing configs and tests keep working.  ``encode_text``
dispatches on the text backbone name (mean-pool for BERT-style,
``text_embeds`` for CLIP) so old checkpoints and new CLIP runs share the
same ``text_encoder.*`` / ``text_proj.*`` state-dict keys.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from ..backbones import (
    build_image_backbone,
    build_text_backbone,
    legacy_to_image_cfg,
    legacy_to_text_cfg,
    text_backend_tag,
)


class CMSHC(nn.Module):
    """Shared ``q``-bit space for image and text.

    Parameters
    ----------
    bit_dim : int
        Hash code length ``q``.
    text_model_name : str
        Hugging Face repo id for the text encoder; ignored when
        ``text_feature_dim`` is set or when a ``text_cfg`` is supplied.
    image_backbone : str
        Legacy kwarg; either ``"alexnet"`` or ``"resnet50"``. Ignored
        when ``image_cfg`` is supplied.
    text_feature_dim : int or None
        Legacy kwarg. If ``int``, a 2-layer MLP is used on fixed-length
        feature vectors (paper default: 1386-dim BOW). If ``None``, a HF
        transformer is used.  Ignored when ``text_cfg`` is supplied.
    freeze_text_encoder : bool
        Whether to freeze the HF encoder body when the transformer text
        path is active.
    local_files_only : bool
        Offline flag forwarded to the HF ``AutoModel``.
    image_cfg, text_cfg : dict or None
        Registry-style configs. When provided they take precedence over
        the legacy kwargs.
    """

    def __init__(
        self,
        bit_dim: int,
        text_model_name: str = "",
        image_backbone: str = "alexnet",
        text_feature_dim: int | None = None,
        freeze_text_encoder: bool = False,
        local_files_only: bool = False,
        *,
        image_cfg: dict[str, Any] | None = None,
        text_cfg: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.bit_dim = int(bit_dim)

        if image_cfg is None:
            image_cfg = legacy_to_image_cfg(image_backbone)
        if text_cfg is None:
            text_cfg = legacy_to_text_cfg(
                text_feature_dim=text_feature_dim,
                text_model_name=text_model_name,
                freeze_text_encoder=freeze_text_encoder,
                local_files_only=local_files_only,
            )

        self.image_cfg = dict(image_cfg)
        self.text_cfg = dict(text_cfg)

        self.image_net = build_image_backbone(self.image_cfg, out_dim=self.bit_dim)

        text_module = build_text_backbone(self.text_cfg, out_dim=self.bit_dim)
        self.text_encoder = getattr(text_module, "encoder", None)
        self.text_proj = text_module.proj
        self.text_backend = text_backend_tag(self.text_cfg)

    def encode_image(self, x):
        return self.image_net(x)

    def encode_text(self, input_ids=None, attention_mask=None, text_features=None):
        if self.text_backend == "mlp":
            if text_features is None:
                raise ValueError(
                    "MLP text backend expects text_features= (B, text_feature_dim)."
                )
            return self.text_proj(text_features)

        assert self.text_encoder is not None
        if attention_mask is None:
            raise ValueError(
                "Transformer text backend expects input_ids and attention_mask."
            )
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        text_name = str(self.text_cfg.get("name", ""))
        if text_name == "clip":
            return self.text_proj(out.text_embeds)

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
