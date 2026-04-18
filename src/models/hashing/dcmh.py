"""Deep Cross-Modal Hashing (Jiang et al., CVPR 2017).

https://openaccess.thecvf.com/content_cvpr_2017/papers/Jiang_Deep_Cross-Modal_Hashing_CVPR_2017_paper.pdf

Backbones are resolved through the registry in :mod:`src.models.backbones`
so swapping the image or text tower is a config change rather than a code
change.  Legacy kwargs (``image_backbone``, ``text_feature_dim``,
``text_model_name``, ``freeze_text_encoder``, ``local_files_only``) are
still accepted and translated internally, so existing configs / tests /
checkpoints continue to work -- the state-dict key layout
(``image_net.*``, ``text_encoder.*``, ``text_proj.*``) is preserved.

Transformer-path ``encode_text`` dispatches on the text backbone name so
each encoder uses its native pooling strategy: mean pooling over the
attention-masked last hidden state for BERT-style encoders and the
EOS-token (pooled ``text_embeds``) output for CLIP.
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


class DCMH(nn.Module):
    """Shared K-bit space for image and text.

    Binary codes ``B = sign(F + G)`` are produced in the trainer.

    Two equivalent ways to construct:

    * Legacy kwargs (kept for back-compat)::

          DCMH(bit_dim, text_model_name, image_backbone="alexnet",
               text_feature_dim=1386, freeze_text_encoder=False,
               local_files_only=False)

    * Registry-style config::

          DCMH(bit_dim,
               image_cfg={"name": "alexnet"},
               text_cfg={"name": "mlp_bow", "in_dim": 1386})

    When both are passed, the explicit ``image_cfg`` / ``text_cfg`` win.
    The module exposes the same attributes as before
    (``image_net``, ``text_encoder``, ``text_proj``, ``text_backend``)
    so the trainer does not need to change.
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

        # Image tower: whole backbone stays as ``image_net`` so its own
        # forward (which may include re-normalization for CLIP) runs as-is.
        self.image_net = build_image_backbone(self.image_cfg, out_dim=self.bit_dim)

        # Text tower: split into encoder body + projection head so the
        # trainer can give them different learning rates (or freeze the
        # body).  ``encode_text`` dispatches on the text backbone name.
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

        # Transformer-style path.
        assert self.text_encoder is not None
        if attention_mask is None:
            raise ValueError(
                "Transformer text backend expects input_ids and attention_mask."
            )
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        text_name = str(self.text_cfg.get("name", ""))
        if text_name == "clip":
            # CLIP's native pooling: EOS-token embedding projected to the
            # joint image/text space (``text_embeds``).
            return self.text_proj(out.text_embeds)

        # Default HF-transformer path: attention-masked mean pooling.
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
