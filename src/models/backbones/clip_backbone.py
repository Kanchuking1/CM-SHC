"""CLIP backbones for cross-modal hashing.

Two classes -- :class:`CLIPImageBackbone` and :class:`CLIPTextBackbone` --
that load one tower of a CLIP checkpoint (default
``openai/clip-vit-base-patch32``) and project its joint-embedding output
to the hash dimension.

Both expose the ``.encoder`` / ``.proj`` attribute convention used by
:class:`src.models.hashing.dcmh.DCMH` and
:class:`src.models.hashing.cm_shc.CMSHC`, so they slot into the
backbone registry and downstream trainer unchanged.

Image preprocessing
-------------------
The default data pipeline applies ImageNet normalization (mean/std
``[0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]``).  CLIP was trained
with slightly different normalization (``[0.48145466, 0.4578275,
0.40821073] / [0.26862954, 0.26130258, 0.27577711]``).  When
``renormalize=True`` (default) the image backbone converts ImageNet-
normalized inputs to CLIP-normalized inputs on the fly, so existing
datasets keep working.  Set ``renormalize=False`` if your data loader
already emits CLIP-normalized tensors.

Text tokenization
-----------------
The text backbone assumes batches produced with the CLIP tokenizer
(``CLIPTokenizer`` or ``AutoTokenizer`` on the same repo id) --
i.e. ``input_ids`` padded to the tokenizer's model_max_length (77 for
ViT-B/32) with an attention mask.  The collator layer (Task #25) is
responsible for emitting these tensors.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _freeze_module(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False


class CLIPImageBackbone(nn.Module):
    """CLIP vision tower with linear projection to ``out_dim``.

    Parameters
    ----------
    model_name : str
        HF repo id, e.g. ``"openai/clip-vit-base-patch32"``.
    out_dim : int
        Hash code length.
    freeze : bool
        Freeze the CLIP encoder body (default True -- the "frozen CLIP"
        baseline).  The ``proj`` head is always trainable.
    local_files_only : bool
        Forwarded to ``from_pretrained`` for offline runs.
    renormalize : bool
        Re-normalize from ImageNet stats to CLIP stats inside the forward
        pass (default True).  Set False when the data loader already
        emits CLIP-normalized tensors.
    """

    def __init__(
        self,
        model_name: str,
        out_dim: int,
        freeze: bool = True,
        local_files_only: bool = False,
        renormalize: bool = True,
    ):
        super().__init__()
        from transformers import CLIPVisionModelWithProjection

        self.encoder = CLIPVisionModelWithProjection.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        feat_dim = int(self.encoder.config.projection_dim)
        self.proj = nn.Linear(feat_dim, int(out_dim))

        if freeze:
            _freeze_module(self.encoder)

        self._renormalize = bool(renormalize)
        if self._renormalize:
            self.register_buffer(
                "_imnet_mean",
                torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "_imnet_std",
                torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "_clip_mean",
                torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "_clip_std",
                torch.tensor(_CLIP_STD).view(1, 3, 1, 1),
                persistent=False,
            )

    def _to_clip_space(self, image: torch.Tensor) -> torch.Tensor:
        # Undo ImageNet normalization, then apply CLIP normalization.
        x = image * self._imnet_std + self._imnet_mean
        x = (x - self._clip_mean) / self._clip_std
        return x

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self._to_clip_space(image) if self._renormalize else image
        out = self.encoder(pixel_values=x)
        # ``image_embeds`` are the joint-space features (post visual_projection).
        feat = out.image_embeds
        return self.proj(feat)


class CLIPTextBackbone(nn.Module):
    """CLIP text tower with linear projection to ``out_dim``.

    Exposes ``.encoder`` (the CLIP text tower) and ``.proj`` (linear head).
    ``forward`` takes ``input_ids`` and ``attention_mask`` produced by a
    CLIP tokenizer and returns ``(B, out_dim)`` logits.
    """

    def __init__(
        self,
        model_name: str,
        out_dim: int,
        freeze: bool = True,
        local_files_only: bool = False,
    ):
        super().__init__()
        from transformers import CLIPTextModelWithProjection

        self.encoder = CLIPTextModelWithProjection.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        feat_dim = int(self.encoder.config.projection_dim)
        self.proj = nn.Linear(feat_dim, int(out_dim))

        if freeze:
            _freeze_module(self.encoder)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # ``text_embeds`` are the joint-space features (post text_projection).
        feat = out.text_embeds
        return self.proj(feat)
