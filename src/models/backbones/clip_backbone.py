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

LoRA mode
---------
When ``lora_cfg`` is a dict, the encoder body is wrapped with
``peft.get_peft_model(LoraConfig(**lora_cfg))`` after being frozen, so
only the LoRA adapter parameters and the projection head are trainable
end-to-end.  The default ``target_modules`` (CLIP attention projections)
can be overridden via the config.  Adapter weights are kept inside
``state_dict()``, so the trainer's checkpoint/resume path works without
any special handling; ``save_pretrained`` is available via
:meth:`save_lora_adapter` for users who want to ship the adapters
separately.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# CLIP's attention sub-modules all share this naming convention for the
# query/key/value/output projections.  Good default for LoRA target modules.
_DEFAULT_CLIP_LORA_TARGETS: tuple[str, ...] = ("q_proj", "k_proj", "v_proj")


def _freeze_module(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False


def _coerce_lora_cfg(lora_cfg: Any) -> dict[str, Any] | None:
    """Normalize an OmegaConf DictConfig / plain dict / None into a plain dict."""
    if lora_cfg is None:
        return None
    # OmegaConf container -> primitive dict (resolves interpolations, too).
    try:
        from omegaconf import DictConfig, OmegaConf  # type: ignore

        if isinstance(lora_cfg, DictConfig):
            lora_cfg = OmegaConf.to_container(lora_cfg, resolve=True)
    except Exception:  # pragma: no cover - omegaconf optional
        pass
    if not isinstance(lora_cfg, dict):
        raise TypeError(
            f"lora_cfg must be a mapping, got {type(lora_cfg).__name__}"
        )
    return dict(lora_cfg)


def _apply_lora(encoder: nn.Module, lora_cfg: dict[str, Any]) -> nn.Module:
    """Freeze ``encoder`` and wrap it with peft LoRA, returning a ``PeftModel``.

    Recognised keys (all optional; remaining keys pass through to
    :class:`peft.LoraConfig` unchanged):

    * ``r`` (int, default 8) -- LoRA rank.
    * ``lora_alpha`` (int, default 16) -- scaling constant.
    * ``lora_dropout`` (float, default 0.0).
    * ``target_modules`` (list[str] or str, default CLIP attention projections).
    * ``bias`` (str, default ``"none"``).
    """
    try:
        from peft import LoraConfig, get_peft_model  # type: ignore
    except ImportError as e:
        raise ImportError(
            "peft is required for LoRA mode.  Install it with "
            "`pip install peft` and re-run."
        ) from e

    cfg = dict(lora_cfg)  # do not mutate caller
    target_modules: Iterable[str] | str = cfg.pop(
        "target_modules", list(_DEFAULT_CLIP_LORA_TARGETS)
    )
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    else:
        target_modules = list(target_modules)

    r = int(cfg.pop("r", 8))
    lora_alpha = int(cfg.pop("lora_alpha", 16))
    lora_dropout = float(cfg.pop("lora_dropout", 0.0))
    bias = str(cfg.pop("bias", "none"))

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        **cfg,
    )

    # Freeze the base encoder first.  peft will automatically flip
    # ``requires_grad=True`` on the LoRA adapter weights it inserts.
    _freeze_module(encoder)
    return get_peft_model(encoder, lora_config)


def _is_peft_model(module: nn.Module) -> bool:
    """Return True iff ``module`` is a peft ``PeftModel``."""
    try:
        from peft import PeftModel  # type: ignore
    except ImportError:
        return False
    return isinstance(module, PeftModel)


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
        baseline).  The ``proj`` head is always trainable.  Ignored when
        ``lora_cfg`` is set: LoRA implies a frozen base encoder with
        trainable adapter weights.
    local_files_only : bool
        Forwarded to ``from_pretrained`` for offline runs.
    renormalize : bool
        Re-normalize from ImageNet stats to CLIP stats inside the forward
        pass (default True).  Set False when the data loader already
        emits CLIP-normalized tensors.
    lora_cfg : dict | None
        When a mapping is supplied, wrap the encoder with peft LoRA.
        See :func:`_apply_lora` for recognized keys; default targets are
        the attention q/k/v projections.
    """

    def __init__(
        self,
        model_name: str,
        out_dim: int,
        freeze: bool = True,
        local_files_only: bool = False,
        renormalize: bool = True,
        lora_cfg: dict[str, Any] | None = None,
    ):
        super().__init__()
        from transformers import CLIPVisionModelWithProjection

        self.encoder = CLIPVisionModelWithProjection.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        feat_dim = int(self.encoder.config.projection_dim)
        self.proj = nn.Linear(feat_dim, int(out_dim))

        lora_cfg = _coerce_lora_cfg(lora_cfg)
        self._uses_lora = lora_cfg is not None
        if lora_cfg is not None:
            # LoRA wraps the (frozen) encoder.  The proj head remains trainable.
            self.encoder = _apply_lora(self.encoder, lora_cfg)
        elif freeze:
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

    def save_lora_adapter(self, save_dir: str) -> None:
        """Save just the LoRA adapter weights to ``save_dir`` (peft format)."""
        _save_lora_adapter(self.encoder, save_dir)

    def load_lora_adapter(self, load_dir: str) -> None:
        """Load LoRA adapter weights from ``load_dir`` onto the wrapped encoder."""
        self.encoder = _load_lora_adapter(self.encoder, load_dir)


class CLIPTextBackbone(nn.Module):
    """CLIP text tower with linear projection to ``out_dim``.

    Exposes ``.encoder`` (the CLIP text tower) and ``.proj`` (linear head).
    ``forward`` takes ``input_ids`` and ``attention_mask`` produced by a
    CLIP tokenizer and returns ``(B, out_dim)`` logits.

    See :class:`CLIPImageBackbone` for the shared ``lora_cfg`` behaviour.
    """

    def __init__(
        self,
        model_name: str,
        out_dim: int,
        freeze: bool = True,
        local_files_only: bool = False,
        lora_cfg: dict[str, Any] | None = None,
    ):
        super().__init__()
        from transformers import CLIPTextModelWithProjection

        self.encoder = CLIPTextModelWithProjection.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        feat_dim = int(self.encoder.config.projection_dim)
        self.proj = nn.Linear(feat_dim, int(out_dim))

        lora_cfg = _coerce_lora_cfg(lora_cfg)
        self._uses_lora = lora_cfg is not None
        if lora_cfg is not None:
            self.encoder = _apply_lora(self.encoder, lora_cfg)
        elif freeze:
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

    def save_lora_adapter(self, save_dir: str) -> None:
        """Save just the LoRA adapter weights to ``save_dir`` (peft format)."""
        _save_lora_adapter(self.encoder, save_dir)

    def load_lora_adapter(self, load_dir: str) -> None:
        """Load LoRA adapter weights from ``load_dir`` onto the wrapped encoder."""
        self.encoder = _load_lora_adapter(self.encoder, load_dir)


def _save_lora_adapter(encoder: nn.Module, save_dir: str) -> None:
    if not _is_peft_model(encoder):
        raise RuntimeError(
            "save_lora_adapter() called on a backbone that was not built "
            "with lora_cfg; there are no adapter weights to save."
        )
    encoder.save_pretrained(save_dir)


def _load_lora_adapter(encoder: nn.Module, load_dir: str) -> nn.Module:
    try:
        from peft import PeftModel  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "peft is required to load a LoRA adapter."
        ) from e

    if _is_peft_model(encoder):
        # Already wrapped: just load the weights into the adapter in place.
        encoder.load_adapter(load_dir, adapter_name="default", is_trainable=True)
        return encoder
    return PeftModel.from_pretrained(encoder, load_dir, is_trainable=True)
