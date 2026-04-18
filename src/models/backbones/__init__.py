"""Backbone registry for cross-modal hashing models.

Exposes a small factory layer so hashing models (DCMH, CM-SHC, future
CLIP-based variants) do not have to know about concrete backbone classes.

Two dicts register the available backbones:

* :data:`IMAGE_BACKBONE_REGISTRY` -- ``name -> factory(out_dim, **kwargs)``
* :data:`TEXT_BACKBONE_REGISTRY`  -- ``name -> factory(out_dim, **kwargs)``

Each factory must return an ``nn.Module`` that exposes two attributes:

* ``.encoder`` -- the (possibly frozen) encoder body, or ``None`` when
  there is no separate pretrained encoder (e.g. BOW MLP).
* ``.proj``    -- the final ``nn.Module`` mapping encoder output to the
  ``out_dim``-dim hash space.

Hashing models consume config dicts of the shape::

    image_cfg = {"name": "alexnet"}
    text_cfg  = {"name": "mlp_bow", "in_dim": 1386}
    text_cfg  = {"name": "hf_transformer", "model_name": "bert-base-uncased",
                 "freeze": False, "local_files_only": True}
    image_cfg = {"name": "clip", "model_name": "openai/clip-vit-base-patch32"}
    text_cfg  = {"name": "clip", "model_name": "openai/clip-vit-base-patch32"}

Legacy DCMH / CM-SHC kwargs (``image_backbone``, ``text_feature_dim``,
``text_model_name``, ``freeze_text_encoder``, ``local_files_only``) are
translated via :func:`legacy_to_image_cfg` / :func:`legacy_to_text_cfg`
so that existing configs and tests keep working.
"""

from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn

from .cnn import AlexNetImageEncoder, ResNet50ImageEncoder
from .text_encoder import HFTransformerTextEncoder, MLPBowTextEncoder

ImageFactory = Callable[..., nn.Module]
TextFactory = Callable[..., nn.Module]


def _alexnet_factory(out_dim: int, **_: Any) -> nn.Module:
    return AlexNetImageEncoder(out_dim)


def _resnet50_factory(out_dim: int, **_: Any) -> nn.Module:
    return ResNet50ImageEncoder(out_dim)


def _mlp_bow_factory(out_dim: int, in_dim: int, **_: Any) -> nn.Module:
    return MLPBowTextEncoder(in_dim=in_dim, out_dim=out_dim)


def _hf_transformer_factory(
    out_dim: int,
    model_name: str,
    freeze: bool = False,
    local_files_only: bool = False,
    **_: Any,
) -> nn.Module:
    return HFTransformerTextEncoder(
        model_name=model_name,
        out_dim=out_dim,
        freeze=freeze,
        local_files_only=local_files_only,
    )


def _clip_image_factory(
    out_dim: int,
    model_name: str = "openai/clip-vit-base-patch32",
    freeze: bool = True,
    local_files_only: bool = False,
    renormalize: bool = True,
    **_: Any,
) -> nn.Module:
    # Lazy import so test suites that do not touch CLIP do not need the
    # newer transformers and peft versions.
    from .clip_backbone import CLIPImageBackbone

    return CLIPImageBackbone(
        model_name=model_name,
        out_dim=out_dim,
        freeze=freeze,
        local_files_only=local_files_only,
        renormalize=renormalize,
    )


def _clip_text_factory(
    out_dim: int,
    model_name: str = "openai/clip-vit-base-patch32",
    freeze: bool = True,
    local_files_only: bool = False,
    **_: Any,
) -> nn.Module:
    from .clip_backbone import CLIPTextBackbone

    return CLIPTextBackbone(
        model_name=model_name,
        out_dim=out_dim,
        freeze=freeze,
        local_files_only=local_files_only,
    )


IMAGE_BACKBONE_REGISTRY: dict[str, ImageFactory] = {
    "alexnet": _alexnet_factory,
    "resnet50": _resnet50_factory,
    "clip": _clip_image_factory,
}

TEXT_BACKBONE_REGISTRY: dict[str, TextFactory] = {
    "mlp_bow": _mlp_bow_factory,
    "hf_transformer": _hf_transformer_factory,
    "clip": _clip_text_factory,
}


def register_image_backbone(name: str, factory: ImageFactory) -> None:
    """Register a new image backbone factory under ``name`` (idempotent on identity)."""
    existing = IMAGE_BACKBONE_REGISTRY.get(name)
    if existing is not None and existing is not factory:
        raise ValueError(f"Image backbone {name!r} is already registered.")
    IMAGE_BACKBONE_REGISTRY[name] = factory


def register_text_backbone(name: str, factory: TextFactory) -> None:
    """Register a new text backbone factory under ``name`` (idempotent on identity)."""
    existing = TEXT_BACKBONE_REGISTRY.get(name)
    if existing is not None and existing is not factory:
        raise ValueError(f"Text backbone {name!r} is already registered.")
    TEXT_BACKBONE_REGISTRY[name] = factory


def _split_name(cfg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if "name" not in cfg:
        raise ValueError(
            f"Backbone config is missing required 'name' key. Got: {cfg!r}"
        )
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return str(cfg["name"]), kwargs


def build_image_backbone(cfg: dict[str, Any], out_dim: int) -> nn.Module:
    """Instantiate an image backbone from a config dict."""
    name, kwargs = _split_name(cfg)
    factory = IMAGE_BACKBONE_REGISTRY.get(name)
    if factory is None:
        raise ValueError(
            f"Unknown image backbone {name!r}. "
            f"Choose from {sorted(IMAGE_BACKBONE_REGISTRY)}"
        )
    return factory(out_dim=out_dim, **kwargs)


def build_text_backbone(cfg: dict[str, Any], out_dim: int) -> nn.Module:
    """Instantiate a text backbone from a config dict."""
    name, kwargs = _split_name(cfg)
    factory = TEXT_BACKBONE_REGISTRY.get(name)
    if factory is None:
        raise ValueError(
            f"Unknown text backbone {name!r}. "
            f"Choose from {sorted(TEXT_BACKBONE_REGISTRY)}"
        )
    return factory(out_dim=out_dim, **kwargs)


def legacy_to_image_cfg(image_backbone: str) -> dict[str, Any]:
    """Translate the legacy ``image_backbone=<str>`` kwarg to a registry cfg."""
    return {"name": str(image_backbone)}


def legacy_to_text_cfg(
    text_feature_dim: int | None,
    text_model_name: str,
    freeze_text_encoder: bool = False,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Translate legacy DCMH / CM-SHC text kwargs to a registry cfg.

    * ``text_feature_dim`` set  -> MLP-on-BOW path
    * ``text_feature_dim`` None -> HF transformer path
    """
    if text_feature_dim is not None:
        return {"name": "mlp_bow", "in_dim": int(text_feature_dim)}
    return {
        "name": "hf_transformer",
        "model_name": str(text_model_name),
        "freeze": bool(freeze_text_encoder),
        "local_files_only": bool(local_files_only),
    }


_TEXT_BACKEND_TAGS: dict[str, str] = {
    "mlp_bow": "mlp",
    "hf_transformer": "transformer",
    "clip": "transformer",
}


def register_text_backend_tag(name: str, tag: str) -> None:
    """Declare which trainer dispatch tag a new text backbone maps to."""
    if tag not in ("mlp", "transformer"):
        raise ValueError(f"text backend tag must be 'mlp' or 'transformer', got {tag!r}")
    _TEXT_BACKEND_TAGS[name] = tag


def text_backend_tag(text_cfg: dict[str, Any]) -> str:
    """Return the short backend tag the trainer switches on.

    * ``"mlp"``         -- fixed-length BOW feature path (takes ``text_features``)
    * ``"transformer"`` -- ``input_ids`` + ``attention_mask`` path
    """
    name = str(text_cfg.get("name", ""))
    tag = _TEXT_BACKEND_TAGS.get(name)
    if tag is None:
        raise ValueError(
            f"No text backend tag known for text backbone {name!r}. "
            "Register a mapping via register_text_backend_tag() if this is a new backbone."
        )
    return tag


__all__ = [
    "AlexNetImageEncoder",
    "ResNet50ImageEncoder",
    "HFTransformerTextEncoder",
    "MLPBowTextEncoder",
    "IMAGE_BACKBONE_REGISTRY",
    "TEXT_BACKBONE_REGISTRY",
    "register_image_backbone",
    "register_text_backbone",
    "build_image_backbone",
    "build_text_backbone",
    "legacy_to_image_cfg",
    "legacy_to_text_cfg",
    "text_backend_tag",
    "register_text_backend_tag",
]
