"""Shared loading for evaluate / retrieve pipelines."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from src.data.collators import (
    load_clip_tokenizer,
    load_hf_tokenizer,
    make_dcmh_collate_fn,
)
from src.data.loaders import get_dataset
from src.data.transforms import imagenet_train_transform
from src.pipelines.train import build_model, resolve_text_backbone_spec
from src.utils.checkpoint import find_latest_training_checkpoint, load_model_weights_only
from src.utils.config import experiment_run_dir, load_experiment
from src.utils.model_paths import torch_home_dir


def resolve_device(cfg_device: str, override: str | None) -> torch.device:
    d = override or cfg_device
    dev = torch.device(d if isinstance(d, str) else str(d))
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev


def load_model_and_dataset_for_eval(
    config_path: str | Path,
    checkpoint_path: str | Path,
    device_override: str | None = None,
    cfg: Any | None = None,
) -> tuple[Any, torch.nn.Module, Any, Any, torch.device, int, dict[str, Any]]:
    """
    Merge config, load weights, build tokenizer + dataset (no DataLoader yet).

    Pass ``cfg`` if you already called ``load_experiment`` (avoids loading YAML twice).

    Text-backbone handling mirrors ``src.pipelines.train.main``: string
    ``cfg.model.backbone.text`` is treated as a HF repo id, dict form
    (e.g. ``{name: clip, model_name: openai/clip-vit-base-patch32, ...}``)
    has ``model_name`` used as the repo id, and CLIP dicts load a
    ``CLIPTokenizer`` with the CLIP-specific ``max_length`` clamp
    applied.

    Returns
    -------
    cfg, model, collate_fn, dataset, device, checkpoint_epoch, meta
    """
    if cfg is None:
        cfg = load_experiment(config_path)
    cache_root = Path(cfg.paths.model_cache)
    os.environ["TORCH_HOME"] = str(torch_home_dir(cache_root))

    device = resolve_device(str(cfg.device), device_override)

    tdim = cfg.model.text_feature_dim
    use_mlp_text = tdim is not None
    text_ref, hf_lfo, is_clip_text = resolve_text_backbone_spec(cfg)

    if use_mlp_text:
        tokenizer = None
    elif is_clip_text:
        tokenizer = load_clip_tokenizer(text_ref, local_files_only=hf_lfo)
    else:
        tokenizer = load_hf_tokenizer(text_ref, local_files_only=hf_lfo)

    max_length = int(cfg.dataset.caption_max_length)
    if is_clip_text and tokenizer is not None:
        clip_cap = int(getattr(tokenizer, "model_max_length", 77))
        max_length = min(max_length, clip_cap)
    collate = make_dcmh_collate_fn(tokenizer, max_length=max_length)

    transform = imagenet_train_transform()
    ds_kwargs: dict = {}
    if hasattr(cfg.dataset, "num_pseudo_classes") and cfg.dataset.num_pseudo_classes is not None:
        ds_kwargs["num_pseudo_classes"] = int(cfg.dataset.num_pseudo_classes)
    text_mode = "bow" if use_mlp_text else "raw"
    ds_kwargs.setdefault("text_mode", text_mode)
    ds = get_dataset(
        str(cfg.dataset.name),
        root_dir=cfg.dataset.root,
        transform=transform,
        **ds_kwargs,
    )

    model = build_model(cfg, text_ref=text_ref, hf_local_files_only=hf_lfo).to(device)
    epoch, meta = load_model_weights_only(model, checkpoint_path, device)
    return cfg, model, collate, ds, device, epoch, meta


def resolve_checkpoint_path(
    cfg: Any,
    checkpoint: str | None,
    use_latest: bool,
) -> Path:
    if checkpoint and use_latest:
        raise ValueError("Pass only one of --checkpoint and --latest.")
    if use_latest:
        run_dir = experiment_run_dir(cfg)
        latest = find_latest_training_checkpoint(run_dir)
        if latest is None:
            raise FileNotFoundError(f"No epoch_*.pt under {run_dir}")
        return latest
    if not checkpoint:
        raise ValueError("Provide --checkpoint PATH or --latest.")
    return Path(checkpoint)
