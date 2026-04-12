"""Shared loading for evaluate / retrieve pipelines."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from src.data.collators import load_hf_tokenizer, make_dcmh_collate_fn
from src.data.loaders import get_dataset
from src.data.transforms import imagenet_train_transform
from src.pipelines.train import build_model
from src.utils.checkpoint import find_latest_training_checkpoint, load_model_weights_only
from src.utils.config import experiment_run_dir, load_experiment
from src.utils.model_paths import local_files_only, resolve_pretrained_ref, torch_home_dir


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

    if use_mlp_text:
        text_ref, hf_lfo = "", False
    else:
        offline = local_files_only(cfg)
        repo = str(cfg.model.backbone.text)
        text_ref, hf_lfo = resolve_pretrained_ref(repo, cache_root, offline)

    if use_mlp_text:
        tokenizer = None
    else:
        tokenizer = load_hf_tokenizer(text_ref, local_files_only=hf_lfo)
    collate = make_dcmh_collate_fn(tokenizer, max_length=int(cfg.dataset.caption_max_length))

    transform = imagenet_train_transform()
    ds_kwargs: dict = {}
    if hasattr(cfg.dataset, "num_pseudo_classes") and cfg.dataset.num_pseudo_classes is not None:
        ds_kwargs["num_pseudo_classes"] = int(cfg.dataset.num_pseudo_classes)
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
