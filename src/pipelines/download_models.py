"""
Download pretrained weights into ``model_cache`` for offline HPC training.

Handles three backbone families:

* HF transformer text encoders (BERT / etc.) referenced via
  ``cfg.model.backbone.text`` -- either a plain string repo id or a dict
  with a ``model_name`` key.
* CLIP vision + text towers, referenced via ``cfg.model.backbone.{image,text}``
  dicts with ``name: clip`` and a shared ``model_name``.
* torchvision ResNet-50 ImageNet weights (TORCH_HOME-cached).

Usage::

    python -m src.pipelines.download_models --config configs/experiments/exp_cmshc_clip_lora_mirflickr25k_128bit.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from src.utils.config import load_experiment, repo_root
from src.utils.model_paths import (
    hf_snapshot_dir,
    hf_snapshot_looks_complete,
    torch_home_dir,
)


def _download_hf(repo_id: str, dest: Path) -> str:
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(dest))
    return str(dest)


def _ensure_hf_snapshot(repo_id: str, cache_root: Path, label: str) -> Path:
    dest = hf_snapshot_dir(repo_id, cache_root)
    if hf_snapshot_looks_complete(dest):
        print(f"[{label}] HF snapshot already present: {dest}")
    else:
        print(f"[{label}] downloading HF model {repo_id} -> {dest}")
        _download_hf(repo_id, dest)
        print(f"[{label}] done: {dest}")
    return dest


def _prime_resnet50_weights(torch_home: Path) -> None:
    os.environ["TORCH_HOME"] = str(torch_home)
    import torchvision.models as models

    try:
        w = models.ResNet50_Weights.IMAGENET1K_V1
        models.resnet50(weights=w)
    except AttributeError:
        models.resnet50(pretrained=True)


def _field_to_dict(field: Any) -> dict[str, Any] | None:
    if field is None:
        return None
    if isinstance(field, str):
        return {"name": field}
    try:
        resolved = OmegaConf.to_container(field, resolve=True)
    except Exception:
        resolved = dict(field) if isinstance(field, dict) else None
    return resolved if isinstance(resolved, dict) else None


def _collect_repo_ids(cfg) -> list[tuple[str, str]]:
    backbone = cfg.model.get("backbone", None)
    if backbone is None:
        return []
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for label, key in (("image", "image"), ("text", "text")):
        entry = _field_to_dict(backbone.get(key, None))
        if not entry:
            continue
        repo = entry.get("model_name")
        if not repo:
            continue
        repo = str(repo)
        if repo not in seen:
            seen.add(repo)
            pairs.append((label, repo))
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default="configs/experiments/exp_cmshc_mirflickr25k_128bit.yaml",
    )
    args = p.parse_args()

    cfg = load_experiment(args.config)
    cache_root = Path(cfg.paths.model_cache)
    cache_root.mkdir(parents=True, exist_ok=True)

    tdim = getattr(cfg.model, "text_feature_dim", None)
    backbone = cfg.model.get("backbone", None)

    # --- New-style backbone dicts (CLIP image + text, HF transformer text) ---
    new_style_pairs = _collect_repo_ids(cfg)
    for label, repo in new_style_pairs:
        _ensure_hf_snapshot(repo, cache_root, label=label)

    # --- Legacy string-valued backbone.text path ---
    if not new_style_pairs:
        text_field = backbone.get("text", None) if backbone is not None else None
        if tdim is not None:
            print("text_feature_dim is set; skipping HF text model download.")
        elif isinstance(text_field, str) and text_field:
            _ensure_hf_snapshot(text_field, cache_root, label="text")
        else:
            print("No HF backbone references found; skipping HF download.")

    # --- torchvision ResNet-50 ImageNet weights ---
    img_field = backbone.get("image", None) if backbone is not None else None
    img_dict = _field_to_dict(img_field)
    img_name = (img_dict or {}).get("name", "") if img_dict else ""
    if isinstance(img_field, str):
        img_name = img_field
    if str(img_name).lower() == "resnet50":
        th = torch_home_dir(cache_root)
        th.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(th)
        print(f"Priming ResNet-50 weights under TORCH_HOME={th}")
        _prime_resnet50_weights(th)
        print("ResNet-50 weights cached.")

    print(f"model_cache root: {cache_root.resolve()}")
    print("Repo root:", repo_root())


if __name__ == "__main__":
    main()
