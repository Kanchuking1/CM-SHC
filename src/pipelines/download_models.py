"""
Download pretrained weights into ``model_cache`` for offline HPC training.

Usage::

    python -m src.pipelines.download_models --config configs/experiments/exp_dcmh_flickr8k.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.utils.config import load_experiment, repo_root
from src.utils.model_paths import (
    hf_snapshot_dir,
    hf_snapshot_looks_complete,
    resolve_hf_pretrained_path,
    torch_home_dir,
)


def _download_hf(repo_id: str, dest: Path) -> str:
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(dest))
    return str(dest)


def _prime_resnet50_weights(torch_home: Path) -> None:
    os.environ["TORCH_HOME"] = str(torch_home)
    import torchvision.models as models

    try:
        w = models.ResNet50_Weights.IMAGENET1K_V1
        models.resnet50(weights=w)
    except AttributeError:
        models.resnet50(pretrained=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/experiments/exp_dcmh_flickr8k.yaml")
    args = p.parse_args()

    cfg = load_experiment(args.config)
    cache_root = Path(cfg.paths.model_cache)
    cache_root.mkdir(parents=True, exist_ok=True)

    # --- Hugging Face text encoder + tokenizer ---
    repo = getattr(cfg.model.backbone, "text", None)
    tdim = getattr(cfg.model, "text_feature_dim", None)
    if tdim is not None:
        print("text_feature_dim is set; skipping HF text model download.")
    elif repo:
        dest = hf_snapshot_dir(str(repo), cache_root)
        if hf_snapshot_looks_complete(dest):
            print(f"HF snapshot already present: {dest}")
        else:
            print(f"Downloading HF model {repo} -> {dest}")
            _download_hf(str(repo), dest)
            print(f"Done: {dest}")
    else:
        print("No model.backbone.text in config; skipping HF download.")

    # --- torchvision ResNet-50 ImageNet weights ---
    img = getattr(cfg.model.backbone, "image", "")
    if str(img).lower() == "resnet50":
        th = torch_home_dir(cache_root)
        th.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(th)
        print(f"Priming ResNet-50 weights under TORCH_HOME={th}")
        _prime_resnet50_weights(th)
        print("ResNet-50 weights cached.")
    else:
        print(f"Backbone image={img!r} — add handling in download_models if needed.")

    print(f"model_cache root: {cache_root.resolve()}")
    print("Repo root:", repo_root())


if __name__ == "__main__":
    main()
