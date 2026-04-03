"""
Train from merged YAML experiment config.

Usage (from repository root)::

    python -m src.pipelines.train --config configs/experiments/exp_dcmh_flickr8k.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.core.trainer import DCMHTrainer
from src.data.collators import build_train_labels_tensor, load_hf_tokenizer, make_dcmh_collate_fn
from src.data.loaders import get_dataset
from src.data.transforms import imagenet_train_transform
from src.models.hashing.dcmh import DCMH
from src.utils.config import experiment_run_dir, load_experiment, repo_root
from src.utils.logger import setup_logging
from src.utils.model_paths import local_files_only, resolve_pretrained_ref, torch_home_dir
from src.utils.seed import set_seed


def _cfg_to_json_dict(cfg) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)


def build_model(cfg, text_ref: str, hf_local_files_only: bool):
    if cfg.model.name != "dcmh":
        raise ValueError(
            f"Only model.name=dcmh is implemented in the pipeline; got {cfg.model.name!r}. "
            "Implement CM-SHC or switch config."
        )
    tdim = cfg.model.text_feature_dim
    if tdim is not None:
        tdim = int(tdim)
    return DCMH(
        bit_dim=int(cfg.model.bit_dim),
        text_model_name=text_ref,
        text_feature_dim=tdim,
        freeze_text_encoder=bool(cfg.model.freeze_text_encoder),
        local_files_only=hf_local_files_only and tdim is None,
    )


def find_latest_training_checkpoint(run_dir: Path) -> Path | None:
    """Pick ``epoch_XXXX.pt`` with largest epoch number."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return None
    best: Path | None = None
    best_n = -1
    for p in run_dir.glob("epoch_*.pt"):
        m = re.match(r"epoch_(\d+)\.pt$", p.name)
        if not m:
            continue
        n = int(m.group(1))
        if n > best_n:
            best_n = n
            best = p
    return best


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default="configs/experiments/exp_dcmh_flickr8k.yaml",
        help="Path to experiment YAML (merged with base, model, dataset).",
    )
    p.add_argument("--device", type=str, default=None, help="Override cfg.device (cuda/cpu)")
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore training checkpoints in the experiment dir even if training.resume is true.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_experiment(args.config)

    cache_root = Path(cfg.paths.model_cache)
    os.environ["TORCH_HOME"] = str(torch_home_dir(cache_root))

    set_seed(int(cfg.seed))
    device = args.device or str(cfg.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    offline = local_files_only(cfg)
    repo = str(cfg.model.backbone.text)
    text_ref, hf_lfo = resolve_pretrained_ref(repo, cache_root, offline)

    run_dir = experiment_run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_root = Path(cfg.output.root) / cfg.output.logs
    log_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_dir=log_root)
    logger.info("Experiment dir: %s", run_dir)
    logger.info("TORCH_HOME=%s", os.environ.get("TORCH_HOME"))
    logger.info("HF pretrained ref: %s (local_files_only=%s)", text_ref, hf_lfo)

    (run_dir / "run_config.json").write_text(
        json.dumps(_cfg_to_json_dict(cfg), indent=2),
        encoding="utf-8",
    )

    transform = imagenet_train_transform()
    train_ds = get_dataset(
        str(cfg.dataset.name),
        root_dir=cfg.dataset.root,
        transform=transform,
        num_pseudo_classes=int(cfg.dataset.num_pseudo_classes),
    )

    tokenizer = load_hf_tokenizer(text_ref, local_files_only=hf_lfo)
    collate = make_dcmh_collate_fn(tokenizer, max_length=int(cfg.dataset.caption_max_length))

    loader = DataLoader(
        train_ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=int(cfg.training.num_workers),
        collate_fn=collate,
        pin_memory=device.startswith("cuda"),
    )

    train_labels = build_train_labels_tensor(train_ds, int(cfg.dataset.num_pseudo_classes))

    model = build_model(cfg, text_ref=text_ref, hf_local_files_only=hf_lfo).to(device)
    trainer = DCMHTrainer(
        model=model,
        train_loader=loader,
        train_labels=train_labels,
        device=device,
        gamma=float(cfg.model.gamma),
        eta=float(cfg.model.eta),
        max_epoch=int(cfg.training.max_epochs),
        lr_img=float(cfg.training.lr_img),
        lr_txt=float(cfg.training.lr_txt),
    )

    start_epoch = 0
    want_resume = bool(cfg.training.get("resume", False)) and not args.no_resume
    if want_resume:
        latest = find_latest_training_checkpoint(run_dir)
        if latest is not None:
            start_epoch = trainer.load_training_checkpoint(latest)
            logger.info("Resumed from %s at start_epoch=%s", latest, start_epoch)

    meta = {
        "experiment_name": str(cfg.experiment_name),
        "model": str(cfg.model.name),
        "dataset": str(cfg.dataset.name),
        "image_backbone": str(cfg.model.backbone.image),
        "text_backbone": str(cfg.model.backbone.text),
        "bit_dim": int(cfg.model.bit_dim),
        "config_path": str(Path(args.config).resolve()),
        "repo_root": str(repo_root()),
        "resume_start_epoch": start_epoch,
    }
    trainer.train(
        checkpoint_dir=run_dir,
        save_every=int(cfg.training.save_every),
        run_meta=meta,
        start_epoch=start_epoch,
    )
    print(f"Done. Checkpoints and run_config under: {run_dir}")


if __name__ == "__main__":
    main()
