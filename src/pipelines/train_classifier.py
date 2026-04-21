"""Train a multi-label classifier and dump per-sample predictions.

The classifier's per-class predictions feed into
``src.hashing.centers.build_classifier_similarity`` to construct the
SHC-style "visually confusable" similarity matrix ``S_clf``. This is a
short, one-shot pre-training step (~10 epochs on ResNet18) invoked before
CM-SHC training.

Usage (from repo root)::

    python -m src.pipelines.train_classifier \
        --config configs/experiments/exp_cmshc_mirflickr25k_128bit.yaml \
        --epochs 10 \
        --output experiments/centers/mirflickr25k_classifier_probs.pt
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from src.data.collators import make_dcmh_collate_fn
from src.data.loaders import get_dataset
from src.data.splits import SplitSubset, make_mirflickr_split
from src.data.transforms import imagenet_train_transform
from src.utils.config import load_experiment, repo_root
from src.utils.seed import set_seed


def build_resnet18(num_classes: int) -> nn.Module:
    try:
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    except AttributeError:
        net = models.resnet18(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def default_output_path(cfg) -> Path:
    res_root = Path(cfg.output.root) / "centers"
    return res_root / f"{cfg.dataset.name}_classifier_probs.pt"


def parse_args():
    p = argparse.ArgumentParser(description="Multi-label classifier for CM-SHC S_clf construction")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default=None, help="Override cfg.device")
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Where to save the probability tensor (default: experiments/centers/{dataset}_classifier_probs.pt)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_experiment(args.config)

    os.environ["TORCH_HOME"] = str(Path(cfg.paths.model_cache) / "torch")
    set_seed(int(cfg.seed))

    device = args.device or str(cfg.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    transform = imagenet_train_transform()
    full_ds = get_dataset(str(cfg.dataset.name), root_dir=cfg.dataset.root, transform=transform)

    split_cfg = getattr(cfg.dataset, "split", None)
    if split_cfg is None:
        raise ValueError("dataset.split is required; CM-SHC uses the DCMH 3-way split")
    q_size = int(split_cfg.query_size)
    t_size = int(split_cfg.train_size)
    _query_idx, train_idx, _db_idx = make_mirflickr_split(
        len(full_ds), query_size=q_size, train_size=t_size, seed=int(cfg.seed),
    )
    train_ds = SplitSubset(full_ds, train_idx)
    print(
        f"Dataset: {len(full_ds)} total -> train={len(train_ds)} (query={q_size} held out)",
        flush=True,
    )

    collate = make_dcmh_collate_fn(tokenizer=None)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=int(cfg.training.num_workers),
        collate_fn=collate,
        pin_memory=device.startswith("cuda"),
    )

    num_classes = int(cfg.dataset.num_classes)
    model = build_resnet18(num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    print(f"Training ResNet18 x {args.epochs} epochs on {device}", flush=True)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        n_batches = 0
        for batch in tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}", leave=False):
            imgs = batch["img"].to(device)
            labels = batch["label"].to(device).float()
            logits = model(imgs)
            loss = criterion(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            n_batches += 1
        print(f"  epoch {epoch + 1}: BCE={total / max(n_batches, 1):.4f}", flush=True)

    # Dump predictions over the training subset (for building S_clf).
    infer_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=int(cfg.training.num_workers),
        collate_fn=collate,
        pin_memory=device.startswith("cuda"),
    )
    model.eval()
    probs_chunks: list[torch.Tensor] = []
    labels_chunks: list[torch.Tensor] = []
    idx_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in tqdm(infer_loader, desc="infer", leave=False):
            imgs = batch["img"].to(device)
            logits = model(imgs)
            probs_chunks.append(torch.sigmoid(logits).cpu())
            labels_chunks.append(batch["label"].cpu().float())
            idx = batch["index"]
            idx_chunks.append(idx.cpu() if torch.is_tensor(idx) else torch.tensor(idx))

    probs = torch.cat(probs_chunks, dim=0)
    labels = torch.cat(labels_chunks, dim=0)
    indices = torch.cat(idx_chunks, dim=0)
    # Sort by local index so rows align with the training split ordering.
    order = indices.argsort()
    probs = probs[order]
    labels = labels[order]
    indices = indices[order]

    out_path = Path(args.output) if args.output else default_output_path(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "probs": probs,
        "labels": labels,
        "indices": indices,
        "dataset": str(cfg.dataset.name),
        "num_classes": num_classes,
        "num_epochs": int(args.epochs),
        "seed": int(cfg.seed),
        "split": {"query_size": q_size, "train_size": t_size},
        "backbone": "resnet18",
    }
    torch.save(payload, out_path)
    print(f"Wrote classifier probs: {out_path}  (N={probs.size(0)}, C={probs.size(1)})", flush=True)

    # Log a tiny summary so we can sanity-check the run.
    report = {
        "output": str(out_path),
        "config": str(Path(args.config).resolve()),
        "num_samples": int(probs.size(0)),
        "num_classes": num_classes,
        "mean_prob_per_class": probs.mean(dim=0).tolist(),
        "positive_rate_per_class": labels.mean(dim=0).tolist(),
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote summary: {meta_path}", flush=True)


if __name__ == "__main__":
    main()
