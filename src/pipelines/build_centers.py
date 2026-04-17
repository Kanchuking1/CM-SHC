"""Build and save CM-SHC hash centers.

Run this once before a CM-SHC training run; the saved file is loaded by
``CMSHCTrainer`` to get the (q, C) center matrix ``H`` plus the
per-sample target codes.

Three ``--method`` options cover the ablations in the plan:

* ``cooccurrence`` -- S from label cosine similarity; centers from SHC solver.
* ``classifier``  -- S from a pre-trained multi-label classifier's predictions
                     (see ``src.pipelines.train_classifier``); centers from SHC solver.
* ``csq``          -- Hadamard/Bernoulli centers (Yuan et al., CVPR 2020). No S.

Usage (from repo root)::

    python -m src.pipelines.build_centers \
        --config configs/experiments/exp_cmshc_mirflickr25k_128bit.yaml \
        --method classifier \
        --classifier-probs experiments/centers/mirflickr25k_classifier_probs.pt

    python -m src.pipelines.build_centers --config ... --method cooccurrence
    python -m src.pipelines.build_centers --config ... --method csq
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from src.data.collators import build_train_labels_tensor
from src.data.loaders import get_dataset
from src.data.splits import SplitSubset, make_mirflickr_split
from src.hashing.centers import (
    build_class_similarity,
    build_classifier_similarity,
    csq_hash_centers,
    multi_label_target,
    optimize_semantic_centers,
)
from src.hashing.gv_bound import gilbert_varshamov_distance
from src.utils.config import load_experiment
from src.utils.seed import set_seed


def default_output_path(cfg, method: str) -> Path:
    centers_dir = Path(cfg.output.root) / "centers"
    q = int(cfg.model.bit_dim)
    return centers_dir / f"{cfg.dataset.name}_{method}_q{q}.pt"


def parse_args():
    p = argparse.ArgumentParser(description="Build CM-SHC hash centers.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument(
        "--method",
        choices=("cooccurrence", "classifier", "csq"),
        default=None,
        help="Similarity source (default: cfg.model.similarity_method)",
    )
    p.add_argument(
        "--classifier-probs",
        type=str,
        default=None,
        help="Path to the .pt saved by train_classifier.py (required when --method=classifier)",
    )
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--mu", type=float, default=None, help="Override cfg.model.mu")
    p.add_argument("--max-iters", type=int, default=2000)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _get_train_labels(cfg) -> torch.Tensor:
    """(N_train, C) multi-hot label matrix for the training subset."""
    full_ds = get_dataset(str(cfg.dataset.name), root_dir=cfg.dataset.root)
    split_cfg = cfg.dataset.split
    _q, train_idx, _db = make_mirflickr_split(
        len(full_ds),
        query_size=int(split_cfg.query_size),
        train_size=int(split_cfg.train_size),
        seed=int(cfg.seed),
    )
    train_ds = SplitSubset(full_ds, train_idx)
    return build_train_labels_tensor(train_ds, int(cfg.dataset.num_classes))


def main():
    args = parse_args()
    cfg = load_experiment(args.config)
    os.environ["TORCH_HOME"] = str(Path(cfg.paths.model_cache) / "torch")
    set_seed(int(cfg.seed))

    method = args.method or str(cfg.model.get("similarity_method", "cooccurrence"))
    q = int(cfg.model.bit_dim)
    C = int(cfg.dataset.num_classes)
    d_min = gilbert_varshamov_distance(q, C)
    print(f"Bit budget q={q}, classes C={C}, GV min distance d_min={d_min}", flush=True)

    Y_train = _get_train_labels(cfg)
    print(f"Loaded training labels: shape {tuple(Y_train.shape)}", flush=True)

    S: torch.Tensor | None = None
    if method == "cooccurrence":
        S = build_class_similarity(Y_train, method="cooccurrence")
        H = optimize_semantic_centers(
            S, q=q, d_min=d_min,
            mu=float(args.mu if args.mu is not None else cfg.model.get("mu", 1.0)),
            lr=args.lr, max_iters=args.max_iters, seed=int(cfg.seed),
            verbose=args.verbose,
        )
    elif method == "classifier":
        if args.classifier_probs is None:
            raise ValueError(
                "--classifier-probs is required when --method=classifier. "
                "Run src.pipelines.train_classifier first."
            )
        payload = torch.load(args.classifier_probs, map_location="cpu", weights_only=False)
        probs = payload["probs"]
        clf_labels = payload["labels"]
        if probs.size(0) != Y_train.size(0):
            raise ValueError(
                f"Classifier probs has {probs.size(0)} rows but training split has "
                f"{Y_train.size(0)} - did you change dataset.split between runs?"
            )
        S = build_classifier_similarity(probs, clf_labels)
        H = optimize_semantic_centers(
            S, q=q, d_min=d_min,
            mu=float(args.mu if args.mu is not None else cfg.model.get("mu", 1.0)),
            lr=args.lr, max_iters=args.max_iters, seed=int(cfg.seed),
            verbose=args.verbose,
        )
    elif method == "csq":
        H = csq_hash_centers(q=q, C=C, seed=int(cfg.seed))
    else:
        raise ValueError(f"Unknown method {method!r}")

    # Pairwise stats for the report.
    inner = H.t() @ H
    dist = (q - inner) / 2
    offdiag = dist + (q + 1) * torch.eye(C)
    min_d = int(offdiag.min().item())
    max_inner = float((inner - q * torch.eye(C)).abs().max().item())
    print(
        f"Centers ready. shape={tuple(H.shape)}  min Hamming={min_d}  "
        f"(target>={d_min})  max |<h_i,h_j>|={max_inner:.1f}",
        flush=True,
    )

    # Per-sample targets for the training subset, cached for the trainer.
    T_train = multi_label_target(Y_train, H, seed=int(cfg.seed))
    print(f"Per-sample targets: {tuple(T_train.shape)}", flush=True)

    out_path = Path(args.output) if args.output else default_output_path(cfg, method)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "H": H,                        # (q, C) ±1 centers
        "T_train": T_train,            # (N_train, q) ±1 per-sample target
        "S": S,                         # (C, C) similarity matrix, or None for csq
        "method": method,
        "q": q,
        "C": C,
        "d_min": d_min,
        "min_pairwise_hamming": min_d,
        "dataset": str(cfg.dataset.name),
        "seed": int(cfg.seed),
        "config_path": str(Path(args.config).resolve()),
    }
    torch.save(payload, out_path)
    print(f"Wrote centers: {out_path}", flush=True)

    meta = {
        "output": str(out_path),
        "method": method,
        "q": q,
        "C": C,
        "gv_d_min": d_min,
        "achieved_min_pairwise_hamming": min_d,
        "dataset": str(cfg.dataset.name),
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
