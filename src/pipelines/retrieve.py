"""
Print top-K retrieval results for one query (image-to-text or text-to-image).

Uses the paper's 3-way split: queries come from the held-out query set,
ranked against the database set.  Results are marked with ``+`` when the
retrieved item shares at least one semantic label with the query.

Usage::

    python -m src.pipelines.retrieve --config configs/experiments/exp_dcmh_mirflickr25k.yaml --latest --query-index 0 --top-k 10
    python -m src.pipelines.retrieve --config ... --checkpoint path/to/epoch_0050.pt --mode t2i --query-index 3
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from src.core.retrieval import encode_paired_dataset, hamming_distance_matrix
from src.data.splits import SplitSubset, make_mirflickr_split
from src.pipelines.inference_utils import (
    load_model_and_dataset_for_eval,
    resolve_checkpoint_path,
)
from src.utils.config import load_experiment


def parse_args():
    p = argparse.ArgumentParser(description="Top-K cross-modal retrieval demo")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--latest", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument(
        "--query-index", type=int, required=True,
        help="Index within the query set (0..query_size-1)",
    )
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument(
        "--mode",
        type=str,
        choices=("i2t", "t2i"),
        default="i2t",
        help="i2t: image query ranks texts; t2i: text query ranks images",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_experiment(args.config)
    ckpt_path = resolve_checkpoint_path(cfg, args.checkpoint, args.latest)

    cfg, model, collate, full_ds, device, ckpt_epoch, _meta = load_model_and_dataset_for_eval(
        args.config,
        ckpt_path,
        device_override=args.device,
        cfg=cfg,
    )

    split_cfg = cfg.dataset.split
    query_idx, _train_idx, db_idx = make_mirflickr_split(
        len(full_ds),
        query_size=int(split_cfg.query_size),
        train_size=int(split_cfg.train_size),
        seed=int(cfg.seed),
    )
    query_ds = SplitSubset(full_ds, query_idx)
    db_ds = SplitSubset(full_ds, db_idx)

    n_query = len(query_ds)
    if args.query_index < 0 or args.query_index >= n_query:
        raise SystemExit(f"query-index must be in [0, {n_query - 1}]")

    bs = args.batch_size if args.batch_size is not None else int(cfg.training.batch_size)
    loader_kwargs = dict(
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        num_workers=int(cfg.training.num_workers),
        collate_fn=collate,
        pin_memory=str(device).startswith("cuda"),
    )

    print("Encoding query set...", flush=True)
    h_img_q, h_txt_q, L_q, _ = encode_paired_dataset(model, DataLoader(query_ds, **loader_kwargs), device)
    print("Encoding database...", flush=True)
    h_img_db, h_txt_db, L_db, _ = encode_paired_dataset(model, DataLoader(db_ds, **loader_kwargs), device)

    qi = args.query_index
    k = min(args.top_k, len(db_ds))
    q_label = L_q[qi]

    if args.mode == "i2t":
        dist = hamming_distance_matrix(h_img_q[qi : qi + 1], h_txt_db)[0]
        ranked = dist.argsort()[:k]
        print(f"\nQuery image (query set index {qi}, epoch {ckpt_epoch})")
        print(f"Query labels: {_label_names(q_label, full_ds)}\n")
        print("Rank  Hamming  Rel  DB labels")
        for r, j in enumerate(ranked.tolist()):
            rel = "+" if (q_label @ L_db[j] > 0).item() else " "
            print(f"{r + 1:4d}   {dist[j].item():5d}   {rel}   {_label_names(L_db[j], full_ds)}")
    else:
        dist = hamming_distance_matrix(h_txt_q[qi : qi + 1], h_img_db)[0]
        ranked = dist.argsort()[:k]
        print(f"\nQuery text (query set index {qi}, epoch {ckpt_epoch})")
        print(f"Query labels: {_label_names(q_label, full_ds)}\n")
        print("Rank  Hamming  Rel  DB labels")
        for r, j in enumerate(ranked.tolist()):
            rel = "+" if (q_label @ L_db[j] > 0).item() else " "
            print(f"{r + 1:4d}   {dist[j].item():5d}   {rel}   {_label_names(L_db[j], full_ds)}")


def _label_names(label_vec: torch.Tensor, ds) -> str:
    """Human-readable label names from a multi-hot vector."""
    classes = getattr(ds, "ANNOTATION_CLASSES", None)
    if classes is None:
        return str(label_vec.nonzero(as_tuple=True)[0].tolist())
    indices = label_vec.nonzero(as_tuple=True)[0].tolist()
    return ", ".join(classes[i] for i in indices)


if __name__ == "__main__":
    main()
