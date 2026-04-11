"""
Print top-K retrieval results for one query index (image-to-text or text-to-image).

Assumes the dataset is paired: sample ``i`` pairs image i with caption i (same as evaluate).

Usage::

    python -m src.pipelines.retrieve --config configs/experiments/exp_dcmh_flickr8k.yaml --latest --query-index 0 --top-k 5
    python -m src.pipelines.retrieve --config ... --checkpoint path/to/epoch_0120.pt --mode t2i --query-index 3
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from src.core.retrieval import encode_paired_dataset, hamming_distance_matrix
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
    p.add_argument("--query-index", type=int, required=True, help="Dataset index of the query")
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

    cfg, model, collate, ds, device, ckpt_epoch, _meta = load_model_and_dataset_for_eval(
        args.config,
        ckpt_path,
        device_override=args.device,
        cfg=cfg,
    )

    n = len(ds)
    if args.query_index < 0 or args.query_index >= n:
        raise SystemExit(f"query-index must be in [0, {n - 1}]")

    bs = args.batch_size if args.batch_size is not None else int(cfg.training.batch_size)
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        num_workers=int(cfg.training.num_workers),
        collate_fn=collate,
        pin_memory=str(device).startswith("cuda"),
    )

    h_img, h_txt, sorted_idx = encode_paired_dataset(model, loader, device)
    assert torch.equal(sorted_idx, torch.arange(n)), "Dataset indices must be 0..N-1 for retrieve"

    k = min(args.top_k, n)
    if args.mode == "i2t":
        dist = hamming_distance_matrix(h_img[args.query_index : args.query_index + 1], h_txt)[0]
        ranked = dist.argsort()[:k]
        qcap = ds[args.query_index]["text"]
        print(f"Query image index={args.query_index} (checkpoint epoch {ckpt_epoch})")
        print(f"True caption: {qcap[:200]}{'...' if len(str(qcap)) > 200 else ''}\n")
        print("Rank  Hamming  Caption (truncated)")
        for r, j in enumerate(ranked.tolist()):
            cap = ds[j]["text"]
            cap_s = str(cap).replace("\n", " ")[:120]
            mark = "*" if j == args.query_index else " "
            print(f"{r + 1:3d}   {dist[j].item():5d}  {mark} {cap_s}")
    else:
        dist = hamming_distance_matrix(h_txt[args.query_index : args.query_index + 1], h_img)[0]
        ranked = dist.argsort()[:k]
        qcap = ds[args.query_index]["text"]
        print(f"Query text index={args.query_index} (checkpoint epoch {ckpt_epoch})")
        print(f"Query: {str(qcap)[:300]}{'...' if len(str(qcap)) > 300 else ''}\n")
        print("Rank  Hamming  Image index")
        for r, j in enumerate(ranked.tolist()):
            mark = "*" if j == args.query_index else " "
            print(f"{r + 1:3d}   {dist[j].item():5d}  {mark} {j}")


if __name__ == "__main__":
    main()
