"""
Cross-modal retrieval metrics on the configured dataset (paired rows: image i matches text i).

Uses Hamming distance on sign(hash) embeddings.
layouts; for official splits, point ``dataset.root`` / future split configs to a test list.

Usage::

    python -m src.pipelines.evaluate --config configs/experiments/exp_dcmh_flickr8k.yaml --latest
    python -m src.pipelines.evaluate --config ... --checkpoint experiments/checkpoints/.../epoch_0120.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from src.core.metrics import mean_reciprocal_rank_hamming, recall_at_k_hamming
from src.core.retrieval import encode_paired_dataset, hamming_distance_matrix
from src.pipelines.inference_utils import (
    load_model_and_dataset_for_eval,
    resolve_checkpoint_path,
)
from src.utils.config import load_experiment


def parse_args():
    p = argparse.ArgumentParser(description="DCMH cross-modal retrieval evaluation")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None, help="Path to epoch_*.pt")
    p.add_argument(
        "--latest",
        action="store_true",
        help="Use latest epoch_*.pt under the experiment run dir from this config",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size (default: training.batch_size from config)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write JSON metrics here (default: under experiments/results/)",
    )
    p.add_argument(
        "--ks",
        type=str,
        default="1,5,10",
        help="Comma-separated K values for Recall@K",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_experiment(args.config)
    ckpt_path = resolve_checkpoint_path(cfg, args.checkpoint, args.latest)

    cfg, model, collate, ds, device, ckpt_epoch, meta = load_model_and_dataset_for_eval(
        args.config,
        ckpt_path,
        device_override=args.device,
        cfg=cfg,
    )

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

    h_img, h_txt, _idx = encode_paired_dataset(model, loader, device)

    dist_i2t = hamming_distance_matrix(h_img, h_txt)
    dist_t2i = hamming_distance_matrix(h_txt, h_img)

    ks = tuple(int(x.strip()) for x in args.ks.split(",") if x.strip())
    r_i2t = recall_at_k_hamming(dist_i2t, ks=ks)
    r_t2i = recall_at_k_hamming(dist_t2i, ks=ks)
    mrr_i2t = mean_reciprocal_rank_hamming(dist_i2t)
    mrr_t2i = mean_reciprocal_rank_hamming(dist_t2i)

    report = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(ckpt_path.resolve()),
        "checkpoint_epoch": ckpt_epoch,
        "num_samples": int(h_img.size(0)),
        "bit_dim": int(h_img.size(1)),
        "recall_at_k": {"image_to_text": {str(k): v for k, v in r_i2t.items()}, "text_to_image": {str(k): v for k, v in r_t2i.items()}},
        "mrr": {"image_to_text": mrr_i2t, "text_to_image": mrr_t2i},
        "meta": meta,
    }

    print(json.dumps(report, indent=2))

    out_path = args.output
    if out_path is None:
        res_root = Path(cfg.output.root) / cfg.output.results
        res_root.mkdir(parents=True, exist_ok=True)
        safe = str(cfg.get("experiment_name", "eval")).replace(" ", "_")
        out_path = str(res_root / f"eval_{safe}_e{ckpt_epoch:04d}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
