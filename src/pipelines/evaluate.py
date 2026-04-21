"""
Cross-modal retrieval evaluation using the DCMH paper's protocol:
  - 3-way split: query / train / database (Section 4.1)
  - MAP with label-based relevance: relevant(i,j) = (L_i . L_j > 0)

Usage::

    python -m src.pipelines.evaluate --config configs/experiments/exp_dcmh_mirflickr25k.yaml --latest
    python -m src.pipelines.evaluate --config ... --checkpoint experiments/checkpoints/.../epoch_0050.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from src.core.metrics import mean_average_precision_hamming
from src.core.retrieval import encode_paired_dataset, hamming_distance_matrix
from src.data.splits import SplitSubset, make_mirflickr_split
from src.pipelines.inference_utils import (
    load_model_and_dataset_for_eval,
    resolve_checkpoint_path,
)
from src.utils.config import load_experiment


def parse_args():
    p = argparse.ArgumentParser(description="DCMH cross-modal retrieval evaluation (MAP)")
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
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_experiment(args.config)
    ckpt_path = resolve_checkpoint_path(cfg, args.checkpoint, args.latest)

    cfg, model, collate, full_ds, device, ckpt_epoch, meta = load_model_and_dataset_for_eval(
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

    bs = args.batch_size if args.batch_size is not None else int(cfg.training.batch_size)
    loader_kwargs = dict(
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        num_workers=int(cfg.training.num_workers),
        collate_fn=collate,
        pin_memory=str(device).startswith("cuda"),
    )
    query_loader = DataLoader(query_ds, **loader_kwargs)
    db_loader = DataLoader(db_ds, **loader_kwargs)

    print(
        f"Split: {len(full_ds)} total -> "
        f"{len(query_ds)} query, {len(db_ds)} database",
        flush=True,
    )

    print("Encoding query set...", flush=True)
    h_img_q, h_txt_q, L_q, _ = encode_paired_dataset(model, query_loader, device)
    print(f"  {h_img_q.size(0)} queries ({h_img_q.size(1)}-bit hashes)", flush=True)

    print("Encoding database...", flush=True)
    h_img_db, h_txt_db, L_db, _ = encode_paired_dataset(model, db_loader, device)
    print(f"  {h_img_db.size(0)} database items", flush=True)

    print("Computing Hamming distances (I->T)...", flush=True)
    dist_i2t = hamming_distance_matrix(h_img_q, h_txt_db)
    print("Computing Hamming distances (T->I)...", flush=True)
    dist_t2i = hamming_distance_matrix(h_txt_q, h_img_db)

    print("Computing MAP (I->T)...", flush=True)
    map_i2t = mean_average_precision_hamming(dist_i2t, L_q, L_db)
    print(f"  MAP I->T = {map_i2t:.4f}", flush=True)

    print("Computing MAP (T->I)...", flush=True)
    map_t2i = mean_average_precision_hamming(dist_t2i, L_q, L_db)
    print(f"  MAP T->I = {map_t2i:.4f}", flush=True)

    report = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(ckpt_path.resolve()),
        "checkpoint_epoch": ckpt_epoch,
        "split": {
            "total": len(full_ds),
            "query": len(query_ds),
            "database": len(db_ds),
            "train": int(split_cfg.train_size),
        },
        "bit_dim": int(h_img_q.size(1)),
        "map": {
            "image_to_text": map_i2t,
            "text_to_image": map_t2i,
        },
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
