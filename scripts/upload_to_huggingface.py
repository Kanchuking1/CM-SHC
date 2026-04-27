"""Upload 7 SSHC model checkpoints and supporting artifacts to HuggingFace Hub.

Lays out a single repo on HF with 7 sub-folders (one per trained model),
the eval JSONs needed to regenerate the symmetry scatter plot, a small
exemplar database for retrieval-based classification in the demo
notebook, and a top-level model card.

Workflow
--------
1.  ``huggingface-cli login``  (one-time)
2.  ``python scripts/upload_to_huggingface.py --repo-id YOUR_USER/sshc-mirflickr25k-128bit``
    Without ``--push`` this stages files into ``.hf_staging`` for inspection.
3.  Add ``--push`` once the staging directory looks right.

Repo layout produced
--------------------
    sshc-mirflickr25k-128bit/
    |-- README.md                          (top-level model card)
    |-- shared/
    |   |-- label_names.json               (24 MIR-Flickr label names)
    |   |-- exemplars/
    |   |   |-- images/                    (200 sample images)
    |   |   |-- labels.npy                 (200, 24) uint8
    |   |   `-- index.json
    |   `-- centers/
    |       |-- csq_128.npy                (24, 128)
    |       |-- clf_128.npy                (24, 128)
    |       `-- cooc_128.npy               (24, 128)
    |
    `-- <model_key>/                       x 7
        |-- model.pt                       state_dict only
        |-- config.yaml                    training config
        |-- eval.json                      I->T / T->I MAP
        `-- card.json                      backbone / loss / bits

Each ``model.pt`` is a dict ``{model_state_dict, epoch, meta}`` that
the demo notebook can load directly with ``torch.load(...)``.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.checkpoint import find_latest_training_checkpoint  # noqa: E402
from src.utils.config import experiment_run_dir, load_experiment   # noqa: E402


# ---------------------------------------------------------------------------
# 7 models to ship
# ---------------------------------------------------------------------------
#
# ``run_dir`` is the on-disk subfolder under ``experiments/checkpoints/`` that
# contains ``epoch_XXXX.pt``. We pin this explicitly per-spec because the
# computed path from ``experiment_run_dir(cfg)`` depends on whatever
# ``bit_dim`` is resolved from base+model+dataset+experiment merging, and the
# AlexNet DCMH run was trained at 128 bits while the model default is 64.
MODELS = [
    {
        "key": "alexnet_dcmh",
        "config": "configs/experiments/exp_dcmh_mirflickr25k.yaml",
        "run_dir": "alexnet_mlp_dcmh_mirflickr25k_128bit",
        "eval_json": "eval_alexnet_mlp_e0500.json",
        "epoch": 500,
        "title": "DCMH on AlexNet+BoW",
        "loss": "DCMH (pairwise)",
        "backbone": "AlexNet + BoW-MLP",
        "bit_dim": 128,
    },
    {
        "key": "alexnet_cmshc",
        "config": "configs/experiments/exp_cmshc_mirflickr25k_128bit.yaml",
        "run_dir": "cmshc_mirflickr25k_128bit_clf_cm_shc_mirflickr25k_128bit",
        "eval_json": "eval_cmshc_mirflickr25k_128bit_clf_e0200.json",
        "epoch": 200,
        "title": "CM-SHC on AlexNet+BoW (S_clf)",
        "loss": "CM-SHC (centers)",
        "backbone": "AlexNet + BoW-MLP",
        "bit_dim": 128,
    },
    {
        "key": "clip_frozen_dcmh",
        "config": "configs/experiments/exp_dcmh_clip_frozen_mirflickr25k_128bit.yaml",
        "run_dir": "dcmh_clip_frozen_mirflickr25k_128bit_dcmh_mirflickr25k_128bit",
        "eval_json": "eval_dcmh_clip_frozen_mirflickr25k_128bit_e0200.json",
        "epoch": 200,
        "title": "DCMH on CLIP frozen",
        "loss": "DCMH (pairwise)",
        "backbone": "CLIP ViT-B/32 frozen",
        "bit_dim": 128,
    },
    {
        "key": "clip_frozen_cmshc",
        "config": "configs/experiments/exp_cmshc_clip_frozen_mirflickr25k_128bit.yaml",
        "run_dir": "cmshc_clip_frozen_mirflickr25k_128bit_cm_shc_mirflickr25k_128bit",
        "eval_json": "eval_cmshc_clip_frozen_mirflickr25k_128bit_e0200.json",
        "epoch": 200,
        "title": "CM-SHC on CLIP frozen",
        "loss": "CM-SHC (centers)",
        "backbone": "CLIP ViT-B/32 frozen",
        "bit_dim": 128,
    },
    {
        "key": "clip_lora_dcmh",
        "config": "configs/experiments/exp_dcmh_clip_lora_mirflickr25k_128bit.yaml",
        "run_dir": "dcmh_clip_lora_mirflickr25k_128bit_dcmh_mirflickr25k_128bit",
        "eval_json": "eval_dcmh_clip_lora_mirflickr25k_128bit_e0200.json",
        "epoch": 200,
        "title": "DCMH on CLIP+LoRA",
        "loss": "DCMH (pairwise)",
        "backbone": "CLIP ViT-B/32 + LoRA r=8",
        "bit_dim": 128,
    },
    {
        "key": "clip_lora_cmshc",
        "config": "configs/experiments/exp_cmshc_clip_lora_mirflickr25k_128bit.yaml",
        "run_dir": "cmshc_clip_lora_mirflickr25k_128bit_cm_shc_mirflickr25k_128bit",
        "eval_json": "eval_cmshc_clip_lora_mirflickr25k_128bit_e0200.json",
        "epoch": 200,
        "title": "CM-SHC on CLIP+LoRA",
        "loss": "CM-SHC (centers)",
        "backbone": "CLIP ViT-B/32 + LoRA r=8",
        "bit_dim": 128,
    },
    {
        "key": "clip_lora_anchored_lc30",
        "config": "configs/experiments/exp_dcmh_anchored_clip_lora_mirflickr25k_128bit_lc30.yaml",
        "run_dir": "dcmh_anchored_clip_lora_mirflickr25k_128bit_lc30_dcmh_anchored_mirflickr25k_128bit",
        "eval_json": "eval_dcmh_anchored_clip_lora_mirflickr25k_128bit_lc30_e0200.json",
        "epoch": 200,
        "title": "Anchored-DCMH (lambda_c = 3.0) on CLIP+LoRA",
        "loss": "Anchored-DCMH (DCMH pair + lambda_c * center BCE)",
        "backbone": "CLIP ViT-B/32 + LoRA r=8",
        "bit_dim": 128,
    },
]


# MIR-Flickr-25k 24-class label set (kept here so the HF repo is self-contained).
MIRFLICKR_LABELS = [
    "animals", "baby", "bird", "car", "clouds", "dog", "female", "flower",
    "food", "indoor", "lake", "male", "night", "people", "plant_life",
    "portrait", "river", "sea", "sky", "structures", "sunset", "transport",
    "tree", "water",
]


# ---------------------------------------------------------------------------
# Per-model staging
# ---------------------------------------------------------------------------
def stage_model(spec: dict, staging_root: Path, results_root: Path) -> dict:
    target = staging_root / spec["key"]
    target.mkdir(parents=True, exist_ok=True)

    # Resolve checkpoint location. Prefer explicit ``run_dir`` from the spec
    # (avoids depending on cfg.bit_dim resolution); fall back to computing
    # from the cfg if the spec doesn't pin it.
    if "run_dir" in spec and spec["run_dir"]:
        run_dir = REPO_ROOT / "experiments" / "checkpoints" / spec["run_dir"]
    else:
        cfg_for_path = load_experiment(REPO_ROOT / spec["config"])
        run_dir = experiment_run_dir(cfg_for_path)

    if spec.get("epoch") is not None:
        ckpt_path = run_dir / f"epoch_{int(spec['epoch']):04d}.pt"
    else:
        ckpt_path = find_latest_training_checkpoint(run_dir)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(
            f"[{spec['key']}] checkpoint not found at {ckpt_path}. "
            f"Run dir: {run_dir}"
        )

    # Save weights only (drop optimizer state, RNG, etc.)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    slim = {
        "model_state_dict": ckpt["model_state_dict"],
        "epoch": int(ckpt.get("epoch", spec.get("epoch", 0))),
        "meta": ckpt.get("meta", {}),
    }
    torch.save(slim, target / "model.pt")

    shutil.copy(REPO_ROOT / spec["config"], target / "config.yaml")

    eval_src = results_root / spec["eval_json"]
    if eval_src.exists():
        shutil.copy(eval_src, target / "eval.json")
    else:
        print(f"  WARNING: eval JSON missing for {spec['key']}: {eval_src}")

    # bit_dim: prefer explicit spec value, fall back to cfg
    if "bit_dim" in spec and spec["bit_dim"] is not None:
        bit_dim = int(spec["bit_dim"])
    else:
        cfg_for_bits = load_experiment(REPO_ROOT / spec["config"])
        bit_dim = int(cfg_for_bits.model.bit_dim)

    card = {
        "key": spec["key"],
        "title": spec["title"],
        "loss": spec["loss"],
        "backbone": spec["backbone"],
        "bit_dim": bit_dim,
        "config_path": spec["config"],
        "run_dir": spec.get("run_dir"),
        "epoch": spec.get("epoch"),
    }
    (target / "card.json").write_text(json.dumps(card, indent=2))

    size_mb = (target / "model.pt").stat().st_size / 1024 / 1024
    print(f"  staged {spec['key']:30s}  ({size_mb:6.1f} MB)")
    return card


# ---------------------------------------------------------------------------
# Shared artefacts (centers + exemplars + label names)
# ---------------------------------------------------------------------------
def stage_shared(staging_root: Path, num_exemplars: int = 200, seed: int = 42) -> None:
    shared = staging_root / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    (shared / "label_names.json").write_text(json.dumps(MIRFLICKR_LABELS, indent=2))

    # Centers (csq / clf / cooc) -- look under experiments/centers/.
    centers_src = REPO_ROOT / "experiments" / "centers"
    if centers_src.exists():
        c_target = shared / "centers"
        c_target.mkdir(parents=True, exist_ok=True)
        for tag in ("csq", "clf", "cooc"):
            for stem in (f"{tag}_128", f"centers_{tag}_128", f"H_{tag}_128"):
                f = centers_src / f"{stem}.npy"
                if f.exists():
                    shutil.copy(f, c_target / f"{tag}_128.npy")
                    break
            else:
                # fall back to any .npy with `tag` in the name
                matches = list(centers_src.glob(f"*{tag}*128*.npy"))
                if matches:
                    shutil.copy(matches[0], c_target / f"{tag}_128.npy")

    # Exemplar set: a random 200-sample subset of the database split.
    # We rely on the dataset loader to find image paths + labels.
    try:
        from src.data.loaders import get_dataset
        from src.data.transforms import imagenet_train_transform
    except Exception as e:
        print(f"  could not import dataset loader: {e}")
        print("  skipping exemplar staging; demo notebook will fall back to single-image mode")
        return

    cfg = load_experiment(REPO_ROOT / MODELS[0]["config"])
    try:
        ds = get_dataset(
            str(cfg.dataset.name),
            root_dir=str(cfg.dataset.root),
            transform=imagenet_train_transform(),
            text_mode="bow",
            split="database",
        )
    except TypeError:
        # older signature
        ds = get_dataset(
            str(cfg.dataset.name),
            root_dir=str(cfg.dataset.root),
            transform=imagenet_train_transform(),
        )

    if not hasattr(ds, "__len__") or len(ds) == 0:
        print("  dataset is empty -- skipping exemplars")
        return

    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(ds)), k=min(num_exemplars, len(ds))))

    exemplars = shared / "exemplars"
    img_dir = exemplars / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    labels = []
    index = []
    for new_id, ds_idx in enumerate(indices):
        sample = ds[ds_idx]
        # samples typically yield {"image_path": ..., "labels": ...} or
        # an object with those attributes; defensive lookup.
        if isinstance(sample, dict):
            img_path = sample.get("image_path") or sample.get("path")
            label_vec = sample.get("labels") or sample.get("label")
        else:
            img_path = getattr(sample, "image_path", None)
            label_vec = getattr(sample, "labels", None)
        if img_path is None or label_vec is None:
            continue
        src = Path(img_path)
        if not src.is_absolute():
            src = REPO_ROOT / src
        if not src.exists():
            continue
        dst = img_dir / f"{new_id:04d}{src.suffix.lower()}"
        shutil.copy(src, dst)
        labels.append(np.asarray(label_vec, dtype=np.uint8))
        index.append({"id": new_id, "filename": dst.name, "source_index": int(ds_idx)})

    if labels:
        np.save(exemplars / "labels.npy", np.stack(labels, axis=0))
        (exemplars / "index.json").write_text(json.dumps(index, indent=2))
        print(f"  staged {len(labels)} exemplars in shared/exemplars/")
    else:
        print("  no exemplars could be staged (dataset format mismatch)")


# ---------------------------------------------------------------------------
# Top-level README (HF model card)
# ---------------------------------------------------------------------------
def write_readme(staging_root: Path, cards: list[dict]) -> None:
    rows = []
    for c in cards:
        rows.append(
            f"| `{c['key']}` | {c['backbone']} | {c['loss']} | {c['bit_dim']} | "
            f"{c['epoch']} |"
        )
    table = "\n".join(rows)

    text = f"""---
license: mit
language: en
tags:
- cross-modal-hashing
- mir-flickr-25k
- clip
- lora
- dcmh
- cm-shc
datasets:
- mir-flickr-25k
---

# SSHC: Cross-Modal Semantic Hashing on MIR-Flickr-25k (128 bits)

Seven trained 128-bit cross-modal hashing checkpoints from a course project on
the interaction between hashing objectives (DCMH pairwise vs. CM-SHC center-based)
and pretrained backbones (AlexNet+BoW vs. CLIP frozen vs. CLIP+LoRA). All models
are trained on MIR-Flickr-25k with the canonical 2,000 / 10,000 / 12,581
(query / train / database) split at seed 42.

## Models

| Key | Backbone | Loss | Bits | Epoch |
| --- | -------- | ---- | ---: | ----: |
{table}

## Demo notebook

A Colab notebook that (a) regenerates the I->T / T->I symmetry scatter plot
across all 7 models from their `eval.json` files, and (b) runs single-image
inference (hash code + retrieval-based multi-label classification) is in the
project's GitHub repo: `notebooks/sshc_demo.ipynb`.

## Loading in Python

```python
from huggingface_hub import snapshot_download
local = snapshot_download(repo_id="YOUR_USER/sshc-mirflickr25k-128bit")

import torch
ckpt = torch.load(f"{{local}}/clip_lora_anchored_lc30/model.pt", map_location="cpu")
state = ckpt["model_state_dict"]
```

The model classes themselves live in the `cm-shc` package on GitHub; install
with `pip install git+https://github.com/Kanchuking/SSHC.git` and follow the
demo notebook for end-to-end loading.

## Citation

This release accompanies the project report *Cross-Modal Semantic Hash Centers
with Strong Contrastive Backbones*, CSCE 689 Algorithmic Foundations of Big
Data, Spring 2026.
"""
    (staging_root / "README.md").write_text(text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="e.g. wahibkapdi/sshc-mirflickr25k-128bit")
    ap.add_argument("--staging-dir", default=".hf_staging")
    ap.add_argument("--results-root", default="experiments/results")
    ap.add_argument("--no-exemplars", action="store_true", help="Skip exemplar staging.")
    ap.add_argument(
        "--push",
        action="store_true",
        help="Actually upload to HF (omit for dry-run / inspection).",
    )
    args = ap.parse_args()

    staging_root = (REPO_ROOT / args.staging_dir).resolve()
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)

    results_root = (REPO_ROOT / args.results_root).resolve()

    print(f"Staging into {staging_root}")
    cards = []
    for spec in MODELS:
        cards.append(stage_model(spec, staging_root, results_root))

    if not args.no_exemplars:
        stage_shared(staging_root)
    else:
        # still drop the label names so the demo can resolve label indices
        (staging_root / "shared").mkdir(exist_ok=True)
        (staging_root / "shared" / "label_names.json").write_text(
            json.dumps(MIRFLICKR_LABELS, indent=2)
        )

    write_readme(staging_root, cards)

    if not args.push:
        print(
            f"\nDry run complete. Inspect {staging_root}; pass --push to upload to "
            f"https://huggingface.co/{args.repo_id}"
        )
        return

    from huggingface_hub import HfApi, create_repo, upload_folder

    create_repo(args.repo_id, repo_type="model", exist_ok=True)
    upload_folder(
        folder_path=str(staging_root),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Upload 7 SSHC checkpoints + symmetry scatter eval JSONs",
    )
    print(f"\nPushed to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
