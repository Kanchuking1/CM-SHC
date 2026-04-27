"""Local hash + reverse-text demo for the seven trained SSHC models.

For a single query image, encodes it with each of the seven checkpoints
under ``experiments/checkpoints/`` and -- on the same model -- also
encodes a list of candidate text labels through the model's text tower,
then ranks the candidates by Hamming distance to the image hash. This
is the natural reverse of image-to-text retrieval: "which text would
this model hash to a code closest to the image?"

Candidates default to the 24 MIR-Flickr-25k label names. AlexNet+BoW
models use a one-hot BoW vector for each label; CLIP-based models
tokenize the raw label string.

Usage::

    python scripts/retrieval_demo.py --image path/to/query.jpg
    python scripts/retrieval_demo.py --image im123.jpg --output hashes.png \
        --top-k 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.collators import load_clip_tokenizer                       # noqa: E402
from src.data.transforms import imagenet_train_transform                 # noqa: E402
from src.pipelines.train import build_model, resolve_text_backbone_spec  # noqa: E402
from src.utils.config import load_experiment                             # noqa: E402


# ---------------------------------------------------------------------------
# 7 models -- same spec as scripts/upload_to_huggingface.py
# ---------------------------------------------------------------------------
MODELS = [
    {
        "key": "alexnet_dcmh",
        "config": "configs/experiments/exp_dcmh_mirflickr25k.yaml",
        "run_dir": "alexnet_mlp_dcmh_mirflickr25k_128bit",
        "epoch": 500,
        "bit_dim": 128,
        "label": "DCMH | AlexNet+BoW",
    },
    {
        "key": "alexnet_cmshc",
        "config": "configs/experiments/exp_cmshc_mirflickr25k_128bit.yaml",
        "run_dir": "cmshc_mirflickr25k_128bit_clf_cm_shc_mirflickr25k_128bit",
        "epoch": 200,
        "bit_dim": 128,
        "label": "CM-SHC | AlexNet+BoW (S_clf)",
    },
    {
        "key": "clip_frozen_dcmh",
        "config": "configs/experiments/exp_dcmh_clip_frozen_mirflickr25k_128bit.yaml",
        "run_dir": "dcmh_clip_frozen_mirflickr25k_128bit_dcmh_mirflickr25k_128bit",
        "epoch": 200,
        "bit_dim": 128,
        "label": "DCMH | CLIP frozen",
    },
    {
        "key": "clip_frozen_cmshc",
        "config": "configs/experiments/exp_cmshc_clip_frozen_mirflickr25k_128bit.yaml",
        "run_dir": "cmshc_clip_frozen_mirflickr25k_128bit_cm_shc_mirflickr25k_128bit",
        "epoch": 200,
        "bit_dim": 128,
        "label": "CM-SHC | CLIP frozen",
    },
    {
        "key": "clip_lora_dcmh",
        "config": "configs/experiments/exp_dcmh_clip_lora_mirflickr25k_128bit.yaml",
        "run_dir": "dcmh_clip_lora_mirflickr25k_128bit_dcmh_mirflickr25k_128bit",
        "epoch": 200,
        "bit_dim": 128,
        "label": "DCMH | CLIP+LoRA",
    },
    {
        "key": "clip_lora_cmshc",
        "config": "configs/experiments/exp_cmshc_clip_lora_mirflickr25k_128bit.yaml",
        "run_dir": "cmshc_clip_lora_mirflickr25k_128bit_cm_shc_mirflickr25k_128bit",
        "epoch": 200,
        "bit_dim": 128,
        "label": "CM-SHC | CLIP+LoRA",
    },
    {
        "key": "clip_lora_anchored_lc30",
        "config": "configs/experiments/exp_dcmh_anchored_clip_lora_mirflickr25k_128bit_lc30.yaml",
        "run_dir": "dcmh_anchored_clip_lora_mirflickr25k_128bit_lc30_dcmh_anchored_mirflickr25k_128bit",
        "epoch": 200,
        "bit_dim": 128,
        "label": "Anchored-DCMH (lc=3) | CLIP+LoRA",
    },
]


CANDIDATE_LABELS = [
    "animals", "baby", "bird", "car", "clouds", "dog", "female", "flower",
    "food", "indoor", "lake", "male", "night", "people", "plant_life",
    "portrait", "river", "sea", "sky", "structures", "sunset", "transport",
    "tree", "water",
]


# ---------------------------------------------------------------------------
# BoW vocabulary loader (for AlexNet+BoW text encoding)
# ---------------------------------------------------------------------------
def load_bow_vocab(dataset_root: Path) -> List[str]:
    """Load the 1386-word vocabulary from ``<root>/doc/common_tags.txt``."""
    vocab_path = Path(dataset_root) / "doc" / "common_tags.txt"
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"BoW vocab not found at {vocab_path}. "
            "AlexNet+BoW text encoding needs MIR-Flickr-25k's doc/common_tags.txt."
        )
    vocab: List[str] = []
    with open(vocab_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                vocab.append(parts[0].lower())
    return vocab


def make_bow_features(vocab: List[str], candidates: List[str]) -> tuple[torch.Tensor, List[bool]]:
    """One-hot BoW vector per candidate. Returns (features, valid_mask)."""
    word2idx = {w: i for i, w in enumerate(vocab)}
    feats = torch.zeros(len(candidates), len(vocab))
    valid: List[bool] = []
    for k, label in enumerate(candidates):
        idx = word2idx.get(label.lower())
        if idx is None:
            idx = word2idx.get(label.lower().replace("_", ""))
        if idx is None:
            # Try splitting on underscore: "plant_life" -> indices for "plant" and "life"
            parts = [p for p in label.lower().split("_") if p]
            hit_any = False
            for p in parts:
                pi = word2idx.get(p)
                if pi is not None:
                    feats[k, pi] = 1.0
                    hit_any = True
            valid.append(hit_any)
        else:
            feats[k, idx] = 1.0
            valid.append(True)
    return feats, valid


# ---------------------------------------------------------------------------
# Model loading + encoding
# ---------------------------------------------------------------------------
def build_and_load(spec: dict, device: torch.device, offline: bool) -> torch.nn.Module:
    cfg = load_experiment(REPO_ROOT / spec["config"])
    if spec.get("bit_dim") is not None:
        cfg.model.bit_dim = int(spec["bit_dim"])
    if hasattr(cfg, "paths") and "offline_mode" in cfg.paths:
        cfg.paths.offline_mode = bool(offline)
    text_ref, hf_lfo, _ = resolve_text_backbone_spec(cfg)
    model = build_model(cfg, text_ref=text_ref, hf_local_files_only=hf_lfo)
    model = model.to(device).eval()

    ckpt_path = (
        REPO_ROOT / "experiments" / "checkpoints" / spec["run_dir"]
        / f"epoch_{int(spec['epoch']):04d}.pt"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"[{spec['key']}] missing checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  [{spec['key']}] {len(missing)} missing keys (e.g. {missing[:2]})")
    if unexpected:
        print(f"  [{spec['key']}] {len(unexpected)} unexpected keys (e.g. {unexpected[:2]})")
    return model


@torch.no_grad()
def encode_query(model: torch.nn.Module, query_tensor: torch.Tensor,
                 device: torch.device) -> np.ndarray:
    h = model.encode_image(query_tensor.to(device))
    return torch.sign(h).cpu().squeeze(0).numpy().astype(int)


@torch.no_grad()
def encode_candidate_texts(
    model: torch.nn.Module,
    candidates: List[str],
    bow_feats: torch.Tensor | None,
    clip_tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """Encode candidate texts with the model's text tower. Returns (N, q) signed codes."""
    backend = getattr(model, "text_backend", None)
    if backend == "mlp":
        if bow_feats is None:
            raise RuntimeError("MLP text path needs BoW features.")
        codes = model.encode_text(text_features=bow_feats.to(device))
    else:
        if clip_tokenizer is None:
            raise RuntimeError("Transformer text path needs a tokenizer.")
        tokens = clip_tokenizer(
            candidates, padding=True, return_tensors="pt",
            max_length=77, truncation=True,
        )
        codes = model.encode_text(
            input_ids=tokens["input_ids"].to(device),
            attention_mask=tokens["attention_mask"].to(device),
        )
    return torch.sign(codes).cpu()


def topk_text_matches(
    query_hash: np.ndarray,
    text_codes: torch.Tensor,
    candidates: List[str],
    valid_mask: List[bool] | None,
    k: int,
) -> List[tuple[str, int]]:
    q = torch.from_numpy(query_hash)
    distances = (q.unsqueeze(0) != text_codes).sum(dim=1).clone()
    if valid_mask is not None:
        big = text_codes.shape[1] + 1  # push invalids to bottom of sort
        for i, ok in enumerate(valid_mask):
            if not ok:
                distances[i] = big
    order = torch.argsort(distances).numpy()
    top = []
    for idx in order[:k]:
        top.append((candidates[int(idx)], int(distances[int(idx)])))
    return top


def bits_to_hex(bits: np.ndarray) -> str:
    binary = (bits > 0).astype(np.uint8)
    out = []
    for i in range(0, len(binary), 4):
        nibble = (binary[i] << 3) | (binary[i + 1] << 2) | (binary[i + 2] << 1) | binary[i + 3]
        out.append(f"{nibble:x}")
    return "".join(out)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def render(query_pil: Image.Image, query_path: Path,
           results: list, output: Path | None) -> None:
    n = len(results)
    fig = plt.figure(figsize=(15.5, 2.6 + 1.05 * n))
    gs = fig.add_gridspec(
        nrows=n + 1, ncols=3,
        width_ratios=[2.6, 3.6, 6.0],
        height_ratios=[2.6] + [1.0] * n,
        hspace=0.45, wspace=0.18,
        left=0.03, right=0.98, top=0.94, bottom=0.04,
    )

    ax_q = fig.add_subplot(gs[0, :])
    ax_q.imshow(query_pil)
    ax_q.set_title(f"Query: {query_path.name}", fontsize=12, color="#1a2744")
    ax_q.axis("off")

    for r, res in enumerate(results, start=1):
        spec = res["spec"]

        ax_label = fig.add_subplot(gs[r, 0])
        digest = res["hex"]
        ax_label.text(
            0.02, 0.5,
            f"{spec['label']}\nhex: {digest[:16]}\n     {digest[16:]}",
            fontsize=8.8, va="center", ha="left",
            transform=ax_label.transAxes,
            color="#1a2744", family="monospace",
        )
        ax_label.axis("off")

        ax_hash = fig.add_subplot(gs[r, 1])
        bits = (res["hash"] > 0).astype(np.uint8)
        ax_hash.imshow(bits.reshape(4, 32), cmap="gray_r", aspect="auto",
                       vmin=0, vmax=1, interpolation="nearest")
        ax_hash.set_xticks([]); ax_hash.set_yticks([])
        for spine in ax_hash.spines.values():
            spine.set_edgecolor("#b9c5d0")

        ax_text = fig.add_subplot(gs[r, 2])
        # "label (Hamming distance)" lines, top-K best matches.
        lines = []
        for label, dist in res["topk_text"]:
            lines.append(f"{label:14s}  d={dist:>3d}")
        text_block = "nearest text (image hash -> text):\n" + "\n".join(lines)
        ax_text.text(
            0.02, 0.5, text_block,
            fontsize=9.2, va="center", ha="left",
            transform=ax_text.transAxes, color="#1a2744",
            family="monospace",
        )
        ax_text.set_xlim(0, 1); ax_text.set_ylim(0, 1)
        ax_text.axis("off")

    fig.suptitle(
        "SSHC hash demo + image-to-text reverse: nearest candidate label by Hamming distance",
        fontsize=12.5, y=0.985,
    )

    if output:
        fig.savefig(output, dpi=130, bbox_inches="tight")
        print(f"Saved figure: {output}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", required=True, help="Path to a query image (jpg/png).")
    ap.add_argument("--top-k", type=int, default=3,
                    help="Top-K nearest candidate texts to display per model (default 3).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default=None, help="Optional path to save the figure (PNG).")
    ap.add_argument(
        "--offline",
        action="store_true",
        help="Enforce paths.offline_mode=True (require a pre-populated HF cache). "
             "Default is False so CLIP downloads on demand on first run.",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    query_path = Path(args.image)
    if not query_path.is_absolute():
        query_path = (REPO_ROOT / query_path).resolve()
    if not query_path.exists():
        raise FileNotFoundError(f"Query image not found: {query_path}")
    query_pil = Image.open(query_path).convert("RGB")

    transform = imagenet_train_transform()
    query_tensor = transform(query_pil).unsqueeze(0)

    # Pre-load the CLIP tokenizer (used by 5 of 7 models) and the BoW vocab
    # (used by 2 of 7). One copy each, reused across models.
    cfg_clip = load_experiment(REPO_ROOT / "configs/experiments/exp_dcmh_clip_frozen_mirflickr25k_128bit.yaml")
    if hasattr(cfg_clip, "paths") and "offline_mode" in cfg_clip.paths:
        cfg_clip.paths.offline_mode = bool(args.offline)
    text_ref_clip, hf_lfo_clip, _ = resolve_text_backbone_spec(cfg_clip)
    print(f"Loading CLIP tokenizer from {text_ref_clip} ...")
    clip_tokenizer = load_clip_tokenizer(text_ref_clip, local_files_only=hf_lfo_clip)

    cfg_alex = load_experiment(REPO_ROOT / "configs/experiments/exp_dcmh_mirflickr25k.yaml")
    print(f"Loading BoW vocab from {cfg_alex.dataset.root}/doc/common_tags.txt ...")
    bow_vocab = load_bow_vocab(Path(cfg_alex.dataset.root))
    bow_feats, bow_valid = make_bow_features(bow_vocab, CANDIDATE_LABELS)
    n_valid = sum(bow_valid)
    print(f"  {n_valid}/{len(CANDIDATE_LABELS)} candidate labels found in BoW vocab")

    results = []
    for spec in MODELS:
        print(f"\n--- {spec['key']} ---")
        model = build_and_load(spec, device, offline=args.offline)

        bits = encode_query(model, query_tensor, device)
        digest = bits_to_hex(bits)
        ones = int((bits > 0).sum())
        print(f"  hash: {digest[:16]} {digest[16:]}  ({ones}/128 ones)")

        text_codes = encode_candidate_texts(
            model, CANDIDATE_LABELS, bow_feats, clip_tokenizer, device,
        )
        is_mlp = getattr(model, "text_backend", None) == "mlp"
        topk = topk_text_matches(
            bits, text_codes, CANDIDATE_LABELS,
            bow_valid if is_mlp else None,
            k=args.top_k,
        )
        topk_str = ", ".join(f"{lbl}(d={d})" for lbl, d in topk)
        print(f"  nearest text: {topk_str}")

        results.append({
            "spec": spec, "hash": bits, "hex": digest, "topk_text": topk,
        })

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_path = Path(args.output) if args.output else None
    render(query_pil, query_path, results, out_path)


if __name__ == "__main__":
    main()
