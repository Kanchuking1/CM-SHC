"""Overlay per-epoch training-loss curves across runs.

Reads one or more ``loss_history.json`` files produced by ``DCMHTrainer``,
``CMSHCTrainer`` or ``DCMHAnchoredTrainer`` and stacks them on shared axes so
different training regimes can be compared directly.

Two history formats are auto-detected from the per-entry keys:

* **DCMH** - ``log_loss``, ``quant_loss``, ``balance_loss``, ``total_scaled``,
  ``full_objective``.
* **CM-SHC** - ``center``, ``quant``, ``cross_modal``, ``balance``, ``total``.
  Anchored DCMH runs additionally log ``center_loss`` alongside the DCMH keys
  and are plotted with the DCMH panel set plus a ``center_loss`` panel.

Typical invocations::

    # Explicit list of runs (each arg is "label=path")
    python scripts/plot_loss_curves.py \\
        --run "DCMH (frozen CLIP)=experiments/checkpoints/dcmh_clip_frozen_mirflickr25k_128bit_dcmh_mirflickr25k_128bit/loss_history.json" \\
        --run "DCMH + LoRA=experiments/checkpoints/dcmh_clip_lora_mirflickr25k_128bit_dcmh_mirflickr25k_128bit/loss_history.json" \\
        --output experiments/results/loss_dcmh_frozen_vs_lora.png

    # Presets baked into the script
    python scripts/plot_loss_curves.py --preset dcmh
    python scripts/plot_loss_curves.py --preset cmshc
    python scripts/plot_loss_curves.py --preset all

The DCMH preset overlays {AlexNet, CLIP-frozen, CLIP+LoRA}; the CM-SHC preset
overlays {CLIP-frozen, CLIP+LoRA}; ``--preset all`` produces both figures plus
a combined "total objective" comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Known per-format key -> human-readable label
# ----------------------------------------------------------------------

DCMH_PANELS: List[Tuple[str, str]] = [
    ("log_loss", "Pair log-likelihood"),
    ("quant_loss", "Quantisation"),
    ("balance_loss", "Balance"),
    ("total_scaled", "Total (scaled)"),
    ("full_objective", "Full objective"),
]

CMSHC_PANELS: List[Tuple[str, str]] = [
    ("center", "Centre BCE"),
    ("quant", "Quantisation"),
    ("cross_modal", "Cross-modal"),
    ("balance", "Balance"),
    ("total", "Total"),
]

# Anchored-DCMH is DCMH + an extra centre-BCE anchor.
DCMH_ANCHORED_PANELS: List[Tuple[str, str]] = DCMH_PANELS + [
    ("center_loss", "Centre-BCE anchor"),
]

# ----------------------------------------------------------------------
# Presets - curated overlays that the thesis report references.
# ----------------------------------------------------------------------

REPO_ROOT_DEFAULT = "experiments/checkpoints"

PRESETS: Dict[str, Dict[str, List[Tuple[str, str]]]] = {
    "dcmh": {
        "title": "DCMH training loss curves (MIR-Flickr-25k, 128 bits)",
        "output": "experiments/results/loss_dcmh_overlay.png",
        "runs": [
            # ("AlexNet+BoW", "alexnet_mlp_dcmh_mirflickr25k_128bit/loss_history.json"),
            (
                "CLIP frozen",
                "dcmh_clip_frozen_mirflickr25k_128bit_dcmh_mirflickr25k_128bit/loss_history.json",
            ),
            (
                "CLIP + LoRA",
                "dcmh_clip_lora_mirflickr25k_128bit_dcmh_mirflickr25k_128bit/loss_history.json",
            ),
        ],
    },
    "cmshc": {
        "title": "CM-SHC training loss curves (MIR-Flickr-25k, 128 bits)",
        "output": "experiments/results/loss_cmshc_overlay.png",
        "runs": [
            (
                "CLIP frozen",
                "cmshc_clip_frozen_mirflickr25k_128bit_cm_shc_mirflickr25k_128bit/loss_history.json",
            ),
            (
                "CLIP + LoRA",
                "cmshc_clip_lora_mirflickr25k_128bit_cm_shc_mirflickr25k_128bit/loss_history.json",
            ),
        ],
    },
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def parse_run_spec(spec: str) -> Tuple[str, str]:
    """Parse a "label=path" argument; fall back to path basename as label."""
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip(), path.strip()
    path = spec.strip()
    label = Path(path).parent.name or Path(path).stem
    return label, path


def load_history(path: str) -> Tuple[List[int], Dict[str, List[float]]]:
    """Load a history JSON (list of per-epoch dicts) into columnar arrays."""
    with open(path, "r") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected list of per-epoch dicts, got {type(raw).__name__}")
    if not raw:
        raise ValueError(f"{path}: empty history file")

    epochs: List[int] = []
    cols: Dict[str, List[float]] = {}
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: entry {i} is not a dict")
        ep = int(entry.get("epoch", i + 1))
        epochs.append(ep)
        for k, v in entry.items():
            if k == "epoch":
                continue
            try:
                val = float(v)
            except (TypeError, ValueError):
                continue
            cols.setdefault(k, []).append(val)
    # Pad any missing values with NaN so every column lines up with `epochs`.
    n = len(epochs)
    for k in list(cols.keys()):
        if len(cols[k]) < n:
            cols[k].extend([float("nan")] * (n - len(cols[k])))
    return epochs, cols


def detect_format(cols: Dict[str, List[float]]) -> str:
    keys = set(cols.keys())
    if "center_loss" in keys and "log_loss" in keys:
        return "dcmh_anchored"
    if {"log_loss", "quant_loss", "balance_loss"}.issubset(keys):
        return "dcmh"
    if {"center", "quant", "cross_modal"}.issubset(keys):
        return "cmshc"
    raise ValueError(f"could not identify history format from keys: {sorted(keys)}")


def panels_for(fmt: str) -> List[Tuple[str, str]]:
    if fmt == "dcmh":
        return DCMH_PANELS
    if fmt == "cmshc":
        return CMSHC_PANELS
    if fmt == "dcmh_anchored":
        return DCMH_ANCHORED_PANELS
    raise ValueError(fmt)


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------


def plot_overlay(
    runs: Sequence[Tuple[str, str]],
    *,
    output: str,
    title: str,
    dpi: int = 150,
    log_scale_keys: Sequence[str] = (
        "log_loss",
        "balance_loss",
        "full_objective",
    ),
) -> str:
    """Overlay per-epoch loss curves across runs.

    All runs must share the same history format (DCMH vs CM-SHC vs
    DCMH-anchored).  The first run determines the panel layout; any extra
    keys present in later runs are ignored silently.
    """
    if not runs:
        raise ValueError("no runs provided")

    loaded: List[Tuple[str, List[int], Dict[str, List[float]], str]] = []
    for label, path in runs:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label}: {path} does not exist")
        epochs, cols = load_history(path)
        fmt = detect_format(cols)
        loaded.append((label, epochs, cols, fmt))

    # Use the first run's format to pick the panels; warn (don't error) on
    # mismatches so the user sees which run was incompatible.
    base_fmt = loaded[0][3]
    panels = panels_for(base_fmt)
    mismatched = [lbl for (lbl, _, _, fmt) in loaded if fmt != base_fmt]
    if mismatched:
        print(
            f"[plot_loss_curves] WARNING: format mismatch - base={base_fmt}, "
            f"other runs: {mismatched}.  Only shared keys will render.",
            file=sys.stderr,
        )

    n_panels = len(panels)
    ncols = 2 if n_panels > 2 else n_panels
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.5 * nrows))
    axes = [axes] if n_panels == 1 else list(axes.flat)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, (key, pretty) in zip(axes, panels):
        any_curve = False
        for i, (label, epochs, cols, _fmt) in enumerate(loaded):
            if key not in cols:
                continue
            ax.plot(
                epochs,
                cols[key],
                label=label,
                color=color_cycle[i % len(color_cycle)],
                linewidth=1.4,
            )
            any_curve = True
        ax.set_title(pretty)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)
        if key in log_scale_keys and any_curve:
            try:
                ax.set_yscale("log")
            except ValueError:
                pass
        if any_curve:
            ax.legend(fontsize=8, loc="best")
        else:
            ax.text(0.5, 0.5, f"(no '{key}' in any run)",
                    transform=ax.transAxes, ha="center", va="center",
                    color="gray")

    # Hide spare axes
    for ax in axes[n_panels:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    Path(os.path.dirname(output) or ".").mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(output)


def plot_total_comparison(
    runs: Sequence[Tuple[str, str]],
    *,
    output: str,
    title: str = "Total training objective across regimes",
    dpi: int = 150,
) -> str:
    """Single-panel overlay of each run's "total" curve (normalised).

    Plots ``total_scaled`` for DCMH histories and ``total`` for CM-SHC
    histories on a shared axis.  Because the two objectives are on different
    absolute scales each curve is also divided by its own epoch-1 value so
    the *shape* of the descent can be compared.
    """
    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(12, 4.2))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (label, path) in enumerate(runs):
        epochs, cols = load_history(path)
        fmt = detect_format(cols)
        if fmt in ("dcmh", "dcmh_anchored"):
            total = cols.get("total_scaled", [])
            ykey = "total_scaled"
        else:
            total = cols.get("total", [])
            ykey = "total"
        if not total:
            continue
        color = color_cycle[i % len(color_cycle)]
        ax_raw.plot(epochs, total, label=f"{label} ({ykey})", color=color, linewidth=1.4)
        norm = [v / total[0] if total[0] not in (0, float("nan")) else float("nan") for v in total]
        ax_norm.plot(epochs, norm, label=label, color=color, linewidth=1.4)

    ax_raw.set_title("Raw total objective")
    ax_raw.set_xlabel("Epoch")
    ax_raw.set_ylabel("loss")
    ax_raw.set_yscale("log")
    ax_raw.grid(True, alpha=0.3)
    ax_raw.legend(fontsize=8, loc="best")

    ax_norm.set_title("Total objective (normalised to epoch 1)")
    ax_norm.set_xlabel("Epoch")
    ax_norm.set_ylabel("loss / loss@epoch 1")
    ax_norm.grid(True, alpha=0.3)
    ax_norm.legend(fontsize=8, loc="best")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    Path(os.path.dirname(output) or ".").mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(output)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def resolve_preset(
    name: str, root: str
) -> Tuple[List[Tuple[str, str]], str, str]:
    cfg = PRESETS[name]
    runs = [(lbl, os.path.join(root, rel)) for (lbl, rel) in cfg["runs"]]
    return runs, cfg["output"], cfg["title"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run specifier as 'label=path/to/loss_history.json'. "
             "May be passed multiple times.",
    )
    p.add_argument(
        "--preset",
        choices=sorted(list(PRESETS.keys()) + ["all"]),
        default=None,
        help="Use a built-in overlay preset.  'all' renders every preset plus "
             "a total-objective comparison.",
    )
    p.add_argument(
        "--checkpoints-root",
        default=REPO_ROOT_DEFAULT,
        help="Root directory for preset paths (default: %(default)s).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output PNG path (required if --run is used without --preset).",
    )
    p.add_argument(
        "--title",
        default=None,
        help="Figure title (defaults to the preset title or a generic one).",
    )
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.preset == "all":
        written: List[str] = []
        all_runs: List[Tuple[str, str]] = []
        for name in PRESETS:
            runs, output, title = resolve_preset(name, args.checkpoints_root)
            runs = [(lbl, p) for (lbl, p) in runs if os.path.exists(p)]
            if not runs:
                print(f"[preset {name}] no history files found; skipping", file=sys.stderr)
                continue
            out = plot_overlay(runs, output=output, title=title, dpi=args.dpi)
            print(f"wrote {out}")
            written.append(out)
            all_runs.extend([(f"{name.upper()}: {lbl}", p) for (lbl, p) in runs])

        if all_runs:
            combo_out = os.path.join(
                os.path.dirname(PRESETS["dcmh"]["output"]) or ".",
                "loss_total_objective_overlay.png",
            )
            out = plot_total_comparison(
                all_runs, output=combo_out,
                title="Training objective across DCMH / CM-SHC regimes (128 bits)",
                dpi=args.dpi,
            )
            print(f"wrote {out}")
        return 0

    if args.preset is not None:
        runs, output, title = resolve_preset(args.preset, args.checkpoints_root)
        runs = [(lbl, p) for (lbl, p) in runs if os.path.exists(p)]
        if not runs:
            print(f"no history files found for preset {args.preset}", file=sys.stderr)
            return 2
        out = plot_overlay(
            runs,
            output=args.output or output,
            title=args.title or title,
            dpi=args.dpi,
        )
        print(f"wrote {out}")
        return 0

    if not args.run:
        print("ERROR: pass --preset <name> or one or more --run 'label=path' args",
              file=sys.stderr)
        return 2
    if args.output is None:
        print("ERROR: --output is required when using --run", file=sys.stderr)
        return 2

    runs = [parse_run_spec(s) for s in args.run]
    out = plot_overlay(
        runs,
        output=args.output,
        title=args.title or "Training loss overlay",
        dpi=args.dpi,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
