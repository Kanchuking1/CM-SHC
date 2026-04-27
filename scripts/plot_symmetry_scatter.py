"""Scatter plots of I->T vs T->I mean MAP, grouped by backbone.

Three side-by-side panels - AlexNet+BoW, CLIP-frozen, CLIP+LoRA - each
plot points at (T->I on x-axis, I->T on y-axis). The dashed y=x
reference line makes any I->T / T->I asymmetry visible immediately.
Points are coloured by method family (DCMH, CM-SHC, Anchored-DCMH
variants) and annotated with short labels.

Reads eval JSON files from ``experiments/results/`` that follow the
``DCMHTrainer`` / ``CMSHCTrainer`` / ``DCMHAnchoredTrainer`` output
schema:

    {
      "map": {"image_to_text": 0.xxx, "text_to_image": 0.xxx},
      ...
    }

Typical invocation::

    python scripts/plot_symmetry_scatter.py
    # writes experiments/results/symmetry_scatter_by_backbone.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ----------------------------------------------------------------------
# Styling - mirror the presentation palette in docs/presentation/index.html
# ----------------------------------------------------------------------

COLOR_DCMH = "#c41e3a"     # crimson   - pair loss (DCMH)
COLOR_CMSHC = "#0f9480"    # teal      - centre loss (CM-SHC)
COLOR_ANCHOR_OK = "#d97706"  # amber   - anchored-DCMH sweep points
COLOR_ANCHOR_BEST = "#1a2744"  # navy   - lambda_c=3 best point
COLOR_ANCHOR_BAD = "#6c757d"  # grey    - failure modes (anchor-only, hi-bal)
COLOR_REF = "#b9c5d0"      # light grey - y=x reference

MARKER_DCMH = "o"
MARKER_CMSHC = "s"
MARKER_ANCHOR = "D"


@dataclass
class Point:
    """One scatter point = one eval_*.json file."""

    label: str
    t2i: float
    i2t: float
    color: str
    marker: str
    method: str
    size: int = 70
    highlight: bool = False


RESULTS_ROOT: Path = Path("experiments/results")


def _eval(name: str) -> Dict:
    path = RESULTS_ROOT / f"eval_{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing eval file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _xy(doc: Dict) -> Tuple[float, float]:
    return doc["map"]["text_to_image"], doc["map"]["image_to_text"]


def _alexnet_panel() -> List[Point]:
    pts: List[Point] = []
    t2i, i2t = _xy(_eval("alexnet_mlp_e0500"))
    pts.append(Point(
        label="DCMH (e500)",
        t2i=t2i, i2t=i2t,
        color=COLOR_DCMH, marker=MARKER_DCMH, method="DCMH",
        highlight=True,
    ))
    specs = [
        ("cmshc_mirflickr25k_128bit_csq_e0200",  "CM-SHC (S_csq)"),
        ("cmshc_mirflickr25k_128bit_clf_e0200",  "CM-SHC (S_clf)"),
        ("cmshc_mirflickr25k_128bit_cooc_e0200", "CM-SHC (S_cooc)"),
    ]
    for fname, label in specs:
        t2i, i2t = _xy(_eval(fname))
        pts.append(Point(
            label=label, t2i=t2i, i2t=i2t,
            color=COLOR_CMSHC, marker=MARKER_CMSHC, method="CM-SHC",
        ))
    return pts


def _clip_frozen_panel() -> List[Point]:
    pts: List[Point] = []
    t2i, i2t = _xy(_eval("dcmh_clip_frozen_mirflickr25k_128bit_e0200"))
    pts.append(Point(
        label="DCMH", t2i=t2i, i2t=i2t,
        color=COLOR_DCMH, marker=MARKER_DCMH, method="DCMH",
    ))
    t2i, i2t = _xy(_eval("cmshc_clip_frozen_mirflickr25k_128bit_e0200"))
    pts.append(Point(
        label="CM-SHC", t2i=t2i, i2t=i2t,
        color=COLOR_CMSHC, marker=MARKER_CMSHC, method="CM-SHC",
    ))
    return pts


def _clip_lora_panel() -> List[Point]:
    pts: List[Point] = []
    t2i, i2t = _xy(_eval("dcmh_clip_lora_mirflickr25k_128bit_e0200"))
    pts.append(Point(
        label="DCMH", t2i=t2i, i2t=i2t,
        color=COLOR_DCMH, marker=MARKER_DCMH, method="DCMH",
    ))
    t2i, i2t = _xy(_eval("cmshc_clip_lora_mirflickr25k_128bit_e0200"))
    pts.append(Point(
        label="CM-SHC", t2i=t2i, i2t=i2t,
        color=COLOR_CMSHC, marker=MARKER_CMSHC, method="CM-SHC",
        highlight=True,
    ))
    sweep = [
        ("dcmh_anchored_clip_lora_mirflickr25k_128bit_e0200",      "lc=0.1", COLOR_ANCHOR_OK),
        ("dcmh_anchored_clip_lora_mirflickr25k_128bit_lc03_e0200", "lc=0.3", COLOR_ANCHOR_OK),
        ("dcmh_anchored_clip_lora_mirflickr25k_128bit_lc05_e0200", "lc=0.5", COLOR_ANCHOR_OK),
        ("dcmh_anchored_clip_lora_mirflickr25k_128bit_lc10_e0200", "lc=1.0", COLOR_ANCHOR_BAD),
        ("dcmh_anchored_clip_lora_mirflickr25k_128bit_lc30_e0200", "lc=3.0", COLOR_ANCHOR_BEST),
    ]
    for fname, label, color in sweep:
        t2i, i2t = _xy(_eval(fname))
        pts.append(Point(
            label=label, t2i=t2i, i2t=i2t,
            color=color, marker=MARKER_ANCHOR, method="Anchored-DCMH",
            highlight=(label == "lc=3.0"),
        ))
    diagnostics = [
        ("dcmh_anchored_clip_lora_mirflickr25k_128bit_anchoronly_e0200", "anchor-only"),
        ("dcmh_anchored_clip_lora_mirflickr25k_128bit_hibal_e0200",      "hi-balance"),
    ]
    for fname, label in diagnostics:
        t2i, i2t = _xy(_eval(fname))
        pts.append(Point(
            label=label, t2i=t2i, i2t=i2t,
            color=COLOR_ANCHOR_BAD, marker="x", method="Anchored-DCMH",
            size=80,
        ))
    return pts


def _nice_limits(points_all: Sequence[Point], pad: float = 0.02) -> Tuple[float, float]:
    xs = [p.t2i for p in points_all]
    ys = [p.i2t for p in points_all]
    lo = min(min(xs), min(ys))
    hi = max(max(xs), max(ys))
    return lo - pad, hi + pad


def _draw_panel(ax, title, points, lim):
    lo, hi = lim
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.plot([lo, hi], [lo, hi], linestyle="--", color=COLOR_REF, linewidth=1.2, zorder=1)

    cluster_tol = 0.015
    prev_xy: Optional[Tuple[float, float]] = None
    flip = False

    for p in points:
        if prev_xy is not None:
            dx = abs(p.t2i - prev_xy[0])
            dy = abs(p.i2t - prev_xy[1])
            if dx < cluster_tol and dy < cluster_tol:
                flip = not flip
            else:
                flip = False
        prev_xy = (p.t2i, p.i2t)

        if p.highlight:
            ax.scatter([p.t2i], [p.i2t],
                       s=p.size * 2.6, facecolors="none",
                       edgecolors=p.color, linewidths=1.8,
                       alpha=0.35, zorder=2)

        kwargs = dict(s=p.size, color=p.color, marker=p.marker, zorder=3)
        if p.marker != "x":
            kwargs.update(edgecolors="white", linewidths=0.8)
        ax.scatter([p.t2i], [p.i2t], **kwargs)

        if flip:
            ox, oy = -0.004, -0.014
            ha, va = "right", "top"
        else:
            ox, oy = 0.004, 0.006
            ha, va = "left", "bottom"
        ax.annotate(
            p.label,
            xy=(p.t2i, p.i2t),
            xytext=(p.t2i + ox, p.i2t + oy),
            fontsize=8.2, color="#1a2744",
            fontweight=("bold" if p.highlight else "normal"),
            ha=ha, va=va, zorder=4,
        )

    ax.set_title(title, fontsize=11, fontweight="600", color="#1a2744", pad=8)
    ax.set_xlabel("T -> I mean MAP", fontsize=9, color="#41516f")
    ax.set_ylabel("I -> T mean MAP", fontsize=9, color="#41516f")
    ax.tick_params(axis="both", which="major", labelsize=8, colors="#41516f")
    ax.grid(True, linestyle=":", linewidth=0.5, color="#dce3ea", zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#b9c5d0")
        spine.set_linewidth(0.8)


def _build_legend(fig):
    handles = [
        Line2D([0], [0], color=COLOR_REF, linestyle="--", label="y = x (symmetric)"),
        Line2D([0], [0], marker=MARKER_DCMH, color="w",
               markerfacecolor=COLOR_DCMH, markeredgecolor="white",
               markersize=8, label="DCMH (pair loss)"),
        Line2D([0], [0], marker=MARKER_CMSHC, color="w",
               markerfacecolor=COLOR_CMSHC, markeredgecolor="white",
               markersize=8, label="CM-SHC (centre loss)"),
        Line2D([0], [0], marker=MARKER_ANCHOR, color="w",
               markerfacecolor=COLOR_ANCHOR_OK, markeredgecolor="white",
               markersize=8, label="Anchored-DCMH (lambda_c sweep)"),
        Line2D([0], [0], marker=MARKER_ANCHOR, color="w",
               markerfacecolor=COLOR_ANCHOR_BEST, markeredgecolor="white",
               markersize=8, label="Anchored-DCMH lambda_c=3 (best)"),
        Line2D([0], [0], marker="x", color=COLOR_ANCHOR_BAD,
               markersize=8, linestyle="None",
               label="Diagnostic failure (anchor-only, hi-bal)"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, -0.01), fontsize=9,
    )


def build_figure(results_root: Path, output: Path) -> Path:
    global RESULTS_ROOT
    RESULTS_ROOT = results_root

    panels = [
        ("AlexNet + BoW (128 bit)", _alexnet_panel()),
        ("CLIP frozen (128 bit)",   _clip_frozen_panel()),
        ("CLIP + LoRA (128 bit)",   _clip_lora_panel()),
    ]

    all_points: List[Point] = []
    for _, pts in panels:
        all_points.extend(pts)
    lim = _nice_limits(all_points, pad=0.025)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), constrained_layout=False)
    fig.suptitle(
        "Cross-modal symmetry by backbone  -  MIR-Flickr-25k, 128 bits",
        fontsize=13, fontweight="600", color="#1a2744", y=0.99,
    )

    for ax, (title, pts) in zip(axes, panels):
        _draw_panel(ax, title, pts, lim)

    _build_legend(fig)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.17, wspace=0.24)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor="white")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="experiments/results")
    parser.add_argument("--output", default="experiments/results/symmetry_scatter_by_backbone.png")
    args = parser.parse_args()
    saved = build_figure(Path(args.results_root).resolve(), Path(args.output).resolve())
    print(f"wrote {saved}")


if __name__ == "__main__":
    main()
