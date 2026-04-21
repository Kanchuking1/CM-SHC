"""Plot the CM-SHC lambda_cm sweep from the eval JSONs on disk.

Reads every matching eval_cmshc_mirflickr25k_128bit_<tag>_e<ep>.json under
experiments/results/, maps <tag> to its lambda_cm value, and produces a
single figure with three lines (i->t, t->i, average) plus a horizontal
reference line for the DCMH baseline.

Usage::

    python scripts/plot_lambda_cm_sweep.py
    python scripts/plot_lambda_cm_sweep.py --results-dir experiments/results \\
        --output experiments/results/lambda_cm_sweep.png

Add new sweep points by extending ``TAG_TO_LAMBDA``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Map of experiment tag -> lambda_cm value. Tags are the suffix in
# experiment_name / config filenames (see configs/experiments/).
TAG_TO_LAMBDA: dict[str, float] = {
    "nocm":   0.0,
    "lcm005": 0.05,
    "lcm01":  0.1,
    "lcm03":  0.3,
    "lcm05":  0.5,
    "clf":    1.0,
}

# Optional: also plot other center-construction methods at lambda_cm=1.0
# as extra markers (they share the clf training regime).
EXTRA_CENTERS_TAGS = ["cooc", "csq"]  # both live at lambda_cm=1.0

# DCMH reference run to draw as a horizontal line.
DCMH_REFERENCE_FNAME = "eval_alexnet_mlp_e0500.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir",
        type=str,
        default="experiments/results",
        help="Directory containing eval_*.json files.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path (default: <results-dir>/lambda_cm_sweep.png).",
    )
    p.add_argument(
        "--title",
        type=str,
        default="CM-SHC $\\lambda_{cm}$ sweep (MIR-Flickr-25k, 128 bits)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Also open the figure in an interactive window.",
    )
    p.add_argument(
        "--print-table",
        action="store_true",
        help="Print the sweep table to stdout (markdown format).",
    )
    return p.parse_args()


def _load_eval(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_eval_for_tag(results_dir: Path, tag: str) -> Path | None:
    """Return the highest-epoch eval_*.json file for a given sweep tag, or None."""
    # Filename pattern: eval_cmshc_mirflickr25k_128bit_<tag>_e<epoch>.json
    candidates = sorted(
        results_dir.glob(f"eval_cmshc_mirflickr25k_128bit_{tag}_e*.json"),
    )
    if not candidates:
        return None
    # Pick the one with the largest epoch.
    def epoch_of(p: Path) -> int:
        stem = p.stem
        try:
            return int(stem.rsplit("_e", 1)[1])
        except (IndexError, ValueError):
            return 0
    return max(candidates, key=epoch_of)


def collect_sweep_points(results_dir: Path) -> list[dict]:
    """Return rows of {tag, lambda_cm, epoch, i2t, t2i, avg, path}."""
    rows: list[dict] = []
    for tag, lam in TAG_TO_LAMBDA.items():
        eval_path = _find_eval_for_tag(results_dir, tag)
        if eval_path is None:
            print(f"[warn] no eval JSON for tag={tag!r} at lambda_cm={lam}", file=sys.stderr)
            continue
        d = _load_eval(eval_path)
        i2t = float(d["map"]["image_to_text"])
        t2i = float(d["map"]["text_to_image"])
        rows.append({
            "tag": tag,
            "lambda_cm": lam,
            "epoch": int(d.get("checkpoint_epoch", 0)),
            "i2t": i2t,
            "t2i": t2i,
            "avg": 0.5 * (i2t + t2i),
            "path": str(eval_path),
        })
    rows.sort(key=lambda r: r["lambda_cm"])
    return rows


def collect_extra_centers_points(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for tag in EXTRA_CENTERS_TAGS:
        eval_path = _find_eval_for_tag(results_dir, tag)
        if eval_path is None:
            continue
        d = _load_eval(eval_path)
        i2t = float(d["map"]["image_to_text"])
        t2i = float(d["map"]["text_to_image"])
        rows.append({
            "tag": tag,
            "lambda_cm": 1.0,
            "epoch": int(d.get("checkpoint_epoch", 0)),
            "i2t": i2t,
            "t2i": t2i,
            "avg": 0.5 * (i2t + t2i),
            "path": str(eval_path),
        })
    return rows


def dcmh_reference(results_dir: Path) -> dict | None:
    p = results_dir / DCMH_REFERENCE_FNAME
    if not p.exists():
        print(f"[warn] DCMH reference not found at {p}", file=sys.stderr)
        return None
    d = _load_eval(p)
    return {
        "i2t": float(d["map"]["image_to_text"]),
        "t2i": float(d["map"]["text_to_image"]),
        "avg": 0.5 * (d["map"]["image_to_text"] + d["map"]["text_to_image"]),
        "epoch": int(d.get("checkpoint_epoch", 0)),
    }


def print_table(rows: list[dict], extras: list[dict], dcmh: dict | None) -> None:
    print()
    print("| tag    | lambda_cm | epoch | i->t mAP | t->i mAP |   avg  |")
    print("|--------|-----------|-------|----------|----------|--------|")
    for r in rows:
        print(
            f"| {r['tag']:<6} | {r['lambda_cm']:<9.2f} | {r['epoch']:<5d} | "
            f"{r['i2t']:.4f}   | {r['t2i']:.4f}   | {r['avg']:.4f} |"
        )
    for r in extras:
        print(
            f"| {r['tag']:<6} | {r['lambda_cm']:<9.2f} | {r['epoch']:<5d} | "
            f"{r['i2t']:.4f}   | {r['t2i']:.4f}   | {r['avg']:.4f} |"
        )
    if dcmh is not None:
        print(
            f"| DCMH   | n/a       | {dcmh['epoch']:<5d} | "
            f"{dcmh['i2t']:.4f}   | {dcmh['t2i']:.4f}   | {dcmh['avg']:.4f} |"
        )
    print()


def make_plot(
    rows: list[dict],
    extras: list[dict],
    dcmh: dict | None,
    output_path: Path,
    title: str,
    dpi: int,
    show: bool,
) -> None:
    # Import matplotlib lazily so --print-table alone doesn't require it.
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        raise RuntimeError("No sweep points found; nothing to plot.")

    xs = [r["lambda_cm"] for r in rows]
    i2t = [r["i2t"] for r in rows]
    t2i = [r["t2i"] for r in rows]
    avg = [r["avg"] for r in rows]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    ax.plot(xs, avg, "-o", color="#1f77b4", label="CM-SHC avg mAP", linewidth=2.0, zorder=3)
    ax.plot(xs, i2t, "--s", color="#2ca02c", label="CM-SHC i$\\rightarrow$t", alpha=0.85)
    ax.plot(xs, t2i, "--^", color="#d62728", label="CM-SHC t$\\rightarrow$i", alpha=0.85)

    # Annotate each sweep point with its tag.
    for r in rows:
        ax.annotate(
            r["tag"],
            xy=(r["lambda_cm"], r["avg"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color="#555555",
        )

    # DCMH reference as a horizontal band (i2t + t2i range) plus a dashed line at the avg.
    if dcmh is not None:
        ax.axhline(dcmh["avg"], color="black", linestyle=":", linewidth=1.2,
                   label=f"DCMH avg mAP ({dcmh['avg']:.3f})", zorder=2)

    # Extra center-construction methods at lambda=1.0 (csq, cooc) as scatter points.
    for r in extras:
        ax.plot(
            r["lambda_cm"], r["avg"], "P",
            color="#9467bd", markersize=9, label=f"CM-SHC {r['tag']} (avg)" if r is extras[0] else None,
            zorder=4,
        )
        # Jitter the annotation slightly so csq/cooc don't collide with the clf point.
        ax.annotate(
            r["tag"],
            xy=(r["lambda_cm"], r["avg"]),
            xytext=(-24, -14 if r["tag"] == "csq" else 6),
            textcoords="offset points",
            fontsize=8,
            color="#9467bd",
        )

    ax.set_xlabel("$\\lambda_{cm}$ (cross-modal MSE weight)")
    ax.set_ylabel("mAP (label overlap relevance)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.08)
    # Pad the y-range so markers at edges aren't clipped.
    all_y = i2t + t2i + avg + ([dcmh["avg"]] if dcmh is not None else []) + [r["avg"] for r in extras]
    ymin, ymax = min(all_y), max(all_y)
    pad = max(0.01, (ymax - ymin) * 0.08)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    print(f"Wrote {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    if not results_dir.is_dir():
        print(f"[error] results dir not found: {results_dir}", file=sys.stderr)
        return 2

    rows = collect_sweep_points(results_dir)
    extras = collect_extra_centers_points(results_dir)
    dcmh = dcmh_reference(results_dir)

    if args.print_table:
        print_table(rows, extras, dcmh)

    if not rows:
        print(
            "[error] no sweep points found. Expected filenames like "
            "eval_cmshc_mirflickr25k_128bit_<tag>_e<epoch>.json",
            file=sys.stderr,
        )
        return 1

    output_path = (
        Path(args.output).resolve()
        if args.output is not None
        else results_dir / "lambda_cm_sweep.png"
    )
    make_plot(rows, extras, dcmh, output_path, args.title, args.dpi, args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
