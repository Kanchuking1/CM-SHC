"""Label and text helpers (dataset-agnostic building blocks)."""

from __future__ import annotations

import pandas as pd


def hash_filename_label(filename: str, num_classes: int) -> int:
    return hash(filename) % num_classes


def image_caption_first_map(df: pd.DataFrame) -> dict[str, str] | None:
    """First caption per image when CSV has image + caption columns."""
    cols = {c.lower().strip(): c for c in df.columns}
    img_key = cols.get("image") or cols.get("img") or cols.get("filename")
    cap_key = cols.get("caption") or cols.get("text")
    if img_key and cap_key:
        first = df.rename(columns={img_key: "_img", cap_key: "_cap"}).groupby("_img")["_cap"].first()
        return {str(k).strip(): str(v) for k, v in first.items()}
    return None
