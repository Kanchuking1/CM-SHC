"""Batch collation for cross-modal training."""

from __future__ import annotations

import torch
from transformers import AutoTokenizer, BertTokenizer


def load_hf_tokenizer(model_id: str, local_files_only: bool = False):
    """Prefer slow BertTokenizer when possible to avoid fast-tokenizer conversion issues."""
    attempts: list[tuple[str, BaseException]] = []
    try:
        return BertTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    except Exception as e:
        attempts.append(("BertTokenizer", e))
    try:
        return AutoTokenizer.from_pretrained(model_id, use_fast=False, local_files_only=local_files_only)
    except Exception as e:
        attempts.append(("AutoTokenizer(use_fast=False)", e))
    try:
        return AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    except Exception as e:
        attempts.append(("AutoTokenizer(fast)", e))
    lines = "\n".join(f"  - {n}: {x}" for n, x in attempts)
    raise RuntimeError(f"Could not load tokenizer for {model_id!r}.\n{lines}") from attempts[-1][1]


def make_dcmh_collate_fn(tokenizer, max_length: int = 64):
    def _collate(samples: list[dict]) -> dict:
        imgs = torch.stack([s["img"] for s in samples], dim=0)
        labels = torch.stack([s["label"] for s in samples], dim=0)
        idx = torch.tensor([s["index"] for s in samples], dtype=torch.long)
        texts = [s["text"] for s in samples]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        out = {
            "img": imgs,
            "label": labels,
            "index": idx,
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }
        return out

    return _collate


def build_train_labels_tensor(dataset, num_classes: int) -> torch.Tensor:
    """(N, num_classes) multi-hot label matrix.

    If the dataset already provides multi-hot vectors (e.g. MIRFlickr25k with
    real annotations), they are stacked directly.  Otherwise scalar class
    indices are converted via one_hot.
    """
    import torch.nn.functional as F

    has_fast_path = callable(getattr(dataset, "get_label", None))
    n = len(dataset)
    if has_fast_path:
        sample = dataset.get_label(0)
        if torch.is_tensor(sample) and sample.dim() >= 1:
            return torch.stack([dataset.get_label(i) for i in range(n)]).float()
        Lidx = torch.tensor([dataset.get_label(i) for i in range(n)], dtype=torch.long)
    else:
        ids = []
        for i in range(n):
            y = dataset[i]["label"]
            if not torch.is_tensor(y):
                y = torch.as_tensor(y)
            if y.dim() >= 1:
                ids.append(y.float())
            else:
                ids.append(y.long().reshape(()))
        stacked = torch.stack(ids)
        if stacked.dim() == 2:
            return stacked.float()
        Lidx = stacked.long()
    return F.one_hot(Lidx, num_classes=num_classes).float()
