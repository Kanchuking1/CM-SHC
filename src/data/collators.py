"""Batch collation for cross-modal training."""

from __future__ import annotations

import torch
from transformers import AutoTokenizer, BertTokenizer


def load_hf_tokenizer(model_id: str):
    """Prefer slow BertTokenizer when possible to avoid fast-tokenizer conversion issues."""
    attempts: list[tuple[str, BaseException]] = []
    try:
        return BertTokenizer.from_pretrained(model_id)
    except Exception as e:
        attempts.append(("BertTokenizer", e))
    try:
        return AutoTokenizer.from_pretrained(model_id, use_fast=False)
    except Exception as e:
        attempts.append(("AutoTokenizer(use_fast=False)", e))
    try:
        return AutoTokenizer.from_pretrained(model_id)
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
    """(N, num_classes) multi-hot from class indices in dataset[i]['label']."""
    import torch.nn.functional as F

    ids = []
    for i in range(len(dataset)):
        y = dataset[i]["label"]
        if not torch.is_tensor(y):
            y = torch.as_tensor(y, dtype=torch.long)
        ids.append(y.long().reshape(()))
    Lidx = torch.stack(ids)
    return F.one_hot(Lidx, num_classes=num_classes).float()
