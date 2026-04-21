"""Tests for the ``text_raw`` data-pipeline path (CLIP / HF transformer collate).

These tests avoid any real pretrained-weight download and any real dataset
on disk by:

* Building samples by hand for the collator path.
* Monkey-patching ``MIRFlickr25kDCMHDataset._read_tags_file`` /
  ``_build_bow`` and faking the filesystem probes so the ``_build_prompts``
  logic can be exercised in isolation.

The goal is to verify that:

1. ``make_dcmh_collate_fn`` with a tokenizer tokenizes ``text_raw`` and
   produces ``input_ids`` / ``attention_mask`` batches.
2. ``make_dcmh_collate_fn`` with no tokenizer falls back to
   ``text_features`` when those are present.
3. The ``text_raw`` path wins when both keys are present *and* a tokenizer
   is supplied (text_mode='both' ablation scenario).
4. ``MIRFlickr25kDCMHDataset`` emits ``text_raw`` / ``text_features`` in
   line with the ``text_mode`` kwarg.
"""

from __future__ import annotations

import pytest
import torch


class _FakeTokenizer:
    """Tiny tokenizer: splits on whitespace, pads to batch max length."""

    model_max_length = 77

    def __call__(
        self,
        texts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    ):
        token_lists = [[hash(w) % 30000 for w in t.split()][:max_length] for t in texts]
        lens = [len(t) for t in token_lists]
        L = max(lens) if lens else 0
        ids = torch.zeros(len(texts), L, dtype=torch.long)
        mask = torch.zeros(len(texts), L, dtype=torch.long)
        for i, toks in enumerate(token_lists):
            if toks:
                ids[i, : len(toks)] = torch.tensor(toks, dtype=torch.long)
                mask[i, : len(toks)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def _sample(idx: int, **extra):
    s = {
        "index": idx,
        "img": torch.randn(3, 8, 8),
        "label": torch.tensor([1.0, 0.0, 1.0]),
    }
    s.update(extra)
    return s


def test_collate_with_tokenizer_tokenizes_text_raw():
    from src.data.collators import make_dcmh_collate_fn

    collate = make_dcmh_collate_fn(_FakeTokenizer(), max_length=16)
    batch = collate([
        _sample(0, text_raw="a photo of a cat"),
        _sample(1, text_raw="a photo of a dog running fast"),
    ])
    assert batch["img"].shape == (2, 3, 8, 8)
    assert batch["label"].shape == (2, 3)
    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    # The shorter caption is zero-padded, so row 0 has fewer ones than row 1.
    assert batch["attention_mask"][0].sum() < batch["attention_mask"][1].sum()


def test_collate_without_tokenizer_uses_text_features():
    from src.data.collators import make_dcmh_collate_fn

    collate = make_dcmh_collate_fn(None)
    batch = collate([
        _sample(0, text_features=torch.randn(7)),
        _sample(1, text_features=torch.randn(7)),
    ])
    assert "input_ids" not in batch
    assert batch["text_features"].shape == (2, 7)


def test_collate_raw_wins_when_both_present():
    """When text_mode='both' and a tokenizer is provided, raw text wins."""
    from src.data.collators import make_dcmh_collate_fn

    collate = make_dcmh_collate_fn(_FakeTokenizer(), max_length=16)
    batch = collate([
        _sample(0, text_raw="a photo of a cat", text_features=torch.randn(5)),
        _sample(1, text_raw="a photo of a dog", text_features=torch.randn(5)),
    ])
    assert "input_ids" in batch
    assert "text_features" not in batch


def test_collate_raises_without_tokenizer_for_raw_text():
    from src.data.collators import make_dcmh_collate_fn

    collate = make_dcmh_collate_fn(None)
    with pytest.raises(RuntimeError, match="no tokenizer was provided"):
        collate([_sample(0, text_raw="a photo")])


def test_collate_raises_when_empty():
    from src.data.collators import make_dcmh_collate_fn

    collate = make_dcmh_collate_fn(None)
    with pytest.raises(RuntimeError, match="no text_features"):
        collate([_sample(0)])


def test_collate_falls_back_to_legacy_text_key():
    """Backward compat: samples with a ``text`` key still work."""
    from src.data.collators import make_dcmh_collate_fn

    collate = make_dcmh_collate_fn(_FakeTokenizer(), max_length=16)
    batch = collate([_sample(0, text="hello world")])
    assert "input_ids" in batch
    assert batch["input_ids"].shape[0] == 1


@pytest.fixture
def fake_mirflickr(tmp_path, monkeypatch):
    """Build a tiny MIRFlickr-like directory tree for dataset tests."""
    root = tmp_path / "mirflickr"
    root.mkdir()
    (root / "annotations").mkdir()
    (root / "meta").mkdir()
    (root / "meta" / "tags").mkdir()
    (root / "doc").mkdir()

    # Three images, each annotated with a single class so the filter keeps them.
    from PIL import Image
    for i in (1, 2, 3):
        Image.new("RGB", (8, 8)).save(root / f"im{i}.jpg")
        (root / "meta" / "tags" / f"tags{i}.txt").write_text(f"tag{i}\nshared\n")

    (root / "annotations" / "sky.txt").write_text("1\n2\n3\n")
    (root / "doc" / "common_tags.txt").write_text("tag1\ntag2\ntag3\nshared\n")
    return root


def test_dataset_text_mode_bow_only(fake_mirflickr):
    from src.data.loaders import MIRFlickr25kDCMHDataset

    ds = MIRFlickr25kDCMHDataset(fake_mirflickr, text_mode="bow")
    s = ds[0]
    assert "text_features" in s
    assert "text_raw" not in s
    assert s["text_features"].shape == (4,)


def test_dataset_text_mode_raw_only(fake_mirflickr):
    from src.data.loaders import MIRFlickr25kDCMHDataset

    ds = MIRFlickr25kDCMHDataset(fake_mirflickr, text_mode="raw")
    s = ds[0]
    assert "text_raw" in s
    assert "text_features" not in s
    assert isinstance(s["text_raw"], str) and s["text_raw"].startswith("a photo of")


def test_dataset_text_mode_both(fake_mirflickr):
    from src.data.loaders import MIRFlickr25kDCMHDataset

    ds = MIRFlickr25kDCMHDataset(fake_mirflickr, text_mode="both")
    s = ds[0]
    assert "text_raw" in s and "text_features" in s


def test_dataset_rejects_invalid_text_mode(fake_mirflickr):
    from src.data.loaders import MIRFlickr25kDCMHDataset

    with pytest.raises(ValueError, match="text_mode must be one of"):
        MIRFlickr25kDCMHDataset(fake_mirflickr, text_mode="nope")


def test_set_dataset_text_mode_via_subset(fake_mirflickr):
    from src.data.loaders import MIRFlickr25kDCMHDataset, set_dataset_text_mode

    ds = MIRFlickr25kDCMHDataset(fake_mirflickr, text_mode="bow")

    class _Subset:
        def __init__(self, d):
            self.dataset = d

    wrapped = _Subset(ds)
    set_dataset_text_mode(wrapped, "raw")
    assert ds.text_mode == "raw"


def test_empty_prompt_fallback(fake_mirflickr):
    """Images whose tags file is empty still emit a non-empty prompt."""
    from src.data.loaders import MIRFlickr25kDCMHDataset

    (fake_mirflickr / "meta" / "tags" / "tags1.txt").write_text("")
    ds = MIRFlickr25kDCMHDataset(fake_mirflickr, text_mode="raw")
    # Find the index of im1 after sorting.
    idx = ds.image_ids.index(1)
    assert ds._prompts[idx] == MIRFlickr25kDCMHDataset.EMPTY_PROMPT
