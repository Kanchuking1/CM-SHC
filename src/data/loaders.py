"""Dataset registry and concrete implementations."""

from __future__ import annotations

import os
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class MIRFlickr25kDCMHDataset(Dataset):
    """MIR-Flickr-25k for DCMH: image tensor + user tags as text.

    Expected layout under *root_dir*::

        im1.jpg  im2.jpg  ...  im25000.jpg
        meta/
          tags/              ← one file per image, one tag per line
            tags1.txt  tags2.txt  ...  tags25000.txt
        annotations/         ← 24-class potential labels from official download
            sky.txt  clouds.txt  ...  (one image-id per line)

    Tags are joined with ``", "`` to form a pseudo-caption fed to the text
    encoder.  Images without a tags file (or with an empty file) get a
    fallback string so the tokenizer never receives an empty input.

    Only images with **at least one** annotation are kept (the paper filters
    to annotated images).
    """

    ANNOTATION_CLASSES: list[str] = [
        "animals", "baby", "bird", "car", "clouds", "dog", "female",
        "flower", "food", "indoor", "lake", "male", "night", "people",
        "plant_life", "portrait", "river", "sea", "sky", "structures",
        "sunset", "transport", "tree", "water",
    ]
    NUM_CLASSES = len(ANNOTATION_CLASSES)
    BOW_VOCAB_FILE = "doc/common_tags.txt"

    def __init__(
        self,
        root_dir: str | os.PathLike,
        transform: Callable | None = None,
        **kwargs,
    ):
        self.root_dir = os.fspath(root_dir)
        self.transform = transform
        self.tags_dir = os.path.join(self.root_dir, "meta", "tags")
        self.ann_dir = os.path.join(self.root_dir, "annotations")

        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(
                f"MIRFlickr25k root not found: {self.root_dir!r}  "
                "Set dataset.root in configs/dataset/mirflickr25k.yaml or "
                "export MIRFLICKR_ROOT=/path/to/mirflickr"
            )
        if not os.path.isdir(self.ann_dir):
            raise FileNotFoundError(
                f"Annotations dir not found: {self.ann_dir!r}  "
                "Download mirflickr25k_annotations_v080.zip from "
                "http://press.liacs.nl/mirflickr/mirdownload.html "
                "and extract into {self.root_dir}/annotations/"
            )

        all_ids: list[int] = []
        for f in os.listdir(self.root_dir):
            if f.startswith("im") and f.lower().endswith(".jpg"):
                try:
                    all_ids.append(int(f[2:].split(".")[0]))
                except ValueError:
                    continue

        if not all_ids:
            raise FileNotFoundError(
                f"No im*.jpg files found in {self.root_dir!r}  "
                "Expected layout: im1.jpg, im2.jpg, …, im25000.jpg"
            )

        max_id = max(all_ids)
        full_labels = torch.zeros(max_id + 1, self.NUM_CLASSES, dtype=torch.float32)
        for cls_idx, cls_name in enumerate(self.ANNOTATION_CLASSES):
            ann_path = os.path.join(self.ann_dir, f"{cls_name}.txt")
            if not os.path.isfile(ann_path):
                continue
            with open(ann_path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            full_labels[int(line), cls_idx] = 1.0
                        except (ValueError, IndexError):
                            continue

        all_ids_set = set(all_ids)
        self.image_ids: list[int] = sorted(
            img_id for img_id in all_ids_set
            if full_labels[img_id].sum() > 0
        )

        self._labels = full_labels[self.image_ids]  # (N_filtered, 24)

        vocab, self._bow = self._build_bow()
        self.bow_vocab: list[str] = vocab
        self.bow_dim: int = len(vocab)

    @staticmethod
    def _read_tags_file(path: str) -> list[str]:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                return [line.strip().lower() for line in f if line.strip()]
        except FileNotFoundError:
            return []

    def _build_bow(self) -> tuple[list[str], torch.Tensor]:
        """Load 1386-word vocabulary from ``doc/common_tags.txt`` and build BOW matrix."""
        vocab_path = os.path.join(self.root_dir, "doc", "common_tags.txt")
        vocab: list[str] = []
        with open(vocab_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    vocab.append(parts[0].lower())
        word2idx = {w: i for i, w in enumerate(vocab)}
        bow = torch.zeros(len(self.image_ids), len(vocab), dtype=torch.float32)
        for row, img_id in enumerate(self.image_ids):
            tag_path = os.path.join(self.tags_dir, f"tags{img_id}.txt")
            for tag in self._read_tags_file(tag_path):
                idx = word2idx.get(tag)
                if idx is not None:
                    bow[row, idx] = 1.0
        return vocab, bow

    def get_label(self, index: int) -> torch.Tensor:
        return self._labels[index]

    def __getitem__(self, index: int) -> dict:
        img_id = self.image_ids[index]
        fname = f"im{img_id}.jpg"
        image_path = os.path.join(self.root_dir, fname)
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self._labels[index]
        return {
            "index": index,
            "img": image,
            "label": label,
            "text_features": self._bow[index],
        }

    def __len__(self) -> int:
        return len(self.image_ids)


def get_dataset(name: str, root_dir: str | os.PathLike, transform: Callable | None = None, **kwargs) -> Dataset:
    key = name.strip().lower().replace("_", "").replace("-", "")
    if key == "mirflickr25k":
        kwargs.pop("num_pseudo_classes", None)
        return MIRFlickr25kDCMHDataset(root_dir, transform=transform, **kwargs)
    if key == "coco":
        raise NotImplementedError("coco: add COCOCrossModalDataset in loaders.py")
    raise ValueError(f"Unknown dataset {name!r}")
