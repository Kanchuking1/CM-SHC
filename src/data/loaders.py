"""Dataset registry and concrete implementations."""

from __future__ import annotations

import os
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .preprocessing import hash_filename_label, image_caption_first_map


class Flickr8KDataset(Dataset):
    """Legacy: PIL image + caption string."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_dir = os.path.join(root_dir, "Images")
        self.label_file = os.path.join(root_dir, "captions.txt")
        self.data_files = sorted(os.listdir(self.data_dir))
        self.labels = pd.read_csv(self.label_file)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_dir, self.data_files[index])
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels.iloc[index]["caption"]

    def __len__(self):
        return len(self.data_files)

class Flickr8KDCMHDataset(Dataset):
    """Flickr8k samples as dicts for DCMH (index, img tensor, label, text)."""

    def __init__(
        self,
        root_dir: str | os.PathLike,
        transform: Callable | None = None,
        num_pseudo_classes: int = 256,
    ):
        self.root_dir = os.fspath(root_dir)
        self.transform = transform
        self.num_pseudo_classes = int(num_pseudo_classes)
        self.data_dir = os.path.join(self.root_dir, "Images")
        self.label_file = os.path.join(self.root_dir, "captions.txt")
        self.data_files = sorted(
            f for f in os.listdir(self.data_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        self._df = pd.read_csv(self.label_file)
        self._cap_by_image = image_caption_first_map(self._df)

    def _caption_for(self, index: int, filename: str) -> str:
        if self._cap_by_image is not None:
            cap = self._cap_by_image.get(filename)
            if cap is None:
                base = filename.rsplit(".", 1)[0]
                cap = self._cap_by_image.get(base + ".jpg") or self._cap_by_image.get(base + ".jpeg")
            if cap is not None:
                return cap
        cap_col = self._df.columns[
            self._df.columns.str.lower().str.contains("caption|text", case=False, regex=True)
        ]
        col = cap_col[0] if len(cap_col) else self._df.columns[-1]
        return self._df.iloc[index][col]

    def __getitem__(self, index: int) -> dict:
        fname = self.data_files[index]
        image_path = os.path.join(self.data_dir, fname)
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        caption = self._caption_for(index, fname)
        label = hash_filename_label(fname, self.num_pseudo_classes)
        return {
            "index": index,
            "img": image,
            "label": torch.tensor(label, dtype=torch.long),
            "text": str(caption),
        }

    def __len__(self) -> int:
        return len(self.data_files)

class MIRFlickr25kDCMHDataset(Dataset):
    """MIR-Flickr-25k for DCMH: image tensor + user tags as text.

    Expected layout under *root_dir*::

        im1.jpg  im2.jpg  ...  im25000.jpg
        meta/
          tags_raw/          ← one file per image, one tag per line
            tags1.txt  tags2.txt  ...  tags25000.txt

    Tags are joined with ``", "`` to form a pseudo-caption fed to the text
    encoder.  Images without a tags file (or with an empty file) get a
    fallback string so the tokenizer never receives an empty input.
    """

    FALLBACK_TEXT = "no tags"

    def __init__(
        self,
        root_dir: str | os.PathLike,
        transform: Callable | None = None,
        num_pseudo_classes: int = 256,
    ):
        self.root_dir = os.fspath(root_dir)
        self.transform = transform
        self.num_pseudo_classes = int(num_pseudo_classes)
        self.tags_dir = os.path.join(self.root_dir, "meta", "tags_raw")

        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(
                f"MIRFlickr25k root not found: {self.root_dir!r}  "
                "Set dataset.root in configs/dataset/mirflickr25k.yaml or "
                "export MIRFLICKR_ROOT=/path/to/mirflickr"
            )

        self.image_ids: list[int] = []
        for f in os.listdir(self.root_dir):
            if f.startswith("im") and f.lower().endswith(".jpg"):
                try:
                    self.image_ids.append(int(f[2:].split(".")[0]))
                except ValueError:
                    continue
        self.image_ids.sort()

        if not self.image_ids:
            raise FileNotFoundError(
                f"No im*.jpg files found in {self.root_dir!r}  "
                "Expected layout: im1.jpg, im2.jpg, …, im25000.jpg"
            )

        self._tags_cache: dict[int, str] = {}

    def _load_tags(self, img_id: int) -> str:
        if img_id in self._tags_cache:
            return self._tags_cache[img_id]
        tag_path = os.path.join(self.tags_dir, f"tags{img_id}.txt")
        try:
            with open(tag_path, encoding="utf-8", errors="replace") as f:
                tags = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            tags = []
        text = ", ".join(tags) if tags else self.FALLBACK_TEXT
        self._tags_cache[img_id] = text
        return text

    def __getitem__(self, index: int) -> dict:
        img_id = self.image_ids[index]
        fname = f"im{img_id}.jpg"
        image_path = os.path.join(self.root_dir, fname)
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        text = self._load_tags(img_id)
        label = hash_filename_label(fname, self.num_pseudo_classes)
        return {
            "index": index,
            "img": image,
            "label": torch.tensor(label, dtype=torch.long),
            "text": text,
        }

    def __len__(self) -> int:
        return len(self.image_ids)


def get_dataset(name: str, root_dir: str | os.PathLike, transform: Callable | None = None, **kwargs) -> Dataset:
    key = name.strip().lower().replace("_", "").replace("-", "")
    if key == "flickr8k":
        return Flickr8KDCMHDataset(root_dir, transform=transform, **kwargs)
    if key == "mirflickr25k":
        return MIRFlickr25kDCMHDataset(root_dir, transform=transform, **kwargs)
    if key == "coco":
        raise NotImplementedError("coco: add COCOCrossModalDataset in loaders.py")
    raise ValueError(f"Unknown dataset {name!r}")
