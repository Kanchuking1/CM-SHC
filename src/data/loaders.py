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


def get_dataset(name: str, root_dir: str | os.PathLike, transform: Callable | None = None, **kwargs) -> Dataset:
    key = name.strip().lower().replace("_", "").replace("-", "")
    if key == "flickr8k":
        return Flickr8KDCMHDataset(root_dir, transform=transform, **kwargs)
    if key == "mirflickr25k":
        raise NotImplementedError("mirflickr25k: extend Flickr8KDCMHDataset or add new class in loaders.py")
    if key == "coco":
        raise NotImplementedError("coco: add COCOCrossModalDataset in loaders.py")
    raise ValueError(f"Unknown dataset {name!r}")
