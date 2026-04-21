"""Dataset transforms."""

from __future__ import annotations

import torchvision.transforms as T


def imagenet_train_transform(image_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
