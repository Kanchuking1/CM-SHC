"""Image CNN backbones for cross-modal hashing."""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as models


class ResNet50ImageEncoder(nn.Module):
    """ImageNet ResNet-50 with final linear to ``out_dim`` hash features."""

    def __init__(self, out_dim: int):
        super().__init__()
        try:
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except AttributeError:
            backbone = models.resnet50(pretrained=True)
        in_f = backbone.fc.in_features
        backbone.fc = nn.Linear(in_f, out_dim)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)
