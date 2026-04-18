"""Tests for :mod:`src.models.backbones` registry and legacy translators.

These tests avoid any pretrained weight download by exercising the MLP-on-BOW
text path and by monkey-patching the AlexNet/ResNet factories where needed.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


def test_text_registry_mlp_bow_roundtrip():
    from src.models.backbones import build_text_backbone, text_backend_tag

    cfg = {"name": "mlp_bow", "in_dim": 17}
    mod = build_text_backbone(cfg, out_dim=8)
    assert mod.encoder is None
    assert isinstance(mod.proj, nn.Module)
    x = torch.randn(3, 17)
    y = mod(x)
    assert y.shape == (3, 8)
    assert text_backend_tag(cfg) == "mlp"


def test_build_text_backbone_unknown_name_raises():
    from src.models.backbones import build_text_backbone

    with pytest.raises(ValueError, match="Unknown text backbone"):
        build_text_backbone({"name": "definitely-not-a-backbone"}, out_dim=4)


def test_build_image_backbone_unknown_name_raises():
    from src.models.backbones import build_image_backbone

    with pytest.raises(ValueError, match="Unknown image backbone"):
        build_image_backbone({"name": "definitely-not-a-backbone"}, out_dim=4)


def test_missing_name_key_raises():
    from src.models.backbones import build_image_backbone, build_text_backbone

    with pytest.raises(ValueError, match="missing required 'name' key"):
        build_image_backbone({}, out_dim=4)
    with pytest.raises(ValueError, match="missing required 'name' key"):
        build_text_backbone({"in_dim": 7}, out_dim=4)


def test_legacy_translators():
    from src.models.backbones import legacy_to_image_cfg, legacy_to_text_cfg

    assert legacy_to_image_cfg("alexnet") == {"name": "alexnet"}
    assert legacy_to_image_cfg("resnet50") == {"name": "resnet50"}

    assert legacy_to_text_cfg(text_feature_dim=1386, text_model_name="") == {
        "name": "mlp_bow",
        "in_dim": 1386,
    }
    cfg = legacy_to_text_cfg(
        text_feature_dim=None,
        text_model_name="bert-base-uncased",
        freeze_text_encoder=True,
        local_files_only=True,
    )
    assert cfg == {
        "name": "hf_transformer",
        "model_name": "bert-base-uncased",
        "freeze": True,
        "local_files_only": True,
    }


def test_register_new_text_backbone():
    """Downstream code (e.g. CLIP) should be able to register a new backbone."""
    from src.models.backbones import (
        TEXT_BACKBONE_REGISTRY,
        build_text_backbone,
        register_text_backbone,
    )

    name = "__test_linear__"

    class _LinearText(nn.Module):
        def __init__(self, out_dim: int, in_dim: int):
            super().__init__()
            self.encoder = None
            self.proj = nn.Linear(in_dim, out_dim)

        def forward(self, x):
            return self.proj(x)

    def _factory(out_dim, in_dim, **_):
        return _LinearText(out_dim=out_dim, in_dim=in_dim)

    try:
        register_text_backbone(name, _factory)
        mod = build_text_backbone({"name": name, "in_dim": 4}, out_dim=3)
        assert mod(torch.randn(2, 4)).shape == (2, 3)
        # Re-registering the same factory object is idempotent.
        register_text_backbone(name, _factory)
        # Registering a different factory under the same name is rejected.
        with pytest.raises(ValueError):
            register_text_backbone(name, lambda out_dim, **_: nn.Linear(1, out_dim))
    finally:
        TEXT_BACKBONE_REGISTRY.pop(name, None)


def test_dcmh_legacy_kwargs_still_work():
    """The DCMH constructor must accept the old kwargs without changes."""
    from src.models.hashing.dcmh import DCMH

    m = DCMH(32, text_model_name="", text_feature_dim=64, image_backbone="alexnet")
    assert m.bit_dim == 32
    assert m.text_backend == "mlp"
    assert m.text_encoder is None
    assert isinstance(m.text_proj, nn.Module)
    # Image backbone config is captured for introspection.
    assert m.image_cfg == {"name": "alexnet"}
    assert m.text_cfg == {"name": "mlp_bow", "in_dim": 64}
    # Forward path for the text tower (avoids downloading AlexNet weights
    # by going directly through the projection head).
    y = m.encode_text(text_features=torch.randn(2, 64))
    assert y.shape == (2, 32)


def test_cmshc_legacy_kwargs_still_work():
    from src.models.hashing.cm_shc import CMSHC

    m = CMSHC(16, text_model_name="", text_feature_dim=32)
    assert m.bit_dim == 16
    assert m.text_backend == "mlp"
    assert m.text_cfg == {"name": "mlp_bow", "in_dim": 32}
    y = m.encode_text(text_features=torch.randn(2, 32))
    assert y.shape == (2, 16)


def test_dcmh_config_driven_construction():
    """When ``image_cfg`` / ``text_cfg`` are passed they override legacy kwargs."""
    from src.models.hashing.dcmh import DCMH

    # Pass a bogus legacy ``image_backbone`` but a valid explicit cfg; the
    # explicit cfg must win.
    m = DCMH(
        8,
        image_backbone="nonexistent-should-be-ignored",
        text_cfg={"name": "mlp_bow", "in_dim": 5},
        image_cfg={"name": "alexnet"},
    )
    assert m.image_cfg == {"name": "alexnet"}
    assert m.text_cfg == {"name": "mlp_bow", "in_dim": 5}
    assert m.text_backend == "mlp"
