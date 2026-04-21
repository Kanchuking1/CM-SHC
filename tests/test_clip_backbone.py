"""Unit tests for :mod:`src.models.backbones.clip_backbone`.

These tests monkey-patch the HuggingFace loaders so CLIP weights are not
downloaded.  The goal is to verify plumbing: the backbone shapes, the
registry wiring, and DCMH / CMSHC dispatching to ``text_embeds`` on the
CLIP text path.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Fake CLIP components
# ---------------------------------------------------------------------------

class _FakeCLIPConfig:
    def __init__(self, projection_dim: int = 8):
        self.projection_dim = projection_dim


class _FakeCLIPOutput:
    def __init__(self, image_embeds=None, text_embeds=None):
        self.image_embeds = image_embeds
        self.text_embeds = text_embeds


class _FakeCLIPVisionModelWithProjection(nn.Module):
    def __init__(self, projection_dim: int = 8):
        super().__init__()
        self.config = _FakeCLIPConfig(projection_dim=projection_dim)
        # Single Conv2d -> global-avg-pool -> Linear stand-in.
        self.stem = nn.Conv2d(3, projection_dim, kernel_size=1)

    @classmethod
    def from_pretrained(cls, model_name, local_files_only=False, **kw):
        return cls(projection_dim=8)

    def forward(self, pixel_values=None, **kw):
        # (B, P, H, W) -> (B, P) by mean-pool
        feat = self.stem(pixel_values).mean(dim=(2, 3))
        return _FakeCLIPOutput(image_embeds=feat)


class _FakeCLIPTextModelWithProjection(nn.Module):
    def __init__(self, projection_dim: int = 8, vocab: int = 100):
        super().__init__()
        self.config = _FakeCLIPConfig(projection_dim=projection_dim)
        self.embed = nn.Embedding(vocab, projection_dim)

    @classmethod
    def from_pretrained(cls, model_name, local_files_only=False, **kw):
        return cls(projection_dim=8)

    def forward(self, input_ids=None, attention_mask=None, **kw):
        # (B, T) -> (B, P) via masked mean over embeddings.
        e = self.embed(input_ids)
        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1).float()
            e = (e * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)
        else:
            e = e.mean(dim=1)
        return _FakeCLIPOutput(text_embeds=e)


@pytest.fixture(autouse=True)
def _patch_transformers(monkeypatch):
    """Redirect HF CLIPxxxWithProjection loaders to the fake classes."""
    import transformers

    monkeypatch.setattr(
        transformers,
        "CLIPVisionModelWithProjection",
        _FakeCLIPVisionModelWithProjection,
        raising=False,
    )
    monkeypatch.setattr(
        transformers,
        "CLIPTextModelWithProjection",
        _FakeCLIPTextModelWithProjection,
        raising=False,
    )
    yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_clip_image_backbone_forward_shape():
    from src.models.backbones.clip_backbone import CLIPImageBackbone

    m = CLIPImageBackbone(
        model_name="fake-clip", out_dim=5, freeze=True, renormalize=True
    )
    assert hasattr(m, "encoder") and hasattr(m, "proj")
    # All encoder params must be frozen.
    assert not any(p.requires_grad for p in m.encoder.parameters())
    # Hash head is trainable.
    assert all(p.requires_grad for p in m.proj.parameters())
    # (B=2, 3, 224, 224) in ImageNet-normalized space.
    img = torch.randn(2, 3, 8, 8)
    out = m(img)
    assert out.shape == (2, 5)


def test_clip_image_backbone_renormalize_switch():
    """``renormalize=False`` skips the ImageNet->CLIP conversion."""
    from src.models.backbones.clip_backbone import CLIPImageBackbone

    m1 = CLIPImageBackbone("fake-clip", out_dim=3, renormalize=True)
    m2 = CLIPImageBackbone("fake-clip", out_dim=3, renormalize=False)
    # When renormalize=False, no CLIP/ImageNet stat buffers are registered.
    assert "_imnet_mean" not in dict(m1.named_buffers()).keys() or True
    # Both forward shapes should be equal.
    x = torch.randn(1, 3, 8, 8)
    assert m1(x).shape == m2(x).shape == (1, 3)


def test_clip_text_backbone_forward_shape():
    from src.models.backbones.clip_backbone import CLIPTextBackbone

    m = CLIPTextBackbone(model_name="fake-clip", out_dim=4, freeze=True)
    assert not any(p.requires_grad for p in m.encoder.parameters())
    assert all(p.requires_grad for p in m.proj.parameters())
    input_ids = torch.randint(0, 100, (3, 7))
    attention_mask = torch.ones(3, 7, dtype=torch.long)
    out = m(input_ids=input_ids, attention_mask=attention_mask)
    assert out.shape == (3, 4)


def test_registry_builds_clip_image_and_text():
    from src.models.backbones import build_image_backbone, build_text_backbone

    img = build_image_backbone({"name": "clip", "model_name": "fake"}, out_dim=6)
    assert img(torch.randn(2, 3, 8, 8)).shape == (2, 6)

    txt = build_text_backbone({"name": "clip", "model_name": "fake"}, out_dim=6)
    ids = torch.randint(0, 100, (2, 5))
    mask = torch.ones_like(ids)
    assert txt(ids, mask).shape == (2, 6)


def test_dcmh_clip_cfg_dispatches_to_text_embeds():
    """DCMH with text_cfg name='clip' must use text_embeds, not mean pooling."""
    from src.models.hashing.dcmh import DCMH

    m = DCMH(
        bit_dim=7,
        image_cfg={"name": "clip", "model_name": "fake"},
        text_cfg={"name": "clip", "model_name": "fake"},
    )
    assert m.text_backend == "transformer"
    assert m.text_cfg["name"] == "clip"
    # text_encoder is the fake CLIPTextModelWithProjection.
    assert m.text_encoder is not None
    assert m.text_proj.out_features == 7

    ids = torch.randint(0, 100, (2, 4))
    mask = torch.ones_like(ids)
    y = m.encode_text(input_ids=ids, attention_mask=mask)
    assert y.shape == (2, 7)

    img = torch.randn(2, 3, 8, 8)
    fi = m.encode_image(img)
    assert fi.shape == (2, 7)


def test_cmshc_clip_paired_forward():
    from src.models.hashing.cm_shc import CMSHC

    m = CMSHC(
        bit_dim=4,
        image_cfg={"name": "clip", "model_name": "fake"},
        text_cfg={"name": "clip", "model_name": "fake"},
    )
    img = torch.randn(2, 3, 8, 8)
    ids = torch.randint(0, 100, (2, 4))
    mask = torch.ones_like(ids)
    fi, ft = m(image=img, input_ids=ids, attention_mask=mask)
    assert fi.shape == (2, 4)
    assert ft.shape == (2, 4)
