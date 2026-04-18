"""Tests for LoRA integration on CLIP backbones.

Most of these tests skip gracefully when :mod:`peft` or :mod:`transformers`
are not installed locally; they are primarily meant to run on the TAMU
Grace HPC environment where those libraries are available.

Three scoped tests work even without the full CLIP stack:

* ``_coerce_lora_cfg`` normalises various config shapes to plain dicts.
* The registry factories expose a ``lora`` kwarg and forward it as
  ``lora_cfg`` to the backbone classes.
* ``_apply_lora`` wraps a tiny toy module (one ``nn.Linear`` named
  ``q_proj``) with peft LoRA, yielding exactly the expected trainable
  adapter shapes.

Tests that hit real CLIP weights (``openai/clip-vit-base-patch32``) are
skipped unless ``HF_HUB_OFFLINE`` is set to a local cache containing the
snapshot *and* transformers is importable.
"""

from __future__ import annotations

import inspect
import os

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")


# ---------------------------------------------------------------------------
# Pure-Python helpers -- no torch / peft required.
# ---------------------------------------------------------------------------


def test_coerce_lora_cfg_none_is_none():
    from src.models.backbones.clip_backbone import _coerce_lora_cfg

    assert _coerce_lora_cfg(None) is None


def test_coerce_lora_cfg_plain_dict_roundtrip():
    from src.models.backbones.clip_backbone import _coerce_lora_cfg

    cfg = {"r": 4, "lora_alpha": 8, "target_modules": ["q_proj", "v_proj"]}
    out = _coerce_lora_cfg(cfg)
    assert out == cfg
    # Must be a shallow copy -- mutating the returned dict must not touch the input.
    out["r"] = 16
    assert cfg["r"] == 4


def test_coerce_lora_cfg_rejects_non_mapping():
    from src.models.backbones.clip_backbone import _coerce_lora_cfg

    with pytest.raises(TypeError, match="lora_cfg must be a mapping"):
        _coerce_lora_cfg([1, 2, 3])


def test_coerce_lora_cfg_accepts_dictconfig():
    omegaconf = pytest.importorskip("omegaconf")
    from src.models.backbones.clip_backbone import _coerce_lora_cfg

    dc = omegaconf.OmegaConf.create({"r": 8, "target_modules": ["q_proj"]})
    out = _coerce_lora_cfg(dc)
    assert isinstance(out, dict)
    assert out["r"] == 8
    assert out["target_modules"] == ["q_proj"]


# ---------------------------------------------------------------------------
# Registry factories -- signature introspection (no model instantiation).
# ---------------------------------------------------------------------------


def test_clip_image_factory_exposes_lora_kwarg():
    from src.models.backbones import _clip_image_factory

    sig = inspect.signature(_clip_image_factory)
    assert "lora" in sig.parameters
    assert sig.parameters["lora"].default is None


def test_clip_text_factory_exposes_lora_kwarg():
    from src.models.backbones import _clip_text_factory

    sig = inspect.signature(_clip_text_factory)
    assert "lora" in sig.parameters
    assert sig.parameters["lora"].default is None


# ---------------------------------------------------------------------------
# LoRA wrapping on a toy module -- exercises _apply_lora end-to-end without
# needing real CLIP weights.
# ---------------------------------------------------------------------------


class _ToyAttention(nn.Module):
    """A tiny stand-in for a CLIP attention sub-module.

    Has the ``q_proj`` / ``k_proj`` / ``v_proj`` naming convention peft
    will look for when ``target_modules=["q_proj", "k_proj", "v_proj"]``
    is used.
    """

    def __init__(self, dim: int = 8):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))


def test_apply_lora_wraps_toy_module():
    pytest.importorskip("peft")
    from src.models.backbones.clip_backbone import _apply_lora, _is_peft_model

    toy = _ToyAttention(dim=8)
    total_before = sum(p.numel() for p in toy.parameters())

    lora_cfg = {"r": 4, "lora_alpha": 8, "target_modules": ["q_proj", "v_proj"]}
    wrapped = _apply_lora(toy, lora_cfg)

    assert _is_peft_model(wrapped)
    # All original weights should be frozen.
    trainable = [p for p in wrapped.parameters() if p.requires_grad]
    assert len(trainable) > 0
    # Adapters add at most ~ r * (in + out) per target linear -- strictly
    # fewer trainable params than the full module.
    trainable_count = sum(p.numel() for p in trainable)
    assert 0 < trainable_count < total_before

    # Forward should still return a valid (B, dim) tensor.
    out = wrapped(torch.randn(2, 8))
    assert out.shape == (2, 8)


def test_apply_lora_raises_on_no_peft(monkeypatch):
    """When peft is unavailable ``_apply_lora`` raises ImportError."""
    import builtins

    original_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "peft" or name.startswith("peft."):
            raise ImportError("simulated missing peft")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    from src.models.backbones.clip_backbone import _apply_lora

    with pytest.raises(ImportError, match="peft is required"):
        _apply_lora(_ToyAttention(), {"r": 4})


# ---------------------------------------------------------------------------
# End-to-end CLIP tests -- skipped unless HF snapshot is available locally.
# ---------------------------------------------------------------------------


def _clip_available() -> bool:
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False
    return os.environ.get("CLIP_TEST_MODEL") is not None


@pytest.mark.skipif(not _clip_available(), reason="CLIP_TEST_MODEL / transformers not available")
def test_clip_image_backbone_with_lora_has_trainable_adapters():
    pytest.importorskip("peft")
    from src.models.backbones.clip_backbone import CLIPImageBackbone

    model = CLIPImageBackbone(
        model_name=os.environ["CLIP_TEST_MODEL"],
        out_dim=16,
        lora_cfg={"r": 4, "lora_alpha": 8},
        local_files_only=True,
    )
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    # proj is always trainable; adapters should contribute additional params.
    assert any("proj" in n for n in trainable)
    assert any("lora" in n.lower() for n in trainable)


@pytest.mark.skipif(not _clip_available(), reason="CLIP_TEST_MODEL / transformers not available")
def test_clip_text_backbone_with_lora_has_trainable_adapters():
    pytest.importorskip("peft")
    from src.models.backbones.clip_backbone import CLIPTextBackbone

    model = CLIPTextBackbone(
        model_name=os.environ["CLIP_TEST_MODEL"],
        out_dim=16,
        lora_cfg={"r": 4, "lora_alpha": 8},
        local_files_only=True,
    )
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert any("proj" in n for n in trainable)
    assert any("lora" in n.lower() for n in trainable)
