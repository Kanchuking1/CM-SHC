"""Regression tests for the text-backbone resolution used by
:mod:`src.pipelines.inference_utils` (and shared with
:mod:`src.pipelines.train`).

The bug these tests lock in: when ``cfg.model.backbone.text`` is a
CLIP-style dict like ``{name: clip, model_name: openai/clip-vit-base-patch32,
freeze: true}``, the previous code in ``inference_utils`` stringified the
whole dict and passed the repr into ``resolve_pretrained_ref`` as the
"repo id", producing a bogus cache path.  The fix is to route both
evaluate/retrieve and train through a single ``resolve_text_backbone_spec``
helper that pulls ``model_name`` out of dict form and also flags CLIP
text so the right tokenizer is loaded.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.pipelines.train import (
    _backbone_field_repr,
    resolve_text_backbone_spec,
    resolve_text_repo,
    text_backbone_name,
)


# ---------------------------------------------------------------------------
# _backbone_field_repr / resolve_text_repo / text_backbone_name
# ---------------------------------------------------------------------------


def test_resolve_text_repo_plain_string():
    assert resolve_text_repo("huawei-noah/TinyBERT_General_4L_312D") == (
        "huawei-noah/TinyBERT_General_4L_312D"
    )


def test_resolve_text_repo_none():
    assert resolve_text_repo(None) == ""


def test_resolve_text_repo_plain_dict():
    d = {"name": "clip", "model_name": "openai/clip-vit-base-patch32", "freeze": True}
    assert resolve_text_repo(d) == "openai/clip-vit-base-patch32"


def test_resolve_text_repo_dict_missing_model_name():
    assert resolve_text_repo({"name": "clip"}) == ""


def test_resolve_text_repo_omegaconf_dictconfig():
    dc = OmegaConf.create({"name": "clip", "model_name": "openai/clip-vit-base-patch32"})
    assert resolve_text_repo(dc) == "openai/clip-vit-base-patch32"


def test_text_backbone_name_detects_clip_dict():
    assert text_backbone_name({"name": "clip", "model_name": "x"}) == "clip"
    assert text_backbone_name(OmegaConf.create({"name": "clip", "model_name": "x"})) == "clip"


def test_text_backbone_name_empty_for_strings_and_none():
    # Legacy-string backbones have no explicit registry name.
    assert text_backbone_name("huawei-noah/TinyBERT_General_4L_312D") == ""
    assert text_backbone_name(None) == ""


def test_backbone_field_repr_serializable():
    # String stays string; DictConfig collapses to plain dict (JSON-friendly).
    assert _backbone_field_repr("alexnet") == "alexnet"
    assert _backbone_field_repr(None) is None
    dc = OmegaConf.create({"name": "clip", "model_name": "x"})
    out = _backbone_field_repr(dc)
    assert isinstance(out, dict)
    assert out["name"] == "clip" and out["model_name"] == "x"


# ---------------------------------------------------------------------------
# resolve_text_backbone_spec -- the function evaluate/retrieve/train share
# ---------------------------------------------------------------------------


def _base_cfg(model_block: dict, cache_root: Path) -> OmegaConf:
    """Minimal cfg with just the fields resolve_text_backbone_spec touches."""
    cfg = OmegaConf.create(
        {
            "paths": {"model_cache": str(cache_root), "offline_mode": False},
            "model": model_block,
        }
    )
    return cfg


def test_spec_returns_mlp_path_for_bow_configs(tmp_path):
    """text_feature_dim set -> MLP path, no HF resolution at all."""
    cfg = _base_cfg(
        {"text_feature_dim": 1386, "backbone": {"image": "alexnet", "text": "anything"}},
        tmp_path,
    )
    text_ref, hf_lfo, is_clip = resolve_text_backbone_spec(cfg)
    assert text_ref == ""
    assert hf_lfo is False
    assert is_clip is False


def test_spec_resolves_legacy_string_text_backbone(tmp_path, monkeypatch):
    cfg = _base_cfg(
        {
            "text_feature_dim": None,
            "backbone": {
                "image": "resnet50",
                "text": "huawei-noah/TinyBERT_General_4L_312D",
            },
        },
        tmp_path,
    )

    calls: list[tuple[str, Path, bool]] = []

    def fake_resolve(repo_id, cache_root, offline):
        calls.append((repo_id, Path(cache_root), bool(offline)))
        return str(Path(cache_root) / "hf" / "stub"), True

    monkeypatch.setattr(
        "src.pipelines.train.resolve_pretrained_ref", fake_resolve, raising=True
    )

    text_ref, hf_lfo, is_clip = resolve_text_backbone_spec(cfg)
    assert calls and calls[0][0] == "huawei-noah/TinyBERT_General_4L_312D"
    assert hf_lfo is True
    assert is_clip is False
    assert Path(text_ref).name == "stub"


def test_spec_extracts_model_name_from_clip_dict(tmp_path, monkeypatch):
    """The pre-fix bug: whole dict was stringified into resolve_pretrained_ref.

    We patch ``resolve_pretrained_ref`` and assert the *first* argument it
    receives is exactly the HF repo id string, not ``str(dict)``.
    """
    cfg = _base_cfg(
        {
            "text_feature_dim": None,
            "backbone": {
                "image": {
                    "name": "clip",
                    "model_name": "openai/clip-vit-base-patch32",
                    "freeze": True,
                    "local_files_only": "auto",
                },
                "text": {
                    "name": "clip",
                    "model_name": "openai/clip-vit-base-patch32",
                    "freeze": True,
                    "local_files_only": "auto",
                },
            },
        },
        tmp_path,
    )

    captured: list[str] = []

    def fake_resolve(repo_id, cache_root, offline):
        # The whole test is about this: we must receive a clean repo id.
        assert isinstance(repo_id, str)
        assert "{" not in repo_id and "}" not in repo_id, (
            f"regression: dict-form backbone got stringified -- repo_id={repo_id!r}"
        )
        captured.append(repo_id)
        return str(Path(cache_root) / "hf" / repo_id.replace("/", "__")), True

    monkeypatch.setattr(
        "src.pipelines.train.resolve_pretrained_ref", fake_resolve, raising=True
    )

    text_ref, hf_lfo, is_clip = resolve_text_backbone_spec(cfg)
    assert captured == ["openai/clip-vit-base-patch32"]
    assert hf_lfo is True
    assert is_clip is True
    assert text_ref.endswith("openai__clip-vit-base-patch32")


def test_spec_raises_when_text_backbone_is_missing(tmp_path):
    cfg = _base_cfg(
        {"text_feature_dim": None, "backbone": {"image": "alexnet", "text": None}},
        tmp_path,
    )
    with pytest.raises(ValueError, match="Text backbone repo id is empty"):
        resolve_text_backbone_spec(cfg)


def test_spec_raises_when_dict_has_no_model_name(tmp_path):
    cfg = _base_cfg(
        {
            "text_feature_dim": None,
            "backbone": {"image": "alexnet", "text": {"name": "clip"}},
        },
        tmp_path,
    )
    with pytest.raises(ValueError, match="Text backbone repo id is empty"):
        resolve_text_backbone_spec(cfg)


# ---------------------------------------------------------------------------
# load_model_and_dataset_for_eval: smoke test that the tokenizer branch
# picks CLIPTokenizer for CLIP dict configs.  We stop short of actually
# calling build_model so the test stays fast and offline.
# ---------------------------------------------------------------------------


def test_eval_loader_picks_clip_tokenizer_for_clip_dict(tmp_path, monkeypatch):
    """When backbone.text is a CLIP dict, inference_utils should reach for
    ``load_clip_tokenizer`` (not ``load_hf_tokenizer``).

    We stub the rest of ``load_model_and_dataset_for_eval`` -- we only
    care that the tokenizer dispatch picks the CLIP path and that the
    collate fn is built successfully.
    """
    import src.pipelines.inference_utils as iu

    cfg = OmegaConf.create(
        {
            "paths": {"model_cache": str(tmp_path), "offline_mode": False},
            "device": "cpu",
            "dataset": {
                "name": "mirflickr25k",
                "root": str(tmp_path),
                "caption_max_length": 512,
            },
            "model": {
                "text_feature_dim": None,
                "name": "cm_shc",
                "bit_dim": 64,
                "backbone": {
                    "image": {
                        "name": "clip",
                        "model_name": "openai/clip-vit-base-patch32",
                        "freeze": True,
                    },
                    "text": {
                        "name": "clip",
                        "model_name": "openai/clip-vit-base-patch32",
                        "freeze": True,
                    },
                },
            },
        }
    )

    # ---- stubs -----------------------------------------------------------

    def fake_resolve(repo_id, cache_root, offline):
        return str(Path(cache_root) / "hf" / "stub"), True

    class FakeTok:
        model_max_length = 77

        def __call__(self, texts, **kw):
            raise NotImplementedError  # not exercised

    picked: dict[str, str] = {}

    def fake_load_clip_tokenizer(model_id, local_files_only=False):
        picked["clip"] = model_id
        return FakeTok()

    def fake_load_hf_tokenizer(model_id, local_files_only=False):
        picked["hf"] = model_id
        return FakeTok()

    class FakeDS:
        def __len__(self):  # pragma: no cover - never iterated
            return 0

    def fake_get_dataset(name, root_dir, transform, **kwargs):
        return FakeDS()

    class FakeModel:
        def to(self, device):
            return self

    def fake_build_model(cfg_, text_ref, hf_local_files_only):
        return FakeModel()

    def fake_load_weights(model, ckpt_path, device):
        return 0, {"stubbed": True}

    monkeypatch.setattr(
        "src.pipelines.train.resolve_pretrained_ref", fake_resolve, raising=True
    )
    monkeypatch.setattr(iu, "load_clip_tokenizer", fake_load_clip_tokenizer)
    monkeypatch.setattr(iu, "load_hf_tokenizer", fake_load_hf_tokenizer)
    monkeypatch.setattr(iu, "get_dataset", fake_get_dataset)
    monkeypatch.setattr(iu, "build_model", fake_build_model)
    monkeypatch.setattr(iu, "load_model_weights_only", fake_load_weights)

    cfg_out, model, collate, ds, device, epoch, meta = (
        iu.load_model_and_dataset_for_eval(
            config_path="unused.yaml", checkpoint_path="unused.pt", cfg=cfg
        )
    )

    # CLIP tokenizer path won; HF path was never called.
    assert "clip" in picked and "hf" not in picked
    assert picked["clip"].endswith("stub")
    # max_length is clamped to CLIP's 77-token cap even though the config
    # asked for 512.  We can't inspect the closure directly, but the
    # collate_fn callable was built successfully.
    assert callable(collate)
    assert epoch == 0
    assert meta == {"stubbed": True}
