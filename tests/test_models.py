"""Smoke tests for importability."""

from __future__ import annotations


def test_dcmh_constructible():
    from src.models.hashing.dcmh import DCMH

    # MLP text path avoids downloading HF weights in CI/smoke tests
    m = DCMH(32, text_model_name="", text_feature_dim=64)
    assert m.bit_dim == 32


def test_config_compose():
    from pathlib import Path

    from src.utils.config import configs_dir, load_experiment

    exp = configs_dir() / "experiments" / "exp_dcmh_flickr8k.yaml"
    assert Path(exp).is_file()
    cfg = load_experiment(exp)
    assert cfg.model.name == "dcmh"
