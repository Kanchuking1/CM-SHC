"""model_cache path helpers."""

from __future__ import annotations

from pathlib import Path

from src.utils.model_paths import hf_repo_dir_name, hf_snapshot_looks_complete, resolve_pretrained_ref


def test_hf_repo_dir_name():
    assert hf_repo_dir_name("org/model") == "org__model"


def test_resolve_uses_local_when_complete(tmp_path):
    repo = "dummy/repo"
    dest = tmp_path / "hf" / hf_repo_dir_name(repo)
    dest.mkdir(parents=True)
    (dest / "config.json").write_text("{}")
    ref, lfo = resolve_pretrained_ref(repo, tmp_path, offline=False)
    assert Path(ref) == dest
    assert lfo is True


def test_hf_snapshot_looks_complete(tmp_path):
    assert hf_snapshot_looks_complete(tmp_path) is False
    (tmp_path / "config.json").write_text("{}")
    assert hf_snapshot_looks_complete(tmp_path) is True
