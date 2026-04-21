"""Resolve pretrained snapshot paths under ``model_cache`` and offline flags."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_OFFLINE_ENV = "CM_SHC_OFFLINE"


def hf_repo_dir_name(repo_id: str) -> str:
    """Stable folder name without slashes (for local snapshot dirs)."""
    return repo_id.replace("/", "__").replace("\\", "_")


def hf_snapshot_dir(repo_id: str, cache_root: Path | str) -> Path:
    root = Path(cache_root)
    return root / "hf" / hf_repo_dir_name(repo_id)


def torch_home_dir(cache_root: Path | str) -> Path:
    """TORCH_HOME for torchvision / torch hub checkpoints under model_cache."""
    return Path(cache_root) / "torch"


def local_files_only(cfg: Any) -> bool:
    if os.environ.get(_OFFLINE_ENV, "").lower() in ("1", "true", "yes"):
        return True
    flag = OmegaConf_select(cfg, "paths.offline_mode")
    if flag is None:
        return False
    return bool(flag)


def OmegaConf_select(cfg: Any, key: str):
    try:
        from omegaconf import OmegaConf

        return OmegaConf.select(cfg, key)
    except Exception:
        return None


def resolve_hf_pretrained_path(cfg: Any) -> str | None:
    """
    Local directory for HF text encoder + tokenizer, or None if MLP text path (no HF).
    """
    tdim = OmegaConf_select(cfg, "model.text_feature_dim")
    if tdim is not None:
        return None
    rid = OmegaConf_select(cfg, "model.backbone.text")
    if not rid:
        return None
    cache = OmegaConf_select(cfg, "paths.model_cache")
    if not cache:
        raise ValueError("paths.model_cache must be set in config for HF pretrained resolution.")
    return str(hf_snapshot_dir(str(rid), Path(cache)))


def assert_hf_cache_ready(local_dir: Path | str, offline: bool) -> None:
    """Fail fast if offline and snapshot is missing or incomplete."""
    p = Path(local_dir)
    if not p.is_dir():
        if offline:
            raise FileNotFoundError(
                f"HF model cache missing: {p}. Run on a machine with internet:\n"
                f"  python -m src.pipelines.download_models --config <experiment.yaml>"
            )
        return
    if not (p / "config.json").is_file():
        if offline:
            raise FileNotFoundError(
                f"HF cache incomplete (no config.json): {p}. Re-run download_models."
            )


def hf_snapshot_looks_complete(local_dir: Path) -> bool:
    if not local_dir.is_dir():
        return False
    return (local_dir / "config.json").is_file()


def resolve_pretrained_ref(repo_id: str, cache_root: Path | str, offline: bool) -> tuple[str, bool]:
    """
    Return (path_or_id, local_files_only) for ``from_pretrained``.
    Prefer a complete local snapshot; otherwise use the hub id unless ``offline`` forces cache.
    """
    root = Path(cache_root)
    dest = hf_snapshot_dir(repo_id, root)
    if hf_snapshot_looks_complete(dest):
        return str(dest), True
    if offline:
        assert_hf_cache_ready(dest, True)
        return str(dest), True
    return repo_id, False
