"""Load and merge experiment configs (OmegaConf, Hydra-free)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

CONFIG_DIRNAME = "configs"

_ENV_OVERRIDES: dict[str, str] = {
    "MIRFLICKR_ROOT": "dataset.root",
    "FLICKR8K_ROOT": "dataset.root",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def configs_dir() -> Path:
    return repo_root() / CONFIG_DIRNAME


def load_experiment(path: str | Path) -> Any:
    """
    Merge: base.yaml + model/{name}.yaml + dataset/{name}.yaml + experiment overrides.

    The experiment file selects the model / dataset YAMLs via top-level keys
    ``model`` and ``dataset``. Two accepted shapes per key:

    * **Scalar** -- ``model: dcmh`` / ``dataset: mirflickr25k``. The value is
      the filename stem under ``configs/model/`` or ``configs/dataset/``.
    * **Mapping with ``name:`` subkey** -- lets an experiment override nested
      model (or dataset) fields without duplicating the top-level key::

          model:
            name: cm_shc
            bit_dim: 128
            similarity_method: classifier

      Any keys besides ``name`` are kept as overrides and merged after the
      base model YAML.
    """
    path = Path(path)
    if not path.is_absolute():
        path = repo_root() / path
    raw = OmegaConf.load(path)
    container: dict = OmegaConf.to_container(raw, resolve=True)

    def _pop_selector(key: str) -> str:
        val = container.pop(key)
        if isinstance(val, dict):
            if "name" not in val:
                raise ValueError(
                    f"{path}: `{key}` is a mapping but has no `name:` subkey to identify "
                    f"the {key} YAML to load."
                )
            sel = val.pop("name")
            # Put the remaining override fields back under the same top-level
            # key so they merge cleanly after base/model/dataset.
            if val:
                container[key] = val
            return sel
        return val

    model_id = _pop_selector("model")
    dataset_id = _pop_selector("dataset")
    overrides = OmegaConf.create(container)

    parts = [
        configs_dir() / "base.yaml",
        configs_dir() / "model" / f"{model_id}.yaml",
        configs_dir() / "dataset" / f"{dataset_id}.yaml",
        overrides,
    ]
    cfg = OmegaConf.create({})
    for p in parts:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(p) if isinstance(p, Path) else p)

    dr = cfg.dataset.get("root")
    if dr is not None and str(dr) and not Path(str(dr)).is_absolute():
        cfg.dataset.root = str((repo_root() / str(dr)).resolve())
    out_root = cfg.output.get("root")
    if out_root is not None and str(out_root) and not Path(str(out_root)).is_absolute():
        cfg.output.root = str((repo_root() / str(out_root)).resolve())

    mc = OmegaConf.select(cfg, "paths.model_cache")
    if mc is not None and str(mc) and not Path(str(mc)).is_absolute():
        cfg.paths.model_cache = str((repo_root() / str(mc)).resolve())

    for env_key, cfg_key in _ENV_OVERRIDES.items():
        val = os.environ.get(env_key)
        if val:
            OmegaConf.update(cfg, cfg_key, val)

    return cfg


def experiment_run_dir(cfg: Any) -> Path:
    """experiments/checkpoints/{experiment_name}_{model}_{dataset}_{bits}"""
    name = cfg.get("experiment_name", "run")
    m = cfg.model.name
    dset = cfg.dataset.name
    bits = cfg.model.bit_dim
    sub = f"{name}_{m}_{dset}_{bits}bit"
    root = Path(cfg.output.root)
    return root / cfg.output.checkpoints / sub
