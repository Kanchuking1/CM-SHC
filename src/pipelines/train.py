"""
Train from merged YAML experiment config.

Usage (from repository root)::

    python -m src.pipelines.train --config configs/experiments/exp_cmshc_mirflickr25k_128bit.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.core.trainer import CMSHCTrainer, DCMHTrainer, count_parameters
from src.data.collators import (
    build_train_labels_tensor,
    load_clip_tokenizer,
    load_hf_tokenizer,
    make_dcmh_collate_fn,
)
from src.data.loaders import get_dataset
from src.data.splits import SplitSubset, make_mirflickr_split
from src.data.transforms import imagenet_train_transform
from src.models.hashing.cm_shc import CMSHC
from src.models.hashing.dcmh import DCMH
from src.utils.checkpoint import find_latest_training_checkpoint
from src.utils.config import experiment_run_dir, load_experiment, repo_root
from src.utils.logger import setup_logging
from src.utils.model_paths import local_files_only, resolve_pretrained_ref, torch_home_dir
from src.utils.seed import set_seed


def _cfg_to_json_dict(cfg) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)


def resolve_text_repo(backbone_text_field) -> str:
    """Extract the HF repo id from a ``cfg.model.backbone.text`` field.

    Accepts the two supported forms:

    * a plain string -- returned unchanged (legacy path, e.g.
      ``huawei-noah/TinyBERT_General_4L_312D``);
    * a mapping with a ``model_name`` key -- e.g. ``{name: clip,
      model_name: openai/clip-vit-base-patch32, ...}``.

    Returns an empty string when the field is absent or malformed.
    """
    if backbone_text_field is None:
        return ""
    if isinstance(backbone_text_field, str):
        return backbone_text_field
    try:
        resolved = OmegaConf.to_container(backbone_text_field, resolve=True)
    except Exception:
        resolved = dict(backbone_text_field) if isinstance(backbone_text_field, dict) else {}
    if isinstance(resolved, dict):
        return str(resolved.get("model_name", ""))
    return str(backbone_text_field)


def text_backbone_name(backbone_text_field) -> str:
    """Return the text backbone's registry ``name`` key (or empty string).

    Plain-string backbone fields are legacy HF repo ids with no explicit
    registry name, so they return ``""`` here.
    """
    if backbone_text_field is None or isinstance(backbone_text_field, str):
        return ""
    try:
        resolved = OmegaConf.to_container(backbone_text_field, resolve=True)
    except Exception:
        resolved = dict(backbone_text_field) if isinstance(backbone_text_field, dict) else {}
    if isinstance(resolved, dict):
        return str(resolved.get("name", ""))
    return ""


def resolve_text_backbone_spec(cfg) -> tuple[str, bool, bool]:
    """Return ``(text_ref, hf_local_files_only, is_clip_text)`` for ``cfg``.

    Handles the three tokenizer / encoder regimes that the pipelines care
    about:

    * MLP-on-BOW path (``text_feature_dim`` set) -- returns ``("", False,
      False)``; no HF snapshot needed.
    * Legacy string ``backbone.text`` -- resolved as an HF repo id.
    * Dict ``backbone.text`` (e.g. CLIP) -- ``model_name`` is used as the
      repo id and ``is_clip_text`` reflects the registry ``name`` key.

    This is the single place evaluate / retrieve / train agree on how
    the text backbone is configured.
    """
    tdim = cfg.model.text_feature_dim
    if tdim is not None:
        return "", False, False

    backbone = cfg.model.get("backbone", None)
    backbone_text_field = backbone.get("text", None) if backbone is not None else None
    is_clip = text_backbone_name(backbone_text_field) == "clip"
    repo = resolve_text_repo(backbone_text_field)
    if not repo:
        raise ValueError(
            "Text backbone repo id is empty. Set cfg.model.backbone.text to either "
            "a HF repo id (str) or a registry dict with a 'model_name' key."
        )
    cache_root = Path(cfg.paths.model_cache)
    offline = local_files_only(cfg)
    text_ref, hf_lfo = resolve_pretrained_ref(repo, cache_root, offline)
    return text_ref, hf_lfo, is_clip


# Back-compat aliases (old private names; prefer the public forms above).
_resolve_text_repo = resolve_text_repo
_text_backbone_name = text_backbone_name


def _backbone_field_repr(field) -> Any:
    """JSON-serialisable summary of a backbone field for run metadata.

    Plain strings stay as strings; OmegaConf DictConfigs are materialised
    to plain dicts so ``json.dump`` does not fall back to ``str(...)`` and
    produce an unreadable blob.
    """
    if field is None:
        return None
    if isinstance(field, str):
        return field
    try:
        resolved = OmegaConf.to_container(field, resolve=True)
    except Exception:
        resolved = dict(field) if isinstance(field, dict) else None
    return resolved if resolved is not None else str(field)


def _backbone_field_to_cfg(field, hf_local_files_only: bool):
    if field is None:
        return None
    if isinstance(field, str):
        return None
    try:
        resolved = OmegaConf.to_container(field, resolve=True)
    except Exception:
        resolved = dict(field) if isinstance(field, dict) else None
    if not isinstance(resolved, dict):
        return None
    lfo = resolved.get("local_files_only", "auto")
    if lfo == "auto" or lfo is None:
        resolved["local_files_only"] = bool(hf_local_files_only)
    return resolved


def _resolve_backbone_model_name(cfg_dict, cache_root, offline_hint: bool):
    """If ``cfg_dict["model_name"]`` is an HF repo id with a complete local
    snapshot under ``cache_root`` (or ``offline_hint`` is True), rewrite it
    to the on-disk path and force ``local_files_only=True``.

    This keeps CLIP / BERT backbones honest about offline mode: the
    registry factories just call ``from_pretrained(model_name,
    local_files_only=...)``, so the ``model_name`` must be a concrete
    local path on HPC nodes without hub access.
    """
    if not isinstance(cfg_dict, dict):
        return cfg_dict
    repo = cfg_dict.get("model_name")
    if not repo:
        return cfg_dict
    ref, lfo = resolve_pretrained_ref(str(repo), cache_root, bool(offline_hint))
    cfg_dict["model_name"] = ref
    if lfo:
        cfg_dict["local_files_only"] = True
    return cfg_dict


def build_model(cfg, text_ref: str, hf_local_files_only: bool):
    name = str(cfg.model.name).lower()
    tdim = cfg.model.text_feature_dim
    if tdim is not None:
        tdim = int(tdim)

    backbone = cfg.model.get("backbone", None)
    raw_image = backbone.get("image", "alexnet") if backbone is not None else "alexnet"
    raw_text = backbone.get("text", None) if backbone is not None else None
    image_cfg = _backbone_field_to_cfg(raw_image, hf_local_files_only)
    text_cfg = _backbone_field_to_cfg(raw_text, hf_local_files_only)

    cache_root = Path(cfg.paths.model_cache)
    offline = local_files_only(cfg)
    image_cfg = _resolve_backbone_model_name(image_cfg, cache_root, offline)
    text_cfg = _resolve_backbone_model_name(text_cfg, cache_root, offline)

    image_backbone = str(raw_image) if isinstance(raw_image, str) else "alexnet"
    freeze_text = bool(cfg.model.get("freeze_text_encoder", False))

    common = dict(
        bit_dim=int(cfg.model.bit_dim),
        text_model_name=text_ref,
        image_backbone=image_backbone,
        text_feature_dim=tdim,
        freeze_text_encoder=freeze_text,
        local_files_only=hf_local_files_only and tdim is None,
    )
    if image_cfg is not None:
        common["image_cfg"] = image_cfg
    if text_cfg is not None:
        common["text_cfg"] = text_cfg

    if name == "dcmh":
        return DCMH(**common)
    if name == "cm_shc":
        return CMSHC(**common)
    raise ValueError(
        f"Unknown model.name={name!r}; expected 'dcmh' or 'cm_shc'."
    )


def _resolve_centers_path(cfg) -> Path:
    explicit = cfg.model.get("centers_path", None)
    if explicit:
        return Path(str(explicit))
    method = str(cfg.model.get("similarity_method", "cooccurrence"))
    dataset = str(cfg.dataset.name)
    q = int(cfg.model.bit_dim)
    return Path(cfg.output.root) / "centers" / f"{dataset}_{method}_q{q}.pt"


def _load_cmshc_targets(cfg, num_train: int, bit_dim: int) -> torch.Tensor:
    path = _resolve_centers_path(cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"CM-SHC centers file not found: {path}\n"
            "Run `python -m src.pipelines.build_centers --config ... --method ...` first."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    T = payload.get("T_train")
    if T is None:
        raise KeyError(f"Payload at {path} is missing 'T_train'.")
    if T.shape != (num_train, bit_dim):
        raise ValueError(
            f"Cached T_train shape {tuple(T.shape)} does not match current split "
            f"(num_train={num_train}, bit_dim={bit_dim}). Did you change dataset.split "
            "or bit_dim between build_centers and train runs?"
        )
    return T.float()


def _optimizer_kwargs_from_cfg(cfg) -> dict:
    """Pull optimizer knobs out of cfg.training and coerce types."""
    tr = cfg.training
    betas_raw = tr.get("betas", [0.9, 0.999])
    if hasattr(betas_raw, "__iter__") and not isinstance(betas_raw, (str, bytes)):
        betas_list = [float(x) for x in list(betas_raw)]
        betas = tuple(betas_list) if len(betas_list) == 2 else (0.9, 0.999)
    else:
        betas = (0.9, 0.999)
    return dict(
        optimizer=str(tr.get("optimizer", "sgd")),
        weight_decay=float(tr.get("weight_decay", 0.0)),
        momentum=float(tr.get("momentum", 0.0)),
        betas=betas,
        eps=float(tr.get("eps", 1e-8)),
    )


def _log_trainable_summary(logger, model, opt_name: str) -> None:
    img_tr, img_total = count_parameters(model.image_net)
    txt_params = list(model.text_proj.parameters())
    if model.text_encoder is not None:
        txt_params += list(model.text_encoder.parameters())
    txt_total = sum(p.numel() for p in txt_params)
    txt_tr = sum(p.numel() for p in txt_params if p.requires_grad)
    logger.info(
        "Trainable params -- image: %s / %s (%.2f%%) | text: %s / %s (%.2f%%) | optimizer=%s",
        f"{img_tr:,}", f"{img_total:,}",
        100.0 * img_tr / max(img_total, 1),
        f"{txt_tr:,}", f"{txt_total:,}",
        100.0 * txt_tr / max(txt_total, 1),
        opt_name,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default="configs/experiments/exp_cmshc_mirflickr25k_128bit.yaml",
        help="Path to experiment YAML (merged with base, model, dataset).",
    )
    p.add_argument("--device", type=str, default=None, help="Override cfg.device (cuda/cpu)")
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore training checkpoints in the experiment dir even if training.resume is true.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_experiment(args.config)

    cache_root = Path(cfg.paths.model_cache)
    os.environ["TORCH_HOME"] = str(torch_home_dir(cache_root))

    set_seed(int(cfg.seed))
    device = args.device or str(cfg.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    tdim = cfg.model.text_feature_dim
    use_mlp_text = tdim is not None
    text_ref, hf_lfo, is_clip_text = resolve_text_backbone_spec(cfg)

    run_dir = experiment_run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_root = Path(cfg.output.root) / cfg.output.logs
    log_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_dir=log_root)
    logger.info("Experiment dir: %s", run_dir)
    logger.info("TORCH_HOME=%s", os.environ.get("TORCH_HOME"))
    if use_mlp_text:
        logger.info("Text encoder: MLP on %d-dim feature vectors", int(tdim))
    else:
        logger.info("HF pretrained ref: %s (local_files_only=%s)", text_ref, hf_lfo)

    (run_dir / "run_config.json").write_text(
        json.dumps(_cfg_to_json_dict(cfg), indent=2),
        encoding="utf-8",
    )

    transform = imagenet_train_transform()
    ds_kwargs = {}
    if hasattr(cfg.dataset, "num_pseudo_classes") and cfg.dataset.num_pseudo_classes is not None:
        ds_kwargs["num_pseudo_classes"] = int(cfg.dataset.num_pseudo_classes)

    text_mode = "bow" if use_mlp_text else "raw"
    ds_kwargs.setdefault("text_mode", text_mode)

    full_ds = get_dataset(
        str(cfg.dataset.name),
        root_dir=cfg.dataset.root,
        transform=transform,
        **ds_kwargs,
    )

    split_cfg = getattr(cfg.dataset, "split", None)
    if split_cfg is not None:
        q_size = int(split_cfg.query_size)
        t_size = int(split_cfg.train_size)
        _query_idx, train_idx, _db_idx = make_mirflickr_split(
            len(full_ds), query_size=q_size, train_size=t_size, seed=int(cfg.seed),
        )
        train_ds = SplitSubset(full_ds, train_idx)
        logger.info(
            "Split: %d total -> %d query (held out), %d train, %d database",
            len(full_ds), q_size, len(train_ds), len(_db_idx),
        )
    else:
        train_ds = full_ds

    if use_mlp_text:
        tokenizer = None
    elif is_clip_text:
        tokenizer = load_clip_tokenizer(text_ref, local_files_only=hf_lfo)
    else:
        tokenizer = load_hf_tokenizer(text_ref, local_files_only=hf_lfo)
    max_length = int(cfg.dataset.caption_max_length)
    if is_clip_text:
        clip_cap = int(getattr(tokenizer, "model_max_length", 77))
        max_length = min(max_length, clip_cap)
    collate = make_dcmh_collate_fn(tokenizer, max_length=max_length)

    loader = DataLoader(
        train_ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=int(cfg.training.num_workers),
        collate_fn=collate,
        pin_memory=device.startswith("cuda"),
    )

    num_cls = int(getattr(cfg.dataset, "num_classes", 0) or getattr(cfg.dataset, "num_pseudo_classes", 0))
    train_labels = build_train_labels_tensor(train_ds, num_cls)

    model = build_model(cfg, text_ref=text_ref, hf_local_files_only=hf_lfo).to(device)

    opt_kwargs = _optimizer_kwargs_from_cfg(cfg)
    _log_trainable_summary(logger, model, opt_kwargs["optimizer"])

    model_name = str(cfg.model.name).lower()
    if model_name == "dcmh":
        trainer = DCMHTrainer(
            model=model,
            train_loader=loader,
            train_labels=train_labels,
            device=device,
            gamma=float(cfg.model.gamma),
            eta=float(cfg.model.eta),
            max_epoch=int(cfg.training.max_epochs),
            lr_img=float(cfg.training.lr_img),
            lr_txt=float(cfg.training.lr_txt),
            **opt_kwargs,
        )
    elif model_name == "cm_shc":
        T_train = _load_cmshc_targets(cfg, num_train=len(train_ds), bit_dim=int(cfg.model.bit_dim))
        logger.info(
            "Loaded CM-SHC targets %s from %s",
            tuple(T_train.shape),
            _resolve_centers_path(cfg),
        )
        trainer = CMSHCTrainer(
            model=model,
            train_loader=loader,
            target_codes=T_train,
            device=device,
            lambda_center=float(cfg.model.get("lambda_center", 1.0)),
            lambda_quant=float(cfg.model.get("lambda_quant", 0.1)),
            lambda_cm=float(cfg.model.get("lambda_cm", 1.0)),
            lambda_bal=float(cfg.model.get("lambda_bal", 0.0)),
            max_epoch=int(cfg.training.max_epochs),
            lr_img=float(cfg.training.lr_img),
            lr_txt=float(cfg.training.lr_txt),
            **opt_kwargs,
        )
    else:
        raise ValueError(f"Unknown model.name={cfg.model.name!r}")

    start_epoch = 0
    resumed_ckpt = None
    want_resume = bool(cfg.training.get("resume", False)) and not args.no_resume
    if want_resume:
        latest = find_latest_training_checkpoint(run_dir)
        if latest is not None:
            start_epoch = trainer.load_training_checkpoint(latest)
            resumed_ckpt = latest
            logger.info("Resumed from %s at start_epoch=%s", latest, start_epoch)

    meta = {
        "experiment_name": str(cfg.experiment_name),
        "model": str(cfg.model.name),
        "dataset": str(cfg.dataset.name),
        "image_backbone": _backbone_field_repr(cfg.model.backbone.image),
        "text_backbone": _backbone_field_repr(cfg.model.backbone.text),
        "bit_dim": int(cfg.model.bit_dim),
        "config_path": str(Path(args.config).resolve()),
        "repo_root": str(repo_root()),
        "resume_start_epoch": start_epoch,
    }
    trainer.train(
        checkpoint_dir=run_dir,
        save_every=int(cfg.training.save_every),
        run_meta=meta,
        start_epoch=start_epoch,
        resumed_checkpoint=resumed_ckpt,
    )
    print(f"Done. Checkpoints and run_config under: {run_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
