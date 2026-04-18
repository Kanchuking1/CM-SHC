"""Training loops.

* ``DCMHTrainer`` -- alternating F / G / B optimization (Jiang et al. 2017).
* ``CMSHCTrainer`` -- single joint mini-batch update against pre-computed
  per-sample target codes (CM-SHC, Stage 3).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.nn.utils import clip_grad_norm_

from ..models.hashing.cm_shc import CMSHC
from ..models.hashing.dcmh import DCMH
from ..models.losses.dcmh_loss import (
    dcmh_batch_loss_image,
    dcmh_batch_loss_text,
)
from ..models.losses.semantic_center_loss import cmshc_full_loss


def trainable_parameters(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Return only the parameters of *module* with ``requires_grad=True``.

    This is what optimizers should ever see when part of the model is
    frozen (frozen CLIP, LoRA with only adapters trainable, etc.):
    including `requires_grad=False` params inflates the optimizer state,
    wastes GPU memory, and -- for stateful optimizers like AdamW -- can
    still produce spurious weight updates via momentum on zero gradients.
    """
    return [p for p in module.parameters() if p.requires_grad]


def count_parameters(module: torch.nn.Module) -> tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts for *module*."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return trainable, total


def _resolve_opt_name(name):
    name = (name or "sgd").strip().lower()
    if name not in ("sgd", "adamw"):
        raise ValueError(
            f"Unsupported optimizer {name!r}; choose from 'sgd' / 'adamw'."
        )
    return name


def build_optimizer(
    params,
    lr: float,
    name: str = "sgd",
    weight_decay: float = 0.0,
    momentum: float = 0.0,
    betas=(0.9, 0.999),
    eps: float = 1e-8,
):
    """Construct an optimizer for ``params`` according to *name*.

    ``params`` must already be filtered to ``requires_grad=True`` parameters
    (call :func:`trainable_parameters` first).  An empty iterable raises a
    ``ValueError`` to catch the "everything frozen by mistake" case early.
    """
    params = list(params)
    if not params:
        raise ValueError(
            "No trainable parameters were supplied to build_optimizer(). "
            "Did you accidentally freeze the entire module?"
        )
    kind = _resolve_opt_name(name)
    if kind == "sgd":
        return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    return AdamW(params, lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay)


def _plot_losses(history, dest: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), tight_layout=True)

    for ax, key, label in zip(
        axes.flat,
        ("total_scaled", "log_loss", "quant_loss", "full_objective"),
        ("Total (scaled)", "Log-likelihood", "Quantization", "Full objective"),
    ):
        vals = [h.get(key, 0.0) for h in history]
        ax.plot(epochs, vals, linewidth=1.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    fig.savefig(dest, dpi=120)
    plt.close(fig)


class DCMHTrainer:
    """
    Expects batch dicts: ``index``, ``img``, ``label``, ``input_ids``, ``attention_mask``
    (and optionally ``text_features`` for MLP text path).
    DataLoader must use ``drop_last=True`` and fixed batch size.
    """

    def __init__(
        self,
        model: DCMH,
        train_loader: DataLoader,
        train_labels: torch.Tensor,
        device = "cuda",
        gamma: float = 1.0,
        eta: float = 1.0,
        max_epoch: int = 500,
        lr_img = None,
        lr_txt = None,
        lr_decay = None,
        optimizer: str = "sgd",
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.model = model
        self.train_loader = train_loader
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.gamma = gamma
        self.eta = eta
        self.max_epoch = max_epoch

        self.batch_size = train_loader.batch_size
        if self.batch_size is None:
            raise ValueError("train_loader must have batch_size set.")
        self.num_train = len(train_loader.dataset)
        self.bit = model.bit_dim

        self.train_L = train_labels.float().to(self.device)
        if self.train_L.dim() == 1:
            nc = int(self.train_L.max().item()) + 1
            self.train_L = torch.nn.functional.one_hot(self.train_L.long(), num_classes=nc).float()

        self.F_buffer = torch.randn(self.num_train, self.bit, device=self.device)
        self.G_buffer = torch.randn(self.num_train, self.bit, device=self.device)
        self.B = torch.sign(self.F_buffer + self.G_buffer)

        self.ones = torch.ones(self.batch_size, 1, device=self.device)
        self.ones_ = torch.ones(self.num_train - self.batch_size, 1, device=self.device)

        if lr_img is None:
            lr_img = 10 ** (-1.5)
        if lr_txt is None:
            lr_txt = 10 ** (-1.5)
        self.lr_img_init = lr_img
        self.lr_txt_init = lr_txt
        self.optimizer_name = _resolve_opt_name(optimizer)

        img_params = trainable_parameters(model.image_net)
        self._img_params = img_params
        if getattr(model, "text_backend", "transformer") == "mlp":
            self._txt_params = trainable_parameters(model.text_proj)
        else:
            assert model.text_encoder is not None
            self._txt_params = trainable_parameters(model.text_encoder) + trainable_parameters(model.text_proj)

        self.optim_img = build_optimizer(
            img_params, lr=lr_img, name=self.optimizer_name,
            weight_decay=weight_decay, momentum=momentum, betas=betas, eps=eps,
        )
        self.optim_txt = build_optimizer(
            self._txt_params, lr=lr_txt, name=self.optimizer_name,
            weight_decay=weight_decay, momentum=momentum, betas=betas, eps=eps,
        )

        if lr_decay is None:
            self.lr_decay = (1e-6 / 10 ** (-1.5)) ** (1.0 / max(self.max_epoch, 1))
        else:
            self.lr_decay = lr_decay

    def lr_schedule(self) -> None:
        for g in self.optim_img.param_groups:
            g["lr"] = max(g["lr"] * self.lr_decay, 1e-6)
        for g in self.optim_txt.param_groups:
            g["lr"] = max(g["lr"] * self.lr_decay, 1e-6)

    @torch.no_grad()
    def refresh_binary_codes(self) -> None:
        self.B = torch.sign(self.F_buffer + self.G_buffer)

    def train_epoch(self):
        self.model.train()
        metrics = {
            "log_loss": 0.0,
            "quant_loss": 0.0,
            "balance_loss": 0.0,
            "total_scaled": 0.0,
            "n_img_steps": 0,
            "n_txt_steps": 0,
        }

        for batch in tqdm(self.train_loader, desc="img", leave=False):
            ind = batch["index"].numpy() if torch.is_tensor(batch["index"]) else np.asarray(batch["index"])
            image = batch["img"].to(self.device)
            sample_L = batch["label"].to(self.device).float()
            if sample_L.dim() == 1:
                sample_L = torch.nn.functional.one_hot(
                    sample_L.long(), num_classes=self.train_L.size(1)
                ).float()

            cur_f = self.model.encode_image(image)
            with torch.no_grad():
                self.F_buffer[ind, :] = cur_f.detach()

            logloss, quant, bal = dcmh_batch_loss_image(
                cur_f, sample_L, self.train_L,
                self.G_buffer, self.F_buffer, self.B,
                ind, self.ones, self.ones_, self.num_train,
            )
            nll_scaled = logloss / (self.num_train * self.batch_size)
            quant_scaled = self.gamma * quant / (self.batch_size * self.bit)
            bal_scaled = self.eta * bal / (self.num_train * self.bit)
            loss = nll_scaled + quant_scaled + bal_scaled

            self.optim_img.zero_grad(set_to_none=True)
            loss.backward()
            if self._img_params:
                clip_grad_norm_(self._img_params, max_norm=5.0)
            self.optim_img.step()

            metrics["log_loss"] += float(logloss.detach())
            metrics["quant_loss"] += float(quant.detach())
            metrics["balance_loss"] += float(bal.detach())
            metrics["total_scaled"] += float(loss.detach())
            metrics["n_img_steps"] += 1

        for batch in tqdm(self.train_loader, desc="txt", leave=False):
            ind = batch["index"].numpy() if torch.is_tensor(batch["index"]) else np.asarray(batch["index"])
            sample_L = batch["label"].to(self.device).float()
            if sample_L.dim() == 1:
                sample_L = torch.nn.functional.one_hot(
                    sample_L.long(), num_classes=self.train_L.size(1)
                ).float()

            if getattr(self.model, "text_backend", "transformer") == "mlp":
                cur_g = self.model.encode_text(text_features=batch["text_features"].to(self.device))
            else:
                input_ids = batch["input_ids"].to(self.device)
                attn = batch["attention_mask"].to(self.device)
                cur_g = self.model.encode_text(input_ids, attn)
            with torch.no_grad():
                self.G_buffer[ind, :] = cur_g.detach()

            logloss, quant, bal = dcmh_batch_loss_text(
                cur_g, sample_L, self.train_L,
                self.F_buffer, self.G_buffer, self.B,
                ind, self.ones, self.ones_, self.num_train,
            )
            nll_scaled = logloss / (self.num_train * self.batch_size)
            quant_scaled = self.gamma * quant / (self.batch_size * self.bit)
            bal_scaled = self.eta * bal / (self.num_train * self.bit)
            loss = nll_scaled + quant_scaled + bal_scaled

            self.optim_txt.zero_grad(set_to_none=True)
            loss.backward()
            if self._txt_params:
                clip_grad_norm_(self._txt_params, max_norm=5.0)
            self.optim_txt.step()

            metrics["log_loss"] += float(logloss.detach())
            metrics["quant_loss"] += float(quant.detach())
            metrics["balance_loss"] += float(bal.detach())
            metrics["total_scaled"] += float(loss.detach())
            metrics["n_txt_steps"] += 1

        self.refresh_binary_codes()
        n = max(metrics["n_img_steps"] + metrics["n_txt_steps"], 1)
        for k in ("log_loss", "quant_loss", "balance_loss", "total_scaled"):
            metrics[k] /= n
        return metrics

    def full_objective_value(self) -> float:
        chunk = 512
        F_cpu = self.F_buffer.detach().cpu().float()
        G_cpu = self.G_buffer.detach().cpu().float()
        B_cpu = self.B.detach().cpu().float()
        L_cpu = self.train_L.detach().cpu().float()

        with torch.no_grad():
            nll = torch.tensor(0.0)
            for i in range(0, self.num_train, chunk):
                f_chunk = F_cpu[i : i + chunk]
                l_chunk = L_cpu[i : i + chunk]
                theta = f_chunk @ G_cpu.t() * 0.5
                sim = (l_chunk @ L_cpu.t() > 0).float()
                nll += torch.sum(torch.nn.functional.softplus(theta) - sim * theta)

            quant = torch.sum((B_cpu - F_cpu) ** 2) + torch.sum((B_cpu - G_cpu) ** 2)
            bal = torch.sum(F_cpu.sum(0) ** 2) + torch.sum(G_cpu.sum(0) ** 2)
            return float(nll + self.gamma * quant + self.eta * bal)

    def save_checkpoint(self, path, epoch: int, meta=None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": int(epoch),
            "model_state_dict": self.model.state_dict(),
            "optim_img": self.optim_img.state_dict(),
            "optim_txt": self.optim_txt.state_dict(),
            "F_buffer": self.F_buffer.detach().cpu(),
            "G_buffer": self.G_buffer.detach().cpu(),
            "B": self.B.detach().cpu(),
        }
        if meta:
            payload["meta"] = meta
        torch.save(payload, path)

    def load_training_checkpoint(self, path) -> int:
        path = Path(path)
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optim_img.load_state_dict(ckpt["optim_img"])
        self.optim_txt.load_state_dict(ckpt["optim_txt"])
        saved_n = ckpt["F_buffer"].shape[0]
        if saved_n == self.num_train:
            self.F_buffer = ckpt["F_buffer"].to(self.device)
            self.G_buffer = ckpt["G_buffer"].to(self.device)
            self.B = ckpt["B"].to(self.device)
        else:
            import logging
            logging.getLogger(__name__).warning(
                "Checkpoint buffers have %d rows but current dataset has %d; "
                "re-initialising F/G/B buffers from scratch.",
                saved_n, self.num_train,
            )
        return int(ckpt["epoch"])

    def train(
        self,
        checkpoint_dir=None,
        save_every: int = 1,
        run_meta=None,
        start_epoch: int = 0,
        resumed_checkpoint=None,
    ) -> None:
        checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        run_meta = dict(run_meta) if run_meta else {}

        history_path = checkpoint_dir / "loss_history.json" if checkpoint_dir else None
        history = []
        if history_path and history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                history = []

        prev_ckpt = Path(resumed_checkpoint) if resumed_checkpoint else None

        for epoch in range(start_epoch, self.max_epoch):
            m = self.train_epoch()
            full = self.full_objective_value()
            print(
                f"epoch {epoch + 1}/{self.max_epoch}  "
                f"log={m['log_loss']:.4f} quant={m['quant_loss']:.4f} bal={m['balance_loss']:.4f}  "
                f"full_obj={full:.4f}"
            )

            history.append({
                "epoch": epoch + 1,
                "log_loss": m["log_loss"],
                "quant_loss": m["quant_loss"],
                "balance_loss": m["balance_loss"],
                "total_scaled": m["total_scaled"],
                "full_objective": full,
            })

            if checkpoint_dir is not None:
                history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
                try:
                    _plot_losses(history, checkpoint_dir / "loss_curve.png")
                except Exception:
                    pass

            if checkpoint_dir is not None and save_every > 0:
                if (epoch + 1) % save_every == 0 or epoch == self.max_epoch - 1:
                    ck_meta = {
                        **run_meta,
                        **m,
                        "full_objective": full,
                        "epoch_saved": epoch + 1,
                    }
                    new_ckpt = checkpoint_dir / f"epoch_{epoch + 1:04d}.pt"
                    self.save_checkpoint(new_ckpt, epoch + 1, ck_meta)
                    if prev_ckpt is not None and prev_ckpt.exists():
                        prev_ckpt.unlink()
                    prev_ckpt = new_ckpt

            self.lr_schedule()


# =============================================================================
# CM-SHC trainer (Stage 3)
# =============================================================================


def _plot_cmshc_losses(history, dest: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), tight_layout=True)

    for ax, key, label in zip(
        axes.flat,
        ("center", "quant", "cross_modal", "total"),
        ("Central BCE", "Quantization (log cosh)", "Cross-modal MSE", "Total (weighted)"),
    ):
        vals = [h.get(key, 0.0) for h in history]
        ax.plot(epochs, vals, linewidth=1.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    fig.savefig(dest, dpi=120)
    plt.close(fig)


class CMSHCTrainer:
    """Single joint update over (image, text) -> shared q-bit space."""

    def __init__(
        self,
        model: CMSHC,
        train_loader: DataLoader,
        target_codes: torch.Tensor,
        device = "cuda",
        lambda_center: float = 1.0,
        lambda_quant: float = 0.1,
        lambda_cm: float = 1.0,
        lambda_bal: float = 0.0,
        max_epoch: int = 200,
        lr_img = None,
        lr_txt = None,
        lr_decay = None,
        grad_clip_norm: float = 5.0,
        optimizer: str = "sgd",
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.model = model
        self.train_loader = train_loader
        self.device = (
            torch.device(device) if not isinstance(device, torch.device) else device
        )

        self.batch_size = train_loader.batch_size
        if self.batch_size is None:
            raise ValueError("train_loader must have batch_size set.")
        self.num_train = len(train_loader.dataset)
        self.bit = model.bit_dim

        if target_codes.shape != (self.num_train, self.bit):
            raise ValueError(
                f"target_codes must have shape (num_train={self.num_train}, "
                f"bit={self.bit}); got {tuple(target_codes.shape)}"
            )
        self.T = target_codes.to(self.device).float()

        self.lambda_center = float(lambda_center)
        self.lambda_quant = float(lambda_quant)
        self.lambda_cm = float(lambda_cm)
        self.lambda_bal = float(lambda_bal)
        self.max_epoch = int(max_epoch)
        self.grad_clip_norm = float(grad_clip_norm)

        if lr_img is None:
            lr_img = 10 ** (-1.5)
        if lr_txt is None:
            lr_txt = 10 ** (-1.5)
        self.lr_img_init = lr_img
        self.lr_txt_init = lr_txt
        self.optimizer_name = _resolve_opt_name(optimizer)

        img_params = trainable_parameters(model.image_net)
        self._img_params = img_params
        if getattr(model, "text_backend", "transformer") == "mlp":
            self._txt_params = trainable_parameters(model.text_proj)
        else:
            assert model.text_encoder is not None
            self._txt_params = trainable_parameters(model.text_encoder) + trainable_parameters(model.text_proj)

        self.optim_img = build_optimizer(
            img_params, lr=lr_img, name=self.optimizer_name,
            weight_decay=weight_decay, momentum=momentum, betas=betas, eps=eps,
        )
        self.optim_txt = build_optimizer(
            self._txt_params, lr=lr_txt, name=self.optimizer_name,
            weight_decay=weight_decay, momentum=momentum, betas=betas, eps=eps,
        )

        if lr_decay is None:
            self.lr_decay = (1e-6 / max(lr_img, 1e-12)) ** (1.0 / max(self.max_epoch, 1))
        else:
            self.lr_decay = lr_decay

    def lr_schedule(self) -> None:
        for g in self.optim_img.param_groups:
            g["lr"] = max(g["lr"] * self.lr_decay, 1e-6)
        for g in self.optim_txt.param_groups:
            g["lr"] = max(g["lr"] * self.lr_decay, 1e-6)

    def _encode_text_batch(self, batch):
        if getattr(self.model, "text_backend", "transformer") == "mlp":
            tf = batch["text_features"].to(self.device)
            return self.model.encode_text(text_features=tf)
        input_ids = batch["input_ids"].to(self.device)
        attn = batch["attention_mask"].to(self.device)
        return self.model.encode_text(input_ids, attn)

    def train_epoch(self):
        self.model.train()
        running = {"center": 0.0, "quant": 0.0, "cross_modal": 0.0, "balance": 0.0, "total": 0.0}
        n_batches = 0

        for batch in tqdm(self.train_loader, desc="cmshc", leave=False):
            ind = batch["index"]
            if torch.is_tensor(ind):
                ind = ind.to(self.device).long()
            else:
                ind = torch.as_tensor(ind, dtype=torch.long, device=self.device)

            image = batch["img"].to(self.device)
            t = self.T.index_select(0, ind)

            f_logits = self.model.encode_image(image)
            g_logits = self._encode_text_batch(batch)

            loss, parts = cmshc_full_loss(
                f_logits, g_logits, t,
                lambda_center=self.lambda_center,
                lambda_quant=self.lambda_quant,
                lambda_cm=self.lambda_cm,
                lambda_bal=self.lambda_bal,
            )

            self.optim_img.zero_grad(set_to_none=True)
            self.optim_txt.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip_norm > 0:
                if self._img_params:
                    clip_grad_norm_(self._img_params, max_norm=self.grad_clip_norm)
                if self._txt_params:
                    clip_grad_norm_(self._txt_params, max_norm=self.grad_clip_norm)
            self.optim_img.step()
            self.optim_txt.step()

            for k in running:
                running[k] += float(parts[k])
            n_batches += 1

        n = max(n_batches, 1)
        return {k: v / n for k, v in running.items()}

    def save_checkpoint(self, path, epoch: int, meta=None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": int(epoch),
            "model_state_dict": self.model.state_dict(),
            "optim_img": self.optim_img.state_dict(),
            "optim_txt": self.optim_txt.state_dict(),
            "lambdas": {
                "center": self.lambda_center,
                "quant": self.lambda_quant,
                "cm": self.lambda_cm,
                "bal": self.lambda_bal,
            },
        }
        if meta:
            payload["meta"] = meta
        torch.save(payload, path)

    def load_training_checkpoint(self, path) -> int:
        path = Path(path)
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optim_img.load_state_dict(ckpt["optim_img"])
        self.optim_txt.load_state_dict(ckpt["optim_txt"])
        return int(ckpt["epoch"])

    def train(
        self,
        checkpoint_dir=None,
        save_every: int = 1,
        run_meta=None,
        start_epoch: int = 0,
        resumed_checkpoint=None,
    ) -> None:
        checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        run_meta = dict(run_meta) if run_meta else {}

        history_path = checkpoint_dir / "loss_history.json" if checkpoint_dir else None
        history = []
        if history_path and history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                history = []

        prev_ckpt = Path(resumed_checkpoint) if resumed_checkpoint else None

        for epoch in range(start_epoch, self.max_epoch):
            m = self.train_epoch()
            print(
                f"epoch {epoch + 1}/{self.max_epoch}  "
                f"center={m['center']:.4f}  quant={m['quant']:.4f}  "
                f"cm={m['cross_modal']:.4f}  total={m['total']:.4f}"
            )

            history.append({"epoch": epoch + 1, **m})

            if checkpoint_dir is not None:
                history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
                try:
                    _plot_cmshc_losses(history, checkpoint_dir / "loss_curve.png")
                except Exception:
                    pass

            if checkpoint_dir is not None and save_every > 0:
                if (epoch + 1) % save_every == 0 or epoch == self.max_epoch - 1:
                    ck_meta = {**run_meta, **m, "epoch_saved": epoch + 1}
                    new_ckpt = checkpoint_dir / f"epoch_{epoch + 1:04d}.pt"
                    self.save_checkpoint(new_ckpt, epoch + 1, ck_meta)
                    if prev_ckpt is not None and prev_ckpt.exists():
                        prev_ckpt.unlink()
                    prev_ckpt = new_ckpt

            self.lr_schedule()
