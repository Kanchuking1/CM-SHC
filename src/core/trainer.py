"""Training loops (DCMH alternating optimization; extend for other methods)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.nn.utils import clip_grad_norm_

from ..models.hashing.dcmh import DCMH
from ..models.losses.dcmh_loss import (
    dcmh_batch_loss_image,
    dcmh_batch_loss_text,
    dcmh_full_loss,
)
from ..hashing.similarity import calc_neighbor


def _plot_losses(history: list[dict], dest: Path) -> None:
    """Save a multi-panel loss curve to *dest* (PNG)."""
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
        device: torch.device | str = "cuda",
        gamma: float = 1.0,
        eta: float = 1.0,
        max_epoch: int = 500,
        lr_img: float | None = None,
        lr_txt: float | None = None,
        lr_decay: float | None = None,
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

        self.Sim = calc_neighbor(self.train_L, self.train_L)

        self.ones = torch.ones(self.batch_size, 1, device=self.device)
        self.ones_ = torch.ones(self.num_train - self.batch_size, 1, device=self.device)

        if lr_img is None:
            lr_img = 10 ** (-1.5)
        if lr_txt is None:
            lr_txt = 10 ** (-1.5)
        self.lr_img_init = lr_img
        self.lr_txt_init = lr_txt
        self.optim_img = SGD(model.image_net.parameters(), lr=lr_img)
        if getattr(model, "text_backend", "transformer") == "mlp":
            self._txt_params = list(model.text_proj.parameters())
        else:
            assert model.text_encoder is not None
            self._txt_params = list(model.text_encoder.parameters()) + list(model.text_proj.parameters())
        self.optim_txt = SGD(self._txt_params, lr=lr_txt)

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

    def train_epoch(self) -> dict[str, float]:
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
                cur_f,
                sample_L,
                self.train_L,
                self.G_buffer,
                self.F_buffer,
                self.B,
                ind,
                self.ones,
                self.ones_,
                self.num_train,
            )
            nll_scaled = logloss / (self.num_train * self.batch_size)
            quant_scaled = self.gamma * quant / (self.batch_size * self.bit)
            bal_scaled = self.eta * bal / (self.num_train * self.bit)
            loss = nll_scaled + quant_scaled + bal_scaled

            self.optim_img.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(self.model.image_net.parameters(), max_norm=5.0)
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
                cur_g,
                sample_L,
                self.train_L,
                self.F_buffer,
                self.G_buffer,
                self.B,
                ind,
                self.ones,
                self.ones_,
                self.num_train,
            )
            nll_scaled = logloss / (self.num_train * self.batch_size)
            quant_scaled = self.gamma * quant / (self.batch_size * self.bit)
            bal_scaled = self.eta * bal / (self.num_train * self.bit)
            loss = nll_scaled + quant_scaled + bal_scaled

            self.optim_txt.zero_grad(set_to_none=True)
            loss.backward()
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
        with torch.no_grad():
            return float(
                dcmh_full_loss(
                    self.B,
                    self.F_buffer,
                    self.G_buffer,
                    self.Sim,
                    self.gamma,
                    self.eta,
                )
            )

    def save_checkpoint(self, path: Path | str, epoch: int, meta: dict | None = None) -> None:
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

    def load_training_checkpoint(self, path: Path | str) -> int:
        """Restore model, optimizers, and hash buffers. Returns next epoch index (0-based).

        If the saved buffers have a different num_train (e.g. dataset filtering
        changed), only model weights and optimizers are restored and buffers are
        kept at their freshly-initialised values.
        """
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
        checkpoint_dir: Path | str | None = None,
        save_every: int = 1,
        run_meta: dict | None = None,
        start_epoch: int = 0,
        resumed_checkpoint: Path | str | None = None,
    ) -> None:
        checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        run_meta = dict(run_meta) if run_meta else {}

        history_path = checkpoint_dir / "loss_history.json" if checkpoint_dir else None
        history: list[dict] = []
        if history_path and history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                history = []

        prev_ckpt: Path | None = Path(resumed_checkpoint) if resumed_checkpoint else None

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
