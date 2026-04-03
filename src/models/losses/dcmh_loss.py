"""DCMH objective (Jiang et al., CVPR 2017)."""

from __future__ import annotations

import numpy as np
import torch

from ...hashing.similarity import calc_neighbor


def dcmh_full_loss(
    B: torch.Tensor,
    F_codes: torch.Tensor,
    G_codes: torch.Tensor,
    Sim: torch.Tensor,
    gamma: float,
    eta: float,
) -> torch.Tensor:
    theta = torch.matmul(F_codes, G_codes.t()) / 2.0
    term1 = torch.sum(torch.log(1.0 + torch.exp(theta)) - Sim * theta)
    term2 = torch.sum((B - F_codes) ** 2) + torch.sum((B - G_codes) ** 2)
    term3 = torch.sum(F_codes.sum(dim=0) ** 2) + torch.sum(G_codes.sum(dim=0) ** 2)
    return term1 + gamma * term2 + eta * term3


def dcmh_batch_loss_image(
    cur_f: torch.Tensor,
    sample_labels: torch.Tensor,
    train_labels: torch.Tensor,
    G_buffer: torch.Tensor,
    F_buffer: torch.Tensor,
    B: torch.Tensor,
    ind: np.ndarray,
    ones: torch.Tensor,
    ones_: torch.Tensor,
    num_train: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    S = calc_neighbor(sample_labels, train_labels)
    theta = 0.5 * torch.matmul(cur_f, G_buffer.t())
    logloss = -torch.sum(S * theta - torch.log(1.0 + torch.exp(theta)))
    quantization = torch.sum((B[ind, :] - cur_f) ** 2)
    unupdated_ind = np.setdiff1d(np.arange(num_train), ind, assume_unique=False)
    sum_batch = cur_f.t().mm(ones)
    sum_rest = F_buffer[unupdated_ind].t().mm(ones_)
    balance = torch.sum((sum_batch + sum_rest) ** 2)
    return logloss, quantization, balance


def dcmh_batch_loss_text(
    cur_g: torch.Tensor,
    sample_labels: torch.Tensor,
    train_labels: torch.Tensor,
    F_buffer: torch.Tensor,
    G_buffer: torch.Tensor,
    B: torch.Tensor,
    ind: np.ndarray,
    ones: torch.Tensor,
    ones_: torch.Tensor,
    num_train: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    S = calc_neighbor(sample_labels, train_labels)
    theta = 0.5 * torch.matmul(cur_g, F_buffer.t())
    logloss = -torch.sum(S * theta - torch.log(1.0 + torch.exp(theta)))
    quantization = torch.sum((B[ind, :] - cur_g) ** 2)
    unupdated_ind = np.setdiff1d(np.arange(num_train), ind, assume_unique=False)
    sum_batch = cur_g.t().mm(ones)
    sum_rest = G_buffer[unupdated_ind].t().mm(ones_)
    balance = torch.sum((sum_batch + sum_rest) ** 2)
    return logloss, quantization, balance
