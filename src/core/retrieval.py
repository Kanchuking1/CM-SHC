"""Encode database and build Hamming distance matrices for I2T / T2I retrieval."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def binary_sign_codes(continuous: torch.Tensor) -> torch.Tensor:
    """Map continuous embeddings to {-1, +1} hash bits (DCMH-style sign)."""
    return continuous.sign()


def hamming_distance_matrix(
    query_bits: torch.Tensor,
    db_bits: torch.Tensor,
    chunk: int = 512,
) -> torch.Tensor:
    """
    Pairwise Hamming distances: (Nq, d) vs (Nd, d) -> (Nq, Nd) int16.
    Bits must be in {-1, +1}.

    Uses ``d - q @ db^T`` (integer matmul) instead of broadcasting a
    ``(chunk, Nd, d)`` boolean tensor, keeping peak memory at O(chunk * Nd).
    """
    d = query_bits.shape[1]
    nq = query_bits.shape[0]
    q_int = query_bits.to(torch.int8)
    db_int = db_bits.to(torch.int8)
    out = torch.empty(nq, db_bits.shape[0], dtype=torch.int16)
    for i in tqdm(range(0, nq, chunk), desc="hamming", leave=False):
        dot = q_int[i : i + chunk].to(torch.int16) @ db_int.t().to(torch.int16)
        out[i : i + chunk] = ((d - dot) // 2).to(torch.int16)
    return out


@torch.no_grad()
def encode_paired_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run the model on every batch; return image hashes, text hashes, labels,
    and sample indices (all in sorted index order).

    Returns
    -------
    h_img, h_txt : (N, bit) float in {-1, +1}
    labels : (N, C) float multi-hot label matrix
    indices : (N,) long
    """
    model.eval()
    chunks_i = []
    chunks_t = []
    chunks_lab = []
    chunks_idx = []
    for batch in tqdm(loader, desc="encode", leave=False):
        idx = batch["index"]
        if torch.is_tensor(idx):
            idx = idx.cpu()
        else:
            idx = torch.tensor(idx, dtype=torch.long)
        img = batch["img"].to(device)
        f = model.encode_image(img)
        if "text_features" in batch:
            g = model.encode_text(text_features=batch["text_features"].to(device))
        else:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            g = model.encode_text(ids, mask)
        chunks_i.append(binary_sign_codes(f).cpu())
        chunks_t.append(binary_sign_codes(g).cpu())
        chunks_lab.append(batch["label"].cpu().float())
        chunks_idx.append(idx)
    all_idx = torch.cat(chunks_idx, dim=0)
    h_img = torch.cat(chunks_i, dim=0)
    h_txt = torch.cat(chunks_t, dim=0)
    labels = torch.cat(chunks_lab, dim=0)
    order = all_idx.argsort()
    return h_img[order], h_txt[order], labels[order], all_idx[order]
