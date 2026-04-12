"""Encode database and build Hamming distance matrices for I2T / T2I retrieval."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def binary_sign_codes(continuous: torch.Tensor) -> torch.Tensor:
    """Map continuous embeddings to {-1, +1} hash bits (DCMH-style sign)."""
    return continuous.sign()


def hamming_distance_matrix(
    query_bits: torch.Tensor,
    db_bits: torch.Tensor,
    chunk: int = 1024,
) -> torch.Tensor:
    """
    Pairwise Hamming distances: (Nq, d) vs (Nd, d) -> (Nq, Nd) long counts.
    Bits are in {-1, +1} or {0, 1}; inequality counts mismatches.

    Computed in row-chunks to avoid materialising an (Nq, Nd, d) broadcast tensor.
    """
    nq = query_bits.shape[0]
    out = torch.empty(nq, db_bits.shape[0], dtype=torch.long)
    for i in range(0, nq, chunk):
        q = query_bits[i : i + chunk].unsqueeze(1)
        out[i : i + chunk] = (q != db_bits.unsqueeze(0)).sum(dim=-1).long()
    return out


@torch.no_grad()
def encode_paired_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run the model on every batch; return image hashes, text hashes, and sample indices (sorted order).

    Returns
    -------
    h_img, h_txt : (N, bit) float in {-1, +1}
    indices : (N,) long — row r corresponds to dataset index indices[r]
    """
    model.eval()
    chunks_i = []
    chunks_t = []
    chunks_idx = []
    for batch in loader:
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
        chunks_idx.append(idx)
    all_idx = torch.cat(chunks_idx, dim=0)
    h_img = torch.cat(chunks_i, dim=0)
    h_txt = torch.cat(chunks_t, dim=0)
    order = all_idx.argsort()
    return h_img[order], h_txt[order], all_idx[order]
