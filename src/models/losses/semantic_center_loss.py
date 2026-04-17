"""CM-SHC losses (Stage 3 of the CM-SHC pipeline).

The training objective combines four terms, each implemented as a standalone
``nn.Module``-free function so callers can pick and choose:

* :func:`central_bce_loss` -- per-sample central BCE against the target code
  ``t_i`` (CSQ / SHC form). Logits in, mean reduction out.
* :func:`log_cosh_quant_loss` -- smooth binary quantization on the sigmoid
  outputs; zero at saturation.
* :func:`cross_modal_consistency_loss` -- L2 between ``tanh(f)`` and
  ``tanh(g)`` that ties the two modality codes.
* :func:`bit_balance_loss` -- DCMH-style bit-balance on the raw continuous
  codes. Optional.

:func:`cmshc_full_loss` is a convenience wrapper that returns the scalar
objective together with a dict of per-term values for logging.

All functions expect ``f_logits, g_logits`` to be the *raw* network outputs
(no activation), with shape ``(B, q)``. Targets ``t`` are in ``{-1, +1}``
with shape ``(B, q)``.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Central BCE
# ---------------------------------------------------------------------------

def central_bce_loss(logits: torch.Tensor, targets_pm1: torch.Tensor) -> torch.Tensor:
    """Per-bit BCE-with-logits against the center target.

    Parameters
    ----------
    logits : (B, q) float tensor
        Raw network output (no sigmoid).
    targets_pm1 : (B, q) float tensor in ``{-1, +1}``
        Per-sample target code obtained by majority vote over class centers.

    Returns
    -------
    torch.Tensor (scalar)
        Mean BCE across batch and bits.

    Notes
    -----
    Targets are mapped from ``{-1, +1}`` to ``{0, 1}`` internally so the loss
    matches CSQ (Yuan et al., 2020) / SHC (Chen et al., 2025). We use the
    numerically stable ``binary_cross_entropy_with_logits`` variant.
    """
    if logits.shape != targets_pm1.shape:
        raise ValueError(
            f"logits and targets must match; got {logits.shape} vs {targets_pm1.shape}"
        )
    targets_01 = (targets_pm1 + 1.0) * 0.5
    return F.binary_cross_entropy_with_logits(logits, targets_01, reduction="mean")


# ---------------------------------------------------------------------------
# Smooth quantization (log cosh) on sigmoid outputs
# ---------------------------------------------------------------------------

def log_cosh_quant_loss(logits: torch.Tensor) -> torch.Tensor:
    """Smooth binary quantization penalty.

    Pushes the sigmoid activation toward 0 or 1, i.e. the signed continuous
    code toward ``{-1, +1}``. Zero when codes are perfectly saturated.

    Uses the CSQ formulation::

        L_quant = mean over (batch, bits) of  log cosh( |2 sigma(x) - 1| - 1 )

    ``|2 sigma(x) - 1|`` lives in ``[0, 1]``; subtracting 1 gives an argument
    in ``[-1, 0]`` which equals 0 at saturation. ``log cosh(0) = 0``.
    """
    a = (2.0 * torch.sigmoid(logits) - 1.0).abs() - 1.0
    # Numerically stable log cosh: log cosh(x) = |x| + log(1 + exp(-2|x|)) - log(2)
    abs_a = a.abs()
    lc = abs_a + torch.log1p(torch.exp(-2.0 * abs_a)) - math.log(2.0)
    return lc.mean()


# ---------------------------------------------------------------------------
# Cross-modal consistency
# ---------------------------------------------------------------------------

def cross_modal_consistency_loss(
    f_logits: torch.Tensor, g_logits: torch.Tensor
) -> torch.Tensor:
    """MSE between ``tanh(f)`` and ``tanh(g)`` -- ties the two modalities.

    Both inputs are raw network outputs of shape ``(B, q)``. Returns the mean
    squared difference over batch and bits.
    """
    if f_logits.shape != g_logits.shape:
        raise ValueError(
            f"image and text logits must match; got {f_logits.shape} vs {g_logits.shape}"
        )
    return F.mse_loss(torch.tanh(f_logits), torch.tanh(g_logits), reduction="mean")


# ---------------------------------------------------------------------------
# Bit balance (optional; inherited from DCMH)
# ---------------------------------------------------------------------------

def bit_balance_loss(codes: torch.Tensor) -> torch.Tensor:
    """DCMH-style balance: penalize bits that are always 0 or always 1 on a batch.

    ``mean_b (sum_i codes[i, b])^2``, normalized so the magnitude stays
    comparable across batch sizes.
    """
    if codes.dim() != 2:
        raise ValueError(f"expected (B, q) codes, got shape {tuple(codes.shape)}")
    B = codes.size(0)
    return (codes.sum(dim=0) ** 2).mean() / (B * B)


# ---------------------------------------------------------------------------
# Combined objective
# ---------------------------------------------------------------------------

def cmshc_full_loss(
    f_logits: torch.Tensor,
    g_logits: torch.Tensor,
    targets_pm1: torch.Tensor,
    lambda_center: float = 1.0,
    lambda_quant: float = 0.1,
    lambda_cm: float = 1.0,
    lambda_bal: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the full CM-SHC training loss.

    Parameters
    ----------
    f_logits, g_logits : (B, q) float tensors
        Raw image / text encoder outputs (pre-activation).
    targets_pm1 : (B, q) float tensor in ``{-1, +1}``
        Per-sample target codes (majority vote over class centers), cached on
        disk during center-build.
    lambda_center, lambda_quant, lambda_cm, lambda_bal : float
        Mixture weights. ``lambda_bal = 0`` disables the balance term (default).

    Returns
    -------
    total : torch.Tensor (scalar)
        Weighted sum of terms.
    parts : dict
        Per-term unweighted scalars for logging / diagnostics.
    """
    center = central_bce_loss(f_logits, targets_pm1) + central_bce_loss(
        g_logits, targets_pm1
    )
    quant = log_cosh_quant_loss(f_logits) + log_cosh_quant_loss(g_logits)
    cross = cross_modal_consistency_loss(f_logits, g_logits)
    if lambda_bal > 0:
        bal = bit_balance_loss(f_logits) + bit_balance_loss(g_logits)
    else:
        bal = torch.zeros((), device=f_logits.device, dtype=f_logits.dtype)

    total = (
        lambda_center * center
        + lambda_quant * quant
        + lambda_cm * cross
        + lambda_bal * bal
    )
    parts = {
        "center": center.detach(),
        "quant": quant.detach(),
        "cross_modal": cross.detach(),
        "balance": bal.detach(),
        "total": total.detach(),
    }
    return total, parts
