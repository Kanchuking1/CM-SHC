"""Tests for CM-SHC loss components and the CMSHC model forward shapes.

These cover Stage 3 of the pipeline (the actual training signal).
"""

from __future__ import annotations

import math

import pytest
import torch

from src.models.hashing.cm_shc import CMSHC
from src.models.losses.semantic_center_loss import (
    bit_balance_loss,
    central_bce_loss,
    cmshc_full_loss,
    cross_modal_consistency_loss,
    log_cosh_quant_loss,
)


# ---------------------------------------------------------------------------
# Central BCE
# ---------------------------------------------------------------------------

def test_central_bce_zero_at_saturation():
    # Logits at ±large saturate σ to {0, 1}; if targets match, BCE → 0.
    q, B = 16, 4
    targets = torch.where(
        torch.randn(B, q) > 0, torch.tensor(1.0), torch.tensor(-1.0)
    )
    logits = targets * 50.0  # huge positive where t=1, huge negative where t=-1
    loss = central_bce_loss(logits, targets)
    assert loss.item() < 1e-6


def test_central_bce_pushes_logits_toward_targets():
    q, B = 8, 6
    targets = torch.where(
        torch.randn(B, q) > 0, torch.tensor(1.0), torch.tensor(-1.0)
    )
    # Start at zero logits; a single gradient step should move each logit
    # in the direction of its target.
    logits = torch.zeros(B, q, requires_grad=True)
    loss = central_bce_loss(logits, targets)
    loss.backward()
    g = logits.grad
    # grad of BCE w.r.t. logit is σ(x) - target_01; at x=0, σ=0.5.
    # target_01 is 1 for t=+1 → grad = -0.5 (push logit up)
    # target_01 is 0 for t=-1 → grad = +0.5 (push logit down)
    expected_sign = -targets.sign()
    assert torch.all(g.sign() == expected_sign), "grad direction is wrong"


def test_central_bce_shape_mismatch_raises():
    with pytest.raises(ValueError):
        central_bce_loss(torch.zeros(4, 8), torch.zeros(4, 7))


# ---------------------------------------------------------------------------
# Log-cosh quantization
# ---------------------------------------------------------------------------

def test_log_cosh_quant_zero_at_saturation():
    # At very large |logits|, sigmoid saturates → |2σ-1| → 1 → arg → 0 → loss → 0.
    q, B = 12, 3
    logits = torch.randn(B, q).sign() * 100.0
    loss = log_cosh_quant_loss(logits)
    assert loss.item() < 1e-5


def test_log_cosh_quant_positive_at_zero():
    # At logits = 0, σ = 0.5 → |2σ-1| = 0 → arg = -1 → log cosh(-1) = log cosh(1) > 0.
    q, B = 8, 4
    logits = torch.zeros(B, q)
    loss = log_cosh_quant_loss(logits)
    expected = math.log(math.cosh(1.0))
    assert loss.item() == pytest.approx(expected, abs=1e-5)


def test_log_cosh_quant_nonneg_everywhere():
    torch.manual_seed(0)
    for _ in range(5):
        logits = torch.randn(8, 16) * 3.0
        loss = log_cosh_quant_loss(logits)
        assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# Cross-modal consistency
# ---------------------------------------------------------------------------

def test_cross_modal_consistency_zero_when_equal():
    torch.manual_seed(0)
    x = torch.randn(6, 16)
    loss = cross_modal_consistency_loss(x, x)
    assert loss.item() < 1e-10


def test_cross_modal_consistency_positive_when_different():
    torch.manual_seed(1)
    f = torch.randn(4, 8)
    g = torch.randn(4, 8)
    loss = cross_modal_consistency_loss(f, g)
    assert loss.item() > 0.0


def test_cross_modal_consistency_symmetric():
    torch.manual_seed(2)
    f = torch.randn(5, 10)
    g = torch.randn(5, 10)
    a = cross_modal_consistency_loss(f, g)
    b = cross_modal_consistency_loss(g, f)
    assert torch.isclose(a, b, atol=1e-7)


def test_cross_modal_shape_mismatch_raises():
    with pytest.raises(ValueError):
        cross_modal_consistency_loss(torch.zeros(3, 8), torch.zeros(3, 4))


# ---------------------------------------------------------------------------
# Bit balance
# ---------------------------------------------------------------------------

def test_bit_balance_zero_when_columns_sum_to_zero():
    # Pair positive/negative rows so each column sums to zero.
    codes = torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]])
    loss = bit_balance_loss(codes)
    assert loss.item() < 1e-10


def test_bit_balance_positive_when_biased():
    codes = torch.ones(4, 3)
    loss = bit_balance_loss(codes)
    assert loss.item() > 0.0


def test_bit_balance_rejects_wrong_shape():
    with pytest.raises(ValueError):
        bit_balance_loss(torch.zeros(4, 3, 2))


# ---------------------------------------------------------------------------
# Full loss wrapper
# ---------------------------------------------------------------------------

def test_cmshc_full_loss_returns_parts_and_scalar():
    torch.manual_seed(3)
    B, q = 8, 32
    f = torch.randn(B, q, requires_grad=True)
    g = torch.randn(B, q, requires_grad=True)
    t = torch.where(torch.randn(B, q) > 0, torch.tensor(1.0), torch.tensor(-1.0))
    total, parts = cmshc_full_loss(f, g, t, lambda_bal=0.5)
    assert total.dim() == 0
    for k in ("center", "quant", "cross_modal", "balance", "total"):
        assert k in parts
    total.backward()
    assert f.grad is not None and g.grad is not None


def test_cmshc_full_loss_lambda_bal_zero_short_circuits():
    # With lambda_bal=0 (default), the balance term must be a zero tensor
    # rather than NaN / a computed value that could cost grads.
    torch.manual_seed(4)
    B, q = 6, 16
    f = torch.randn(B, q, requires_grad=True)
    g = torch.randn(B, q, requires_grad=True)
    t = torch.where(torch.randn(B, q) > 0, torch.tensor(1.0), torch.tensor(-1.0))
    _total, parts = cmshc_full_loss(f, g, t)
    assert parts["balance"].item() == 0.0


def test_cmshc_full_loss_decreases_with_one_step():
    # Sanity: plain SGD on the combined loss should reduce it on the first
    # step when starting from a random point.
    torch.manual_seed(5)
    B, q = 8, 32
    f = torch.randn(B, q, requires_grad=True)
    g = torch.randn(B, q, requires_grad=True)
    t = torch.where(torch.randn(B, q) > 0, torch.tensor(1.0), torch.tensor(-1.0))
    total0, _ = cmshc_full_loss(f, g, t)
    total0.backward()
    with torch.no_grad():
        f -= 0.1 * f.grad
        g -= 0.1 * g.grad
    total1, _ = cmshc_full_loss(f.detach(), g.detach(), t)
    assert total1.item() <= total0.item()


# ---------------------------------------------------------------------------
# CMSHC model forward shapes (MLP path)
# ---------------------------------------------------------------------------

def test_cmshc_forward_mlp_path_shapes():
    torch.manual_seed(6)
    q = 16
    model = CMSHC(bit_dim=q, text_model_name="", text_feature_dim=64)
    model.eval()
    x = torch.randn(3, 3, 224, 224)
    tf = torch.randn(3, 64)
    with torch.no_grad():
        f, g = model(image=x, text_features=tf)
    assert f.shape == (3, q)
    assert g.shape == (3, q)


def test_cmshc_encode_image_only():
    q = 8
    model = CMSHC(bit_dim=q, text_model_name="", text_feature_dim=32)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        f = model(image=x)
    assert f.shape == (2, q)


def test_cmshc_encode_text_only_mlp():
    q = 8
    model = CMSHC(bit_dim=q, text_model_name="", text_feature_dim=32)
    model.eval()
    tf = torch.randn(5, 32)
    with torch.no_grad():
        g = model(text_features=tf)
    assert g.shape == (5, q)


def test_cmshc_requires_some_input():
    q = 4
    model = CMSHC(bit_dim=q, text_model_name="", text_feature_dim=16)
    with pytest.raises(ValueError):
        model()
