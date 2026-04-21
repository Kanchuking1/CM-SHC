"""Unit tests for GV bound, class similarity, center optimization, and
multi-label target construction (CM-SHC stages 1-3)."""

from __future__ import annotations

import math

import pytest
import torch

from src.hashing.centers import (
    _sylvester_hadamard,
    build_class_similarity,
    build_classifier_similarity,
    csq_hash_centers,
    multi_label_target,
    optimize_semantic_centers,
)
from src.hashing.gv_bound import gilbert_varshamov_distance, hamming_ball_volume


# ---------------------------------------------------------------------------
# Gilbert-Varshamov bound
# ---------------------------------------------------------------------------

def test_hamming_ball_volume_edges():
    assert hamming_ball_volume(0, 0) == 1
    assert hamming_ball_volume(5, -1) == 0
    assert hamming_ball_volume(5, 10) == 2 ** 5
    # V(4, 2) = C(4,0)+C(4,1)+C(4,2) = 1+4+6 = 11
    assert hamming_ball_volume(4, 2) == 11


def test_gv_distance_two_codewords_nontrivial():
    # Sanity: GV always certifies a strictly positive minimum distance when
    # there is room for 2 codewords. We do NOT claim d == q here; that is
    # achievable by hand-construction (0...0 vs 1...1) but is stronger than
    # the GV lower bound.
    for q in (4, 8, 16, 32):
        d = gilbert_varshamov_distance(q, 2)
        assert d >= 1
        # For these q, GV gives at least ~q/3 -- just a floor check.
        assert d >= q // 4


def test_gv_distance_single_codeword_is_q():
    # Trivial case: one codeword has no pairwise distance constraint.
    assert gilbert_varshamov_distance(q=10, num_codewords=1) == 10


def test_gv_distance_returns_zero_when_too_many_codewords():
    # 2**3 = 8 codewords is the max size of length-3 binary codes;
    # asking for 16 has no feasible d >= 1.
    assert gilbert_varshamov_distance(q=3, num_codewords=16) == 0


def test_gv_distance_monotone_in_codewords():
    # Increasing num_codewords can only shrink the feasible distance.
    prev = float("inf")
    for M in (2, 4, 8, 16, 32, 64):
        d = gilbert_varshamov_distance(q=16, num_codewords=M)
        assert d <= prev
        prev = d


def test_gv_distance_mirflickr_setting():
    # q=64 bits, C=24 classes -- the CM-SHC MIR-Flickr setting.
    d = gilbert_varshamov_distance(q=64, num_codewords=24)
    assert d >= 16, f"expected a comfortable margin; got d={d}"
    assert d <= 64


# ---------------------------------------------------------------------------
# Class similarity
# ---------------------------------------------------------------------------

def test_cooccurrence_identity_when_labels_disjoint():
    # Each sample has exactly one class -> columns of Y are orthogonal,
    # co-occurrence matrix is diag -> cosine similarity is identity.
    Y = torch.tensor(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1],
         [1, 0, 0],
         [0, 1, 0]], dtype=torch.float32,
    )
    S = build_class_similarity(Y, method="cooccurrence")
    assert torch.allclose(S, torch.eye(3))


def test_cooccurrence_symmetric_and_diag_one():
    torch.manual_seed(0)
    Y = (torch.rand(200, 5) > 0.6).float()
    # guarantee every class is used at least once
    Y[0] = 1.0
    S = build_class_similarity(Y, method="cooccurrence")
    assert torch.allclose(S, S.t(), atol=1e-6)
    assert torch.allclose(torch.diag(S), torch.ones(5), atol=1e-6)
    assert (S >= 0).all() and (S <= 1).all()


def test_cooccurrence_detects_copies():
    # Classes 0 and 1 co-occur perfectly -> similarity 1.
    Y = torch.tensor(
        [[1, 1, 0],
         [1, 1, 0],
         [0, 0, 1],
         [1, 1, 0]], dtype=torch.float32,
    )
    S = build_class_similarity(Y, method="cooccurrence")
    assert S[0, 1].item() == pytest.approx(1.0, abs=1e-6)
    assert S[0, 2].item() == pytest.approx(0.0, abs=1e-6)


def test_classifier_similarity_shape_and_diag():
    torch.manual_seed(1)
    N, C = 50, 4
    probs = torch.softmax(torch.randn(N, C), dim=1)
    Y = torch.zeros(N, C)
    Y[torch.arange(N), torch.randint(C, (N,))] = 1.0
    S = build_classifier_similarity(probs, Y)
    assert S.shape == (C, C)
    assert torch.allclose(torch.diag(S), torch.ones(C), atol=1e-6)
    assert torch.allclose(S, S.t(), atol=1e-6)
    assert (S >= 0).all() and (S <= 1).all()


# ---------------------------------------------------------------------------
# Semantic hash center optimization
# ---------------------------------------------------------------------------

def test_optimize_returns_pm_one_codes():
    C = 6
    S = torch.eye(C) * 1.0 + 0.2 * (1 - torch.eye(C))
    H = optimize_semantic_centers(S, q=32, max_iters=400, seed=0)
    assert H.shape == (32, C)
    vals = torch.unique(H)
    assert torch.all((vals == -1.0) | (vals == 1.0)), f"non-binary values: {vals.tolist()}"


def test_optimize_respects_minimum_distance():
    # With q = 64 and C = 10 the GV bound is large; the solver should at
    # least produce distinct codewords, and should usually satisfy d_min / 2.
    C = 10
    q = 64
    S = torch.eye(C) + 0.3 * (1 - torch.eye(C))
    H = optimize_semantic_centers(S, q=q, max_iters=800, seed=0)
    inner = H.t() @ H
    dist = (q - inner) / 2
    offdiag = dist + (q + 1) * torch.eye(C)
    min_d = int(offdiag.min().item())
    assert min_d >= 1, "columns must be distinct"
    # GV distance is 22+ for (64, 10); allow a relaxed floor because the
    # solver is a soft relaxation (not hard-projected).
    gv = gilbert_varshamov_distance(q, C)
    assert min_d >= gv // 4, f"min distance {min_d} too far below GV floor {gv}"


def test_optimize_reflects_similarity_ordering():
    # If S[0,1] is much larger than S[0,2], h0 should be closer to h1 than to h2
    # in Hamming distance after optimization (statistically).
    C = 4
    q = 32
    S = torch.tensor(
        [[1.0, 0.9, 0.1, 0.1],
         [0.9, 1.0, 0.1, 0.1],
         [0.1, 0.1, 1.0, 0.9],
         [0.1, 0.1, 0.9, 1.0]],
    )
    H = optimize_semantic_centers(S, q=q, max_iters=1500, seed=0)
    inner = H.t() @ H
    dist = (q - inner) / 2
    # The similar pairs (0,1) and (2,3) should be closer than the dissimilar
    # cross pairs (0,2), (0,3), (1,2), (1,3).
    similar_pairs = [(0, 1), (2, 3)]
    dissimilar_pairs = [(0, 2), (0, 3), (1, 2), (1, 3)]
    mean_sim = sum(dist[i, j].item() for i, j in similar_pairs) / len(similar_pairs)
    mean_dis = sum(dist[i, j].item() for i, j in dissimilar_pairs) / len(dissimilar_pairs)
    assert mean_sim < mean_dis, f"expected {mean_sim} < {mean_dis}"


# ---------------------------------------------------------------------------
# CSQ Hadamard / Bernoulli baseline centers
# ---------------------------------------------------------------------------

def test_sylvester_hadamard_orthogonality():
    for n in (1, 2, 4, 8, 16):
        H = _sylvester_hadamard(n)
        assert H.shape == (n, n)
        assert set(torch.unique(H).tolist()).issubset({-1.0, 1.0})
        # H @ H^T == n * I
        assert torch.allclose(H @ H.t(), n * torch.eye(n), atol=1e-5)


def test_csq_centers_power_of_two_distance():
    # For q a power of 2 and C <= q we should get exact pairwise Hamming
    # distance q/2 between distinct centers (Hadamard rows).
    for q, C in [(4, 4), (8, 4), (8, 8), (16, 10), (64, 24)]:
        H = csq_hash_centers(q=q, C=C, seed=0)
        assert H.shape == (q, C)
        assert set(torch.unique(H).tolist()).issubset({-1.0, 1.0})
        inner = H.t() @ H
        offdiag = inner - q * torch.eye(C)
        # All off-diagonal inner products should be zero (Hadamard rows orthogonal)
        # which corresponds to Hamming distance q/2.
        assert torch.allclose(offdiag, torch.zeros_like(offdiag), atol=1e-5)


def test_csq_centers_non_power_of_two_falls_back_to_bernoulli():
    # q=12 is not a power of 2 -> Bernoulli fallback.
    H = csq_hash_centers(q=12, C=6, seed=0)
    assert H.shape == (12, 6)
    assert set(torch.unique(H).tolist()).issubset({-1.0, 1.0})
    # Centers should still be distinct (with overwhelmingly high probability).
    inner = H.t() @ H
    offdiag = inner - 12 * torch.eye(6)
    assert (offdiag.abs() < 12).all()


def test_csq_centers_extended_when_C_up_to_2q():
    # When C is in (q, 2q] we use [H; -H] concatenation; min distance
    # remains q/2 and negated-row pairs achieve distance q.
    q, C = 8, 16
    H = csq_hash_centers(q=q, C=C, seed=0)
    assert H.shape == (q, C)
    inner = H.t() @ H
    dist = (q - inner) / 2
    offdiag = dist + (q + 1) * torch.eye(C)  # mask diag
    assert int(offdiag.min().item()) >= q // 2


# ---------------------------------------------------------------------------
# Multi-label target (majority vote)
# ---------------------------------------------------------------------------

def test_multi_label_target_single_label_returns_center():
    C, q = 5, 16
    H = torch.tensor([[1, -1], [-1, 1], [1, 1], [-1, -1], [1, -1]], dtype=torch.float32).repeat_interleave(1, dim=1)
    # rebuild H with clean shape
    H = torch.where(torch.randn(q, C) > 0, torch.tensor(1.0), torch.tensor(-1.0))
    Y = torch.eye(C)  # each row is a one-hot label
    T = multi_label_target(Y, H)
    for c in range(C):
        assert torch.equal(T[c], H[:, c]), f"row {c} should equal H[:, {c}]"


def test_multi_label_target_shapes():
    torch.manual_seed(2)
    N, C, q = 13, 7, 24
    Y = (torch.rand(N, C) > 0.7).float()
    Y[0] = 0  # force a tie-break branch on sample 0
    H = torch.where(torch.randn(q, C) > 0, torch.tensor(1.0), torch.tensor(-1.0))
    T = multi_label_target(Y, H)
    assert T.shape == (N, q)
    assert set(torch.unique(T).tolist()).issubset({-1.0, 1.0})


def test_multi_label_target_tie_break_is_deterministic():
    C, q = 4, 10
    H = torch.where(torch.randn(q, C) > 0, torch.tensor(1.0), torch.tensor(-1.0))
    # all-zero label -> every bit is a tie
    Y = torch.zeros(3, C)
    T1 = multi_label_target(Y, H, seed=123)
    T2 = multi_label_target(Y, H, seed=123)
    T3 = multi_label_target(Y, H, seed=999)
    assert torch.equal(T1, T2)
    assert not torch.equal(T1, T3)  # different seed -> different tie break (almost surely)
