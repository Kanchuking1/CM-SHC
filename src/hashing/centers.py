"""Semantic hash center construction for CM-SHC.

Three public entry points:

* ``build_class_similarity`` -- build a ``(C, C)`` class-similarity matrix
  ``S`` from labels (``method="cooccurrence"``) or from a pre-fitted
  multi-label classifier's predictions (``method="classifier"``).
* ``optimize_semantic_centers`` -- solve the relaxed SHC center
  optimization, returning ``H in {-1,+1}^{q x C}`` whose column inner
  products approximate ``2S - 1`` subject to the Gilbert-Varshamov bound.
* ``multi_label_target`` -- per-sample ``{-1,+1}^q`` code via bit-wise
  majority vote over class centers (CSQ multi-label rule).
"""

from __future__ import annotations

from typing import Literal

import torch

from .gv_bound import gilbert_varshamov_distance


SimilarityMethod = Literal["cooccurrence", "classifier"]


# ---------------------------------------------------------------------------
# Stage 1: class similarity matrix
# ---------------------------------------------------------------------------

def build_class_similarity(
    labels: torch.Tensor,
    method: SimilarityMethod = "cooccurrence",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute a ``(C, C)`` class-similarity matrix ``S in [0, 1]``.

    Parameters
    ----------
    labels : (N, C) float or long tensor
        Multi-hot label matrix for ``method='cooccurrence'``. For
        ``method='classifier'`` this is instead the ``(N, C)`` matrix of
        per-class prediction probabilities.
    method : {'cooccurrence', 'classifier'}
        * ``'cooccurrence'``: cosine similarity of class columns in the
          ground-truth label matrix ``Y``. Cheap, deterministic,
          label-only.
        * ``'classifier'``: SHC-style, built from a classifier's
          per-class prediction vectors. For each sample we zero out the
          argmax class, re-normalize, then average per ground-truth class.
          Captures visual confusability rather than just co-occurrence.
    eps : float
        Numerical floor for denominators.

    Returns
    -------
    (C, C) float tensor with diagonal ``== 1`` and values in ``[0, 1]``.
    """
    Y = labels.float()
    if Y.dim() != 2:
        raise ValueError(f"labels must be 2-D, got shape {tuple(Y.shape)}")

    if method == "cooccurrence":
        co = Y.t() @ Y  # (C, C) pairwise co-occurrence counts
        diag_sqrt = torch.sqrt(torch.diag(co).clamp(min=eps))
        S = co / (diag_sqrt.unsqueeze(0) * diag_sqrt.unsqueeze(1))
    elif method == "classifier":
        # Expect labels == probabilities (N, C); caller must pass them in.
        # Mask argmax per sample, renormalize, then average per true class.
        # Requires a paired ground-truth tensor -- use the companion helper.
        raise ValueError(
            "method='classifier' requires build_classifier_similarity(probs, true_labels). "
            "Call that function directly."
        )
    else:
        raise ValueError(f"Unknown method {method!r}")

    S = torch.clamp(S, 0.0, 1.0)
    S = 0.5 * (S + S.t())  # enforce symmetry
    S.fill_diagonal_(1.0)
    return S


def build_classifier_similarity(
    probs: torch.Tensor,
    true_labels: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """SHC-style class similarity from a pre-fitted multi-label classifier.

    For each sample with predictions ``p in R^C``: zero out its top-1
    entry, renormalize the remaining ``C - 1`` probabilities to sum to 1,
    then assign the resulting vector to every ground-truth class of that
    sample. Average across samples per class to obtain one row per class,
    then symmetrize and clip.

    Parameters
    ----------
    probs : (N, C) float tensor of per-class probabilities (sigmoid or softmax).
    true_labels : (N, C) multi-hot ground-truth label matrix.

    Returns
    -------
    (C, C) float tensor -- row ``c`` is the average "residual" distribution
    over the other classes for samples labelled with class ``c``.
    """
    if probs.shape != true_labels.shape:
        raise ValueError(
            f"probs {tuple(probs.shape)} and true_labels {tuple(true_labels.shape)} "
            "must have the same shape"
        )
    N, C = probs.shape
    masked = probs.clone().float()
    top1 = masked.argmax(dim=1)
    masked.scatter_(1, top1.unsqueeze(1), 0.0)
    masked = masked / masked.sum(dim=1, keepdim=True).clamp(min=eps)  # (N, C)

    Y = true_labels.float()
    counts = Y.sum(dim=0).clamp(min=eps)              # (C,)
    per_class = (Y.t() @ masked) / counts.unsqueeze(1)  # (C, C)

    S = 0.5 * (per_class + per_class.t())
    S = torch.clamp(S, 0.0, 1.0)
    S.fill_diagonal_(1.0)
    return S


# ---------------------------------------------------------------------------
# Stage 2: semantic hash center optimization
# ---------------------------------------------------------------------------

def optimize_semantic_centers(
    S: torch.Tensor,
    q: int,
    d_min: int | None = None,
    mu: float = 1.0,
    lr: float = 0.05,
    max_iters: int = 2000,
    beta_start: float = 1e-3,
    beta_end: float = 10.0,
    seed: int = 42,
    verbose: bool = False,
) -> torch.Tensor:
    """Solve the SHC center optimization and return ``H in {-1,+1}^{q x C}``.

    Minimizes (on a continuous relaxation ``H_tilde in R^{q x C}``):

        L = || (1/q) H^T H  -  (2S - 1) ||_F^2                    # semantic alignment
          + mu/(C*(C-1)) * sum_{i != j} ((h_i^T h_j) / q)^2       # soft spread
          + beta(t) * mean( (H_tilde^2 - 1)^2 )                   # push toward +/-1

    with ``beta`` annealed from ``beta_start`` to ``beta_end``. We then
    round ``H = sign(H_tilde)``.

    Parameters
    ----------
    S : (C, C) float tensor, symmetric with diag == 1, values in [0, 1].
    q : int
        Bit length of each center.
    d_min : int or None
        Desired minimum Hamming distance between centers. If None, use
        the Gilbert-Varshamov bound ``gilbert_varshamov_distance(q, C)``.
        Post-optimization we report whether the bound was met (we do not
        hard-project during training to keep the loss smooth).
    mu : float
        Weight of the soft-spread term.
    lr : float
        Adam learning rate.
    max_iters : int
        Number of optimization steps.
    beta_start, beta_end : float
        Annealing schedule for the binary push.
    seed : int
        Random init seed.
    verbose : bool
        If True, print loss/min-distance diagnostics.

    Returns
    -------
    H : (q, C) float tensor with entries in ``{-1, +1}``.
    """
    if S.dim() != 2 or S.size(0) != S.size(1):
        raise ValueError(f"S must be square, got shape {tuple(S.shape)}")
    C = S.size(0)
    if d_min is None:
        d_min = gilbert_varshamov_distance(q, C)

    R = (2.0 * S - 1.0).float()  # (C, C) in [-1, 1]

    gen = torch.Generator(device=S.device).manual_seed(seed)
    H = 0.3 * torch.randn(q, C, generator=gen, device=S.device, dtype=torch.float32)
    H.requires_grad_(True)

    opt = torch.optim.Adam([H], lr=lr)
    offdiag_mask = 1.0 - torch.eye(C, device=S.device)

    betas = torch.linspace(beta_start, beta_end, max_iters, device=S.device)

    for it in range(max_iters):
        G = (H.t() @ H) / q                        # (C, C) approx of center inner-products
        loss_sem = ((G - R) ** 2).mean()
        loss_spread = ((G * offdiag_mask) ** 2).sum() / (C * (C - 1))
        loss_bin = ((H * H - 1.0) ** 2).mean()
        loss = loss_sem + mu * loss_spread + betas[it] * loss_bin

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if verbose and (it == 0 or (it + 1) % max(1, max_iters // 10) == 0):
            print(
                f"iter {it + 1:5d}/{max_iters}  "
                f"L_sem={loss_sem.item():.4f}  "
                f"L_spread={loss_spread.item():.4f}  "
                f"L_bin={loss_bin.item():.4f}"
            )

    with torch.no_grad():
        H_bin = torch.sign(H.detach())
        # Replace any exact-zero bits with +1 (almost never hit).
        H_bin[H_bin == 0] = 1.0

    if verbose:
        min_d, mean_d = _pairwise_hamming_stats(H_bin)
        print(
            f"final min pairwise distance = {min_d} (target >= {d_min}), "
            f"mean = {mean_d:.2f}"
        )

    return H_bin


def _pairwise_hamming_stats(H: torch.Tensor) -> tuple[int, float]:
    """Min and mean pairwise Hamming distance across the columns of ``H``."""
    q, C = H.shape
    inner = H.t() @ H           # (C, C) in [-q, q]
    dist = (q - inner) / 2      # (C, C) Hamming distances
    dist_offdiag = dist + (q + 1) * torch.eye(C, device=H.device)  # mask diagonal
    min_d = int(dist_offdiag.min().item())
    n_pairs = C * (C - 1)
    mean_d = float((dist.sum() - dist.diag().sum()).item()) / max(n_pairs, 1)
    return min_d, mean_d


# ---------------------------------------------------------------------------
# CSQ-style data-agnostic centers (ablation baseline)
# ---------------------------------------------------------------------------

def _sylvester_hadamard(n: int) -> torch.Tensor:
    """Sylvester Hadamard matrix of order ``n`` (``n`` must be a power of 2).

    Built by recursive Kronecker-doubling:
    ``H_{2n} = [[1, 1], [1, -1]] (x) H_n``. All entries are in ``{-1, +1}``
    and all pairs of rows are orthogonal.
    """
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"n must be a positive power of 2, got {n}")
    H = torch.tensor([[1.0]])
    base = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    k = 1
    while k < n:
        H = torch.kron(base, H)
        k *= 2
    return H


def csq_hash_centers(q: int, C: int, seed: int = 42) -> torch.Tensor:
    """Data-agnostic hash centers following CSQ (Yuan et al., CVPR 2020).

    * When ``q`` is a power of 2 and ``C <= 2 * q`` we use rows of the
      Sylvester Hadamard matrix ``H_q`` (or ``[H_q; -H_q]`` for
      ``q < C <= 2 * q``). This guarantees minimum pairwise Hamming
      distance ``q / 2``.
    * Otherwise we fall back to i.i.d. ``Bernoulli(0.5)`` sampling, whose
      expected pairwise distance is ``q / 2``.

    Parameters
    ----------
    q : int
        Bit length per center.
    C : int
        Number of centers (classes).
    seed : int
        Seed for the Bernoulli fallback.

    Returns
    -------
    H : (q, C) float tensor with entries in ``{-1, +1}``.
    """
    if q <= 0 or C <= 0:
        raise ValueError(f"q and C must be positive; got q={q}, C={C}")

    is_pow2 = (q & (q - 1)) == 0
    if is_pow2 and C <= 2 * q:
        Hq = _sylvester_hadamard(q)  # (q, q) with entries in {+1, -1}
        if C <= q:
            rows = Hq[:C]  # (C, q)
        else:
            rows = torch.cat([Hq, -Hq], dim=0)[:C]  # (C, q), distance still q/2
        return rows.t().contiguous()  # (q, C)

    gen = torch.Generator().manual_seed(seed)
    bits = (torch.rand(q, C, generator=gen) < 0.5).float() * 2 - 1
    return bits


# ---------------------------------------------------------------------------
# Stage 3 helper: per-sample target code
# ---------------------------------------------------------------------------

def multi_label_target(
    Y: torch.Tensor,
    H: torch.Tensor,
    seed: int = 42,
) -> torch.Tensor:
    """Per-sample target code via bit-wise majority vote (CSQ multi-label rule).

    For sample ``i`` with multi-hot label ``y_i in {0,1}^C`` and centers
    ``H in {-1,+1}^{q x C}``::

        t_i = sign( sum_c y_{i,c} * h_c )   in {-1, +1}^q

    Ties (zero sum) are broken by independent fair coin flips (seeded).

    Parameters
    ----------
    Y : (N, C) multi-hot label matrix (float or bool / 0-1 long).
    H : (q, C) center matrix in ``{-1, +1}``.
    seed : int
        RNG seed for deterministic tie-breaks.

    Returns
    -------
    T : (N, q) float tensor with entries in ``{-1, +1}``.
    """
    Y = Y.float()
    if Y.dim() != 2 or H.dim() != 2:
        raise ValueError(
            f"Y must be (N, C) and H must be (q, C); got {tuple(Y.shape)}, {tuple(H.shape)}"
        )
    if Y.size(1) != H.size(1):
        raise ValueError(
            f"Class dim mismatch: Y has C={Y.size(1)}, H has C={H.size(1)}"
        )

    scores = Y @ H.t()  # (N, q)
    sign = torch.sign(scores)

    ties = sign == 0
    if ties.any():
        gen = torch.Generator(device=scores.device).manual_seed(seed)
        rand = torch.rand(scores.shape, generator=gen, device=scores.device)
        rand_bits = torch.where(
            rand < 0.5,
            torch.tensor(-1.0, device=scores.device),
            torch.tensor(1.0, device=scores.device),
        )
        sign = torch.where(ties, rand_bits, sign)
    return sign
