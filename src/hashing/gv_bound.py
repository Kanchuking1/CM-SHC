"""Gilbert-Varshamov bound utilities for hash-center design.

The classical GV lower bound: there exists a binary code of length ``q``,
size ``M``, and minimum Hamming distance ``d`` whenever

    M * V(q, d - 1)  <  2**q                (eq. GV)

where ``V(q, r) = sum_{i=0}^{r} C(q, i)`` is the volume of a Hamming ball
of radius ``r`` in ``{0,1}^q`` (equivalently in ``{-1,+1}^q``).

For hash-center design we fix ``q`` (bit budget) and ``M = C`` (number of
classes) and ask for the *largest* achievable minimum distance ``d``. This
``d`` is what the CM-SHC optimizer uses as a constraint on pairwise
center inner products: ``h_i . h_j <= q - 2 * d``.
"""

from __future__ import annotations

from math import comb


def hamming_ball_volume(q: int, r: int) -> int:
    """``V(q, r) = sum_{i=0}^{r} C(q, i)`` -- number of binary strings of
    length ``q`` within Hamming distance ``r`` of a fixed codeword.

    Returns 0 when ``r < 0`` and ``2**q`` when ``r >= q``.
    """
    if q < 0:
        raise ValueError(f"q must be non-negative, got {q}")
    if r < 0:
        return 0
    r = min(r, q)
    return sum(comb(q, i) for i in range(r + 1))


def gilbert_varshamov_distance(q: int, num_codewords: int) -> int:
    """Largest ``d`` for which the GV bound guarantees a binary code with
    ``num_codewords`` codewords of length ``q`` and minimum distance ``>= d``.

    Uses the strict inequality ``C * V(q, d-1) < 2**q``. Searches over
    ``d = 1, 2, ..., q`` and returns the largest ``d`` for which the bound
    holds. Always returns at least 1 (distinct codewords) when feasible.

    Parameters
    ----------
    q : int
        Code length (hash-bit budget).
    num_codewords : int
        Number of codewords desired (``C``, the class count).

    Returns
    -------
    int
        Largest feasible minimum distance. If no ``d >= 1`` satisfies the
        bound (e.g. ``num_codewords > 2**q``), returns 0.

    Examples
    --------
    >>> gilbert_varshamov_distance(q=64, num_codewords=24)   # 24 classes, 64 bits
    24
    >>> gilbert_varshamov_distance(q=8, num_codewords=2)
    8
    """
    if q <= 0:
        raise ValueError(f"q must be positive, got {q}")
    if num_codewords <= 0:
        raise ValueError(f"num_codewords must be positive, got {num_codewords}")
    if num_codewords == 1:
        return q  # trivial; one codeword has no pairwise distance constraint

    capacity = 1 << q  # 2**q as an int
    best_d = 0
    for d in range(1, q + 1):
        if num_codewords * hamming_ball_volume(q, d - 1) < capacity:
            best_d = d
        else:
            break  # V(q, .) is monotone increasing, so we can stop
    return best_d
