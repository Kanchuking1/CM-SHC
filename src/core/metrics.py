"""Retrieval metrics (mAP, precision@k) — extend for benchmark scripts."""

from __future__ import annotations


def mean_average_precision(*args, **kwargs):
    raise NotImplementedError("Wire mAP from retrieval bench in src.core.evaluator")
