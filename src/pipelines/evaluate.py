"""Evaluate a checkpoint (wire to src.core.evaluator when retrieval is implemented)."""

from __future__ import annotations

import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    return p.parse_args()


def main():
    parse_args()
    raise NotImplementedError("Evaluation pipeline not yet wired; see src.core.evaluator")


if __name__ == "__main__":
    main()
