"""Evaluation utilities for Phase 3.

Val is 152 tiles from one WSI; a bare-mean Dice on that pool is
high-variance, so:
    bootstrap_dice         95% CI on mean per-tile Dice
    paired_wilcoxon_dice   one-sided paired test, baseline vs treatment

The hypothesis (stain-aug > baseline) is paired by tile, hence a
non-parametric paired test rather than two-sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class BootstrapCI:
    mean: float
    lo: float
    hi: float
    n: int
    n_boot: int


def per_tile_dice(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    """Per-tile binary Dice. pred and target are [N, H, W] in {0, 1}."""
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape} target={target.shape}")
    p = pred.astype(np.float64).reshape(pred.shape[0], -1)
    t = target.astype(np.float64).reshape(target.shape[0], -1)
    inter = (p * t).sum(axis=1)
    denom = p.sum(axis=1) + t.sum(axis=1)
    return (2.0 * inter + eps) / (denom + eps)


def per_tile_iou(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    """Per-tile binary IoU = TP / (TP + FP + FN).

    `pred` and `target` are binary [N, H, W] arrays already thresholded
    at 0.5 on logits by the caller. Returns a 1-D float64 array of
    length N in the same tile order as `per_tile_dice`.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape} target={target.shape}")
    p = pred.astype(np.float64).reshape(pred.shape[0], -1)
    t = target.astype(np.float64).reshape(target.shape[0], -1)
    inter = (p * t).sum(axis=1)
    union = p.sum(axis=1) + t.sum(axis=1) - inter
    return (inter + eps) / (union + eps)


def bootstrap_dice(
    per_tile: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """Bootstrap 95% (or other) CI on the mean per-tile Dice."""
    if per_tile.ndim != 1:
        raise ValueError(f"expected 1-D per-tile Dice array, got shape {per_tile.shape}")
    rng = np.random.default_rng(seed)
    n = len(per_tile)
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = per_tile[idx].mean()
    alpha = 1.0 - ci
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapCI(
        mean=float(per_tile.mean()),
        lo=float(lo),
        hi=float(hi),
        n=n,
        n_boot=n_boot,
    )


def paired_wilcoxon_dice(
    treatment: np.ndarray,
    baseline: np.ndarray,
    alternative: str = "greater",
) -> Tuple[float, float]:
    """Paired Wilcoxon signed-rank test on per-tile Dice. Returns (stat, p)."""
    from scipy.stats import wilcoxon

    if treatment.shape != baseline.shape:
        raise ValueError(
            f"shape mismatch: treatment={treatment.shape} baseline={baseline.shape}"
        )
    res = wilcoxon(treatment, baseline, alternative=alternative)
    return float(res.statistic), float(res.pvalue)
