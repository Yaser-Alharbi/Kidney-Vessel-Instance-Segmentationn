"""Rendering helpers for the demo: labels, mask overlays and Altair charts.

Holds only presentation, and is the single source of each arm's display
label, colour and blurb (the live lab and the story site both read it).
The list of arms comes from the config (`cfg.raw["augs"]`); this module
raises if an arm has no label.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import altair as alt
import numpy as np
import pandas as pd

# story site (plain-language headline)
TITLE = "Teaching AI to Find Blood Vessels in Kidney Tissue"
SUBTITLE = "...even when the stain colour changes from lab to lab"
# live lab (technical headline)
LAB_TITLE = "Blood Vessel Segmentation in Kidney Histology"
LAB_SUBTITLE = (
    "U-Net with a ResNet-34 encoder, trained under four stain-augmentation strategies "
    "and evaluated for robustness to inter-laboratory stain variation."
)

# technical name, display name, chart/overlay colour (each >= 5:1 on the dark theme)
ARM_STYLE: Dict[str, Tuple[str, str, str]] = {
    "rgb_aug": ("rgb_aug", "Baseline", "#a09cad"),
    "hed_only": ("hed_only", "Stain jitter", "#5b93ef"),
    "macenko_only": ("macenko_only", "Stain normalisation", "#d6a03c"),
    "full_stain_aware": ("full_stain_aware", "Jitter + normalisation", "#de6b93"),
}

# one-line plain explanation of each arm
ARM_BLURB: Dict[str, str] = {
    "rgb_aug": "Flips, rotations and brightness/contrast tweaks. "
               "The standard recipe the others are compared against.",
    "hed_only": "Randomly varies the strength of the purple and pink stains while "
                "training, so the model sees many lab colours.",
    "macenko_only": "Recolours every tile to one fixed reference stain, "
                    "in training and at test time.",
    "full_stain_aware": "Stain jitter while training, then recolours to the "
                        "reference stain at test time.",
}

# why each arm is in the study (story site)
ARM_ROLE: Dict[str, str] = {
    "rgb_aug": "The control. Standard geometric and brightness augmentation, no stain "
               "handling, so every other recipe is measured against it.",
    "hed_only": "First hypothesis. If the model sees many stain strengths in training, it "
                "should cope better with slides it has never seen.",
    "macenko_only": "Second hypothesis. Recolouring everything to one reference removes "
                    "colour differences, but the reference was designed for H&E, not "
                    "the PAS stain used here.",
    "full_stain_aware": "The additivity test. Tellez et al. (2019) found augmentation and "
                        "normalisation add up on H&E. Does that hold on PAS?",
}

# train / eval transform per arm, mirroring CLAUDE.md
ARM_PIPELINE: Dict[str, Tuple[str, str]] = {
    "rgb_aug": ("flips + rot90 + brightness/contrast", "plain"),
    "hed_only": ("flips + rot90 + HEDJitter(sigma=0.05)", "plain"),
    "macenko_only": ("flips + rot90 + MacenkoNormalize", "MacenkoNormalize"),
    "full_stain_aware": ("flips + rot90 + HEDJitter(sigma=0.05)", "MacenkoNormalize"),
}

GT_COLOR = "#19a974"
FP_COLOR = "#e2483d"
FN_COLOR = "#3fa7ff"


def check_arms(arms: Iterable[str]) -> None:
    """Raise if any configured arm has no label/colour here."""
    missing = [a for a in arms if a not in ARM_STYLE]
    if missing:
        raise ValueError(
            f"demo/views.py has no label for arm(s) {missing}; add them to ARM_STYLE"
        )


def arm_label(arm: str) -> str:
    """Display name for an arm."""
    return ARM_STYLE[arm][1]


def arm_color(arm: str) -> str:
    """Hex colour for an arm."""
    return ARM_STYLE[arm][2]


def _rgb(hex_color: str) -> np.ndarray:
    """'#rrggbb' -> float array of 3 channels."""
    h = hex_color.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)


def _paint(out: np.ndarray, where: np.ndarray, color: str, alpha: float) -> None:
    """Alpha-blend a flat colour into `out` wherever `where` is True."""
    out[where] = out[where] * (1.0 - alpha) + _rgb(color) * alpha


def overlay(  # pylint: disable=too-many-arguments
    img: np.ndarray,
    pred: Optional[np.ndarray] = None,
    gt: Optional[np.ndarray] = None,
    *,
    color: str = GT_COLOR,
    alpha: float = 0.5,
    show_gt: bool = True,
    error_view: bool = False,
) -> np.ndarray:
    """Draw prediction and/or ground-truth masks on an RGB tile.

    Error view (needs both masks): green = hit, red = extra, blue = missed.
    Otherwise the prediction is drawn in `color` and, if `show_gt`, the
    ground truth not covered by the prediction in green.
    """
    out = img.astype(np.float64).copy()
    p = pred.astype(bool) if pred is not None else None
    g = gt.astype(bool) if gt is not None else None
    if error_view and p is not None and g is not None:
        _paint(out, p & g, GT_COLOR, alpha)
        _paint(out, p & ~g, FP_COLOR, alpha)
        _paint(out, ~p & g, FN_COLOR, alpha)
    else:
        if g is not None and show_gt:
            _paint(out, g if p is None else g & ~p, GT_COLOR, alpha)
        if p is not None:
            _paint(out, p, color, alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def diff_overlay(
    img: np.ndarray,
    preds: Tuple[np.ndarray, np.ndarray],
    colors: Tuple[str, str],
    alpha: float = 0.6,
) -> np.ndarray:
    """Pixels only arm A finds in A's colour, only arm B finds in B's colour."""
    out = img.astype(np.float64).copy()
    a, b = preds[0].astype(bool), preds[1].astype(bool)
    _paint(out, a & ~b, colors[0], alpha)
    _paint(out, b & ~a, colors[1], alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def to_gray_image(arr: np.ndarray) -> np.ndarray:
    """Scale a float map to uint8 using its 1st-99th percentile range."""
    lo, hi = np.percentile(arr, [1, 99])
    scaled = (arr - lo) / max(hi - lo, 1e-8)
    return (np.clip(scaled, 0, 1) * 255).astype(np.uint8)


def _arm_scale(arms: Sequence[str]) -> alt.Scale:
    """Colour scale keyed by display label, in arm order."""
    return alt.Scale(
        domain=[arm_label(a) for a in arms],
        range=[arm_color(a) for a in arms],
    )


def seed_strip_chart(
    per_seed: Dict[str, Dict[str, float]],
    cis: Dict[str, Tuple[float, float, float]],
    arms: Sequence[str],
    metric: str,
) -> alt.Chart:
    """One dot per seed per arm, with the cross-seed mean and 95% CI."""
    dots = pd.DataFrame(
        [
            {"arm": arm_label(a), "seed": s, metric: v}
            for a in arms
            for s, v in per_seed[a].items()
        ]
    )
    bars = pd.DataFrame(
        [
            {"arm": arm_label(a), "mean": m, "lo": lo, "hi": hi}
            for a in arms
            for (m, lo, hi) in [cis[a]]
        ]
    )
    order = [arm_label(a) for a in arms]
    y = alt.Y("arm:N", sort=order, title=None)
    pts = (
        alt.Chart(dots)
        .mark_circle(size=70, opacity=0.75)
        .encode(
            x=alt.X(f"{metric}:Q", title=f"mean per-tile {metric} per training run"),
            y=y,
            yOffset=alt.YOffset("jitter:Q"),
            color=alt.Color("arm:N", scale=_arm_scale(arms), legend=None),
            tooltip=["arm", "seed", alt.Tooltip(f"{metric}:Q", format=".4f")],
        )
        .transform_calculate(jitter="random() * 20 - 10")
    )
    rule = alt.Chart(bars).mark_rule(strokeWidth=3, color="#888").encode(
        x="lo:Q", x2="hi:Q", y=y
    )
    tick = alt.Chart(bars).mark_tick(thickness=3, size=26, color="#888").encode(
        x="mean:Q",
        y=y,
        tooltip=[
            "arm",
            alt.Tooltip("mean:Q", format=".4f"),
            alt.Tooltip("lo:Q", format=".4f"),
            alt.Tooltip("hi:Q", format=".4f"),
        ],
    )
    return (rule + tick + pts).properties(height=260)


def sweep_chart(
    rows: List[Dict[str, float]],
    arms: Sequence[str],
    x_title: str,
    band: Tuple[float, float],
) -> alt.Chart:
    """Dice vs stain shift for each arm, with the training range shaded."""
    df = pd.DataFrame(
        [
            {"shift": r["shift"], "arm": arm_label(a), "Dice": r[a]}
            for r in rows
            for a in arms
        ]
    )
    shade = (
        alt.Chart(pd.DataFrame([{"lo": band[0], "hi": band[1]}]))
        .mark_rect(opacity=0.12, color="#19a974")
        .encode(x="lo:Q", x2="hi:Q")
    )
    x_range = [min(df["shift"].min(), band[0]), max(df["shift"].max(), band[1])]
    lines = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("shift:Q", title=x_title,
                    scale=alt.Scale(domain=x_range, zero=False, nice=False)),
            y=alt.Y("Dice:Q"),
            color=alt.Color(
                "arm:N", scale=_arm_scale(arms), legend=alt.Legend(title=None, orient="bottom")
            ),
            tooltip=["arm", "shift", alt.Tooltip("Dice:Q", format=".4f")],
        )
    )
    return (shade + lines).properties(height=280, width="container")
