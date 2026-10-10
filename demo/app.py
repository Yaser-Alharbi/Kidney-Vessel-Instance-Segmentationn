"""Live lab: hands-on tools for the kidney blood-vessel segmentation study.

    streamlit run demo/app.py

Three tools (tile explorer, stain stress test, upload) for the four
stain-augmentation recipes. The story, methods and full results live on
the static site (`data.SITE_URL`). Stored scores are read from
`artifacts_v2/phase3_summary.json`; images and live inference use the
seed-42 models.
"""

from __future__ import annotations

import io
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# pylint: disable=wrong-import-position
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from demo import data, inference, views
from src.data.transforms import hed_shift

# pylint: enable=wrong-import-position

BASE_ARM = "rgb_aug"
JITTER_ARM = "hed_only"
SCALE_BAND = (0.95, 1.05)  # HEDJitter(sigma=0.05): alpha in 1 +/- 0.05
BIAS_BAND = (-0.05, 0.05)  # HEDJitter(sigma=0.05): beta in +/- 0.05
SWEEP_POINTS = np.round(np.linspace(0.7, 1.3, 7), 2)
TILE_BLOCK_WIDTH = 1600 # px: tile + 2x2 grid

SPLIT_LABELS = {
    "val": "Validation slide (expert labels)",
    "test": "Test slides (auto-generated labels)",
}
SHIFT_LABELS = {
    "h_scale": "Purple stain strength",
    "e_scale": "Pink stain strength",
    "h_bias": "purple offset",
    "e_bias": "pink offset",
}
DICE_HELP = (
    "Dice measures how well the model's vessels overlap the expert outline: "
    "2 x shared pixels / (model pixels + expert pixels). "
    "1.0 is a perfect match and 0 is no overlap."
)
SHIFT_DICE_HELP = (
    DICE_HELP + " The number in brackets is the change from the tile's original colours."
)
ABOUT = f"""**{views.LAB_TITLE}**

{views.LAB_SUBTITLE} This app runs the four seed-42 models live on
kidney tissue tiles.

- Full story, methods and results: {data.SITE_URL}
- Code: {data.REPO_URL}
"""


@dataclass(frozen=True)
class Selection:
    """Tile and display choices shared by every tab."""

    split: str
    tile_id: str
    alpha: float
    show_gt: bool


# ---------------------------------------------------------------- cached data


@st.cache_resource
def _cfg():
    """Demo config (v2)."""
    return data.load_demo_config()


@st.cache_resource
def _arms() -> List[str]:
    """Arms from the config, checked against the label table."""
    arms = data.arms_from_config(_cfg())
    views.check_arms(arms)
    return arms


@st.cache_resource
def _summary() -> Dict:
    """Committed v2 summary (about 6 MB, loaded once per process)."""
    return data.load_summary(_cfg())


@st.cache_resource
def _cube(split: str) -> data.Cube:
    """Per-tile Dice/IoU for every (arm, seed) on one split."""
    return data.build_cube(_summary(), _arms(), split)


@st.cache_resource
def _device():
    """Torch device for inference."""
    return inference.resolve_device()


# LRU cap on loaded models (about 93 MiB each); not one-per-arm by design.
@st.cache_resource(max_entries=4, show_spinner=False)
def _model(arm: str, seed: int):
    """Loaded checkpoint for (arm, seed), or None if it is not on disk."""
    path = data.checkpoint_path(_cfg(), arm, seed)
    return None if path is None else inference.load_model(path, _device())


@st.cache_data(max_entries=64, show_spinner=False)
def _tile(tile_id: str):
    """(RGB tile, ground-truth mask)."""
    return data.load_tile(_cfg(), tile_id)


def _run_model(arm: str, img: np.ndarray, macenko_input: bool = False) -> Optional[np.ndarray]:
    """Uncached seed-42 forward pass for an arbitrary image."""
    model = _model(arm, data.LIVE_SEED)
    if model is None:
        return None
    if macenko_input and arm not in inference.MACENKO_AT_EVAL:
        img = inference.macenko(img)
    return inference.predict(model, arm, img, _device())


@st.cache_data(max_entries=256, show_spinner=False)
def _predict(arm: str, img: np.ndarray, macenko_input: bool = False) -> Optional[np.ndarray]:
    """Cached `_run_model`."""
    return _run_model(arm, img, macenko_input)


def _tile_prediction(arm: str, tile_id: str) -> Optional[np.ndarray]:
    """Stored seed-42 mask if present (deployed), else live inference."""
    pred = data.stored_prediction(arm, tile_id)
    if pred is not None:
        return pred
    return _predict(arm, _tile(tile_id)[0])


def _preload() -> None:
    """Load the seed-42 model for every arm at startup."""
    with st.spinner("Loading the four models (the first visit can take a moment)..."):
        for arm in _arms():
            _model(arm, data.LIVE_SEED)


# ------------------------------------------------------------------- helpers


def _arm_header(arm: str) -> None:
    """Coloured dot, display name and technical id."""
    # fixed two-line header so every image in the 2x2 grid starts at the same height
    st.markdown(
        f"<div style='white-space:nowrap;line-height:1.35'>"
        f"<span style='color:{views.arm_color(arm)}'>&#9679;</span> "
        f"<b>{views.arm_label(arm)}</b><br><code style='font-size:0.72rem'>{arm}</code></div>",
        unsafe_allow_html=True,
    )


def _arm_grid(draw: Callable[[str], None]) -> None:
    """One card per arm in a 2x2 grid; `draw` fills the card below its header."""
    arms = _arms()
    for start in range(0, len(arms), 2):
        for col, arm in zip(st.columns(2), arms[start:start + 2]):
            with col:
                _arm_header(arm)
                draw(arm)


def _legend(error_view: bool) -> None:
    """Overlay colour key."""
    if error_view:
        items = [(views.GT_COLOR, "found correctly"),
                 (views.FP_COLOR, "marked, but not a vessel"),
                 (views.FN_COLOR, "vessel the model missed")]
        st.markdown(
            " &nbsp; ".join(f"<span style='color:{c}'>&#9632;</span> {t}" for c, t in items),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "Each model's vessels are drawn in its recipe colour. "
            f"<span style='color:{views.GT_COLOR}'>&#9632;</span> expert-outlined vessel "
            "the model missed.",
            unsafe_allow_html=True,
        )


def _two_columns():
    """Left column for the tile, right column for the 2x2 grid.

    5:4 makes the grid (two rows of half-width images plus captions) about
    as tall as the single tile on the left. The block is capped at
    TILE_BLOCK_WIDTH so the images stay compact on wide screens.
    """
    with st.container(width=TILE_BLOCK_WIDTH):
        return st.columns([5, 4], gap="large")


# ---------------------------------------------------------------- page parts


def _header() -> None:
    """Title, subtitle and links to the story site and the code."""
    st.title(views.LAB_TITLE)
    st.markdown(views.LAB_SUBTITLE)
    st.markdown(
        "Compare the four models on held-out tiles, test them under synthetic stain shifts, "
        "or run them on your own image.  \n"
        f"[Methods and results]({data.SITE_URL}) &nbsp;·&nbsp; [Source code]({data.REPO_URL})"
    )


def _picker() -> Selection:
    """Slide, ranking, tile and display settings, shared by the tabs."""
    c1, c2, c3, c4 = st.columns([2.5, 4, 2, 1.3], vertical_alignment="bottom")
    split = c1.selectbox("Slide", list(SPLIT_LABELS), format_func=SPLIT_LABELS.get)
    cube = _cube(split)
    sort = c2.selectbox(
        "Show me", data.SORTS,
        help=f"Rankings use the average Dice of all {len(cube.seeds)} training runs. The images "
             f"show one run (seed {data.LIVE_SEED}), so a single tile can disagree.",
    )
    ids = data.rank_tiles(cube, sort, data.available_tiles(_cfg(), cube, split))
    if not ids:
        st.error("No tiles available for this slide.")
        st.stop()
    rank = {t: i for i, t in enumerate(ids, start=1)}
    tile_id = c3.selectbox("Tile", ids, format_func=lambda t: f"#{rank[t]}  ·  {t}")
    with c4.popover("Display", width="stretch", wrap=False):
        alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.5, 0.05)
        show_gt = st.checkbox("Show expert outline", value=True)
    if split == "test":
        st.caption(
            "Test slides use automatically generated (noisy) labels. Scores there compare "
            "recipes with each other; they are not absolute accuracy."
        )
    return Selection(split, tile_id, alpha, show_gt)


# ------------------------------------------------------------------ explorer


def _tab_explorer(sel: Selection) -> None:
    """The tile, the expert outline and every recipe's prediction."""
    img, gt = _tile(sel.tile_id)
    error_view = st.toggle("Show mistakes", help="Colour each pixel by hit, extra or missed.")
    _legend(error_view)
    left, right = _two_columns()
    with left:
        st.markdown("**Tissue tile**")
        st.image(views.overlay(img, gt=gt, alpha=sel.alpha, show_gt=sel.show_gt),
                 width="stretch")
        st.caption(f"Tile {sel.tile_id} · vessels cover {gt.mean():.1%} of this tile")
    preds: Dict[str, Optional[np.ndarray]] = {}

    def draw(arm: str) -> None:
        preds[arm] = pred = _tile_prediction(arm, sel.tile_id)
        if pred is None:
            st.info("No model or stored prediction for this recipe.")
            return
        st.image(views.overlay(img, pred, gt, color=views.arm_color(arm), alpha=sel.alpha,
                               show_gt=sel.show_gt, error_view=error_view), width="stretch")
        st.markdown(f"Dice **{inference.scores(pred, gt)[0]:.3f}**", help=DICE_HELP)

    with right:
        _arm_grid(draw)
    with st.expander("Compare two recipes"):
        _arm_diff(img, gt, preds, sel.alpha)
    with st.expander("All scores for this tile"):
        _score_table(sel, preds, gt)


def _arm_diff(img: np.ndarray, gt: np.ndarray, preds: Dict, alpha: float) -> None:
    """Pixels that only one of two recipes marks as vessel."""
    ready = [a for a, p in preds.items() if p is not None]
    if len(ready) < 2:
        st.caption("Needs at least two recipes with a prediction.")
        return
    cols = st.columns([1, 1, 2])
    a = cols[0].selectbox("Recipe A", ready, format_func=views.arm_label,
                          index=ready.index(JITTER_ARM) if JITTER_ARM in ready else 0)
    b = cols[1].selectbox("Recipe B", ready, format_func=views.arm_label,
                          index=ready.index(BASE_ARM) if BASE_ARM in ready else 1)
    pa, pb, g = preds[a].astype(bool), preds[b].astype(bool), gt.astype(bool)
    with cols[2]:
        st.image(views.diff_overlay(img, (pa, pb), (views.arm_color(a), views.arm_color(b)),
                                    alpha), width=420)
    for col, arm, only in ((cols[0], a, pa & ~pb), (cols[1], b, pb & ~pa)):
        col.markdown(f"<span style='color:{views.arm_color(arm)}'>&#9632;</span> only "
                     f"{views.arm_label(arm)}: {int(only.sum())} px", unsafe_allow_html=True)
        col.caption(f"{int((only & g).sum())} px of them are expert-outlined vessel")


def _score_table(sel: Selection, preds: Dict, gt: np.ndarray) -> None:
    """Live and stored scores for the selected tile."""
    cube = _cube(sel.split)
    seed = st.selectbox("Stored scores from training run", cube.seeds,
                        index=cube.seeds.index(data.LIVE_SEED), format_func=lambda s: f"seed {s}")
    ti, si = cube.tile_index(sel.tile_id), cube.seed_index(seed)
    rows = []
    for ai, arm in enumerate(_arms()):
        pred = preds.get(arm)
        dice, iou = inference.scores(pred, gt) if pred is not None else (np.nan, np.nan)
        runs = cube.dice[ai, :, ti]
        rows.append({
            "Recipe": views.arm_label(arm),
            "Dice (model shown)": round(dice, 3),
            "IoU (model shown)": round(iou, 3),
            f"Dice (seed {seed})": round(float(cube.dice[ai, si, ti]), 3),
            f"Dice, {len(runs)}-run average": f"{runs.mean():.3f} ± {runs.std():.3f}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        f"Model shown: the seed-{data.LIVE_SEED} model behind the images above. "
        "IoU is a stricter overlap score: shared pixels / pixels marked by either. "
        f"± is the standard deviation across the {len(cube.seeds)} training runs."
    )


# --------------------------------------------------------------- stress test


def _shift_controls() -> Dict[str, float]:
    """Stain sliders; offsets live in an expander."""
    c1, c2 = st.columns(2)
    p = {
        "h_scale": c1.slider(f"{SHIFT_LABELS['h_scale']} (haematoxylin)", 0.5, 1.5, 1.0, 0.05,
                             help="1.0 is the tile as scanned."),
        "e_scale": c2.slider(f"{SHIFT_LABELS['e_scale']} (eosin)", 0.5, 1.5, 1.0, 0.05,
                             help="1.0 is the tile as scanned."),
        "h_bias": 0.0,
        "e_bias": 0.0,
    }
    with st.expander("Advanced: stain offsets"):
        b1, b2 = st.columns(2)
        p["h_bias"] = b1.slider("Purple offset", -0.3, 0.3, 0.0, 0.01)
        p["e_bias"] = b2.slider("Pink offset", -0.3, 0.3, 0.0, 0.01)
    st.caption(
        f"Stain-jitter training covered strengths {SCALE_BAND[0]} to {SCALE_BAND[1]} and "
        f"offsets {BIAS_BAND[0]} to {BIAS_BAND[1]}. Beyond that, every model is extrapolating."
    )
    outside = [SHIFT_LABELS[k].lower() for k, v in p.items()
               if not (SCALE_BAND if "scale" in k else BIAS_BAND)[0] <= v
               <= (SCALE_BAND if "scale" in k else BIAS_BAND)[1]]
    if outside:
        st.warning("Outside the training range: " + ", ".join(outside))
    return p


def _shifted(img: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    """Apply the shift; identity settings return the tile untouched."""
    if p["h_scale"] == 1.0 and p["e_scale"] == 1.0 and p["h_bias"] == 0.0 and p["e_bias"] == 0.0:
        return img
    return hed_shift(img, **p)


def _tab_stress(sel: Selection) -> None:
    """Live stain-shift robustness on the selected tile."""
    img, gt = _tile(sel.tile_id)
    st.markdown(
        "Labs stain tissue differently. Change this tile's stain and see which models hold "
        f"up. Every image here is a live prediction from the seed-{data.LIVE_SEED} models. "
        "This is an illustration."
    )
    p = _shift_controls()
    with st.container(horizontal=True, gap="large"):
        mac = st.toggle(
            "Standardise colours first (Macenko)",
            help="Applies to Baseline and Stain jitter only; the other two recipes "
                 "already standardise colours at test time.",
        )
        error_view = st.toggle("Show mistakes", key="stress_errors")
    shifted = _shifted(img, p)
    left, right = _two_columns()
    with left:
        st.markdown("**Re-stained tile**")
        st.image(shifted, width="stretch")
        st.caption(f"Tile {sel.tile_id} (picked above)")

    def draw(arm: str) -> None:
        pred, ref = _predict(arm, shifted, mac), _predict(arm, img, mac)
        if pred is None:
            st.info("No model for this recipe.")
            return
        st.image(views.overlay(shifted, pred, gt, color=views.arm_color(arm), alpha=sel.alpha,
                               show_gt=sel.show_gt, error_view=error_view), width="stretch")
        d, d0 = inference.scores(pred, gt)[0], inference.scores(ref, gt)[0]
        st.markdown(f"Dice **{d:.3f}** ({d - d0:+.3f})", help=SHIFT_DICE_HELP)
        if mac and arm in inference.MACENKO_AT_EVAL:
            st.caption("Already standardises colours at test time.")

    with right:
        _arm_grid(draw)
    _sweep(sel, img, gt, mac)


def _sweep(sel: Selection, img: np.ndarray, gt: np.ndarray, mac: bool) -> None:
    """Dice vs one stain strength for every arm, with a live run log beside it."""
    left, right = st.columns([5, 4], gap="large")
    with left:
        st.subheader("Sweep one stain")
        st.caption(
            f"Runs every model at {len(SWEEP_POINTS)} stain strengths on this tile and plots "
            "its Dice. The shaded band is the range stain jitter saw in training."
        )
        axis = st.radio("Stain", ["h_scale", "e_scale"], horizontal=True,
                        format_func=SHIFT_LABELS.get)
        clicked = st.button(
            f"Run sweep ({len(_arms())} models × {len(SWEEP_POINTS)} strengths)")
        chart_slot = st.empty()
    with right:
        st.markdown("**Run log**")
        with st.container(height=330, border=True):
            log_slot = st.empty()
    key = ("sweep", sel.tile_id, axis, mac)
    if clicked:
        st.session_state[key] = _run_sweep(sel.tile_id, img, gt, axis, mac, log_slot)
    rows, log = st.session_state.get(key, (None, []))
    log_slot.code("\n".join(log) if log else "No sweep run yet for these settings.",
                  language=None, wrap_lines=True)
    if rows:
        chart_slot.altair_chart(
            views.sweep_chart(rows, _arms(), SHIFT_LABELS[axis], SCALE_BAND), width="stretch")


def _run_sweep(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    tile_id: str, img: np.ndarray, gt: np.ndarray, axis: str, mac: bool, log_slot
):
    """Uncached forward pass per (strength, arm), logging each one as it finishes."""
    log: List[str] = []

    def note(line: str) -> None:
        log.append(line)
        log_slot.code("\n".join(log), language=None, wrap_lines=True)

    total, start = len(SWEEP_POINTS) * len(_arms()), time.perf_counter()
    note(f"{time.strftime('%H:%M:%S')} sweep started")
    note(f"tile {tile_id} · {SHIFT_LABELS[axis].lower()}")
    note(f"Macenko input {'on' if mac else 'off'} · device {_device()}")
    note("run   strength  model                  Dice  time")
    rows = []
    for s in SWEEP_POINTS:
        p = {"h_scale": 1.0, "e_scale": 1.0, "h_bias": 0.0, "e_bias": 0.0, axis: float(s)}
        shifted = _shifted(img, p)
        row = {"shift": float(s)}
        for arm in _arms():
            t0 = time.perf_counter()
            pred = _run_model(arm, shifted, mac)
            row[arm] = inference.scores(pred, gt)[0] if pred is not None else np.nan
            note(f"{len(rows) * len(_arms()) + len(row) - 1:2d}/{total} {s:.2f}  "
                 f"{views.arm_label(arm):<22} {row[arm]:.3f} "
                 f"{(time.perf_counter() - t0) * 1000:3.0f}ms")
        rows.append(row)
    note(f"{time.strftime('%H:%M:%S')} done · {total} forward passes "
         f"in {time.perf_counter() - start:.1f} s")
    return rows, log


# -------------------------------------------------------------------- upload


def _tab_upload(sel: Selection) -> None:
    """Run all arms on an uploaded image (kept in memory only)."""
    st.markdown(
        "Upload a kidney tissue tile (H&E or PAS stain). It is resized to 512 × 512, kept in "
        "memory only and never saved. This tab uses your image instead of the tile picked "
        "above. There is no expert outline for it, so no score is shown."
    )
    f = st.file_uploader("Image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if f is None:
        return
    try:
        img = inference.resize_upload(Image.open(io.BytesIO(f.getvalue())))
    except (OSError, ValueError) as exc:
        st.error(f"Could not read that image ({exc}). Try a PNG or JPEG.")
        return
    left, right = _two_columns()
    with left:
        st.markdown("**Your image**")
        st.image(img, width="stretch")

    def draw(arm: str) -> None:
        pred = _predict(arm, img)
        if pred is None:
            st.info("No model for this recipe.")
            return
        st.image(views.overlay(img, pred, color=views.arm_color(arm), alpha=sel.alpha),
                 width="stretch")
        st.caption(f"Marked as vessel: {pred.mean():.1%} of the image")

    with right:
        _arm_grid(draw)


def main() -> None:
    """Page layout."""
    st.set_page_config(
        page_title=views.LAB_TITLE,
        page_icon="🔬",
        layout="wide",
        menu_items={"About": ABOUT, "Report a bug": f"{data.REPO_URL}/issues"},
    )
    _header()
    _preload()
    sel = _picker()
    explorer, stress, upload = st.tabs(["Tile explorer", "Stain stress test", "Try your own image"])
    with explorer:
        _tab_explorer(sel)
    with stress:
        _tab_stress(sel)
    with upload:
        _tab_upload(sel)


if __name__ == "__main__":
    main()
