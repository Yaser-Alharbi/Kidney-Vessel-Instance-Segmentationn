"""Vessel Stain Lab: Streamlit app for the four-arm stain-augmentation ablation.

    streamlit run demo/app.py

Technical view by default, with a sidebar toggle for a plain-language
view. Every number shown is read from `artifacts_v2/phase3_summary.json`
or computed live from a prediction mask.
"""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# pylint: disable=wrong-import-position
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from demo import data, inference, views
from src.data.transforms import hed_channels, hed_shift
from src.training.evaluate import bootstrap_dice

# pylint: enable=wrong-import-position

BASE_ARM = "rgb_aug"
SCALE_BAND = (0.95, 1.05)  # HEDJitter(sigma=0.05): alpha in 1 +/- 0.05
BIAS_BAND = (-0.05, 0.05)  # HEDJitter(sigma=0.05): beta in +/- 0.05
SWEEP_POINTS = np.round(np.linspace(0.7, 1.3, 7), 2)


@dataclass(frozen=True)
class Options:
    """Sidebar state shared by every tab."""

    plain: bool
    split: str
    tile_id: str
    seed: int
    alpha: float
    show_gt: bool
    error_view: bool


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


@st.cache_data(max_entries=256, show_spinner=False)
def _predict(arm: str, img: np.ndarray, macenko_input: bool = False) -> Optional[np.ndarray]:
    """Live seed-42 prediction for an arbitrary image."""
    model = _model(arm, data.LIVE_SEED)
    if model is None:
        return None
    if macenko_input and arm not in inference.MACENKO_AT_EVAL:
        img = inference.macenko(img)
    return inference.predict(model, arm, img, _device())


def _tile_prediction(arm: str, tile_id: str) -> Optional[np.ndarray]:
    """Stored seed-42 mask if present (deployed), else live inference."""
    pred = data.stored_prediction(arm, tile_id)
    if pred is not None:
        return pred
    return _predict(arm, _tile(tile_id)[0])


def _preload() -> None:
    """Load the seed-42 model for every arm at startup."""
    remote = not data.local_mode(_cfg()) and data.weights_repo() is not None
    msg = ("Downloading and loading models (first start only)..." if remote
           else "Loading models...")
    with st.spinner(msg):
        for arm in _arms():
            _model(arm, data.LIVE_SEED)


# ------------------------------------------------------------------- helpers


def _arm_header(arm: str, plain: bool) -> None:
    """Coloured arm name."""
    st.markdown(
        f"<span style='color:{views.arm_color(arm)}'>&#9679;</span> "
        f"**{views.arm_label(arm, plain)}**",
        unsafe_allow_html=True,
    )


def _legend(error_view: bool) -> None:
    """Overlay colour key."""
    if error_view:
        items = [(views.GT_COLOR, "hit"), (views.FP_COLOR, "extra (false positive)"),
                 (views.FN_COLOR, "missed (false negative)")]
    else:
        items = [(views.GT_COLOR, "expected vessels not found"),
                 ("#888", "prediction in arm colour")]
    st.markdown(
        " &nbsp; ".join(f"<span style='color:{c}'>&#9632;</span> {t}" for c, t in items),
        unsafe_allow_html=True,
    )


def _noisy_note(split: str) -> None:
    """Caveat for the auto-labelled test slides."""
    if split == "test":
        st.info(
            "Test slides use automatically generated (noisy) labels. Scores there are "
            "for comparing recipes with each other, not for absolute accuracy."
        )


# ------------------------------------------------------------------- sidebar


def _sidebar() -> Options:
    """Draw the sidebar and return its state."""
    sb = st.sidebar
    sb.title("Vessel Stain Lab")
    plain = sb.radio("View", ["Technical", "Plain language"], horizontal=True) == "Plain language"
    split = sb.radio(
        "Slide",
        ["val", "test"],
        format_func=lambda s: {
            "val": "Validation (1 WSI, expert labels)",
            "test": "Test (2 WSIs, noisy auto labels)",
        }[s],
    )
    cube = _cube(split)
    allowed = data.available_tiles(_cfg(), cube, split)
    sort = sb.selectbox("Sort tiles by", data.SORTS)
    ids = data.rank_tiles(cube, sort, allowed)
    arms = _arms()
    mean = {a: cube.dice[i].mean(axis=0) for i, a in enumerate(arms)}

    def _fmt(t: str) -> str:
        i = cube.tile_index(t)
        return f"{t}  HED {mean['hed_only'][i]:.2f} / RGB {mean[BASE_ARM][i]:.2f}"

    tile_id = sb.selectbox("Tile (20-seed mean Dice)", ids, format_func=_fmt)
    seed = sb.selectbox(
        "Seed for stored scores", cube.seeds, index=cube.seeds.index(data.LIVE_SEED)
    )
    alpha = sb.slider("Overlay opacity", 0.0, 1.0, 0.5, 0.05)
    show_gt = sb.checkbox("Show expected vessels", value=True)
    error_view = sb.checkbox("Error view (hit / extra / missed)", value=False)
    sb.caption(
        f"U-Net + ResNet-34 · {len(arms)} arms × {len(cube.seeds)} seeds · "
        f"{len(ids)} of {len(cube.tile_ids)} {split} tiles shown · "
        f"{'local data' if data.local_mode(_cfg()) else 'curated tiles'} · "
        f"device {_device()}"
    )
    return Options(plain, split, tile_id, int(seed), alpha, show_gt, error_view)


# ---------------------------------------------------------------------- tabs


def _tab_explorer(o: Options) -> None:  # pylint: disable=too-many-locals
    """Tile, ground truth and every arm's prediction."""
    img, gt = _tile(o.tile_id)
    cube = _cube(o.split)
    ti, si = cube.tile_index(o.tile_id), cube.seed_index(o.seed)
    _noisy_note(o.split)
    if o.seed != data.LIVE_SEED:
        st.warning(f"Predictions: seed {data.LIVE_SEED}. Scores: seed {o.seed}.")
    _legend(o.error_view)
    cols = st.columns(len(_arms()) + 1)
    with cols[0]:
        st.markdown("**Tissue tile**")
        st.image(views.overlay(img, gt=gt, alpha=o.alpha, show_gt=o.show_gt), width="stretch")
        st.caption(f"{o.tile_id} · vessel pixels {gt.mean():.1%}")
    preds = {}
    for col, (ai, arm) in zip(cols[1:], enumerate(_arms())):
        with col:
            _arm_header(arm, o.plain)
            preds[arm] = pred = _tile_prediction(arm, o.tile_id)
            if pred is None:
                st.info("No seed-42 checkpoint or stored mask for this arm.")
                continue
            st.image(
                views.overlay(img, pred, gt, color=views.arm_color(arm), alpha=o.alpha,
                              show_gt=o.show_gt, error_view=o.error_view),
                width="stretch",
            )
            d42, i42 = inference.scores(pred, gt)
            seeds_d = cube.dice[ai, :, ti]
            if o.plain:
                st.markdown(f"Overlap score **{cube.dice[ai, si, ti]:.2f}**  \n"
                            f"Average over {len(seeds_d)} runs: {seeds_d.mean():.2f}")
            else:
                st.caption(
                    f"seed {data.LIVE_SEED} mask: Dice {d42:.3f} · IoU {i42:.3f}  \n"
                    f"stored seed {o.seed}: Dice {cube.dice[ai, si, ti]:.3f} · "
                    f"IoU {cube.iou[ai, si, ti]:.3f}  \n"
                    f"{len(seeds_d)} seeds: Dice {seeds_d.mean():.3f} ± {seeds_d.std():.3f}"
                )
    _arm_diff(img, gt, preds, o)


def _arm_diff(  # pylint: disable=too-many-locals
    img: np.ndarray, gt: np.ndarray, preds: Dict, o: Options
) -> None:
    """Where two arms disagree."""
    st.subheader("Where two recipes disagree" if o.plain else "Arm vs arm")
    ready = [a for a, p in preds.items() if p is not None]
    if len(ready) < 2:
        return
    c1, c2, c3 = st.columns([1, 1, 2])

    def fmt(arm: str) -> str:
        return views.arm_label(arm, o.plain)

    a = c1.selectbox("Arm A", ready, format_func=fmt,
                     index=ready.index("hed_only") if "hed_only" in ready else 0)
    b = c2.selectbox("Arm B", ready, format_func=fmt,
                     index=ready.index(BASE_ARM) if BASE_ARM in ready else 1)
    pa, pb = preds[a].astype(bool), preds[b].astype(bool)
    with c3:
        st.image(views.diff_overlay(img, (pa, pb), (views.arm_color(a), views.arm_color(b)),
                                    o.alpha), width=420)
    only_a, only_b, g = pa & ~pb, pb & ~pa, gt.astype(bool)
    c1.markdown(f"<span style='color:{views.arm_color(a)}'>&#9632;</span> only {fmt(a)}: "
                f"{int(only_a.sum())} px", unsafe_allow_html=True)
    c2.markdown(f"<span style='color:{views.arm_color(b)}'>&#9632;</span> only {fmt(b)}: "
                f"{int(only_b.sum())} px", unsafe_allow_html=True)
    if not o.plain:
        c1.caption(f"of which on GT: {int((only_a & g).sum())} px")
        c2.caption(f"of which on GT: {int((only_b & g).sum())} px")


def _shift_controls(plain: bool) -> Dict[str, float]:
    """Stain-shift sliders; bias sliders only in technical view."""
    c = st.columns(4 if not plain else 2)
    p = {
        "h_scale": c[0].slider("Purple (haematoxylin) strength" if plain else "H scale",
                               0.5, 1.5, 1.0, 0.05),
        "e_scale": c[1].slider("Pink (eosin) strength" if plain else "E scale",
                               0.5, 1.5, 1.0, 0.05),
        "h_bias": 0.0,
        "e_bias": 0.0,
    }
    if not plain:
        p["h_bias"] = c[2].slider("H bias", -0.3, 0.3, 0.0, 0.01)
        p["e_bias"] = c[3].slider("E bias", -0.3, 0.3, 0.0, 0.01)
    st.caption(
        f"Training range of HEDJitter(sigma=0.05): scale {SCALE_BAND[0]} to {SCALE_BAND[1]}, "
        f"bias {BIAS_BAND[0]} to {BIAS_BAND[1]}. The D channel is left unchanged here."
    )
    outside = [k for k, v in p.items()
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


def _tab_stress(o: Options) -> None:
    """Live stain-shift robustness on the selected tile."""
    img, gt = _tile(o.tile_id)
    st.markdown(
        "Labs stain tissue differently. Shift this tile's stain and see which models hold up. "
        f"Every image here is a live seed-{data.LIVE_SEED} forward pass. One tile, so this is "
        "illustrative, not evidence."
    )
    p = _shift_controls(o.plain)
    mac = st.toggle(
        "Macenko-normalise the input",
        help="Applies to rgb_aug and hed_only only; macenko_only and full_stain_aware "
             "already normalise at eval.",
    )
    shifted = _shifted(img, p)
    cols = st.columns(len(_arms()) + 1)
    with cols[0]:
        st.markdown("**Shifted tile**")
        st.image(shifted, width="stretch")
    for col, arm in zip(cols[1:], _arms()):
        with col:
            _arm_header(arm, o.plain)
            pred, ref = _predict(arm, shifted, mac), _predict(arm, img, mac)
            if pred is None:
                st.info("No seed-42 checkpoint for this arm.")
                continue
            st.image(views.overlay(shifted, pred, gt, color=views.arm_color(arm), alpha=o.alpha,
                                   show_gt=o.show_gt, error_view=o.error_view), width="stretch")
            d, d0 = inference.scores(pred, gt)[0], inference.scores(ref, gt)[0]
            st.caption(f"Dice {d:.3f} ({d - d0:+.3f} vs unshifted)")
            if mac and arm in inference.MACENKO_AT_EVAL:
                st.caption("already Macenko at eval")
    _sweep(o, img, gt, mac)


def _sweep(o: Options, img: np.ndarray, gt: np.ndarray, mac: bool) -> None:
    """Dice vs H or E scale for every arm (4 arms x 7 points)."""
    st.subheader("Sweep")
    axis = st.radio("Shift", ["h_scale", "e_scale"], horizontal=True,
                    format_func=lambda k: {"h_scale": "H scale", "e_scale": "E scale"}[k])
    key = ("sweep", o.tile_id, axis, mac)
    if st.button(f"Run sweep ({len(_arms())} arms × {len(SWEEP_POINTS)} points)"):
        rows = []
        with st.spinner("Running..."):
            for s in SWEEP_POINTS:
                p = {"h_scale": 1.0, "e_scale": 1.0, "h_bias": 0.0, "e_bias": 0.0, axis: float(s)}
                shifted = _shifted(img, p)
                row = {"shift": float(s)}
                for arm in _arms():
                    pred = _predict(arm, shifted, mac)
                    row[arm] = inference.scores(pred, gt)[0] if pred is not None else np.nan
                rows.append(row)
        st.session_state[key] = rows
    if key in st.session_state:
        st.altair_chart(
            views.sweep_chart(st.session_state[key], _arms(), o.plain,
                              "H scale" if axis == "h_scale" else "E scale", SCALE_BAND),
            width="stretch",
        )


def _results_table(o: Options, metric: str, values: Dict, cis: Dict) -> pd.DataFrame:
    """Cross-seed table, technical or plain."""
    stats = _summary()["stats"][o.split]
    cs = stats["cross_seed"][metric]
    wil = stats["wilcoxon_cross_seed"][metric]
    base = cs[BASE_ARM]["mean"]
    rows = []
    for arm in _arms():
        m = cs[arm]
        w = wil.get(f"{arm}_gt_{BASE_ARM}")
        wins = data.wins_vs_base(values, arm, BASE_ARM) if arm != BASE_ARM else None
        change = (m["mean"] - base) / base * 100 if arm != BASE_ARM else None
        if o.plain:
            verdict = "baseline" if w is None else (
                "clearly better" if w["p"] < 0.05 else "not better")
            rows.append({
                "Recipe": views.arm_label(arm, True),
                "Average score": round(m["mean"], 3),
                "Change vs baseline": "" if change is None else f"{change:+.1f}%",
                "Better than baseline": "" if wins is None else f"{wins} of {m['n']} runs",
                "Verdict": verdict,
            })
        else:
            rows.append({
                "Arm": arm,
                "Mean": round(m["mean"], 4),
                "Std (summary)": round(m["std"], 4),
                "Cross-seed 95% CI": f"{cis[arm].lo:.4f} to {cis[arm].hi:.4f}",
                f"vs {BASE_ARM}": "" if change is None else f"{change:+.1f}%",
                "Wins": "" if wins is None else f"{wins} / {m['n']}",
                "Wilcoxon p (one-sided)": "" if w is None else f"{w['p']:.4g}",
            })
    return pd.DataFrame(rows)


def _seed42_block(o: Options, metric: str) -> None:
    """Seed-42 per-tile bootstrap CIs and Wilcoxon (technical only)."""
    stats = _summary()["stats"][o.split]
    cis = stats["cis"] if metric == "dice" else stats["iou"]
    n = next(iter(cis.values()))["n"]
    st.subheader(f"Seed {data.LIVE_SEED}, per tile (n = {n} tiles)")
    rows = []
    for arm in _arms():
        c = cis[arm]
        w = stats["wilcoxon"].get(f"{arm}_gt_{BASE_ARM}") if metric == "dice" else None
        rows.append({
            "Arm": arm, "Mean": round(c["mean"], 4),
            "Per-tile 95% CI": f"{c['lo']:.4f} to {c['hi']:.4f}",
            "Wilcoxon p (one-sided, per tile)": "" if w is None else f"{w['p']:.4g}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        f"Bootstrap over the {n} tiles of one run (n_boot = {next(iter(cis.values()))['n_boot']}). "
        "This measures tile-to-tile spread, not seed-to-seed spread, so it is narrower than "
        "the cross-seed CI above."
    )


def _additivity(o: Options, metric: str) -> None:
    """Is the combined arm's gain the sum of its parts? (cross-seed means)."""
    cs = _summary()["stats"][o.split]["cross_seed"][metric]
    needed = {BASE_ARM, "hed_only", "macenko_only", "full_stain_aware"}
    if not needed <= set(cs):
        return
    base = cs[BASE_ARM]["mean"]
    g_hed, g_mac = cs["hed_only"]["mean"] - base, cs["macenko_only"]["mean"] - base
    g_full = cs["full_stain_aware"]["mean"] - base
    st.subheader("Do the two stain tricks add up?" if o.plain else "Additivity check")
    st.markdown(
        f"HED gain {g_hed:+.4f} + Macenko gain {g_mac:+.4f} = expected {g_hed + g_mac:+.4f}; "
        f"observed combined gain {g_full:+.4f} (cross-seed means, {metric})."
    )
    if o.plain:
        st.caption("Combining both did worse than either alone, so the two do not simply add up."
                   if g_full < min(g_hed, g_mac) else
                   "The combined recipe sits between or above the single recipes.")


def _tab_results(o: Options) -> None:
    """Cross-seed results, seed-42 per-tile stats and learning curves."""
    _noisy_note(o.split)
    metric = "dice" if o.plain else st.radio("Metric", ["dice", "iou"], horizontal=True)
    values = data.cross_seed_values(_summary(), o.split, metric)
    cis = {a: bootstrap_dice(np.array(list(values[a].values())), n_boot=2000, seed=42)
           for a in _arms()}
    st.markdown(
        f"Each dot is one training run's mean per-tile {metric} on the {o.split} split. "
        "Grey bar: cross-seed mean and 95% bootstrap CI over the seed means."
    )
    chart_cis = {a: (c.mean, c.lo, c.hi) for a, c in cis.items()}
    st.altair_chart(views.seed_strip_chart(values, chart_cis, _arms(), o.plain, metric),
                    width="stretch")
    st.dataframe(_results_table(o, metric, values, cis), hide_index=True, width="stretch")
    if not o.plain:
        st.caption("Cross-seed CI: bootstrap (n_boot = 2000, seed 42) over per-seed means. "
                   "Wins: seeds where the arm beat rgb_aug trained with the same seed. "
                   "Wilcoxon: paired, one-sided, over seeds. Std as stored in the summary.")
        _seed42_block(o, metric)
    _additivity(o, metric)
    if not o.plain:
        _learning_curves(o)


def _learning_curves(o: Options) -> None:
    """Loss and val Dice curves for one run."""
    st.subheader("Learning curves")
    c1, c2 = st.columns(2)
    arm = c1.selectbox("Arm", _arms(), key="lc_arm")
    seed = c2.selectbox("Seed", _cube(o.split).seeds, key="lc_seed")
    tag = data.run_tag(arm, seed)
    run = _summary()["runs"][tag]
    st.altair_chart(
        views.learning_curve_chart(run, f"{tag} · best epoch {run['best_epoch']}"),
        width="stretch",
    )


def _tab_methods(o: Options) -> None:
    """Stain decomposition and Macenko on the selected tile (technical only)."""
    img, _ = _tile(o.tile_id)
    ch = hed_channels(img)
    cols = st.columns(4)
    for col, (title, im) in zip(cols, [
        ("Raw tile", img),
        ("H channel (optical density)", views.to_gray_image(ch["H"])),
        ("E channel (optical density)", views.to_gray_image(ch["E"])),
        ("Macenko-normalised", inference.macenko(img)),
    ]):
        col.markdown(f"**{title}**")
        col.image(im, width="stretch")
    st.caption("Ruifrok & Johnston HED matrix; Macenko with a fixed reference stain matrix "
               "and 99th-percentile concentrations (src/data/transforms.py).")
    rows = [{"Arm": a, "Train": views.ARM_PIPELINE[a][0], "Eval": views.ARM_PIPELINE[a][1]}
            for a in _arms()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _tab_upload(o: Options) -> None:
    """Run all arms on an uploaded image (kept in memory only)."""
    st.markdown("Upload an H&E/PAS tile. It is resized to 512 × 512 and never saved. "
                "No ground truth, so no score.")
    f = st.file_uploader("Image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if f is None:
        return
    try:
        img = inference.resize_upload(Image.open(io.BytesIO(f.getvalue())))
    except (OSError, ValueError) as exc:
        st.error(f"Could not read that image ({exc}). Try a PNG or JPEG.")
        return
    cols = st.columns(len(_arms()) + 1)
    cols[0].markdown("**Upload**")
    cols[0].image(img, width="stretch")
    for col, arm in zip(cols[1:], _arms()):
        with col:
            _arm_header(arm, o.plain)
            pred = _predict(arm, img)
            if pred is None:
                st.info("No seed-42 checkpoint for this arm.")
                continue
            st.image(views.overlay(img, pred, color=views.arm_color(arm), alpha=o.alpha),
                     width="stretch")
            st.caption(f"vessel pixels {pred.mean():.1%}")


def main() -> None:
    """Page layout."""
    st.set_page_config(page_title="Vessel Stain Lab", page_icon="🔬", layout="wide")
    o = _sidebar()
    _preload()
    st.title("Vessel Stain Lab")
    if o.plain:
        st.markdown("Four ways of recolouring training images, one U-Net each, 20 training runs "
                    "per recipe. Which recipe finds blood vessels in kidney tissue best?")
        with st.expander("What are the four recipes?"):
            for arm in _arms():
                st.markdown(f"**{views.arm_label(arm, True)}**: {views.ARM_BLURB[arm]}")
    else:
        st.markdown("Four-arm stain-augmentation ablation, U-Net + ResNet-34, HuBMAP vasculature. "
                    "v2 run: 20 seeds per arm, test = 2 held-out WSIs.")
    names = ["Tile explorer", "Stain stress test", "Results (20 seeds)"]
    funcs = [_tab_explorer, _tab_stress, _tab_results]
    if not o.plain:
        names.append("Methods")
        funcs.append(_tab_methods)
    names.append("Try your own image")
    funcs.append(_tab_upload)
    for tab, fn in zip(st.tabs(names), funcs):
        with tab:
            fn(o)


if __name__ == "__main__":
    main()
