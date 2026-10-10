"""Build the static story site into docs/ (served by GitHub Pages).

    python -m demo.build_site                        # full build (raw data + checkpoints)
    python -m demo.build_site --showcase <tile_id>   # choose the hero tile
    python -m demo.build_site --skip-images          # re-render the HTML only

Every number on the page is computed here, never typed into the template:
statistics come from the committed `artifacts_v2/phase3_summary.json`,
`artifacts/inspection_summary.json` and `artifacts/mask_audit.json`; values
that need raw data (split sizes, the showcase tile, the stain sweep) are
computed during the image pass and saved to `docs/build_meta.json`, which
`--skip-images` reads back. After rendering, every number in the page text
is checked against the values the template was given.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import altair as alt
import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape
from PIL import Image, ImageDraw
from scipy.stats import wilcoxon

from demo import data, inference, views
from src.data.transforms import hed_channels, hed_shift
from src.training.evaluate import bootstrap_dice
from src.utils.paths import project_root

DOCS = project_root() / "docs"
SITE_SRC = Path(__file__).resolve().parent / "site"
CANDIDATES_PNG = project_root() / "results_v2" / "site_candidates.png"
BASE_ARM, JITTER_ARM = "rgb_aug", "hed_only"
MAC_ARM, FULL_ARM = "macenko_only", "full_stain_aware"
CI_LEVEL = 95  # bootstrap_dice default ci=0.95
THRESHOLD = 0.5  # inference.predict: sigmoid > 0.5
SPLITS = ("val", "test")
STAIN_VARIANTS = (0.7, 1.0, 1.3)  # stain strengths shown in "The problem"
SWEEP_POINTS = [round(float(s), 2) for s in np.linspace(0.7, 1.3, 7)]
SCALE_BAND = (0.95, 1.05)  # HEDJitter(sigma=0.05): alpha in 1 +/- 0.05
N_TRAIN_SAMPLES = 3
MACENKO_ODD_DIFF = 20  # mean-colour difference (0-255) that counts as a different recolouring
CELL, PAD = 256, 28  # candidate contact sheet: image size, caption height
POWER_N = 3  # "with only a few runs" example for the power argument
CDN = {"vega": "6.3.1", "vega-lite": "6.4.1", "vega-embed": "7.2.0"}
CITATION_YEARS = {"2001", "2009", "2019"}  # Ruifrok, Macenko, Tellez

# chart styling that matches the site's dark tokens (demo/site/style.css)
CHART_TEXT, CHART_MUTED, CHART_GRID = "#c9ced6", "#9aa4b0", "#232a33"


# ------------------------------------------------------------------ formatting


class Numbers:
    """Jinja filters that format numbers and remember every string they emit."""

    def __init__(self) -> None:
        self.emitted: Set[str] = set()

    def _keep(self, text: str) -> str:
        self.emitted.add(text)
        return text

    def f3(self, x: float) -> str:
        """Three decimals (Dice, IoU)."""
        return self._keep(f"{x:.3f}")

    def signed(self, x: float) -> str:
        """Three decimals with a sign (differences)."""
        return self._keep(f"{x:+.3f}")

    def pval(self, p: float) -> str:
        """Two decimals from 0.1 up, two significant figures below, scientific below 1e-4."""
        if p < 1e-4:
            return self._keep(f"{p:.1e}")
        if p >= 0.1:
            return self._keep(f"{p:.2f}")
        return self._keep(f"{p:.{1 - math.floor(math.log10(p))}f}")

    def pct(self, x: float) -> str:
        """Fraction as a percentage with one decimal."""
        return self._keep(f"{x * 100:.1f}%")

    def rel(self, x: float) -> str:
        """Relative change as a signed percentage with one decimal."""
        return self._keep(f"{x * 100:+.1f}%")

    def n(self, x: float) -> str:
        """Integer with thousands separators."""
        return self._keep(f"{int(round(x)):,}")

    def num(self, x: float) -> str:
        """Plain number, shortest form (config values such as 1e-4 or 0.05)."""
        return self._keep(f"{x:g}")

    def filters(self) -> Dict[str, Callable]:
        """Filter table for Jinja."""
        return {k: getattr(self, k) for k in ("f3", "signed", "pval", "pct", "rel", "n", "num")}


class _TextOnly(HTMLParser):
    """Collect visible text, skipping <script> and <style>."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip -= 1

    def handle_data(self, data):  # pylint: disable=redefined-outer-name
        if not self._skip:
            self.parts.append(data)


_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?(?:e[+-]?\d+)?")


def unexplained_numbers(html: str, emitted: Set[str]) -> List[str]:
    """Numbers in the page text that no filter produced (should be empty)."""
    parser = _TextOnly()
    parser.feed(html)
    text = " ".join(parser.parts)
    allowed = set(CITATION_YEARS) | {"34"}  # "ResNet-34" is an architecture name
    for value in emitted:
        allowed.update(_NUMBER.findall(value))
    for train, test in views.ARM_PIPELINE.values():  # transform names read from code
        allowed.update(_NUMBER.findall(train + " " + test))
    return sorted(set(_NUMBER.findall(text)) - allowed)


# ------------------------------------------------------------------ statistics


def _arm_stats(summary: Dict, arms: List[str], split: str, metric: str) -> Dict[str, Dict]:
    """Across-run and per-tile statistics for every arm on one split and metric."""
    stats = summary["stats"][split]
    cross = stats["cross_seed"][metric]
    across_p = stats["wilcoxon_cross_seed"][metric]
    tile_ci = stats["cis"] if metric == "dice" else stats["iou"]
    n_boot = next(iter(tile_ci.values()))["n_boot"]
    values = data.cross_seed_values(summary, split, metric)
    base = cross[BASE_ARM]["mean"]
    tile_p = stats["wilcoxon"] if metric == "dice" else {}
    out = {}
    for arm in arms:
        ci = bootstrap_dice(np.array(list(values[arm].values())), n_boot=n_boot, seed=42)
        out[arm] = {
            "mean": cross[arm]["mean"], "std": cross[arm]["std"], "n": cross[arm]["n"],
            "lo": ci.lo, "hi": ci.hi, "n_boot": n_boot,
            "gain": cross[arm]["mean"] - base,
            "rel": (cross[arm]["mean"] - base) / base,
            "wins": None if arm == BASE_ARM else data.wins_vs_base(values, arm, BASE_ARM),
            "p": across_p.get(f"{arm}_gt_{BASE_ARM}", {}).get("p"),
            "tile_mean": tile_ci[arm]["mean"], "tile_lo": tile_ci[arm]["lo"],
            "tile_hi": tile_ci[arm]["hi"], "tile_n": tile_ci[arm]["n"],
            "tile_p": tile_p.get(f"{arm}_gt_{BASE_ARM}", {}).get("p"),
        }
    return out


def _power() -> Dict[str, float]:
    """Smallest one-sided exact Wilcoxon p when one recipe wins every run."""
    def best_p(n: int) -> float:
        return float(wilcoxon(np.arange(1, n + 1), alternative="greater",
                              method="exact").pvalue)
    return {"small_n": POWER_N, "small_p": best_p(POWER_N)}


def build_context(summary: Dict, arms: List[str], meta: Dict) -> Dict[str, Any]:
    """Everything the template needs, as raw numbers and plain strings."""
    stats = {s: {m: _arm_stats(summary, arms, s, m) for m in ("dice", "iou")} for s in SPLITS}
    snap = summary["config_snapshot"]
    with open(project_root() / "artifacts" / "inspection_summary.json", encoding="utf-8") as f:
        inspection = json.load(f)
    with open(project_root() / "artifacts" / "mask_audit.json", encoding="utf-8") as f:
        coverage = json.load(f)["coverage"]
    power = _power()
    power["full_p"] = float(wilcoxon(np.arange(1, len(summary["seeds"]) + 1),
                                     alternative="greater", method="exact").pvalue)
    return {
        "title": views.TITLE, "subtitle": views.SUBTITLE, "lab_title": views.LAB_TITLE,
        "urls": {"repo": data.REPO_URL, "lab": data.LAB_URL, "weights": data.WEIGHTS_URL},
        "code": _code_links(),
        "cdn": CDN,
        "arms": [{"id": a, "label": views.arm_label(a), "color": views.arm_color(a),
                  "blurb": views.ARM_BLURB[a], "role": views.ARM_ROLE[a],
                  "train": views.ARM_PIPELINE[a][0], "eval": views.ARM_PIPELINE[a][1]}
                 for a in arms],
        "base": BASE_ARM, "jitter": JITTER_ARM, "mac": MAC_ARM, "full": FULL_ARM,
        "ci_level": CI_LEVEL, "threshold": THRESHOLD, "stats": stats, "power": power,
        "n_seeds": len(summary["seeds"]), "n_arms": len(arms),
        "n_runs": len(summary["seeds"]) * len(arms), "seed": data.LIVE_SEED,
        "train_hours": summary["wall_time_seconds"] / 3600,
        "cfg": {k: snap[k] for k in ("epochs", "early_stop_patience", "lr", "weight_decay",
                                     "batch_size", "image_size")},
        "hed": meta["hed"], "inspection": inspection, "coverage": coverage,
        "meta": meta, "scale_band": SCALE_BAND, "colors": {
            "gt": views.GT_COLOR, "fp": views.FP_COLOR, "fn": views.FN_COLOR},
        "charts": _charts(summary, arms, meta, stats),
    }


def _code_links() -> Dict[str, str]:
    """GitHub links to the line where each algorithm lives (found by searching)."""
    targets = {
        "hed_matrix": ("src/data/transforms.py", "_RGB_FROM_HED = "),
        "hed_jitter": ("src/data/transforms.py", "class HEDJitter"),
        "hed_builder": ("src/data/transforms.py", "HEDJitter(sigma="),
        "macenko": ("src/data/transforms.py", "class MacenkoNormalize"),
        "loss": ("src/training/losses.py", "class DiceBCELoss"),
        "model": ("src/models/__init__.py", "smp.Unet("),
        "bootstrap": ("src/training/evaluate.py", "def bootstrap_dice"),
        "wilcoxon": ("src/training/evaluate.py", "def paired_wilcoxon_dice"),
        "train": ("src/training/train.py", "CosineAnnealingLR(optimizer"),
    }
    links = {}
    for key, (rel, needle) in targets.items():
        lines = (project_root() / rel).read_text(encoding="utf-8").splitlines()
        line = next(i for i, text in enumerate(lines, start=1) if needle in text)
        links[key] = f"{data.REPO_URL}/blob/main/{rel}#L{line}"
    return links


# ---------------------------------------------------------------------- charts


def _dark(chart: alt.TopLevelMixin) -> Dict:
    """Vega-Lite dict styled for the dark page."""
    styled = (
        chart.configure(background="transparent", font="Inter")
        .configure_axis(labelColor=CHART_MUTED, titleColor=CHART_TEXT, gridColor=CHART_GRID,
                        domainColor="#3a424d", tickColor="#3a424d", labelFontSize=12,
                        titleFontSize=12, titleFontWeight=500)
        .configure_axisX(tickCount=8, labelOverlap=True)
        .configure_axisY(labelLimit=220)
        .configure_legend(labelColor=CHART_TEXT, labelFontSize=12)
        .configure_view(stroke=None)
        .properties(autosize=alt.AutoSizeParams(type="fit", contains="padding"))
    )
    return styled.to_dict()


def _charts(summary: Dict, arms: List[str], meta: Dict, stats: Dict) -> Dict[str, Dict]:
    """Vega-Lite specs embedded in the page."""
    charts = {}
    for split in SPLITS:
        values = data.cross_seed_values(summary, split, "dice")
        cis = {a: (s["mean"], s["lo"], s["hi"]) for a, s in stats[split]["dice"].items()}
        chart = views.seed_strip_chart(values, cis, arms, "dice").properties(width="container")
        charts[f"seeds_{split}"] = _dark(chart)
    for axis, title in (("h_scale", "Purple stain strength (haematoxylin)"),
                        ("e_scale", "Pink stain strength (eosin)")):
        chart = views.sweep_chart(meta["sweep"][axis], arms, title, SCALE_BAND)
        charts[f"sweep_{axis}"] = _dark(chart)
    return charts


# ---------------------------------------------------------------------- images


def _save(img: np.ndarray, name: str, size: Optional[int] = None) -> str:
    """Write an RGB uint8 image as WebP under docs/img and return its relative path."""
    out = DOCS / "img" / f"{name}.webp"
    out.parent.mkdir(parents=True, exist_ok=True)
    pil = Image.fromarray(img)
    if size:
        pil = pil.resize((size, size), Image.Resampling.LANCZOS)
    pil.save(out, quality=86, method=6)
    return f"img/{name}.webp"


def _denormalise(tensor) -> np.ndarray:
    """Undo ImageNet normalisation: transform output tensor -> RGB uint8."""
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    arr = tensor.numpy().transpose(1, 2, 0) * std + mean
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def _train_samples(arm: str, img: np.ndarray, gt: np.ndarray) -> List[str]:
    """The tile passed through the arm's training transform, a few distinct draws.

    Seeds are tried in order and a draw is kept only if it differs from the
    draws already kept, so the thumbnails show the augmentation's variety
    (some seeds apply no random op at all). Deterministic for a given tile.
    """
    transform = inference.EVAL_BUILDERS[arm](image_size=img.shape[0], train=True)
    kept: List[np.ndarray] = []
    for seed in range(100):
        np.random.seed(seed)  # HEDJitter draws from the global numpy RNG
        transform.compose.set_random_seed(seed)
        sample = _denormalise(transform(image=img, mask=gt)["image"])
        if all(np.abs(sample.astype(int) - k).mean() > 2 for k in kept):
            kept.append(sample)
        if len(kept) == N_TRAIN_SAMPLES:
            break
    return [_save(k, f"train_{arm}_{i}", size=256) for i, k in enumerate(kept)]


def _macenko_orientation_check(img: np.ndarray, samples: List[str]) -> Dict[str, Any]:
    """How often MacenkoNormalize recolours a flipped/rotated copy of the tile differently.

    Flips and rotations only move pixels, so a stable normalisation would give
    the same mean colour for all eight orientations. Returns how many of the
    eight differ from the unrotated result, and which saved training thumbnails
    are affected (by mean colour, threshold MACENKO_ODD_DIFF).
    """
    ref = inference.macenko(img).reshape(-1, 3).mean(axis=0)
    def odd(rgb: np.ndarray) -> bool:
        return bool(np.abs(rgb.reshape(-1, 3).mean(axis=0) - ref).max() > MACENKO_ODD_DIFF)
    variants = [np.rot90(f, k) for f in (img, img[:, ::-1]) for k in range(4)]
    thumbs = [np.asarray(Image.open(DOCS / src).convert("RGB")) for src in samples]
    return {"n_orient": len(variants),
            "n_odd": sum(odd(inference.macenko(np.ascontiguousarray(v))) for v in variants),
            "odd_thumbs": [i for i, t in enumerate(thumbs) if odd(t)]}


def _showcase_candidates(cfg, cube: data.Cube, k: int = 5) -> List[Dict[str, Any]]:
    """Val tiles with the largest 20-run stain-jitter gain, above-median vessel
    coverage, and a seed-42 result that agrees (so the hero image is not misleading)."""
    a = {arm: i for i, arm in enumerate(cube.arms)}
    mean = cube.dice.mean(axis=1)
    gain = mean[a[JITTER_ARM]] - mean[a[BASE_ARM]]
    si = cube.seed_index(data.LIVE_SEED)
    cov = np.array([data.load_tile(cfg, t)[1].mean() for t in cube.tile_ids])
    keep = [i for i in np.argsort(-gain)
            if cov[i] > np.median(cov) and cube.dice[a[JITTER_ARM], si, i]
            > cube.dice[a[BASE_ARM], si, i]]
    return [{"id": cube.tile_ids[i], "gain": float(gain[i]), "coverage": float(cov[i])}
            for i in keep[:k]]


def _contact_sheet(cfg, candidates: List[Dict], models: Dict, device) -> None:
    """Rows of (expert outline, baseline, stain jitter) for each candidate tile."""
    sheet = Image.new("RGB", (3 * CELL, len(candidates) * (CELL + PAD)), "#0d1014")
    draw = ImageDraw.Draw(sheet)
    for row, cand in enumerate(candidates):
        img, gt = data.load_tile(cfg, cand["id"])
        panels = [views.overlay(img, gt=gt)] + [
            views.overlay(img, inference.predict(models[arm], arm, img, device),
                          color=views.arm_color(arm)) for arm in (BASE_ARM, JITTER_ARM)]
        y = row * (CELL + PAD)
        draw.text((6, y + 6), f"#{row + 1} {cand['id']}  gain {cand['gain']:+.3f}  "
                              f"coverage {cand['coverage']:.1%}", fill="#e8eaed")
        for col, panel in enumerate(panels):
            sheet.paste(Image.fromarray(panel).resize((CELL, CELL)), (col * CELL, y + PAD))
    CANDIDATES_PNG.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(CANDIDATES_PNG)


def _sweep(models: Dict, img: np.ndarray, gt: np.ndarray, device) -> Dict[str, List[Dict]]:
    """Seed-42 Dice at each stain strength, for both stains."""
    out = {}
    for axis in ("h_scale", "e_scale"):
        rows = []
        for s in SWEEP_POINTS:
            shift = {"h_scale": 1.0, "e_scale": 1.0, axis: s}
            shifted = img if s == 1.0 else hed_shift(img, **shift)
            row = {"shift": s}
            for arm, model in models.items():
                row[arm] = inference.scores(inference.predict(model, arm, shifted, device), gt)[0]
            rows.append(row)
        out[axis] = rows
    return out


def _tile_images(img: np.ndarray, gt: np.ndarray, models: Dict, device) -> Dict[str, Any]:
    """Hero, stain-variant, channel, training-sample and prediction images."""
    preds = {arm: inference.predict(m, arm, img, device) for arm, m in models.items()}
    samples = {arm: _train_samples(arm, img, gt) for arm in models}
    channels = hed_channels(img)
    return {
        "hero": {"tile": _save(img, "hero_tile"),
                 "gt": _save(views.overlay(img, gt=gt, alpha=0.6), "hero_gt"),
                 "pred": _save(views.overlay(img, preds[JITTER_ARM], alpha=0.6,
                                             color=views.arm_color(JITTER_ARM)), "hero_pred")},
        "variants": [{"strength": s, "src": _save(img if s == 1.0 else hed_shift(img, s, s),
                                                  f"stain_{i}", size=384)}
                     for i, s in enumerate(STAIN_VARIANTS)],
        "channels": {"H": _save(views.to_gray_image(channels["H"]), "channel_h", size=384),
                     "E": _save(views.to_gray_image(channels["E"]), "channel_e", size=384),
                     "macenko": _save(inference.macenko(img), "macenko", size=384)},
        "train_samples": samples,
        "macenko_check": _macenko_orientation_check(img, samples[MAC_ARM]),
        "preds": {arm: {"src": _save(views.overlay(img, p, gt, error_view=True, alpha=0.6),
                                     f"pred_{arm}", size=384),
                        "dice": inference.scores(p, gt)[0]} for arm, p in preds.items()},
    }


def image_pass(summary: Dict, arms: List[str], showcase: Optional[str]) -> Dict[str, Any]:
    """Write every image and return the values that need raw data (build meta)."""
    cfg = data.load_demo_config()
    if not data.local_mode(cfg):
        raise RuntimeError("the image pass needs data/raw and the masks; use --skip-images")
    device = inference.resolve_device()
    models = {a: inference.load_model(data.checkpoint_path(cfg, a, data.LIVE_SEED), device)
              for a in arms}
    candidates = _showcase_candidates(cfg, data.build_cube(summary, arms, "val"))
    _contact_sheet(cfg, candidates, models, device)
    tile_id = showcase or candidates[0]["id"]
    img, gt = data.load_tile(cfg, tile_id)
    with open(cfg.paths.splits, encoding="utf-8") as f:
        splits = json.load(f)
    hed = next(t for t in inference.EVAL_BUILDERS[JITTER_ARM](train=True).compose.transforms
               if type(t).__name__ == "HEDJitter")
    return {
        "showcase": {"id": tile_id, "coverage": float(gt.mean())},
        "candidates": candidates,
        "splits": {k: (len(v) if isinstance(v, list) else v) for k, v in splits.items()
                   if k in ("train", "val", "test", "train_wsis", "val_wsis", "test_wsis")},
        "images": _tile_images(img, gt, models, device),
        "sweep": _sweep(models, img, gt, device),
        "hed": {"sigma": hed.sigma, "p": hed.p},
    }


# ---------------------------------------------------------------------- render


def _to_builtin(obj: Any) -> Any:
    """numpy scalars -> Python numbers for JSON."""
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def render(context: Dict[str, Any]) -> List[str]:
    """Render docs/index.html, copy static files; return unexplained numbers."""
    numbers = Numbers()
    env = Environment(loader=FileSystemLoader(SITE_SRC), autoescape=select_autoescape())
    env.filters.update(numbers.filters())
    html = env.get_template("index.html.j2").render(**context)
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "index.html").write_text(html, encoding="utf-8")
    for name in ("style.css", "site.js"):
        shutil.copy2(SITE_SRC / name, DOCS / name)
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    return unexplained_numbers(html, numbers.emitted)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-images", action="store_true",
                        help="re-render HTML from docs/build_meta.json (no raw data needed)")
    parser.add_argument("--showcase", help="val tile id for the hero image")
    args = parser.parse_args()
    cfg = data.load_demo_config()
    arms = data.arms_from_config(cfg)
    views.check_arms(arms)
    summary = data.load_summary(cfg)
    meta_path = DOCS / "build_meta.json"
    if args.skip_images:
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} is missing; run a full build first")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = _to_builtin(image_pass(summary, arms, args.showcase))
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[build_site] showcase candidates: {CANDIDATES_PNG}")
        for i, c in enumerate(meta["candidates"], start=1):
            print(f"  #{i} {c['id']}  gain {c['gain']:+.3f}  coverage {c['coverage']:.1%}")
        print(f"[build_site] showcase tile: {meta['showcase']['id']}")
    stray = render(_to_builtin(build_context(summary, arms, meta)))
    print(f"[build_site] wrote {DOCS / 'index.html'}")
    if stray:
        raise SystemExit(f"[build_site] numbers in the page with no source: {stray}")
    print("[build_site] number check passed: every number in the page text has a source")


if __name__ == "__main__":
    main()
