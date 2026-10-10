"""Data access for the demo: config, summary, per-tile cube, tiles and masks.

All statistics come from the committed `artifacts_v2/phase3_summary.json`.
Images come from one of two places:

- local mode: the repo's `data/raw/train/*.tif` + `data/processed/masks/*.png`
  and checkpoints under `results_v2/checkpoints/`,
- deployed mode: the curated `demo_assets/` folder written by `demo/build_deploy.py`,
  with checkpoints downloaded from the HF model repo named in its manifest.

Functions here are plain (no Streamlit); `demo/app.py` adds caching.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from src.utils.paths import Config, load_config, project_root

MODEL_NAME = "unet_resnet34"
LIVE_SEED = 42
CONFIG_PATH = project_root() / "configs" / "rerun_v2.yaml"
ASSETS_DIR = project_root() / "demo_assets"
REPO_URL = "https://github.com/Yaser-Alharbi/kidney-blood-vessel-segmentation"
SITE_URL = "https://yaser-alharbi.github.io/kidney-blood-vessel-segmentation/"
LAB_URL = "https://kidney-blood-vessel-segmentation.streamlit.app/"
WEIGHTS_URL = "https://huggingface.co/Yaser-Alharbi/vessel-stain-lab-weights"
FORCE_CURATED_ENV = "DEMO_FORCE_CURATED"  # "1" simulates the deployed app locally

# tile rankings (rank_tiles keys on position, so keep the order)
SORTS = (
    "Largest stain-jitter gain (where it helps most)",
    "Largest stain-jitter loss (where it hurts most)",
    "Lowest mean Dice (hardest tiles)",
    "Highest mean Dice (easiest tiles)",
    "Highest Dice std across runs (least consistent)",
    "Tile id (all tiles)",
)


def run_tag(arm: str, seed: int) -> str:
    """Run tag as written by the training loop."""
    return f"{MODEL_NAME}_{arm}_seed{seed}"


def load_demo_config() -> Config:
    """The v2 config (20 seeds, two test WSIs)."""
    return load_config(CONFIG_PATH)


def arms_from_config(cfg: Config) -> List[str]:
    """Arm list in config order."""
    return list(cfg.raw["augs"])


def load_summary(cfg: Config) -> Dict[str, Any]:
    """The committed v2 phase-3 summary."""
    with open(cfg.paths.artifacts / "phase3_summary.json", "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class Cube:
    """Per-tile scores for one split: values[arm][seed_index][tile_index]."""

    tile_ids: List[str]
    seeds: List[int]
    arms: List[str]
    dice: np.ndarray
    iou: np.ndarray

    def tile_index(self, tile_id: str) -> int:
        """Column of a tile id."""
        return self.tile_ids.index(tile_id)

    def seed_index(self, seed: int) -> int:
        """Row of a seed."""
        return self.seeds.index(seed)


def build_cube(summary: Dict[str, Any], arms: List[str], split: str) -> Cube:
    """Stack per-tile Dice/IoU of every (arm, seed) run, aligned by tile id."""
    seeds = [int(s) for s in summary["seeds"]]
    runs = summary["runs"]
    ref_ids = list(runs[run_tag(arms[0], seeds[0])][f"{split}_tile_ids"])
    dice = np.full((len(arms), len(seeds), len(ref_ids)), np.nan)
    iou = np.full_like(dice, np.nan)
    for ai, arm in enumerate(arms):
        for si, seed in enumerate(seeds):
            run = runs[run_tag(arm, seed)]
            col = {t: i for i, t in enumerate(run[f"{split}_tile_ids"])}
            order = [col[t] for t in ref_ids]
            dice[ai, si] = np.asarray(run[f"{split}_per_tile_dice"])[order]
            iou[ai, si] = np.asarray(run[f"{split}_per_tile_iou"])[order]
    return Cube(tile_ids=ref_ids, seeds=seeds, arms=list(arms), dice=dice, iou=iou)


def rank_tiles(cube: Cube, sort: str, allowed: Optional[List[str]] = None) -> List[str]:
    """Tile ids ordered by one of SORTS, restricted to `allowed` if given."""
    keep = [i for i, t in enumerate(cube.tile_ids) if allowed is None or t in allowed]
    mean = cube.dice.mean(axis=1)  # [arm, tile]
    a = {arm: i for i, arm in enumerate(cube.arms)}
    gap = mean[a["hed_only"]] - mean[a["rgb_aug"]] if {"hed_only", "rgb_aug"} <= set(a) else None
    keys = {
        SORTS[0]: -gap if gap is not None else None,
        SORTS[1]: gap,
        SORTS[2]: mean.mean(axis=0),
        SORTS[3]: -mean.mean(axis=0),
        SORTS[4]: -cube.dice.std(axis=1).mean(axis=0),
    }
    key = keys.get(sort)
    if key is None:
        return sorted(cube.tile_ids[i] for i in keep)
    return [cube.tile_ids[i] for i in sorted(keep, key=lambda i: key[i])]


def local_mode(cfg: Config) -> bool:
    """True when the repo's raw tiles and masks are on disk (and not overridden)."""
    if os.environ.get(FORCE_CURATED_ENV) == "1":
        return False
    return (cfg.paths.raw / "train").is_dir() and cfg.paths.masks.is_dir()


def load_manifest() -> Dict[str, Any]:
    """The deployed curated-tile manifest (empty when absent)."""
    path = ASSETS_DIR / "manifest.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def available_tiles(cfg: Config, cube: Cube, split: str) -> List[str]:
    """Tiles that can be shown: all in local mode, curated ones otherwise."""
    if local_mode(cfg):
        return list(cube.tile_ids)
    tiles = load_manifest().get("tiles", [])
    return [t["id"] for t in tiles if t["split"] == split]


def _read_mask(path: Path) -> np.ndarray:
    """PNG -> uint8 [H, W] in {0, 1}."""
    m = np.array(Image.open(path))
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.uint8)


def load_tile(cfg: Config, tile_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """RGB tile (uint8 HWC) and ground-truth mask (uint8 HW, 0/1).

    Mirrors `HuBMAPDataset._load_pair` in local mode.
    """
    if local_mode(cfg):
        img_path = cfg.paths.raw / "train" / f"{tile_id}.tif"
        mask_path = cfg.paths.masks / f"{tile_id}.png"
    else:
        img_path = ASSETS_DIR / "tiles" / f"{tile_id}.png"
        mask_path = ASSETS_DIR / "masks" / f"{tile_id}.png"
    img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    return img, _read_mask(mask_path)


def stored_prediction(arm: str, tile_id: str) -> Optional[np.ndarray]:
    """Precomputed seed-42 prediction mask from demo_assets, if present."""
    path = ASSETS_DIR / "preds" / arm / f"{tile_id}.png"
    return _read_mask(path) if path.exists() else None


def weights_repo() -> Optional[str]:
    """HF model repo holding the seed-42 checkpoints (deployed mode only)."""
    return load_manifest().get("weights_repo")


def checkpoint_path(cfg: Config, arm: str, seed: int) -> Optional[Path]:
    """Checkpoint for (arm, seed): results_v2, demo_assets, then the HF weights repo.

    Only the seed-42 checkpoints are published, so other seeds resolve to None
    when not on local disk. Downloads land in the HF cache.
    """
    name = f"{run_tag(arm, seed)}.pth"
    for path in (cfg.paths.results / "checkpoints" / name, ASSETS_DIR / "checkpoints" / name):
        if path.exists():
            return path
    repo = weights_repo()
    if repo is None or seed != LIVE_SEED:
        return None
    # lazy import: only deployed mode needs huggingface_hub
    from huggingface_hub import hf_hub_download  # pylint: disable=import-outside-toplevel

    return Path(hf_hub_download(repo_id=repo, filename=name))


def cross_seed_values(
    summary: Dict[str, Any], split: str, metric: str
) -> Dict[str, Dict[str, float]]:
    """{arm: {seed: mean per-tile score}} from stats.<split>.cross_seed."""
    block = summary["stats"][split]["cross_seed"][metric]
    return {arm: dict(v["values_per_seed"]) for arm, v in block.items()}


def wins_vs_base(values: Dict[str, Dict[str, float]], arm: str, base: str) -> int:
    """Seeds on which `arm` scored strictly higher than `base`."""
    return sum(1 for s, v in values[arm].items() if v > values[base][s])
