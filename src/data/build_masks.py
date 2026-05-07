"""Rasterize blood-vessel polygons into binary 512x512 masks.

For each line of polygons.jsonl, draw every "blood_vessel" polygon
with cv2.fillPoly(value=1) onto a zero mask and write it to
data/processed/masks/{tile_id}.png. Other annotation types
(glomerulus, unsure) are counted but not drawn.

Idempotent: tiles whose mask already exists are skipped.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from tqdm import tqdm

from src.utils.paths import Config, ensure_dirs, load_config


@dataclass
class MaskBuildSummary:
    total_tiles_in_jsonl: int
    masks_written: int
    masks_skipped_existing: int
    annotation_counts: Dict[str, int]
    blood_vessel_polygons_drawn: int
    mean_vessel_pixel_fraction: float
    min_vessel_pixel_fraction: float
    max_vessel_pixel_fraction: float


def _polygons_to_mask(
    annotations: List[dict],
    image_size: int,
    target_type: str = "blood_vessel",
) -> tuple[np.ndarray, int]:
    """Rasterize all `target_type` polygons in `annotations` into a uint8 mask."""
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    drawn = 0
    for ann in annotations:
        if ann.get("type") != target_type:
            continue
        for polygon in ann.get("coordinates", []):
            pts = np.asarray(polygon, dtype=np.int32)
            if pts.ndim != 2 or pts.shape[0] < 3:
                continue
            cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], color=1)
            drawn += 1
    return mask, drawn


def build_masks(cfg: Config) -> MaskBuildSummary:
    """Build masks for every tile referenced in `polygons.jsonl`."""
    ensure_dirs(cfg)
    polygons_path = cfg.paths.raw / "polygons.jsonl"
    if not polygons_path.exists():
        raise FileNotFoundError(f"Missing polygons file: {polygons_path}")

    masks_dir: Path = cfg.paths.masks
    image_size = cfg.image_size

    annotation_counts: Counter[str] = Counter()
    fractions: List[float] = []
    polygons_drawn = 0
    masks_written = 0
    masks_skipped = 0
    total_tiles = 0

    with open(polygons_path, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Building masks", unit="tile"):
        total_tiles += 1
        obj = json.loads(line)
        tile_id = obj["id"]
        annotations = obj.get("annotations", [])

        for ann in annotations:
            annotation_counts[ann.get("type", "unknown")] += 1

        out_path = masks_dir / f"{tile_id}.png"
        if out_path.exists():
            masks_skipped += 1
            existing = cv2.imread(str(out_path), cv2.IMREAD_GRAYSCALE)
            if existing is not None:
                fractions.append(float(existing.mean()))
            continue

        mask, drawn = _polygons_to_mask(annotations, image_size, "blood_vessel")
        polygons_drawn += drawn
        ok = cv2.imwrite(str(out_path), mask)
        if not ok:
            raise RuntimeError(f"cv2.imwrite failed for {out_path}")
        masks_written += 1
        fractions.append(float(mask.mean()))

    skipped_non_vessel = {
        k: int(v) for k, v in annotation_counts.items() if k != "blood_vessel"
    }
    summary = MaskBuildSummary(
        total_tiles_in_jsonl=total_tiles,
        masks_written=masks_written,
        masks_skipped_existing=masks_skipped,
        annotation_counts=dict(annotation_counts),
        blood_vessel_polygons_drawn=polygons_drawn,
        mean_vessel_pixel_fraction=float(np.mean(fractions)) if fractions else 0.0,
        min_vessel_pixel_fraction=float(np.min(fractions)) if fractions else 0.0,
        max_vessel_pixel_fraction=float(np.max(fractions)) if fractions else 0.0,
    )

    print("[build_masks] Summary:")
    print(f"  tiles in polygons.jsonl   : {summary.total_tiles_in_jsonl}")
    print(f"  masks newly written       : {summary.masks_written}")
    print(f"  masks skipped (existing)  : {summary.masks_skipped_existing}")
    print(f"  blood_vessel polygons     : {summary.blood_vessel_polygons_drawn}")
    print(f"  annotation counts         : {summary.annotation_counts}")
    print(
        f"  vessel coverage (frac)    : "
        f"mean={summary.mean_vessel_pixel_fraction:.4f}  "
        f"min={summary.min_vessel_pixel_fraction:.4f}  "
        f"max={summary.max_vessel_pixel_fraction:.4f}"
    )
    print(f"  skipped non-vessel types  : {skipped_non_vessel}")

    summary_path = cfg.paths.processed / "mask_build_summary.json"
    with open(summary_path, "w") as f:
        json.dump(asdict(summary), f, indent=2)

    return summary


if __name__ == "__main__":
    build_masks(load_config())
