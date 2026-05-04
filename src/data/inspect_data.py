"""Dataset inspection.

Prints headline counts (tiles, polygons, datasets, WSIs, annotation
types) and saves results/data_samples.png: a 2x2 grid of random
dataset-1 tiles with vessel polygons overlaid.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.utils.paths import Config, ensure_dirs, load_config


@dataclass
class InspectionSummary:
    n_tiles_on_disk: int
    n_polygon_lines: int
    dataset_value_counts: Dict[int, int]
    source_wsi_value_counts: Dict[int, int]
    dataset1_wsi_value_counts: Dict[int, int]
    sample_polygon_keys: List[str]
    sample_polygon_first_annotation_keys: List[str]
    annotation_type_counts: Dict[str, int]
    samples_figure_path: str


def _load_polygons_index(polygons_path: Path) -> Dict[str, list]:
    """Return {tile_id: [annotations,...]} from polygons.jsonl."""
    index: Dict[str, list] = {}
    with open(polygons_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            index[obj["id"]] = obj.get("annotations", [])
    return index


def inspect_data(cfg: Config) -> InspectionSummary:
    ensure_dirs(cfg)
    raw = cfg.paths.raw
    train_dir = raw / "train"
    polygons_path = raw / "polygons.jsonl"
    tile_meta_path = raw / "tile_meta.csv"

    for p in (train_dir, polygons_path, tile_meta_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing expected raw asset: {p}")

    n_tiles = sum(1 for _ in train_dir.glob("*.tif"))
    polygons_index = _load_polygons_index(polygons_path)
    n_poly_lines = len(polygons_index)

    df = pd.read_csv(tile_meta_path)
    print("[inspect] tile_meta.csv head:")
    print(df.head().to_string(index=False))

    ds_counts = df["dataset"].value_counts().sort_index().to_dict()
    wsi_counts = df["source_wsi"].value_counts().sort_index().to_dict()
    df1 = df[df["dataset"] == 1]
    wsi1_counts = df1["source_wsi"].value_counts().sort_index().to_dict()

    print(f"[inspect] tiles on disk in train/        : {n_tiles}")
    print(f"[inspect] lines in polygons.jsonl        : {n_poly_lines}")
    print(f"[inspect] dataset value counts           : {ds_counts}")
    print(f"[inspect] source_wsi value counts        : {wsi_counts}")
    print(f"[inspect] dataset==1 source_wsi counts   : {wsi1_counts}")

    ann_type_counts: Dict[str, int] = {}
    for anns in polygons_index.values():
        for a in anns:
            ann_type_counts[a.get("type", "unknown")] = (
                ann_type_counts.get(a.get("type", "unknown"), 0) + 1
            )
    print(f"[inspect] annotation type counts         : {ann_type_counts}")

    sample_id = next(iter(polygons_index))
    sample_obj: Dict[str, Any] = {
        "id": sample_id,
        "annotations": polygons_index[sample_id],
    }
    sample_keys = list(sample_obj.keys())
    first_ann_keys = (
        list(sample_obj["annotations"][0].keys()) if sample_obj["annotations"] else []
    )
    print(f"[inspect] sample polygon entry id        : {sample_id}")
    print(f"[inspect] sample polygon top-level keys  : {sample_keys}")
    print(f"[inspect] sample first-annotation keys   : {first_ann_keys}")

    fig_path = cfg.paths.results / "data_samples.png"
    _save_sample_grid(cfg, df1, polygons_index, fig_path)

    summary = InspectionSummary(
        n_tiles_on_disk=int(n_tiles),
        n_polygon_lines=int(n_poly_lines),
        dataset_value_counts={int(k): int(v) for k, v in ds_counts.items()},
        source_wsi_value_counts={int(k): int(v) for k, v in wsi_counts.items()},
        dataset1_wsi_value_counts={int(k): int(v) for k, v in wsi1_counts.items()},
        sample_polygon_keys=sample_keys,
        sample_polygon_first_annotation_keys=first_ann_keys,
        annotation_type_counts={k: int(v) for k, v in ann_type_counts.items()},
        samples_figure_path=str(fig_path),
    )

    summary_path = cfg.paths.processed / "inspection_summary.json"
    with open(summary_path, "w") as f:
        json.dump(asdict(summary), f, indent=2)

    return summary


def _save_sample_grid(
    cfg: Config,
    df1: pd.DataFrame,
    polygons_index: Dict[str, list],
    out_path: Path,
) -> None:
    """Render 4 random dataset-1 tiles with vessel polygons overlaid."""
    rng = random.Random(cfg.seed)
    candidates = [tid for tid in df1["id"].tolist() if tid in polygons_index]
    if not candidates:
        candidates = list(polygons_index.keys())

    n = min(4, len(candidates))
    sampled = rng.sample(candidates, n)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    for ax, tile_id in zip(axes, sampled):
        img_path = cfg.paths.raw / "train" / f"{tile_id}.tif"
        if not img_path.exists():
            ax.set_visible(False)
            continue
        img = np.asarray(Image.open(img_path).convert("RGB"))
        ax.imshow(img)
        for ann in polygons_index.get(tile_id, []):
            if ann.get("type") != "blood_vessel":
                continue
            for poly in ann.get("coordinates", []):
                pts = np.asarray(poly)
                if pts.size == 0:
                    continue
                ax.plot(pts[:, 0], pts[:, 1], color="red", linewidth=1.0)
        ax.set_title(tile_id, fontsize=10)
        ax.axis("off")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("HuBMAP — random tiles with blood_vessel polygons", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[inspect] sample grid saved to            : {out_path}")


if __name__ == "__main__":
    inspect_data(load_config())
