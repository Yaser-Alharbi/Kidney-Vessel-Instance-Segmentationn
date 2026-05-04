"""Build a deterministic, WSI-grouped train/val/test split.

Train and val come from dataset 1 (clean labels), one WSI each.
Test is a held-out WSI from dataset 2 (noisy labels), giving a
cross-WSI evaluation set without sharing stain with train or val.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, List

import pandas as pd

from src.utils.paths import Config, ensure_dirs, load_config


@dataclass
class SplitSummary:
    train_wsis: List[int]
    val_wsis: List[int]
    n_train_tiles: int
    n_val_tiles: int
    overlap_count: int
    # Defaulted for back-compat with older cached artifacts.
    n_noisy_train_tiles: int = 0
    noisy_train_wsis: List[int] = field(default_factory=list)
    val_leak_excluded: int = 0
    n_test_tiles: int = 0
    test_wsis: List[int] = field(default_factory=list)
    test_strategy: str = "none"


def build_splits(cfg: Config) -> SplitSummary:
    ensure_dirs(cfg)
    meta_path = cfg.paths.raw / "tile_meta.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing tile_meta.csv: {meta_path}")

    df = pd.read_csv(meta_path)
    df1 = df[df["dataset"] == 1].copy()
    if df1.empty:
        raise RuntimeError("No tiles with dataset == 1 found in tile_meta.csv")

    wsis: List[int] = sorted(df1["source_wsi"].unique().tolist())
    rng = random.Random(cfg.seed)

    if len(wsis) == 2:
        train_wsis = [wsis[0]]
        val_wsis = [wsis[1]]
    else:
        shuffled = list(wsis)
        rng.shuffle(shuffled)
        n_val = max(1, round(0.2 * len(shuffled)))
        val_wsis = sorted(shuffled[:n_val])
        train_wsis = sorted(shuffled[n_val:])

    train_ids = sorted(df1[df1["source_wsi"].isin(train_wsis)]["id"].tolist())
    val_ids = sorted(df1[df1["source_wsi"].isin(val_wsis)]["id"].tolist())

    # Test = first dataset-2 WSI that is not already in train or val.
    df2 = df[df["dataset"] == 2].copy()
    n_d2_total = int(len(df2))
    d2_wsis = sorted(df2["source_wsi"].unique().tolist())
    held_out = [w for w in d2_wsis if w not in train_wsis and w not in val_wsis]
    if not held_out:
        raise RuntimeError(
            f"No dataset-2 WSI available for test (all dataset-2 WSIs {d2_wsis} "
            f"overlap train {train_wsis} or val {val_wsis})."
        )
    test_wsi = held_out[0]
    test_wsis = [test_wsi]
    test_ids = sorted(df2[df2["source_wsi"] == test_wsi]["id"].tolist())
    test_strategy = (
        f"held-out dataset-2 WSI {test_wsi} (noisy labels, "
        f"sort-first of {held_out})"
    )

    overlap_train_val = set(train_ids) & set(val_ids)
    overlap_train_test = set(train_ids) & set(test_ids)
    overlap_val_test = set(val_ids) & set(test_ids)
    overlap_count = (
        len(overlap_train_val) + len(overlap_train_test) + len(overlap_val_test)
    )
    assert overlap_count == 0, (
        f"Split overlap: train_val={len(overlap_train_val)} "
        f"train_test={len(overlap_train_test)} val_test={len(overlap_val_test)}"
    )

    # Noisy training pool = dataset-2 tiles in train_wsis only.
    train_wsi_set = set(train_wsis)
    val_wsi_set = set(val_wsis)
    df2_train = df2[df2["source_wsi"].isin(train_wsi_set)]
    df2_leak = df2[df2["source_wsi"].isin(val_wsi_set)]
    train_noisy_ids = sorted(df2_train["id"].tolist())
    noisy_wsis = sorted(df2_train["source_wsi"].unique().tolist())
    val_leak_excluded = int(len(df2_leak))

    assert not (set(train_noisy_ids) & set(val_ids)), (
        "Noisy train pool overlaps with val IDs"
    )
    assert not (set(train_noisy_ids) & set(test_ids)), (
        "Noisy train pool overlaps with test IDs"
    )

    out: Dict[str, object] = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "train_noisy": train_noisy_ids,
        "train_wsis": train_wsis,
        "val_wsis": val_wsis,
        "test_wsis": test_wsis,
        "noisy_train_wsis": noisy_wsis,
        "seed": cfg.seed,
        "test_strategy": test_strategy,
        "source": "tile_meta.csv (dataset==1 train/val; dataset==2 test/noisy)",
    }
    splits_path = cfg.paths.splits
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_path, "w") as f:
        json.dump(out, f, indent=2)

    summary = SplitSummary(
        train_wsis=train_wsis,
        val_wsis=val_wsis,
        n_train_tiles=len(train_ids),
        n_val_tiles=len(val_ids),
        overlap_count=overlap_count,
        n_noisy_train_tiles=len(train_noisy_ids),
        noisy_train_wsis=noisy_wsis,
        val_leak_excluded=val_leak_excluded,
        n_test_tiles=len(test_ids),
        test_wsis=test_wsis,
        test_strategy=test_strategy,
    )

    print("[splits] WSI-grouped split:")
    print(f"  dataset-1 WSIs     : {wsis}")
    print(f"  train WSI(s)       : {train_wsis}  ({len(train_ids)} tiles, dataset 1)")
    print(f"  val   WSI(s)       : {val_wsis}  ({len(val_ids)} tiles, dataset 1)")
    print(f"  test  WSI(s)       : {test_wsis}  ({len(test_ids)} tiles, dataset 2 noisy)")
    print(f"  test strategy      : {test_strategy}")
    print(f"  pairwise overlap   : {overlap_count} (assert PASSED)")
    print("[splits] dataset-2 noisy training pool:")
    print(f"  total dataset-2 tiles    : {n_d2_total}")
    print(f"  excluded (in val WSIs)   : {val_leak_excluded}")
    print(f"  noisy train pool         : {len(train_noisy_ids)} tiles "
          f"across WSIs {noisy_wsis}")
    print(f"  saved                    : {splits_path}")

    summary_path = cfg.paths.processed / "splits_summary.json"
    with open(summary_path, "w") as f:
        json.dump(asdict(summary), f, indent=2)
    return summary


if __name__ == "__main__":
    build_splits(load_config())
