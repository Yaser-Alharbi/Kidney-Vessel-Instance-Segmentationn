"""HuBMAP Hacking the Human Vasculature: main entrypoint."""

from __future__ import annotations

import json
import platform
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import torch

from src.data.build_masks import MaskBuildSummary, build_masks
from src.data.dataset import HuBMAPDataset, _save_sample_item_figure
from src.data.inspect_data import InspectionSummary, inspect_data
from src.data.splits import SplitSummary, build_splits
from src.utils.paths import Config, ensure_dirs, load_config
from src.utils.seed import set_seed


# Cached run outputs. Committed to git so CI can replay them when
# data/raw is absent (the 4 GB dataset is gitignored).
ARTIFACT_SUMMARY_FILES = [
    "inspection_summary.json",
    "mask_build_summary.json",
    "splits_summary.json",
    "phase2_summary.json",
]
ARTIFACT_FIGURE_FILES = [
    "data_samples.png",
    "sample_dataset_item.png",
]


def _device_string(cfg: Config) -> str:
    requested = cfg.device.lower()
    if requested == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def _write_results_md(
    cfg: Config,
    inspection: InspectionSummary,
    masks: MaskBuildSummary,
    splits: SplitSummary,
    train_len: int,
    val_len: int,
    test_len: int = 0,
) -> Path:
    """Render the Phase 2 report to ~/Desktop/hubmap_phase2_results.md."""
    md_path = cfg.paths.external_results_md
    md_path.parent.mkdir(parents=True, exist_ok=True)

    ds_counts = inspection.dataset_value_counts
    n_total = inspection.n_tiles_on_disk
    n_d1 = ds_counts.get(1, 0)
    n_d2 = ds_counts.get(2, 0)
    n_d3 = ds_counts.get(3, 0)
    d1_wsis = sorted(inspection.dataset1_wsi_value_counts.keys())

    ann = inspection.annotation_type_counts
    n_glom = ann.get("glomerulus", 0)
    n_unsure = ann.get("unsure", 0)

    overlap_status = "PASSED" if splits.overlap_count == 0 else "FAILED"
    masks_dir_count = masks.masks_written + masks.masks_skipped_existing
    device_label = (
        "MPS (Apple Silicon)"
        if _device_string(cfg) == "mps"
        else _device_string(cfg).upper()
    )

    md = f"""# HuBMAP Phase 2: Setup Results

## Hypothesis
On the HuBMAP Vasculature dataset, stain-aware augmentation (HED colour jitter + Macenko stain normalization) yields higher Dice than standard RGB augmentation, because the training set comes from only 2 WSIs and stain variability is the dominant domain shift in PAS histology.

## Category
Training Strategies → Data Augmentation

## Dataset summary
- Total tiles: {n_total}
- Dataset 1 (expert-verified) tiles: {n_d1}
- Dataset 2 (noisy) tiles: {n_d2}
- Dataset 3 (unlabelled) tiles: {n_d3}
- Source WSIs in dataset 1: {d1_wsis}
- Mean vessel pixel fraction: {masks.mean_vessel_pixel_fraction:.6f}

## Splits
- Train WSI(s): {splits.train_wsis}, {splits.n_train_tiles} tiles (dataset 1, clean labels)
- Val WSI(s): {splits.val_wsis}, {splits.n_val_tiles} tiles (dataset 1, clean labels)
- Test WSI(s): {splits.test_wsis}, {splits.n_test_tiles} tiles (dataset 2, **noisy labels**)
- Test strategy: {splits.test_strategy}
- Pairwise overlap check: {overlap_status}

Test labels are dataset 2 (auto-generated, noisy). Use the test set for
cross-WSI generalization comparisons between models, not for absolute
Dice (the score is biased by label noise).

## Mask generation
- Masks created: {masks.masks_written + masks.masks_skipped_existing} (newly written this run: {masks.masks_written}; skipped existing: {masks.masks_skipped_existing})
- Annotations skipped (non blood_vessel): glomerulus={n_glom}, unsure={n_unsure}

## Environment
- Device: {device_label} ({platform.platform()})
- Seed: {cfg.seed}
- Determinism flag set: yes (torch.use_deterministic_algorithms(True, warn_only=True))

## Files generated
- `data/processed/masks/` ({masks_dir_count} files)
- `data/processed/splits.json`
- `results/data_samples.png`
- `results/sample_dataset_item.png`

## Dataset objects
- HuBMAPDataset(train) length: {train_len}
- HuBMAPDataset(val) length:   {val_len}
- HuBMAPDataset(test) length:  {test_len}

## Next phase
Phase 3: model development (U-Net + SegFormer baselines, augmentation pipelines, training loop).
"""
    md_path.write_text(md)
    return md_path


def _save_artifacts(cfg: Config) -> None:
    """Copy run outputs into artifacts/ for CI replay."""
    art = cfg.paths.artifacts
    art.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_SUMMARY_FILES:
        src = cfg.paths.processed / name
        if src.exists():
            shutil.copy2(src, art / name)
    for name in ARTIFACT_FIGURE_FILES:
        src = cfg.paths.results / name
        if src.exists():
            shutil.copy2(src, art / name)
    print(f"[main] artifacts cached      : {art}")


def _str_keys_to_int(d: Dict[Any, Any]) -> Dict[int, Any]:
    """JSON object keys are strings; cast back to int for our dataclasses."""
    return {int(k): v for k, v in d.items()}


def _replay_from_artifacts(cfg: Config) -> None:
    """Regenerate outputs from cached artifacts (CI path with no raw data)."""
    art = cfg.paths.artifacts
    summary_path = art / "phase2_summary.json"
    if not summary_path.exists():
        print(
            f"[main] no cached artifacts at {art} (need phase2_summary.json), "
            "exiting cleanly."
        )
        return

    with open(summary_path, "r") as f:
        payload = json.load(f)

    insp = dict(payload["inspection"])
    insp["dataset_value_counts"] = _str_keys_to_int(insp["dataset_value_counts"])
    insp["source_wsi_value_counts"] = _str_keys_to_int(insp["source_wsi_value_counts"])
    insp["dataset1_wsi_value_counts"] = _str_keys_to_int(insp["dataset1_wsi_value_counts"])
    inspection = InspectionSummary(**insp)
    masks = MaskBuildSummary(**payload["masks"])
    splits = SplitSummary(**payload["splits"])

    cfg.paths.results.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FIGURE_FILES:
        src = art / name
        if src.exists():
            shutil.copy2(src, cfg.paths.results / name)

    md_path = _write_results_md(
        cfg,
        inspection=inspection,
        masks=masks,
        splits=splits,
        train_len=int(payload["train_len"]),
        val_len=int(payload["val_len"]),
        test_len=int(payload.get("test_len", 0)),
    )
    print(f"[main] replay complete; report at {md_path}")


def main():
    """
    This function must execute the complete experimental workflow developed
    for the selected computer vision competition and research hypothesis.

    The automated grading system will call this function. Therefore:
    - The function signature must not be changed.
    - It must not require any user input.
    - It must run deterministically (fixed random seeds).
    - All outputs (metrics, logs, plots) must be saved to disk.

    The workflow should include:

        1. Dataset loading and preparation
           - Download or load the competition dataset
           - Apply preprocessing and data augmentation (if applicable)
           - Create training / validation / test splits

        2. Model construction
           - Build the baseline model
           - Build the proposed model(s) used to test the hypothesis

        3. Training
           - Train model(s) using defined hyperparameters
           - Log training and validation performance

        4. Evaluation
           - Evaluate on validation/test data
           - Compute relevant metrics (e.g., accuracy, F1, etc.)
           - Compare models if testing a hypothesis

        5. Analysis and visualisation
           - Generate and save plots used in the report
           - Save final metrics to disk (e.g., JSON/CSV)

    The purpose of this function is to reproduce all experimental evidence
    presented in the report in a fully automated and reproducible manner.

    Phase 2 implements section 1 (data pipeline) plus the setup-only parts
    of section 5 (preview figures and report). Sections 2-4 are Phase 3.
    """
    cfg = load_config()
    ensure_dirs(cfg)

    print(f"[main] project root : {cfg.root}")
    print(f"[main] device       : {_device_string(cfg)}  (configured: {cfg.device})")
    print(f"[main] seed         : {cfg.seed}")
    set_seed(cfg.seed)

    # No raw data on disk: assume CI and replay from artifacts/.
    if not (cfg.paths.raw / "polygons.jsonl").exists():
        print("[main] no raw data, entering replay mode")
        _replay_from_artifacts(cfg)
        return

    # 1. Dataset loading and preparation.
    print("\n=== 1. Dataset loading and preparation ===")
    print("\n[1a] inspect raw data")
    inspection = inspect_data(cfg)

    print("\n[1b] build blood-vessel masks")
    masks = build_masks(cfg)

    print("\n[1c] build WSI-grouped splits")
    splits = build_splits(cfg)

    print("\n[1d] instantiate datasets")
    with open(cfg.paths.splits, "r") as f:
        split_payload = json.load(f)
    img_dir = cfg.paths.raw / "train"
    train_ds = HuBMAPDataset(split_payload["train"], img_dir, cfg.paths.masks)
    val_ds = HuBMAPDataset(split_payload["val"], img_dir, cfg.paths.masks)
    test_ds = HuBMAPDataset(
        split_payload.get("test", []), img_dir, cfg.paths.masks
    )
    print(f"[main] train dataset length : {len(train_ds)}")
    print(f"[main] val   dataset length : {len(val_ds)}")
    print(f"[main] test  dataset length : {len(test_ds)}")

    sample = train_ds[0]
    print(
        f"[main] sample image  : shape={tuple(sample['image'].shape)}"
        f" dtype={sample['image'].dtype}"
    )
    print(
        f"[main] sample mask   : shape={tuple(sample['mask'].shape)}"
        f" dtype={sample['mask'].dtype}"
    )

    # 2. Model construction.        Phase 3.
    # 3. Training.                  Phase 3.
    # 4. Evaluation.                Phase 3.

    # 5. Analysis and visualisation.
    # Phase 2 writes the setup figures and the report. Training plots come in Phase 3.
    print("\n=== 5. Analysis and visualisation ===\n")
    sample_fig = cfg.paths.results / "sample_dataset_item.png"
    _save_sample_item_figure(sample, sample_fig)
    print(f"[main] sample preview saved : {sample_fig}")

    md_path = _write_results_md(
        cfg,
        inspection=inspection,
        masks=masks,
        splits=splits,
        train_len=len(train_ds),
        val_len=len(val_ds),
        test_len=len(test_ds),
    )
    print(f"[main] external report saved : {md_path}")

    summary_path = cfg.paths.processed / "phase2_summary.json"
    payload = {
        "inspection": asdict(inspection),
        "masks": asdict(masks),
        "splits": asdict(splits),
        "train_len": len(train_ds),
        "val_len": len(val_ds),
        "test_len": len(test_ds),
    }
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[main] phase2 summary json   : {summary_path}")

    _save_artifacts(cfg)

    print("\nPhase 2 setup complete.")


if __name__ == "__main__":
    main()
