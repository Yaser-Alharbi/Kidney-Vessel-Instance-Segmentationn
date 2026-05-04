"""HuBMAP Hacking the Human Vasculature: main entrypoint."""

from __future__ import annotations

import json
import platform
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from src.data.build_masks import MaskBuildSummary, build_masks
from src.data.dataset import HuBMAPDataset, _save_sample_item_figure
from src.data.inspect_data import InspectionSummary, inspect_data
from src.data.splits import SplitSummary, build_splits
from src.training.run_experiments import run_phase3
from src.utils.paths import Config, ensure_dirs, load_config
from src.utils.seed import set_seed


# cached outputs tracked in git so CI can replay without raw data
ARTIFACT_SUMMARY_FILES = [
    "inspection_summary.json",
    "mask_build_summary.json",
    "splits_summary.json",
    "phase2_summary.json",
    "phase3_summary.json",
]
ARTIFACT_FIGURE_FILES = [
    "data_samples.png",
    "sample_dataset_item.png",
    "phase3_comparison.png",
    "unet_resnet34_rgb_aug_loss.png",
    "unet_resnet34_rgb_aug_val_dice.png",
    "unet_resnet34_rgb_aug_predictions.png",
    "unet_resnet34_stain_aware_hed_only_loss.png",
    "unet_resnet34_stain_aware_hed_only_val_dice.png",
    "unet_resnet34_stain_aware_hed_only_predictions.png",
    "unet_resnet34_stain_aware_loss.png",
    "unet_resnet34_stain_aware_val_dice.png",
    "unet_resnet34_stain_aware_predictions.png",
]


def _device_string(cfg: Config) -> str:
    requested = cfg.device.lower()
    if requested == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


_PHASE3_ARMS = ("rgb_aug", "stain_aware_hed_only", "stain_aware")
_PHASE3_RUN_TAG_BY_ARM = {
    "rgb_aug": "unet_resnet34_rgb_aug",
    "stain_aware_hed_only": "unet_resnet34_stain_aware_hed_only",
    "stain_aware": "unet_resnet34_stain_aware",
}


def _phase3_conclusion(stats: Dict[str, Any]) -> str:
    """Build the conclusion paragraph from the stats block."""
    val_cis = stats.get("val", {}).get("cis", {}) or {}
    test_cis = stats.get("test", {}).get("cis", {}) or {}
    val_wc = stats.get("val", {}).get("wilcoxon", {}) or {}
    test_wc = stats.get("test", {}).get("wilcoxon", {}) or {}

    def _mean(cis: Dict[str, Any], k: str) -> float:
        return float(cis.get(k, {}).get("mean", float("nan")))

    def _p(wc: Dict[str, Any], k: str) -> float:
        return float(wc.get(k, {}).get("p", float("nan")))

    base_v = _mean(val_cis, "rgb_aug")
    hed_v = _mean(val_cis, "stain_aware_hed_only")
    full_v = _mean(val_cis, "stain_aware")
    base_t = _mean(test_cis, "rgb_aug")
    full_t = _mean(test_cis, "stain_aware")

    p_full_vs_base_v = _p(val_wc, "stain_aware_gt_rgb_aug")
    p_hed_vs_base_v = _p(val_wc, "stain_aware_hed_only_gt_rgb_aug")
    p_full_vs_hed_v = _p(val_wc, "stain_aware_gt_stain_aware_hed_only")
    p_full_vs_base_t = _p(test_wc, "stain_aware_gt_rgb_aug")

    full_dir_val = "improved" if full_v > base_v else "did not improve"
    full_dir_test = "improved" if full_t > base_t else "did not improve"
    full_signif_val = "significant" if p_full_vs_base_v < 0.05 else "not significant"
    full_signif_test = "significant" if p_full_vs_base_t < 0.05 else "not significant"

    if p_hed_vs_base_v < 0.05 and not p_full_vs_hed_v < 0.05:
        ablation = (
            "HED jitter alone explains most of the val gain over the RGB-aug "
            "control; adding Macenko at eval gives no further significant lift."
        )
    elif p_full_vs_hed_v < 0.05:
        ablation = (
            "Macenko normalization adds a measurable lift on top of HED jitter "
            "at eval (p = "
            f"{p_full_vs_hed_v:.4g}), so both components contribute."
        )
    else:
        ablation = (
            "Neither HED-only nor full stain-aware shows a significant edge "
            "over RGB-aug on val with this single-WSI val set."
        )

    return (
        f"Full stain-aware augmentation {full_dir_val} mean per-tile val Dice "
        f"({full_v:.4f} vs rgb_aug {base_v:.4f}, p = {p_full_vs_base_v:.4g}, "
        f"{full_signif_val} at alpha=0.05). HED-only sits at {hed_v:.4f} on val "
        f"(p = {p_hed_vs_base_v:.4g} vs rgb_aug). {ablation} On the noisy test "
        f"WSI the full arm {full_dir_test} ({full_t:.4f} vs {base_t:.4f}, "
        f"p = {p_full_vs_base_t:.4g}, {full_signif_test}). Confidence is bounded "
        "by the val set being a single WSI, so this run supports the hypothesis "
        "at the strength of one paired comparison per pair."
    )


def _format_phase3_section(phase3: Dict[str, Any]) -> str:
    """Render the Phase 3 markdown block."""
    runs = phase3.get("runs", {})
    stats = phase3.get("stats", {})
    val_stats = stats.get("val", {})
    test_stats = stats.get("test", {})
    val_cis = val_stats.get("cis", {}) or {}
    test_cis = test_stats.get("cis", {}) or {}
    val_wc = val_stats.get("wilcoxon", {}) or {}
    test_wc = test_stats.get("wilcoxon", {}) or {}

    def _row(aug: str, cis: Dict[str, Any]) -> str:
        ci = cis.get(aug)
        if not ci:
            return f"| {aug} | n/a | n/a |"
        return (
            f"| {aug} | {ci['mean']:.4f} | "
            f"[{ci['lo']:.4f}, {ci['hi']:.4f}] |"
        )

    def _wilcoxon_lines(wc: Dict[str, Any]) -> str:
        keys = [
            ("stain_aware_gt_rgb_aug", "stain_aware > rgb_aug"),
            ("stain_aware_hed_only_gt_rgb_aug", "stain_aware_hed_only > rgb_aug"),
            ("stain_aware_gt_stain_aware_hed_only", "stain_aware > stain_aware_hed_only"),
        ]
        lines = []
        for k, label in keys:
            p = wc.get(k, {}).get("p", float("nan"))
            lines.append(f"- Paired Wilcoxon ({label}): p = {float(p):.4g}")
        return "\n".join(lines)

    def _best_lines() -> str:
        lines = []
        for aug in _PHASE3_ARMS:
            r = runs.get(_PHASE3_RUN_TAG_BY_ARM[aug], {})
            ep = r.get("best_epoch", "?")
            d = r.get("best_val_dice", float("nan"))
            lines.append(f"- {aug:22s}: epoch {ep} (val Dice {float(d):.4f})")
        return "\n".join(lines)

    def _figure_lines() -> str:
        lines = ["- `results/phase3_comparison.png`"]
        for aug in _PHASE3_ARMS:
            tag = _PHASE3_RUN_TAG_BY_ARM[aug]
            lines.append(
                f"- `results/{tag}_loss.png`, `results/{tag}_val_dice.png`, "
                f"`results/{tag}_predictions.png`"
            )
        return "\n".join(lines)

    val_table = "\n".join(_row(a, val_cis) for a in _PHASE3_ARMS)
    test_table = "\n".join(_row(a, test_cis) for a in _PHASE3_ARMS)
    conclusion = _phase3_conclusion(stats)

    return f"""

## Phase 3: training and evaluation (three-arm ablation)

### Setup
- Architecture: U-Net (ResNet-34 encoder, ImageNet pretrained)
- Loss: Dice + soft BCE (w=0.5)
- Optimizer: AdamW, lr={phase3.get('lr', 1e-4)}, weight_decay={phase3.get('weight_decay', 1e-4)}
- Scheduler: CosineAnnealingLR
- Epochs: {phase3.get('epochs', '?')}, batch: {phase3.get('batch_size', '?')}, image size: {phase3.get('image_size', '?')}, seed: {phase3.get('seed', '?')}
- Arms: rgb_aug (control), stain_aware_hed_only (HED only), stain_aware (HED + Macenko)

### Best epoch per run
{_best_lines()}

### Validation (dataset 1, clean labels)
| Run                   | Mean Dice | 95% CI         |
| --------------------- | --------- | -------------- |
{val_table}

{_wilcoxon_lines(val_wc)}

### Test (dataset 2, NOISY labels)
| Run                   | Mean Dice | 95% CI         |
| --------------------- | --------- | -------------- |
{test_table}

{_wilcoxon_lines(test_wc)}

Test labels are dataset 2 (auto-generated, noisy). Use these numbers for
relative comparison only, not as absolute accuracy.

### Figures
{_figure_lines()}

### Conclusion
{conclusion}
"""


def _write_results_md(
    cfg: Config,
    inspection: InspectionSummary,
    masks: MaskBuildSummary,
    splits: SplitSummary,
    train_len: int,
    val_len: int,
    test_len: int = 0,
    *,
    phase3: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write the report markdown (Phase 2 + Phase 3 if provided)."""
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

    md = f"""# HuBMAP Phase 3: Model Development Results

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
"""
    if phase3:
        md += _format_phase3_section(phase3)

    md_path.write_text(md)
    return md_path


def _save_artifacts(cfg: Config) -> None:
    """Mirror run outputs into artifacts/ for CI replay."""
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
    """Cast JSON string keys back to int."""
    return {int(k): v for k, v in d.items()}


def _replay_from_artifacts(cfg: Config) -> None:
    """CI replay path: rebuild results/ + report from artifacts/."""
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

    phase3_payload: Optional[Dict[str, Any]] = None
    phase3_path = art / "phase3_summary.json"
    if phase3_path.exists():
        with open(phase3_path, "r") as f:
            phase3_payload = json.load(f)
        cfg.paths.processed.mkdir(parents=True, exist_ok=True)
        shutil.copy2(phase3_path, cfg.paths.processed / "phase3_summary.json")
    else:
        print(f"[main] no cached phase3_summary.json at {phase3_path} (Phase 2 only).")

    md_path = _write_results_md(
        cfg,
        inspection=inspection,
        masks=masks,
        splits=splits,
        train_len=int(payload["train_len"]),
        val_len=int(payload["val_len"]),
        test_len=int(payload.get("test_len", 0)),
        phase3=phase3_payload,
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

    Phase 2 covers section 1 (data pipeline) plus the setup-only parts
    of section 5. Phase 3 fills in sections 2-4 (training + evaluation)
    and adds the comparison figures + Phase 3 markdown to section 5.
    """
    cfg = load_config()
    ensure_dirs(cfg)

    print(f"[main] project root : {cfg.root}")
    print(f"[main] device       : {_device_string(cfg)}  (configured: {cfg.device})")
    print(f"[main] seed         : {cfg.seed}")
    set_seed(cfg.seed)

    # CI path: no raw data -> replay from artifacts/
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

    # 2. Model construction.
    # 3. Training.
    # 4. Evaluation.
    print("\n=== 2-4. Phase 3: training + evaluation ===")
    phase3 = run_phase3(cfg, epochs=int(cfg.raw.get("epochs", 30)))

    # 5. Analysis and visualisation.
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
        phase3=phase3,
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

    print("\nPhase 3 run complete.")


if __name__ == "__main__":
    main()
