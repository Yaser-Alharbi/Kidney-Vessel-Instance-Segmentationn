"""Phase 3 runner: 3 aug arms x 1 model, paired stats, summary JSON.

Arms: rgb_aug (control), stain_aware_hed_only (ablation), stain_aware.
Outputs land in `results/` plus `data/processed/phase3_summary.json`.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.training.evaluate import (
    BootstrapCI,
    bootstrap_dice,
    paired_wilcoxon_dice,
)
from src.training.train import TrainResult, train_one_run
from src.utils.paths import Config


_ARM_COLORS = {
    "rgb_aug": "#1f77b4",
    "stain_aware_hed_only": "#2ca02c",
    "stain_aware": "#ff7f0e",
}
_ARM_LABELS = {
    "rgb_aug": "rgb_aug (control)",
    "stain_aware_hed_only": "stain_aware_hed_only (HED only)",
    "stain_aware": "stain_aware (HED + Macenko)",
}
_DEFAULT_ARMS: List[Tuple[str, str]] = [
    ("rgb_aug", "unet_resnet34_rgb_aug"),
    ("stain_aware_hed_only", "unet_resnet34_stain_aware_hed_only"),
    ("stain_aware", "unet_resnet34_stain_aware"),
]


def _ci_to_dict(ci: BootstrapCI) -> Dict[str, float]:
    return {
        "mean": ci.mean,
        "lo": ci.lo,
        "hi": ci.hi,
        "n": ci.n,
        "n_boot": ci.n_boot,
    }


def _result_summary(name: str, model_name: str, aug_name: str, res: TrainResult) -> Dict[str, Any]:
    return {
        "run_tag": name,
        "model": model_name,
        "aug": aug_name,
        "best_epoch": res.best_epoch,
        "best_val_dice": res.best_val_dice,
        "train_loss": res.train_loss,
        "val_loss": res.val_loss,
        "val_dice": res.val_dice,
        "val_per_tile_dice": res.per_tile_val_dice_at_best.tolist(),
        "test_per_tile_dice": res.per_tile_test_dice_at_best.tolist(),
        "val_tile_ids": list(res.val_tile_ids),
        "test_tile_ids": list(res.test_tile_ids),
        "weights_path": str(res.weights_path),
    }


def _save_comparison_plot(
    val_arrays: Dict[str, np.ndarray],
    test_arrays: Dict[str, np.ndarray],
    out_path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    bins = np.linspace(0.0, 1.0, 21)

    for ax, (arrs, title) in zip(
        axes,
        [
            (val_arrays, "val (dataset 1, clean)"),
            (test_arrays, "test (dataset 2, NOISY)"),
        ],
    ):
        for aug_name, arr in arrs.items():
            color = _ARM_COLORS.get(aug_name, "#888888")
            label = _ARM_LABELS.get(aug_name, aug_name)
            ax.hist(arr, bins=bins, alpha=0.5, label=label, color=color, edgecolor="white")
            ax.axvline(float(np.mean(arr)), color=color, linestyle="--", linewidth=1.2)
        ax.set_xlabel("per-tile Dice")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    axes[0].set_ylabel("# tiles")
    fig.suptitle("Phase 3: per-tile Dice histogram by augmentation arm")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _wilcoxon_block(
    arms: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """Three one-sided paired Wilcoxon tests on per-tile Dice."""
    pairs = [
        ("stain_aware", "rgb_aug"),
        ("stain_aware_hed_only", "rgb_aug"),
        ("stain_aware", "stain_aware_hed_only"),
    ]
    out: Dict[str, Dict[str, float]] = {}
    for treat, base in pairs:
        if treat not in arms or base not in arms:
            continue
        if arms[treat].shape != arms[base].shape:
            raise RuntimeError(
                f"shape mismatch in pairwise test: {treat}={arms[treat].shape} "
                f"{base}={arms[base].shape}"
            )
        stat, p = paired_wilcoxon_dice(arms[treat], arms[base], alternative="greater")
        out[f"{treat}_gt_{base}"] = {"stat": float(stat), "p": float(p)}
    return out


def run_phase3(cfg: Config, epochs: int = 30) -> Dict[str, Any]:
    """Train all 3 arms, compute CIs + pairwise Wilcoxon, write summary."""
    epochs = int(epochs)
    model_name = "unet_resnet34"

    runs: Dict[str, Any] = {}
    train_results: Dict[str, TrainResult] = {}

    for aug_name, run_tag in _DEFAULT_ARMS:
        print(f"\n[phase3] === run {run_tag} (model={model_name} aug={aug_name}) ===")
        res = train_one_run(
            cfg,
            model_name=model_name,
            aug_name=aug_name,
            epochs=epochs,
            run_tag=run_tag,
        )
        train_results[run_tag] = res
        runs[run_tag] = _result_summary(run_tag, model_name, aug_name, res)

    val_arrays: Dict[str, np.ndarray] = {
        aug: train_results[run_tag].per_tile_val_dice_at_best
        for aug, run_tag in _DEFAULT_ARMS
    }
    test_arrays: Dict[str, np.ndarray] = {
        aug: train_results[run_tag].per_tile_test_dice_at_best
        for aug, run_tag in _DEFAULT_ARMS
    }

    val_cis = {
        aug: _ci_to_dict(bootstrap_dice(arr, n_boot=2000, ci=0.95, seed=cfg.seed))
        for aug, arr in val_arrays.items()
    }
    test_cis = {
        aug: _ci_to_dict(bootstrap_dice(arr, n_boot=2000, ci=0.95, seed=cfg.seed))
        for aug, arr in test_arrays.items()
    }
    val_wilcoxon = _wilcoxon_block(val_arrays)
    test_wilcoxon = _wilcoxon_block(test_arrays)

    comparison_png = cfg.paths.results / "phase3_comparison.png"
    _save_comparison_plot(val_arrays, test_arrays, comparison_png)
    print(f"[phase3] comparison figure : {comparison_png}")

    summary: Dict[str, Any] = {
        "epochs": epochs,
        "model": model_name,
        "seed": cfg.seed,
        "image_size": cfg.image_size,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "lr": float(cfg.raw.get("lr", 1e-4)),
        "weight_decay": float(cfg.raw.get("weight_decay", 1e-4)),
        "augs": [aug for aug, _ in _DEFAULT_ARMS],
        "config_snapshot": copy.deepcopy(cfg.raw),
        "runs": runs,
        "stats": {
            "val": {
                "cis": val_cis,
                "wilcoxon": val_wilcoxon,
            },
            "test": {
                "cis": test_cis,
                "wilcoxon": test_wilcoxon,
                "labels_note": "dataset 2 noisy auto-labels; relative comparison only.",
            },
        },
    }

    summary_path = cfg.paths.processed / "phase3_summary.json"
    cfg.paths.processed.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[phase3] summary json      : {summary_path}")

    for split_name, cis, wc in [
        ("val", val_cis, val_wilcoxon),
        ("test", test_cis, test_wilcoxon),
    ]:
        for aug, ci in cis.items():
            tag = "(noisy)" if split_name == "test" else ""
            print(
                f"[phase3] {split_name:4s}  {aug:22s} "
                f"{ci['mean']:.4f} [{ci['lo']:.4f}, {ci['hi']:.4f}] {tag}".rstrip()
            )
        for key, v in wc.items():
            print(f"[phase3] {split_name:4s}  Wilcoxon {key:38s} p = {v['p']:.4g}")

    return summary
