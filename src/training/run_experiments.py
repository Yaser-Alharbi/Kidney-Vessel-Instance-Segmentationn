"""Phase 3 runner: 4 aug arms x 1 model, paired stats, summary JSON.

Arms:
    rgb_aug           control
    hed_only          ablation: HED jitter on train, no Macenko
    macenko_only      ablation: Macenko stain norm on train+eval, no HED
    full_stain_aware  HED jitter on train + Macenko on eval (combined)

Outputs land in `results/` plus `data/processed/phase3_summary.json`.

If a per-tile JSON already exists at `results/{run_tag}_per_tile.json`
the run is loaded from cache instead of re-trained, so we can add the
new `macenko_only` arm without re-running the previous three.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    "hed_only": "#2ca02c",
    "macenko_only": "#d62728",
    "full_stain_aware": "#ff7f0e",
}
_ARM_LABELS = {
    "rgb_aug": "rgb_aug (control)",
    "hed_only": "hed_only (HED jitter)",
    "macenko_only": "macenko_only (Macenko)",
    "full_stain_aware": "full_stain_aware (HED + Macenko)",
}
_DEFAULT_ARMS: List[Tuple[str, str]] = [
    ("rgb_aug", "unet_resnet34_rgb_aug"),
    ("hed_only", "unet_resnet34_hed_only"),
    ("macenko_only", "unet_resnet34_macenko_only"),
    ("full_stain_aware", "unet_resnet34_full_stain_aware"),
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


def _load_cached_train_result(
    cfg: Config,
    run_tag: str,
    model_name: str,
    aug_name: str,
) -> Optional[TrainResult]:
    """Reconstruct a TrainResult from cached artifacts if both exist.

    Reads per-tile dice from `results/{run_tag}_per_tile.json` and the
    loss/Dice curves (if present) from a previous `phase3_summary.json`.
    """
    per_tile_path = cfg.paths.results / f"{run_tag}_per_tile.json"
    if not per_tile_path.exists():
        return None

    with open(per_tile_path, "r") as f:
        pt = json.load(f)

    val_per_tile = np.asarray(pt.get("val_per_tile_dice", []), dtype=np.float64)
    test_per_tile = np.asarray(pt.get("test_per_tile_dice", []), dtype=np.float64)
    if val_per_tile.size == 0 and test_per_tile.size == 0:
        return None

    train_loss: List[float] = []
    val_loss: List[float] = []
    val_dice: List[float] = []
    # legacy run-tag aliases so older summary JSONs still backfill the curves
    legacy_aliases = {
        "unet_resnet34_hed_only": "unet_resnet34_stain_aware_hed_only",
        "unet_resnet34_full_stain_aware": "unet_resnet34_stain_aware",
    }
    summary_path = cfg.paths.processed / "phase3_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r") as f:
                prev = json.load(f)
            runs_block = prev.get("runs") or {}
            run = runs_block.get(run_tag) or runs_block.get(legacy_aliases.get(run_tag, ""), {})
            train_loss = list(run.get("train_loss") or [])
            val_loss = list(run.get("val_loss") or [])
            val_dice = list(run.get("val_dice") or [])
        except (OSError, json.JSONDecodeError):
            pass

    weights_path = cfg.paths.results / "checkpoints" / f"{run_tag}.pth"
    print(
        f"[phase3] cache hit  {run_tag}: val_n={val_per_tile.size} "
        f"test_n={test_per_tile.size} (skip training)"
    )

    return TrainResult(
        train_loss=train_loss,
        val_loss=val_loss,
        val_dice=val_dice,
        best_epoch=int(pt.get("best_epoch", 0)),
        best_val_dice=float(pt.get("best_val_dice", float("nan"))),
        per_tile_val_dice_at_best=val_per_tile,
        per_tile_test_dice_at_best=test_per_tile,
        weights_path=Path(weights_path),
        val_tile_ids=list(pt.get("val_tile_ids", [])),
        test_tile_ids=list(pt.get("test_tile_ids", [])),
    )


def _save_comparison_plot(
    val_arrays: Dict[str, np.ndarray],
    test_arrays: Dict[str, np.ndarray],
    out_path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
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
        ax.legend(loc="upper right", fontsize=8)

    axes[0].set_ylabel("# tiles")
    fig.suptitle("Phase 3: per-tile Dice histogram by augmentation arm")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _wilcoxon_block(
    arms: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """One-sided paired Wilcoxon: every treatment arm vs rgb_aug control."""
    base = "rgb_aug"
    out: Dict[str, Dict[str, float]] = {}
    if base not in arms:
        return out
    for treat in ("hed_only", "macenko_only", "full_stain_aware"):
        if treat not in arms:
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
    """Train all 4 arms (cached arms reuse per_tile JSON), CIs + Wilcoxon."""
    epochs = int(epochs)
    model_name = "unet_resnet34"

    runs: Dict[str, Any] = {}
    train_results: Dict[str, TrainResult] = {}

    for aug_name, run_tag in _DEFAULT_ARMS:
        cached = _load_cached_train_result(cfg, run_tag, model_name, aug_name)
        if cached is not None:
            train_results[run_tag] = cached
            runs[run_tag] = _result_summary(run_tag, model_name, aug_name, cached)
            continue

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
