"""Phase 3 runner: 4 aug arms x 3 seeds, paired stats, summary JSON.

Arms (control + 3 stain-aware variants):
    rgb_aug           control
    hed_only          ablation: HED jitter on train, no Macenko
    macenko_only      ablation: Macenko stain norm on train+eval, no HED
    full_stain_aware  HED jitter on train + Macenko on eval (combined)

Seeds: [42, 1, 2]. Outer loop is seeds, inner loop is arms.

Caching:
    For any (seed, arm) pair, the run is loaded from cached artifacts
    if `results/{run_tag}_per_tile.json` carries both Dice AND IoU
    arrays. If the per-tile JSON has Dice only (legacy seed-42), and a
    `results/checkpoints/{run_tag}.pth` exists, we recompute Dice +
    IoU via forward passes (no retraining).

The seed-42 entries in any pre-existing summary use the legacy
`unet_resnet34_<arm>` tag; on first call they are migrated to
`unet_resnet34_<arm>_seed<seed>` non-destructively, then never
overwritten by training.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import HuBMAPDataset
from src.data.transforms import (
    get_full_stain_aware_transforms,
    get_hed_only_transforms,
    get_macenko_only_transforms,
    get_rgb_aug_transforms,
)
from src.models import build_model
from src.training.evaluate import (
    BootstrapCI,
    bootstrap_dice,
    paired_wilcoxon_dice,
    per_tile_dice,
    per_tile_iou,
)
from src.training.losses import DiceBCELoss
from src.training.train import TrainResult, train_one_run, _resolve_device
from src.utils.paths import Config
from src.utils.seed import set_seed


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
_ARMS: Tuple[str, ...] = (
    "rgb_aug",
    "hed_only",
    "macenko_only",
    "full_stain_aware",
)
_DEFAULT_SEEDS: Tuple[int, ...] = (42, 1, 2)
_DEFAULT_TIME_BUDGET_S: float = 3.0 * 3600.0


_AUG_BUILDERS = {
    "rgb_aug": get_rgb_aug_transforms,
    "hed_only": get_hed_only_transforms,
    "macenko_only": get_macenko_only_transforms,
    "full_stain_aware": get_full_stain_aware_transforms,
}


def _run_tag(arm: str, seed: int) -> str:
    return f"unet_resnet34_{arm}_seed{seed}"


def _ci_to_dict(ci: BootstrapCI) -> Dict[str, float]:
    return {
        "mean": ci.mean,
        "lo": ci.lo,
        "hi": ci.hi,
        "n": ci.n,
        "n_boot": ci.n_boot,
    }


def _migrate_legacy_seed42_keys(payload: Dict[str, Any]) -> bool:
    """Rename `unet_resnet34_<arm>` -> `unet_resnet34_<arm>_seed42` in-place.

    Returns True if anything was rewritten.
    """
    runs = payload.get("runs") or {}
    if not runs:
        return False
    new_runs: Dict[str, Any] = {}
    changed = False
    for tag, run_data in runs.items():
        if "_seed" not in tag and tag.startswith("unet_resnet34_"):
            new_tag = f"{tag}_seed42"
            run_data = dict(run_data)
            run_data["run_tag"] = new_tag
            run_data["seed"] = 42
            wp = run_data.get("weights_path")
            if isinstance(wp, str):
                run_data["weights_path"] = wp.replace(
                    f"/{tag}.pth", f"/{new_tag}.pth"
                )
            new_runs[new_tag] = run_data
            changed = True
            print(f"[phase3] migrate run-tag: {tag} -> {new_tag}")
        else:
            new_runs[tag] = run_data
    if changed:
        payload["runs"] = new_runs
    return changed


def _load_existing_summary(cfg: Config) -> Dict[str, Any]:
    """Load and migrate any existing phase3 summary; prefer processed/."""
    candidates = [
        cfg.paths.processed / "phase3_summary.json",
        cfg.paths.artifacts / "phase3_summary.json",
    ]
    payload: Dict[str, Any] = {}
    src: Optional[Path] = None
    for p in candidates:
        if p.exists():
            try:
                with open(p, "r") as f:
                    payload = json.load(f)
                src = p
                break
            except (OSError, json.JSONDecodeError):
                continue
    if src is not None:
        print(f"[phase3] loaded existing summary from {src}")
        _migrate_legacy_seed42_keys(payload)
    return payload


def _per_tile_json_path(cfg: Config, run_tag: str) -> Path:
    return cfg.paths.results / f"{run_tag}_per_tile.json"


def _ckpt_path(cfg: Config, run_tag: str) -> Path:
    return cfg.paths.results / "checkpoints" / f"{run_tag}.pth"


def _load_cached_arrays(
    cfg: Config, run_tag: str
) -> Optional[Dict[str, Any]]:
    """Read per-tile JSON if it carries Dice (and ideally IoU)."""
    p = _per_tile_json_path(cfg, run_tag)
    if not p.exists():
        return None
    try:
        with open(p, "r") as f:
            pt = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not pt.get("val_per_tile_dice") and not pt.get("test_per_tile_dice"):
        return None
    return pt


def _build_eval_loaders(
    cfg: Config, aug_name: str
) -> Tuple[DataLoader, DataLoader]:
    """Val + test eval loaders for one arm (deterministic, no shuffle)."""
    image_size = int(cfg.image_size)
    batch_size = int(cfg.batch_size)
    img_dir = cfg.paths.raw / "train"
    mask_dir = cfg.paths.masks
    with open(cfg.paths.splits, "r") as f:
        splits = json.load(f)

    if aug_name not in _AUG_BUILDERS:
        raise ValueError(f"unknown aug '{aug_name}'")
    eval_tf = _AUG_BUILDERS[aug_name](image_size=image_size, train=False)

    val_ds = HuBMAPDataset(
        splits["val"], img_dir, mask_dir, transform=eval_tf, cache=True
    )
    test_ds = HuBMAPDataset(
        splits.get("test", []), img_dir, mask_dir, transform=eval_tf, cache=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return val_loader, test_loader


@torch.no_grad()
def _forward_metrics(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Per-tile (Dice, IoU) at thr=0.5 + ordered tile_ids over a loader."""
    model.eval()
    dice_chunks: List[np.ndarray] = []
    iou_chunks: List[np.ndarray] = []
    ids: List[str] = []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=False).float()
        mask = batch["mask"].to(device, non_blocking=False).long()
        logits = model(image)
        pred = (
            (torch.sigmoid(logits) > 0.5).squeeze(1).to("cpu").numpy().astype(np.uint8)
        )
        gt = mask.to("cpu").numpy().astype(np.uint8)
        dice_chunks.append(per_tile_dice(pred, gt))
        iou_chunks.append(per_tile_iou(pred, gt))
        bids = batch.get("tile_id", [])
        if isinstance(bids, (list, tuple)):
            ids.extend(list(bids))
        else:
            ids.extend([str(t) for t in bids])
    dice = np.concatenate(dice_chunks) if dice_chunks else np.array([], dtype=np.float64)
    iou = np.concatenate(iou_chunks) if iou_chunks else np.array([], dtype=np.float64)
    return dice, iou, ids


def _recompute_metrics_from_ckpt(
    cfg: Config, arm: str, seed: int, ckpt_path: Path
) -> Dict[str, Any]:
    """Load a checkpoint, run val+test loaders, return Dice+IoU arrays."""
    device = _resolve_device(cfg)
    print(
        f"[phase3] recompute Dice+IoU from checkpoint  arm={arm} seed={seed} "
        f"({ckpt_path.name})"
    )
    val_loader, test_loader = _build_eval_loaders(cfg, arm)
    model = build_model("unet_resnet34").to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    val_d, val_i, val_ids = _forward_metrics(model, val_loader, device)
    test_d, test_i, test_ids = _forward_metrics(model, test_loader, device)
    return {
        "val_per_tile_dice": val_d.tolist(),
        "val_per_tile_iou": val_i.tolist(),
        "val_tile_ids": val_ids,
        "test_per_tile_dice": test_d.tolist(),
        "test_per_tile_iou": test_i.tolist(),
        "test_tile_ids": test_ids,
        "best_val_dice": float(val_d.mean()) if val_d.size else float("nan"),
    }


def _result_summary(
    name: str,
    model_name: str,
    aug_name: str,
    seed: int,
    res: TrainResult,
) -> Dict[str, Any]:
    return {
        "run_tag": name,
        "model": model_name,
        "aug": aug_name,
        "seed": int(seed),
        "best_epoch": res.best_epoch,
        "best_val_dice": res.best_val_dice,
        "train_loss": res.train_loss,
        "val_loss": res.val_loss,
        "val_dice": res.val_dice,
        "val_per_tile_dice": np.asarray(res.per_tile_val_dice_at_best).tolist(),
        "test_per_tile_dice": np.asarray(res.per_tile_test_dice_at_best).tolist(),
        "val_per_tile_iou": np.asarray(res.per_tile_val_iou_at_best).tolist(),
        "test_per_tile_iou": np.asarray(res.per_tile_test_iou_at_best).tolist(),
        "val_tile_ids": list(res.val_tile_ids),
        "test_tile_ids": list(res.test_tile_ids),
        "weights_path": str(res.weights_path),
    }


def _ensure_per_tile_json(
    cfg: Config, run_tag: str, arm: str, seed: int, payload: Dict[str, Any]
) -> None:
    """Persist the per-tile JSON for a (run_tag) so cache works next time."""
    out = {
        "run_tag": run_tag,
        "model": "unet_resnet34",
        "aug": arm,
        "seed": int(seed),
        "best_epoch": int(payload.get("best_epoch", 0)),
        "best_val_dice": float(payload.get("best_val_dice", float("nan"))),
        "val_tile_ids": list(payload.get("val_tile_ids", [])),
        "val_per_tile_dice": list(payload.get("val_per_tile_dice", [])),
        "val_per_tile_iou": list(payload.get("val_per_tile_iou", [])),
        "test_tile_ids": list(payload.get("test_tile_ids", [])),
        "test_per_tile_dice": list(payload.get("test_per_tile_dice", [])),
        "test_per_tile_iou": list(payload.get("test_per_tile_iou", [])),
    }
    p = _per_tile_json_path(cfg, run_tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, indent=2)


def _save_comparison_plot(
    val_arrays: Dict[str, np.ndarray],
    test_arrays: Dict[str, np.ndarray],
    out_path,
) -> None:
    """Per-tile Dice histogram, four arms (uses seed-42 arrays)."""
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
    fig.suptitle("Phase 3: per-tile Dice histogram by augmentation arm (seed 42)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _wilcoxon_per_tile(
    arms: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """One-sided per-tile paired Wilcoxon: every treatment vs rgb_aug."""
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


def _wilcoxon_cross_seed(
    per_seed_means: Dict[str, Dict[int, float]], seeds: List[int]
) -> Dict[str, Dict[str, Any]]:
    """n=3 paired Wilcoxon on per-seed means, treatment vs rgb_aug.

    With n=3 the smallest two-sided p is 0.25 (so any p < 0.25 is
    not actually achievable); reported anyway for completeness.
    """
    from scipy.stats import wilcoxon

    base = "rgb_aug"
    out: Dict[str, Dict[str, Any]] = {}
    if base not in per_seed_means:
        return out
    base_vals = np.array(
        [per_seed_means[base][s] for s in seeds if s in per_seed_means[base]],
        dtype=np.float64,
    )
    for treat in ("hed_only", "macenko_only", "full_stain_aware"):
        if treat not in per_seed_means:
            continue
        treat_vals = np.array(
            [per_seed_means[treat][s] for s in seeds if s in per_seed_means[treat]],
            dtype=np.float64,
        )
        if treat_vals.shape != base_vals.shape or len(treat_vals) < 1:
            continue
        try:
            res = wilcoxon(treat_vals, base_vals, alternative="greater")
            stat = float(res.statistic)
            p = float(res.pvalue)
        except ValueError:
            stat, p = float("nan"), float("nan")
        out[f"{treat}_gt_{base}"] = {
            "stat": stat,
            "p": p,
            "n": int(len(treat_vals)),
            "treat_mean": float(treat_vals.mean()),
            "base_mean": float(base_vals.mean()),
        }
    return out


def _print_table(
    title: str,
    rows: List[str],
    cols: List[str],
    cell: Dict[Tuple[str, str], float],
) -> None:
    """Pretty-print a 2-D table with row x col labels."""
    print(f"\n{title}")
    header = "{:<18s}".format("arm \\ col") + "".join(f"{c:>14s}" for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = "{:<18s}".format(r)
        for c in cols:
            v = cell.get((r, c), float("nan"))
            line += f"{v:>14.4f}" if v == v else f"{'nan':>14s}"
        print(line)


def run_phase3(
    cfg: Config,
    epochs: int = 30,
    seeds: Tuple[int, ...] = _DEFAULT_SEEDS,
    max_wall_seconds: float = _DEFAULT_TIME_BUDGET_S,
) -> Dict[str, Any]:
    """Train all (seed, arm) pairs, with caching, IoU, cross-seed stats."""
    epochs = int(epochs)
    seeds = tuple(int(s) for s in seeds)
    model_name = "unet_resnet34"
    start = time.time()

    existing = _load_existing_summary(cfg)
    runs: Dict[str, Any] = dict(existing.get("runs", {}) or {})

    # Save migrated keys back so the on-disk summary is consistent
    # before any training starts (protects against a crash mid-run).
    if existing:
        existing["runs"] = runs
        _save_summary(cfg, existing)

    pending: List[Tuple[int, str]] = []

    for seed in seeds:
        for arm in _ARMS:
            tag = _run_tag(arm, seed)
            cached = _load_cached_arrays(cfg, tag)
            has_iou = bool(
                cached
                and cached.get("val_per_tile_iou")
                and cached.get("test_per_tile_iou")
            )

            # Already complete in summary AND on-disk JSON has IoU
            if tag in runs and has_iou:
                run = runs[tag]
                if run.get("val_per_tile_iou") and run.get("test_per_tile_iou"):
                    print(f"[phase3] cache complete  {tag}  (skip)")
                    continue
                # backfill summary entry from on-disk per-tile JSON
                run.update(
                    {
                        "val_per_tile_dice": cached["val_per_tile_dice"],
                        "test_per_tile_dice": cached["test_per_tile_dice"],
                        "val_per_tile_iou": cached["val_per_tile_iou"],
                        "test_per_tile_iou": cached["test_per_tile_iou"],
                        "val_tile_ids": cached.get("val_tile_ids", run.get("val_tile_ids", [])),
                        "test_tile_ids": cached.get("test_tile_ids", run.get("test_tile_ids", [])),
                    }
                )
                runs[tag] = run
                continue

            elapsed = time.time() - start
            if elapsed > max_wall_seconds:
                print(
                    f"[phase3] WALL-TIME BUDGET HIT ({elapsed:.0f}s > {max_wall_seconds:.0f}s); "
                    f"marking ({seed}, {arm}) as pending."
                )
                pending.append((seed, arm))
                continue

            ckpt = _ckpt_path(cfg, tag)
            if seed == 42 and ckpt.exists() and (cached or True):
                # Recompute Dice + IoU from existing checkpoint (no retrain).
                metrics = _recompute_metrics_from_ckpt(cfg, arm, seed, ckpt)
                run_entry = dict(runs.get(tag, {}))
                run_entry.update(
                    {
                        "run_tag": tag,
                        "model": model_name,
                        "aug": arm,
                        "seed": int(seed),
                        "weights_path": str(ckpt),
                        "val_per_tile_dice": metrics["val_per_tile_dice"],
                        "test_per_tile_dice": metrics["test_per_tile_dice"],
                        "val_per_tile_iou": metrics["val_per_tile_iou"],
                        "test_per_tile_iou": metrics["test_per_tile_iou"],
                        "val_tile_ids": metrics["val_tile_ids"],
                        "test_tile_ids": metrics["test_tile_ids"],
                    }
                )
                # preserve any pre-existing best_epoch / best_val_dice / curves
                run_entry.setdefault(
                    "best_val_dice",
                    float(np.asarray(metrics["val_per_tile_dice"]).mean())
                    if metrics["val_per_tile_dice"]
                    else float("nan"),
                )
                run_entry.setdefault("best_epoch", 0)
                runs[tag] = run_entry
                _ensure_per_tile_json(cfg, tag, arm, seed, run_entry)
                _save_summary(cfg, {**existing, "runs": runs})
                continue

            # Fresh training run.
            print(
                f"\n[phase3] === train run_tag={tag} (model={model_name} "
                f"aug={arm} seed={seed} epochs={epochs}) ==="
            )
            set_seed(int(seed))
            res = train_one_run(
                cfg,
                model_name=model_name,
                aug_name=arm,
                epochs=epochs,
                run_tag=tag,
            )
            runs[tag] = _result_summary(tag, model_name, arm, seed, res)
            _save_summary(cfg, {**existing, "runs": runs})

    # ---- Stats ----
    seeds_list = list(seeds)

    # seed-42 per-tile arrays (kept for back-compat tables in the report)
    val42: Dict[str, np.ndarray] = {}
    test42: Dict[str, np.ndarray] = {}
    val42_iou: Dict[str, np.ndarray] = {}
    test42_iou: Dict[str, np.ndarray] = {}
    for arm in _ARMS:
        tag = _run_tag(arm, 42)
        run = runs.get(tag, {})
        if run.get("val_per_tile_dice"):
            val42[arm] = np.asarray(run["val_per_tile_dice"], dtype=np.float64)
        if run.get("test_per_tile_dice"):
            test42[arm] = np.asarray(run["test_per_tile_dice"], dtype=np.float64)
        if run.get("val_per_tile_iou"):
            val42_iou[arm] = np.asarray(run["val_per_tile_iou"], dtype=np.float64)
        if run.get("test_per_tile_iou"):
            test42_iou[arm] = np.asarray(run["test_per_tile_iou"], dtype=np.float64)

    val_cis = {
        arm: _ci_to_dict(bootstrap_dice(arr, n_boot=2000, ci=0.95, seed=42))
        for arm, arr in val42.items()
    }
    test_cis = {
        arm: _ci_to_dict(bootstrap_dice(arr, n_boot=2000, ci=0.95, seed=42))
        for arm, arr in test42.items()
    }
    val_iou_cis = {
        arm: _ci_to_dict(bootstrap_dice(arr, n_boot=2000, ci=0.95, seed=42))
        for arm, arr in val42_iou.items()
    }
    test_iou_cis = {
        arm: _ci_to_dict(bootstrap_dice(arr, n_boot=2000, ci=0.95, seed=42))
        for arm, arr in test42_iou.items()
    }
    val_wilcoxon = _wilcoxon_per_tile(val42)
    test_wilcoxon = _wilcoxon_per_tile(test42)

    # Cross-seed per-arm metrics
    per_seed_val_dice: Dict[str, Dict[int, float]] = {a: {} for a in _ARMS}
    per_seed_test_dice: Dict[str, Dict[int, float]] = {a: {} for a in _ARMS}
    per_seed_val_iou: Dict[str, Dict[int, float]] = {a: {} for a in _ARMS}
    per_seed_test_iou: Dict[str, Dict[int, float]] = {a: {} for a in _ARMS}
    per_seed_best_val: Dict[str, Dict[int, float]] = {a: {} for a in _ARMS}

    for arm in _ARMS:
        for s in seeds_list:
            tag = _run_tag(arm, s)
            run = runs.get(tag)
            if not run:
                continue
            v = run.get("val_per_tile_dice") or []
            t = run.get("test_per_tile_dice") or []
            vi = run.get("val_per_tile_iou") or []
            ti = run.get("test_per_tile_iou") or []
            if v:
                per_seed_val_dice[arm][s] = float(np.mean(v))
            if t:
                per_seed_test_dice[arm][s] = float(np.mean(t))
            if vi:
                per_seed_val_iou[arm][s] = float(np.mean(vi))
            if ti:
                per_seed_test_iou[arm][s] = float(np.mean(ti))
            if "best_val_dice" in run:
                per_seed_best_val[arm][s] = float(run["best_val_dice"])

    def _agg(per_seed: Dict[str, Dict[int, float]]) -> Dict[str, Dict[str, Any]]:
        agg: Dict[str, Dict[str, Any]] = {}
        for arm in _ARMS:
            vals = [per_seed[arm][s] for s in seeds_list if s in per_seed[arm]]
            arr = np.asarray(vals, dtype=np.float64)
            agg[arm] = {
                "mean": float(arr.mean()) if arr.size else float("nan"),
                "std": float(arr.std(ddof=0)) if arr.size else float("nan"),
                "n": int(arr.size),
                "values_per_seed": {
                    str(s): per_seed[arm][s]
                    for s in seeds_list
                    if s in per_seed[arm]
                },
            }
        return agg

    val_cross_seed = {
        "best_val_dice": _agg(per_seed_best_val),
        "dice": _agg(per_seed_val_dice),
        "iou": _agg(per_seed_val_iou),
    }
    test_cross_seed = {
        "dice": _agg(per_seed_test_dice),
        "iou": _agg(per_seed_test_iou),
    }
    val_wc_cs = {
        "dice": _wilcoxon_cross_seed(per_seed_val_dice, seeds_list),
        "iou": _wilcoxon_cross_seed(per_seed_val_iou, seeds_list),
    }
    test_wc_cs = {
        "dice": _wilcoxon_cross_seed(per_seed_test_dice, seeds_list),
        "iou": _wilcoxon_cross_seed(per_seed_test_iou, seeds_list),
    }

    if val42 and test42:
        comparison_png = cfg.paths.results / "phase3_comparison.png"
        _save_comparison_plot(val42, test42, comparison_png)
        print(f"[phase3] comparison figure : {comparison_png}")

    summary: Dict[str, Any] = {
        "epochs": epochs,
        "model": model_name,
        "seeds": list(seeds),
        "seed": 42,
        "image_size": cfg.image_size,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "lr": float(cfg.raw.get("lr", 1e-4)),
        "weight_decay": float(cfg.raw.get("weight_decay", 1e-4)),
        "augs": list(_ARMS),
        "config_snapshot": copy.deepcopy(cfg.raw),
        "runs": runs,
        "stats": {
            "val": {
                "cis": val_cis,
                "wilcoxon": val_wilcoxon,
                "iou": val_iou_cis,
                "cross_seed": val_cross_seed,
                "wilcoxon_cross_seed": val_wc_cs,
            },
            "test": {
                "cis": test_cis,
                "wilcoxon": test_wilcoxon,
                "iou": test_iou_cis,
                "cross_seed": test_cross_seed,
                "wilcoxon_cross_seed": test_wc_cs,
                "labels_note": "dataset 2 noisy auto-labels; relative comparison only.",
            },
        },
        "pending": [
            {"seed": s, "arm": a} for s, a in pending
        ],
        "wall_time_seconds": float(time.time() - start),
    }

    _save_summary(cfg, summary)

    _print_phase3_outputs(summary, seeds_list, pending)

    return summary


def _save_summary(cfg: Config, summary: Dict[str, Any]) -> None:
    """Write summary to data/processed (and copy to artifacts/)."""
    cfg.paths.processed.mkdir(parents=True, exist_ok=True)
    p = cfg.paths.processed / "phase3_summary.json"
    with open(p, "w") as f:
        json.dump(summary, f, indent=2)
    cfg.paths.artifacts.mkdir(parents=True, exist_ok=True)
    p2 = cfg.paths.artifacts / "phase3_summary.json"
    with open(p2, "w") as f:
        json.dump(summary, f, indent=2)


def _print_phase3_outputs(
    summary: Dict[str, Any],
    seeds_list: List[int],
    pending: List[Tuple[int, str]],
) -> None:
    """Print the rubric-required tables to stdout."""
    runs = summary.get("runs", {})
    stats = summary.get("stats", {})

    # 1) per-arm per-seed best validation Dice
    cell_v: Dict[Tuple[str, str], float] = {}
    cell_t: Dict[Tuple[str, str], float] = {}
    for arm in _ARMS:
        for s in seeds_list:
            tag = _run_tag(arm, s)
            run = runs.get(tag, {})
            if "best_val_dice" in run:
                cell_v[(arm, f"seed{s}")] = float(run["best_val_dice"])
            t = run.get("test_per_tile_dice") or []
            if t:
                cell_t[(arm, f"seed{s}")] = float(np.mean(t))
    cols = [f"seed{s}" for s in seeds_list]
    _print_table(
        "[phase3] best validation Dice (rows=arms, cols=seeds)",
        list(_ARMS), cols, cell_v,
    )
    _print_table(
        "[phase3] test mean per-tile Dice (rows=arms, cols=seeds)",
        list(_ARMS), cols, cell_t,
    )

    # 3) cross-seed mean +/- std
    print("\n[phase3] cross-seed mean +/- std (n={}):".format(len(seeds_list)))
    val_cs = stats.get("val", {}).get("cross_seed", {})
    test_cs = stats.get("test", {}).get("cross_seed", {})
    print(
        f"  {'arm':<18s} {'val_dice':>20s} {'test_dice':>20s} "
        f"{'val_iou':>20s} {'test_iou':>20s}"
    )
    for arm in _ARMS:
        vd = val_cs.get("dice", {}).get(arm, {})
        td = test_cs.get("dice", {}).get(arm, {})
        vi = val_cs.get("iou", {}).get(arm, {})
        ti = test_cs.get("iou", {}).get(arm, {})

        def _fmt(b: Dict[str, Any]) -> str:
            m = b.get("mean", float("nan"))
            sd = b.get("std", float("nan"))
            if m != m:
                return f"{'nan':>20s}"
            return f"{m:>10.4f} +/- {sd:6.4f}"

        print(f"  {arm:<18s} {_fmt(vd)} {_fmt(td)} {_fmt(vi)} {_fmt(ti)}")

    # 4) cross-seed Wilcoxon
    print("\n[phase3] cross-seed Wilcoxon (n=3 paired) treatment > rgb_aug:")
    for split_name in ("val", "test"):
        block = stats.get(split_name, {}).get("wilcoxon_cross_seed", {})
        for metric in ("dice", "iou"):
            for treat_key, v in block.get(metric, {}).items():
                print(
                    f"  {split_name:<4s}  {metric:<4s}  {treat_key:<32s} "
                    f"p = {v.get('p', float('nan')):.4g}  "
                    f"(stat={v.get('stat', float('nan')):.2f}, "
                    f"n={v.get('n', 0)})"
                )

    # also keep printing the per-tile (seed-42) Wilcoxon block for the existing report
    for split_name, key in [("val", "wilcoxon"), ("test", "wilcoxon")]:
        block = stats.get(split_name, {}).get(key, {})
        for treat_key, v in block.items():
            print(
                f"[phase3] {split_name:<4s}  per-tile Wilcoxon  {treat_key:<32s} "
                f"p = {v.get('p', float('nan')):.4g}"
            )

    if pending:
        print("\n[phase3] PENDING (time-budget cut):")
        for s, a in pending:
            print(f"  seed={s} arm={a}")
