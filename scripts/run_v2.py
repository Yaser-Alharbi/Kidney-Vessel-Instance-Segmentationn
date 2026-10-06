"""Rerun v2 entrypoint: CUDA, N seeds from config, multi-WSI test set.

Mirrors the full-run branch of `main()` but against `configs/rerun_v2.yaml`,
writing to `results_v2/` and `artifacts_v2/`. `main.py`, `configs/default.yaml`
and the committed `artifacts/` replay path are untouched.

    python -m scripts.run_v2              # full sweep (20 seeds, 80 runs)
    python -m scripts.run_v2 --pilot      # seed 42 only (4 runs)

CUBLAS_WORKSPACE_CONFIG must be set before the CUDA context is created,
so it is set here ahead of every torch-touching import.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# pylint: disable=wrong-import-position
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import torch

from src.data.build_masks import build_masks
from src.data.dataset import HuBMAPDataset
from src.data.inspect_data import inspect_data
from src.data.splits import build_splits
from src.training.run_experiments import run_phase3
from src.training.train import _resolve_device
from src.utils.paths import Config, ensure_dirs, load_config, project_root
from src.utils.seed import set_seed


def _config_path(pilot: bool) -> Path:
    name = "rerun_v2_pilot.yaml" if pilot else "rerun_v2.yaml"
    return project_root() / "configs" / name


def _report_device(cfg: Config) -> torch.device:
    """Resolve and print the device; fail loudly if CUDA was asked for but absent."""
    device = _resolve_device(cfg)
    print(f"[run_v2] configured device : {cfg.device}")
    print(f"[run_v2] resolved device   : {device}")
    print(f"[run_v2] CUBLAS_WORKSPACE_CONFIG : "
          f"{os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")
    if cfg.device.lower() == "cuda" and device.type != "cuda":
        raise RuntimeError(
            "config asks for cuda but torch.cuda.is_available() is False. "
            "Refusing to fall back to CPU: an 80-run CPU sweep would take days."
        )
    if device.type == "cuda":
        print(f"[run_v2] gpu               : {torch.cuda.get_device_name(0)}")
        print(f"[run_v2] capability        : {torch.cuda.get_device_capability(0)}")
        torch.cuda.reset_peak_memory_stats()
    return device


def run(pilot: bool = False) -> Dict[str, Any]:
    """Run the v2 pipeline end to end and return the phase 3 summary."""
    cfg_path = _config_path(pilot)
    cfg = load_config(cfg_path)
    ensure_dirs(cfg)

    seeds = tuple(int(s) for s in cfg.raw.get("seeds", (42, 1, 2)))
    epochs = int(cfg.raw.get("epochs", 30))
    budget_s = float(cfg.raw.get("max_wall_hours", 24)) * 3600.0
    arms = 4

    print(f"[run_v2] config            : {cfg_path}")
    print(f"[run_v2] results           : {cfg.paths.results}")
    print(f"[run_v2] artifacts         : {cfg.paths.artifacts}")
    print(f"[run_v2] seeds ({len(seeds)})        : {list(seeds)}")
    print(f"[run_v2] planned runs      : {len(seeds) * arms}")
    print(f"[run_v2] wall budget       : {budget_s / 3600.0:.1f} h")
    device = _report_device(cfg)

    if not (cfg.paths.raw / "polygons.jsonl").exists():
        raise FileNotFoundError(
            f"No raw data at {cfg.paths.raw}. Run `python -m scripts.fetch_data` "
            "first -- there is nothing to replay from for v2."
        )

    set_seed(cfg.seed)

    print("\n=== 1. Dataset preparation ===")
    inspection = inspect_data(cfg)
    masks = build_masks(cfg)
    splits = build_splits(cfg)

    print(f"\n[run_v2] test WSIs         : {splits.test_wsis}")
    print(f"[run_v2] test tiles        : {splits.n_test_tiles}")
    print(f"[run_v2] split overlap     : {splits.overlap_count}")

    with open(cfg.paths.splits, "r") as f:
        split_payload = json.load(f)
    img_dir = cfg.paths.raw / "train"
    train_ds = HuBMAPDataset(split_payload["train"], img_dir, cfg.paths.masks)
    val_ds = HuBMAPDataset(split_payload["val"], img_dir, cfg.paths.masks)
    test_ds = HuBMAPDataset(split_payload.get("test", []), img_dir, cfg.paths.masks)
    print(f"[run_v2] train/val/test    : "
          f"{len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    print("\n=== 2-4. Phase 3 sweep ===")
    start = time.time()
    phase3 = run_phase3(
        cfg, epochs=epochs, seeds=seeds, max_wall_seconds=budget_s
    )
    elapsed = time.time() - start

    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"\n[run_v2] peak GPU memory   : {peak_gb:.2f} GB")

    n_runs = len(phase3.get("runs", {}))
    pend = phase3.get("pending", [])
    print(f"[run_v2] runs in summary   : {n_runs}")
    print(f"[run_v2] pending           : {len(pend)}")
    print(f"[run_v2] wall clock        : {elapsed:.0f} s "
          f"({elapsed / 3600.0:.2f} h)")
    if n_runs:
        print(f"[run_v2] mean per run      : {elapsed / n_runs:.0f} s")
    if pend:
        print("[run_v2] WARNING: wall budget cut the sweep; rerun to resume.")

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
    print(f"[run_v2] phase2 summary    : {summary_path}")

    print("\n=== 5. Report ===")
    from scripts.report_v2 import report

    report(cfg)
    return phase3


def main() -> None:
    """CLI shim: `--pilot` selects the single-seed config."""
    run(pilot="--pilot" in sys.argv[1:])


if __name__ == "__main__":
    main()
