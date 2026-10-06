"""Live progress readout for the v2 sweep.

The sweep's stdout is block-buffered when redirected to a file, so a
tailed log looks frozen even while the GPU is saturated. This reads
`artifacts_v2/phase3_summary.json` instead, which `run_phase3` rewrites
after every completed run, and is therefore an accurate live signal.

    python -m scripts.progress_v2              # one snapshot
    python -m scripts.progress_v2 --watch      # refresh every 30 s
    python -m scripts.progress_v2 --watch 10   # refresh every 10 s
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from src.utils.paths import Config, load_config, project_root

ARMS: Tuple[str, ...] = ("rgb_aug", "hed_only", "macenko_only", "full_stain_aware")


def _load_cfg() -> Config:
    return load_config(project_root() / "configs" / "rerun_v2.yaml")


def _completed(cfg: Config) -> Tuple[Set[str], float]:
    """Run tags with test metrics, plus the summary's mtime."""
    path = cfg.paths.artifacts / "phase3_summary.json"
    if not path.exists():
        return set(), 0.0
    try:
        with open(path, "r") as f:
            payload: Dict[str, Any] = json.load(f)
    except (OSError, json.JSONDecodeError):
        # mid-write; treat as no new information rather than crashing
        return set(), path.stat().st_mtime
    runs = payload.get("runs") or {}
    done = {tag for tag, run in runs.items() if run.get("test_per_tile_dice")}
    return done, path.stat().st_mtime


def _gpu_line() -> str:
    """One-line nvidia-smi summary, or a note if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,"
             "memory.total,temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=20,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "(nvidia-smi unavailable)"


def _fmt_eta(done_new: int, total_new: int, elapsed: float) -> str:
    if done_new <= 0:
        return "ETA           : (waiting for the first run to finish)"
    per_run = elapsed / done_new
    remaining = (total_new - done_new) * per_run
    end = time.strftime("%H:%M:%S", time.localtime(time.time() + remaining))
    return (
        f"per run       : {per_run:.0f} s\n"
        f"ETA           : {remaining / 3600.0:.2f} h remaining, "
        f"finishes ~{end}"
    )


def snapshot(cfg: Config, started: float) -> int:
    """Print one progress snapshot; returns the number of completed runs."""
    seeds: List[int] = [int(s) for s in cfg.raw.get("seeds", [])]
    done, mtime = _completed(cfg)
    total = len(seeds) * len(ARMS)

    ckpt_dir = cfg.paths.results / "checkpoints"
    n_ckpt = len(list(ckpt_dir.glob("*.pth"))) if ckpt_dir.exists() else 0

    bar_w = 40
    filled = int(bar_w * len(done) / total) if total else 0
    bar = "#" * filled + "-" * (bar_w - filled)

    print(f"runs complete : {len(done)} / {total}  [{bar}]  "
          f"{100.0 * len(done) / total:.0f}%" if total else "no seeds configured")
    print(f"checkpoints   : {n_ckpt}")
    if mtime:
        print(f"last update   : {time.strftime('%H:%M:%S', time.localtime(mtime))}"
              f"  ({time.time() - mtime:.0f} s ago)")
    print(f"gpu           : {_gpu_line()}")

    # Runs finished since this process started, for an honest rate estimate:
    # counting cached runs would make the ETA far too optimistic.
    done_new = len([t for t in done if t not in snapshot.baseline])
    print(_fmt_eta(done_new, total - len(snapshot.baseline), time.time() - started))

    print()
    print("seed | " + " ".join(f"{a[:4]:>4s}" for a in ARMS))
    print("-" * (7 + 5 * len(ARMS)))
    for seed in seeds:
        cells = []
        for arm in ARMS:
            tag = f"unet_resnet34_{arm}_seed{seed}"
            if tag in snapshot.baseline:
                cells.append("   ~")      # already done before this watch
            elif tag in done:
                cells.append("  OK")
            else:
                cells.append("   .")
        print(f"{seed:>4d} | " + " ".join(cells))
    print("\nlegend: OK done this session, ~ already cached, . pending")
    return len(done)


snapshot.baseline = set()  # type: ignore[attr-defined]


def main() -> None:
    """One snapshot, or a refreshing watch with `--watch [seconds]`."""
    argv = sys.argv[1:]
    watch = "--watch" in argv
    interval = 30.0
    if watch:
        idx = argv.index("--watch")
        if idx + 1 < len(argv):
            try:
                interval = float(argv[idx + 1])
            except ValueError:
                pass

    cfg = _load_cfg()
    snapshot.baseline, _ = _completed(cfg)  # type: ignore[attr-defined]
    started = time.time()
    total = len(cfg.raw.get("seeds", [])) * len(ARMS)

    while True:
        print("=" * 60)
        print(f"v2 sweep progress  {time.strftime('%H:%M:%S')}")
        print("=" * 60)
        n_done = snapshot(cfg, started)
        if not watch or n_done >= total:
            if watch:
                print("\nsweep complete.")
            return
        time.sleep(interval)


if __name__ == "__main__":
    main()
