"""Cross-seed TEST-set report for the v2 rerun.

Reads `artifacts_v2/phase3_summary.json` and writes a markdown report
plus a per-seed CSV. Everything here is on the test set and aggregated
over seeds, which is what the extra seeds were run for; the seed-42
per-tile tables stay in `main.py`'s report.

    python -m scripts.report_v2

Test labels are dataset-2 auto-generated (noisy), so these Dice values
support relative comparison between arms, never absolute accuracy.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.training.evaluate import bootstrap_dice
from src.utils.paths import Config, load_config, project_root

ARMS: Tuple[str, ...] = ("rgb_aug", "hed_only", "macenko_only", "full_stain_aware")
BASE = "rgb_aug"


def _run_tag(arm: str, seed: int) -> str:
    return f"unet_resnet34_{arm}_seed{seed}"


def _load_summary(cfg: Config) -> Dict[str, Any]:
    path = cfg.paths.artifacts / "phase3_summary.json"
    if not path.exists():
        path = cfg.paths.processed / "phase3_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No phase3_summary.json in {cfg.paths.artifacts} or {cfg.paths.processed}"
        )
    with open(path, "r") as f:
        return json.load(f)


def _collect(summary: Dict[str, Any]) -> Tuple[List[int], Dict[str, Dict[int, float]], int]:
    """Per-seed mean test Dice per arm, plus the shared test-tile count."""
    seeds = [int(s) for s in summary.get("seeds", [])]
    runs = summary.get("runs", {})
    per_seed: Dict[str, Dict[int, float]] = {a: {} for a in ARMS}
    tile_counts: set = set()

    for arm in ARMS:
        for seed in seeds:
            run = runs.get(_run_tag(arm, seed))
            if not run:
                continue
            vals = run.get("test_per_tile_dice") or []
            if not vals:
                continue
            per_seed[arm][seed] = float(np.mean(vals))
            tile_counts.add(len(vals))

    if not tile_counts:
        raise RuntimeError("no test per-tile Dice arrays found in the summary")
    if len(tile_counts) > 1:
        raise RuntimeError(
            f"runs disagree on test-set size: {sorted(tile_counts)}. "
            "Arms are not comparable; the summary mixes different test sets."
        )
    return seeds, per_seed, tile_counts.pop()


def _stats(per_seed: Dict[str, Dict[int, float]], seeds: List[int]) -> Dict[str, Any]:
    """Mean, CI over seeds, % change and Wilcoxon vs the control arm."""
    from scipy.stats import wilcoxon

    base_seeds = [s for s in seeds if s in per_seed[BASE]]
    base_vals = np.array([per_seed[BASE][s] for s in base_seeds], dtype=np.float64)
    base_mean = float(base_vals.mean()) if base_vals.size else float("nan")

    out: Dict[str, Any] = {}
    for arm in ARMS:
        shared = [s for s in seeds if s in per_seed[arm] and s in per_seed[BASE]]
        vals = np.array([per_seed[arm][s] for s in shared], dtype=np.float64)
        if not vals.size:
            continue

        # CI over per-seed means, not over tiles: the claim is about
        # seed-to-seed reproducibility, so the seed is the sample unit.
        ci = bootstrap_dice(vals, n_boot=10000, ci=0.95, seed=42)

        entry: Dict[str, Any] = {
            "n_seeds": int(vals.size),
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if vals.size > 1 else float("nan"),
            "ci_lo": ci.lo,
            "ci_hi": ci.hi,
            "pct_change": (
                float("nan") if arm == BASE or base_mean != base_mean or base_mean == 0
                else (float(vals.mean()) - base_mean) / base_mean * 100.0
            ),
        }

        if arm != BASE:
            paired_base = np.array([per_seed[BASE][s] for s in shared], dtype=np.float64)
            try:
                res = wilcoxon(vals, paired_base, alternative="greater")
                entry["wilcoxon_stat"] = float(res.statistic)
                entry["wilcoxon_p"] = float(res.pvalue)
            except ValueError:
                entry["wilcoxon_stat"] = float("nan")
                entry["wilcoxon_p"] = float("nan")
            wins = [s for s in shared if per_seed[arm][s] > per_seed[BASE][s]]
            entry["wins"] = len(wins)
            entry["losses"] = [s for s in shared if s not in wins]
            entry["beats_base_every_seed"] = len(wins) == len(shared)
        out[arm] = entry
    return out


def _write_csv(path: Path, seeds: List[int], per_seed: Dict[str, Dict[int, float]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed"] + list(ARMS) + [f"{a}_minus_{BASE}" for a in ARMS if a != BASE])
        for s in seeds:
            row: List[Any] = [s]
            for arm in ARMS:
                row.append(per_seed[arm].get(s, ""))
            for arm in ARMS:
                if arm == BASE:
                    continue
                a, b = per_seed[arm].get(s), per_seed[BASE].get(s)
                row.append(a - b if a is not None and b is not None else "")
            w.writerow(row)


def _markdown(
    summary: Dict[str, Any],
    seeds: List[int],
    stats: Dict[str, Any],
    n_tiles: int,
) -> str:
    runs = summary.get("runs", {})
    pending = summary.get("pending", [])
    wall = float(summary.get("wall_time_seconds", float("nan")))
    n_seeds = max((v["n_seeds"] for v in stats.values()), default=0)

    lines: List[str] = []
    lines.append("# Phase 3 rerun (v2): stain augmentation, TEST set")
    lines.append("")
    lines.append(
        "Test labels are dataset-2 auto-generated and noisy. These numbers "
        "compare arms against each other; they are not absolute accuracy."
    )
    lines.append("")
    lines.append("## Run inventory")
    lines.append("")
    lines.append(f"- total runs: **{len(runs)}** ({len(ARMS)} arms x {len(seeds)} seeds)")
    lines.append(f"- pending (wall-budget cut): **{len(pending)}**")
    lines.append(f"- test tiles per run: **{n_tiles}** (identical across all runs)")
    lines.append(f"- total test-tile evaluations: **{n_tiles * len(runs)}**")
    lines.append(f"- wall clock: **{wall:.0f} s ({wall / 3600.0:.2f} h)**")
    lines.append(f"- seeds: `{seeds}`")
    lines.append("")

    lines.append(f"## Mean test Dice per arm (n={n_seeds} seeds)")
    lines.append("")
    lines.append(
        "95% CI is a bootstrap over the per-seed means (10000 resamples), so the "
        "sample unit is the seed, not the tile."
    )
    lines.append("")
    lines.append("| arm | mean Dice | 95% CI | SD | % change vs rgb_aug |")
    lines.append("| --- | --- | --- | --- | --- |")
    for arm in ARMS:
        e = stats.get(arm)
        if not e:
            continue
        pct = "-" if arm == BASE else f"{e['pct_change']:+.2f}%"
        lines.append(
            f"| `{arm}` | {e['mean']:.4f} | [{e['ci_lo']:.4f}, {e['ci_hi']:.4f}] | "
            f"{e['std']:.4f} | {pct} |"
        )
    lines.append("")

    lines.append(f"## Paired Wilcoxon across seeds (n={n_seeds}), arm > rgb_aug")
    lines.append("")
    lines.append("| arm | statistic | p (one-sided) | seeds won |")
    lines.append("| --- | --- | --- | --- |")
    for arm in ARMS:
        if arm == BASE:
            continue
        e = stats.get(arm)
        if not e:
            continue
        lines.append(
            f"| `{arm}` | {e['wilcoxon_stat']:.1f} | {e['wilcoxon_p']:.4g} | "
            f"{e['wins']}/{e['n_seeds']} |"
        )
    lines.append("")

    hed = stats.get("hed_only")
    if hed:
        lines.append("## Does hed_only beat rgb_aug in every seed?")
        lines.append("")
        if hed["beats_base_every_seed"]:
            lines.append(
                f"**Yes** - `hed_only` > `rgb_aug` in all {hed['n_seeds']}/"
                f"{hed['n_seeds']} seeds."
            )
        else:
            lines.append(
                f"**No** - `hed_only` wins {hed['wins']}/{hed['n_seeds']} seeds. "
                f"It loses or ties on seeds: `{hed['losses']}`."
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def report(cfg: Config | None = None) -> Path:
    """Build the v2 test-set report; returns the markdown path."""
    if cfg is None:
        cfg = load_config(project_root() / "configs" / "rerun_v2.yaml")
    summary = _load_summary(cfg)
    seeds, per_seed, n_tiles = _collect(summary)
    stats = _stats(per_seed, seeds)

    cfg.paths.results.mkdir(parents=True, exist_ok=True)
    csv_path = cfg.paths.results / "phase3_per_seed.csv"
    _write_csv(csv_path, seeds, per_seed)

    md = _markdown(summary, seeds, stats, n_tiles)
    md_path = cfg.paths.results / "phase3_report_v2.md"
    with open(md_path, "w") as f:
        f.write(md)

    json_path = cfg.paths.results / "phase3_cross_seed_stats.json"
    with open(json_path, "w") as f:
        json.dump(
            {"seeds": seeds, "n_test_tiles": n_tiles, "test_dice": stats},
            f, indent=2,
        )

    print(md)
    print(f"[report_v2] markdown : {md_path}")
    print(f"[report_v2] csv      : {csv_path}")
    print(f"[report_v2] json     : {json_path}")
    return md_path


def main() -> None:
    """Entry point for `python -m scripts.report_v2`."""
    report()


if __name__ == "__main__":
    main()
