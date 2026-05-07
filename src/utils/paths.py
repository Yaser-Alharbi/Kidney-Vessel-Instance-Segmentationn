"""YAML config loader and path resolution.

Paths in the YAML are resolved relative to the project root (the
directory containing main.py), unless they start with ~ or /.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


def project_root() -> Path:
    """Repo root: two levels above this file."""
    return Path(__file__).resolve().parents[2]


def _resolve(p: str, root: Path) -> Path:
    """Expand ~, then resolve relative paths against the project root."""
    pp = Path(p).expanduser()
    if pp.is_absolute():
        return pp
    return (root / pp).resolve()


@dataclass(frozen=True)
class Paths:
    raw: Path
    processed: Path
    masks: Path
    splits: Path
    results: Path
    artifacts: Path
    external_results_md: Path


@dataclass(frozen=True)
class Config:
    seed: int
    image_size: int
    batch_size: int
    num_workers: int
    device: str
    paths: Paths
    raw: Dict[str, Any]

    @property
    def root(self) -> Path:
        return project_root()


def load_config(config_path: str | Path | None = None) -> Config:
    """Load configs/default.yaml (or the given path) and resolve all paths."""
    root = project_root()
    if config_path is None:
        config_path = root / "configs" / "default.yaml"
    config_path = Path(config_path)

    with open(config_path, "r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    raw_paths = raw.get("paths", {})
    paths = Paths(
        raw=_resolve(raw_paths["raw"], root),
        processed=_resolve(raw_paths["processed"], root),
        masks=_resolve(raw_paths["masks"], root),
        splits=_resolve(raw_paths["splits"], root),
        results=_resolve(raw_paths["results"], root),
        artifacts=_resolve(raw_paths.get("artifacts", "artifacts"), root),
        external_results_md=_resolve(raw_paths["external_results_md"], root),
    )

    return Config(
        seed=int(raw["seed"]),
        image_size=int(raw["image_size"]),
        batch_size=int(raw["batch_size"]),
        num_workers=int(raw["num_workers"]),
        device=str(raw["device"]),
        paths=paths,
        raw=raw,
    )


def ensure_dirs(cfg: Config) -> None:
    """Make sure processed, masks, results and artifacts dirs exist."""
    cfg.paths.processed.mkdir(parents=True, exist_ok=True)
    cfg.paths.masks.mkdir(parents=True, exist_ok=True)
    cfg.paths.results.mkdir(parents=True, exist_ok=True)
    cfg.paths.artifacts.mkdir(parents=True, exist_ok=True)
