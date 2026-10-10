"""Stage the demo for Streamlit Community Cloud + an HF model repo. Never pushes.

    python -m demo.build_deploy --out ~/vsl-deploy

Writes two folders into an empty `--out` outside the repo:

    app/      -> its own GitHub repo, deployed on Streamlit Community Cloud
        demo/ + the src/ subset the app imports, configs/rerun_v2.yaml,
        artifacts_v2/phase3_summary.json,
        demo_assets/{tiles,masks,preds/<arm>}/ + manifest.json (curated tiles,
            0/1 masks, seed-42 0/1 prediction masks, weights repo id),
        requirements.txt (CPU torch wheels by URL, other pins from this env),
        .streamlit/config.toml, .gitignore, README.md
    weights/  -> HF model repo: the four seed-42 checkpoints + README.md

The app downloads the checkpoints from the weights repo on first start.
Needs the local raw tiles, masks and results_v2 checkpoints.
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from demo import data, inference
from src.utils.paths import Config, project_root

CODE_FILES = (
    "demo/__init__.py",
    "demo/app.py",
    "demo/data.py",
    "demo/inference.py",
    "demo/views.py",
    "src/__init__.py",
    "src/models/__init__.py",
    "src/data/__init__.py",
    "src/data/transforms.py",
    "src/training/__init__.py",
    "src/training/evaluate.py",
    "src/utils/__init__.py",
    "src/utils/paths.py",
    "configs/rerun_v2.yaml",
    "artifacts_v2/phase3_summary.json",
)

DEFAULT_WEIGHTS_REPO = "Yaser-Alharbi/vessel-stain-lab-weights"
COURSEWORK_URL = "https://github.com/Yaser-Alharbi/Kidney-Vessel-Instance-Segmentationn"
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
PYTHONS = ("3.11", "3.12", "3.13")
MAX_UPLOAD_MB = 5

STREAMLIT_CONFIG = f"""[server]
maxUploadSize = {MAX_UPLOAD_MB}

[browser]
gatherUsageStats = false
"""

GITIGNORE = "__pycache__/\n*.pyc\n.venv/\n*.pth\n"


def select_tiles(cube: data.Cube, per_sort: int) -> Dict[str, List[str]]:
    """Top `per_sort` tiles for each ranking (except 'Tile id'), deduplicated."""
    picked: Dict[str, List[str]] = {}
    for sort in data.SORTS[:-1]:
        for tile_id in data.rank_tiles(cube, sort)[:per_sort]:
            picked.setdefault(tile_id, []).append(sort)
    return picked


def _save_mask(mask: np.ndarray, path: Path) -> None:
    """Write a 0/1 uint8 mask as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8)).save(path, optimize=True)


def _wheel_url(package: str, ver: str, py: str) -> str:
    """PyTorch CPU wheel URL for one package, version and CPython version."""
    tag = "cp" + py.replace(".", "")
    return f"{TORCH_INDEX}/{package}-{ver}%2Bcpu-{tag}-{tag}-manylinux_2_28_x86_64.whl"


def _check_url(url: str) -> None:
    """HEAD the URL; raise if it does not resolve."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                raise RuntimeError(f"{url} -> HTTP {resp.status}")
    except OSError as exc:
        raise RuntimeError(f"wheel URL does not resolve: {url} ({exc})") from exc


def _requirements() -> str:
    """Community Cloud requirements.

    torch/torchvision come as direct CPU wheel URLs, one per Python version
    (markers), so uv never consults the PyTorch index for other packages and a
    forced Python upgrade still installs. Everything else is pinned to this env.
    """
    lines = []
    for package in ("torch", "torchvision"):
        ver = version(package).split("+")[0]
        for py in PYTHONS:
            url = _wheel_url(package, ver, py)
            _check_url(url)
            lines.append(f'{package} @ {url} ; python_version == "{py}"')
    lines += [
        f"segmentation-models-pytorch=={version('segmentation-models-pytorch')}",
        f"timm=={version('timm')}",
        f"albumentations=={version('albumentations')}",
        f"opencv-python-headless=={version('opencv-python')}",
        f"numpy=={version('numpy')}",
        f"PyYAML=={version('PyYAML')}",
        f"Pillow=={version('Pillow')}",
        f"streamlit=={version('streamlit')}",
        f"huggingface_hub=={version('huggingface_hub')}",
    ]
    return "\n".join(lines) + "\n"


def _results_rows(summary: Dict[str, Any], arms: List[str]) -> str:
    """Markdown table rows of cross-seed test Dice, read from the summary."""
    cs = summary["stats"]["test"]["cross_seed"]["dice"]
    wil = summary["stats"]["test"]["wilcoxon_cross_seed"]["dice"]
    values = data.cross_seed_values(summary, "test", "dice")
    rows = []
    for arm in arms:
        w = wil.get(f"{arm}_gt_rgb_aug")
        wins = ("" if arm == "rgb_aug"
                else f"{data.wins_vs_base(values, arm, 'rgb_aug')} / {cs[arm]['n']}")
        p = "" if w is None else f"{w['p']:.3g}"
        rows.append(f"| `{arm}` | {cs[arm]['mean']:.3f} | {cs[arm]['std']:.3f} | {wins} | {p} |")
    return "\n".join(rows)


def _app_readme(summary: Dict[str, Any], arms: List[str]) -> str:
    """Portfolio-facing README for the app repo."""
    return f"""# Vessel Stain Lab

Interactive demo of a stain-augmentation study for blood-vessel segmentation
in kidney histology (HuBMAP "Hacking the Human Vasculature" data).

A U-Net with a ResNet-34 encoder was trained four ways, differing only in how
training images are recoloured, 20 times each with different random seeds:

| Arm | Train-time augmentation | Eval-time transform |
| --- | --- | --- |
| `rgb_aug` | flips, rot90, brightness/contrast | none |
| `hed_only` | flips, rot90, HED stain jitter | none |
| `macenko_only` | flips, rot90, Macenko normalisation | Macenko |
| `full_stain_aware` | flips, rot90, HED stain jitter | Macenko |

## Result on the held-out test slides (20 seeds per arm)

| Arm | Mean Dice | Std | Beats `rgb_aug` | Wilcoxon p (one-sided) |
| --- | --- | --- | --- | --- |
{_results_rows(summary, arms)}

Test labels are auto-generated and noisy, so these scores compare arms with
each other; they are not absolute accuracy.

## What the app shows

- **Tile explorer**: the tissue tile, the expected vessels and each arm's
  prediction, with an error view and an arm-vs-arm difference view.
- **Stain stress test**: shift the haematoxylin/eosin balance and watch each
  model react, with live inference.
- **Results**: per-seed distributions, confidence intervals and paired tests.
- **Try your own image**: upload a tile and run all four models.

A sidebar toggle switches between a technical and a plain-language view.

Model weights: [huggingface.co/{{weights}}](https://huggingface.co/{{weights}}).
Training code and full write-up: [{COURSEWORK_URL}]({COURSEWORK_URL}).
"""


def _weights_readme() -> str:
    """Model card for the weights repo."""
    return f"""---
library_name: segmentation-models-pytorch
tags:
- image-segmentation
- histology
---

# Vessel Stain Lab weights

Four U-Net + ResNet-34 checkpoints (seed {data.LIVE_SEED}), one per stain-augmentation
arm: `rgb_aug`, `hed_only`, `macenko_only`, `full_stain_aware`. Binary
blood-vessel segmentation on 512x512 HuBMAP kidney tiles, sigmoid > 0.5.

Load with `smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3,
classes=1)` and `load_state_dict(torch.load(path))`.

Training code: {COURSEWORK_URL}
"""


def _write_text(path: Path, text: str) -> None:
    """Write UTF-8 text, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _copy_checkpoints(cfg: Config, arms: List[str], weights: Path, device) -> Dict[str, Any]:
    """Copy each arm's seed-42 checkpoint to `weights` and return the loaded models."""
    models = {}
    weights.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        ckpt = data.checkpoint_path(cfg, arm, data.LIVE_SEED)
        if ckpt is None:
            raise FileNotFoundError(f"missing seed-{data.LIVE_SEED} checkpoint for {arm}")
        shutil.copy2(ckpt, weights / ckpt.name)
        models[arm] = inference.load_model(ckpt, device)
    return models


def _stage_tile(cfg: Config, tile_id: str, models: Dict[str, Any], assets: Path, device) -> None:
    """Write one tile, its mask and every arm's seed-42 prediction mask."""
    img, gt = data.load_tile(cfg, tile_id)
    (assets / "tiles").mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(assets / "tiles" / f"{tile_id}.png", optimize=True)
    _save_mask(gt, assets / "masks" / f"{tile_id}.png")
    for arm, model in models.items():
        pred = inference.predict(model, arm, img, device)
        _save_mask(pred, assets / "preds" / arm / f"{tile_id}.png")


def _check_out(out: Path) -> None:
    """Refuse a target inside the repo or one that already has files."""
    root = project_root()
    if out.resolve().is_relative_to(root):
        raise ValueError(f"--out must be outside the repo ({root})")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"--out {out} is not empty; pick a new folder")


def build(  # pylint: disable=too-many-locals
    out: Path, per_sort: int, weights_repo: str
) -> Dict[str, int]:
    """Stage app/ and weights/ into `out` and return counts."""
    _check_out(out)
    cfg = data.load_demo_config()
    if not data.local_mode(cfg):
        raise RuntimeError("raw tiles/masks not found; build_deploy needs local data")
    arms = data.arms_from_config(cfg)
    summary = data.load_summary(cfg)
    app, assets = out / "app", out / "app" / "demo_assets"
    requirements = _requirements()  # fail early if a wheel URL is missing
    device = inference.resolve_device()
    models = _copy_checkpoints(cfg, arms, out / "weights", device)

    manifest = {"seed": data.LIVE_SEED, "arms": arms, "weights_repo": weights_repo, "tiles": []}
    for split in ("val", "test"):
        picked = select_tiles(data.build_cube(summary, arms, split), per_sort)
        for tile_id, reasons in picked.items():
            _stage_tile(cfg, tile_id, models, assets, device)
            manifest["tiles"].append({"id": tile_id, "split": split, "reasons": reasons})
    _write_text(assets / "manifest.json", json.dumps(manifest, indent=2))
    readme = _app_readme(summary, arms).replace("{weights}", weights_repo)
    _write_app_files(app, requirements, readme)
    _write_text(out / "weights" / "README.md", _weights_readme())
    counts = {f"{s}_tiles": sum(t["split"] == s for t in manifest["tiles"])
              for s in ("val", "test")}
    counts["checkpoints"] = len(models)
    return counts


def _write_app_files(app: Path, requirements: str, readme: str) -> None:
    """Copy the code subset and write the app repo's support files."""
    for rel in CODE_FILES:
        (app / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project_root() / rel, app / rel)
    _write_text(app / "requirements.txt", requirements)
    _write_text(app / ".streamlit" / "config.toml", STREAMLIT_CONFIG)
    _write_text(app / ".gitignore", GITIGNORE)
    _write_text(app / "README.md", readme)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="empty folder outside the repo")
    parser.add_argument("--per-sort", type=int, default=4, help="tiles per ranking per split")
    parser.add_argument("--weights-repo", default=DEFAULT_WEIGHTS_REPO,
                        help="HF model repo id the app downloads checkpoints from")
    args = parser.parse_args()
    counts = build(args.out.expanduser(), args.per_sort, args.weights_repo)
    print(f"[build_deploy] staged {args.out}: {counts}")
    print("[build_deploy] not pushed. app/ -> GitHub + Streamlit Community Cloud; "
          f"weights/ -> HF model repo {args.weights_repo}.")


if __name__ == "__main__":
    main()
