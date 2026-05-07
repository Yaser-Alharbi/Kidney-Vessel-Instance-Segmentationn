"""UMAP of penultimate-layer activations across the 4 augmentation arms.

Read-only: never retrains. Loads each arm's seed-42 checkpoint
(`results/checkpoints/unet_resnet34_<arm>_seed42.pth`), registers a
forward hook on the ResNet-34 encoder's `layer4`, runs the validation
split through the model, global-average-pools the layer4 feature map
to a 512-D vector, and stacks all 4 x N_val vectors.

A single UMAP (n_neighbors=15, min_dist=0.1, random_state=42) is fit on
the stacked matrix; the 2-D embedding is scattered with one colour per
arm and saved to `figures/umap_activations.pdf`. Prints the silhouette
score (arm labels as cluster assignments) to stdout.

Usage:
    python -m scripts.umap_activations
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# allow `python scripts/umap_activations.py` from the repo root
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.dataset import HuBMAPDataset
from src.data.transforms import (
    get_full_stain_aware_transforms,
    get_hed_only_transforms,
    get_macenko_only_transforms,
    get_rgb_aug_transforms,
)
from src.models import build_model
from src.utils.paths import Config, ensure_dirs, load_config


_ARMS: Tuple[str, ...] = (
    "rgb_aug",
    "hed_only",
    "macenko_only",
    "full_stain_aware",
)
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
_AUG_BUILDERS = {
    "rgb_aug": get_rgb_aug_transforms,
    "hed_only": get_hed_only_transforms,
    "macenko_only": get_macenko_only_transforms,
    "full_stain_aware": get_full_stain_aware_transforms,
}


def _resolve_device(cfg: Config) -> torch.device:
    requested = cfg.device.lower()
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _val_loader_for_arm(cfg: Config, arm: str) -> DataLoader:
    """Eval-style val loader for `arm` (deterministic, no shuffle)."""
    image_size = int(cfg.image_size)
    batch_size = int(cfg.batch_size)
    img_dir = cfg.paths.raw / "train"
    mask_dir = cfg.paths.masks
    with open(cfg.paths.splits, "r") as f:
        splits = json.load(f)
    eval_tf = _AUG_BUILDERS[arm](image_size=image_size, train=False)
    ds = HuBMAPDataset(splits["val"], img_dir, mask_dir, transform=eval_tf, cache=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def _extract_pooled_activations(
    cfg: Config, arm: str, device: torch.device
) -> Tuple[np.ndarray, List[str]]:
    """Run val loader through the seed-42 checkpoint, return (N, 512)."""
    ckpt = cfg.paths.results / "checkpoints" / f"unet_resnet34_{arm}_seed42.pth"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Missing checkpoint for arm={arm}: {ckpt}. "
            "Run main.py first or restore checkpoints."
        )

    model = build_model("unet_resnet34").to(device)
    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    # smp Unet exposes the resnet via .encoder; resnet has .layer4.
    layer4 = model.encoder.layer4

    cache: Dict[str, torch.Tensor] = {}

    def _hook(_mod, _inp, output):
        # global average pool spatial dims to (B, C)
        cache["feat"] = output.detach().mean(dim=(-1, -2)).to("cpu")

    handle = layer4.register_forward_hook(_hook)

    feats: List[np.ndarray] = []
    ids: List[str] = []
    loader = _val_loader_for_arm(cfg, arm)
    try:
        with torch.no_grad():
            for batch in loader:
                image = batch["image"].to(device).float()
                _ = model(image)
                feats.append(cache["feat"].numpy().astype(np.float64))
                bids = batch.get("tile_id", [])
                if isinstance(bids, (list, tuple)):
                    ids.extend(list(bids))
                else:
                    ids.extend([str(t) for t in bids])
    finally:
        handle.remove()

    if not feats:
        return np.zeros((0, 512), dtype=np.float64), ids
    arr = np.concatenate(feats, axis=0)
    print(f"[umap] arm={arm}: activations shape={arr.shape}  tiles={len(ids)}")
    return arr, ids


def _fit_umap(stacked: np.ndarray) -> np.ndarray:
    """UMAP: n_neighbors=15, min_dist=0.1, n_components=2, seed=42."""
    try:
        from umap import UMAP  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "umap-learn is required for this script. "
            "Run: pip install umap-learn"
        ) from e
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            random_state=42,
            n_jobs=1,
        )
        embedding = reducer.fit_transform(stacked)
    return np.asarray(embedding)


def _scatter_plot(
    embedding: np.ndarray, labels: List[str], out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    arr_labels = np.asarray(labels)
    for arm in _ARMS:
        mask = arr_labels == arm
        if not mask.any():
            continue
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=22,
            alpha=0.75,
            c=_ARM_COLORS[arm],
            label=_ARM_LABELS[arm],
            edgecolors="white",
            linewidths=0.4,
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(
        "UMAP of layer-4 GAP activations across 4 augmentation arms (val split, seed 42)"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    device = _resolve_device(cfg)
    print(f"[umap] device={device.type}")

    feats: List[np.ndarray] = []
    labels: List[str] = []
    for arm in _ARMS:
        arr, _ids = _extract_pooled_activations(cfg, arm, device)
        feats.append(arr)
        labels.extend([arm] * arr.shape[0])

    stacked = np.concatenate(feats, axis=0)
    print(f"[umap] stacked activations shape: {stacked.shape}")

    embedding = _fit_umap(stacked)
    print(f"[umap] embedding shape: {embedding.shape}")

    figures_dir = _REPO_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "umap_activations.pdf"
    _scatter_plot(embedding, labels, out_path)
    print(f"[umap] figure saved: {out_path}")

    try:
        from sklearn.metrics import silhouette_score

        sil = float(silhouette_score(embedding, labels))
    except ImportError:  # pragma: no cover
        sil = float("nan")
        print("[umap] scikit-learn missing; cannot compute silhouette")
    print(f"[umap] silhouette_score (arm labels, 2D embedding) = {sil:.4f}")


if __name__ == "__main__":
    main()
