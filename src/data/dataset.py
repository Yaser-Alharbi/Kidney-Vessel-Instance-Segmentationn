"""PyTorch dataset for HuBMAP vasculature tiles.

Without a transform, samples are torch tensors:
    image: float32 [3, H, W] in [0, 1]
    mask:  int64   [H, W]    in {0, 1}

With a transform, it is called as `transform(image=img, mask=mask)`
(Albumentations style) and the dict is returned as-is.

Every sample carries a `loss_weight` (default 1.0). Pass a
`{tile_id: weight}` map to down-weight noisy tiles in Phase 3.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class HuBMAPDataset(Dataset):
    """HuBMAP tile dataset."""

    def __init__(
        self,
        tile_ids: List[str],
        img_dir: str | Path,
        mask_dir: str | Path,
        transform: Optional[Callable] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        cache: bool = False,
    ) -> None:
        self.tile_ids = list(tile_ids)
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.loss_weights = loss_weights or {}
        self.cache = bool(cache)
        self._cache: Dict[int, dict] = {}

    def __len__(self) -> int:
        return len(self.tile_ids)

    def _load_pair(self, tile_id: str) -> Tuple[np.ndarray, np.ndarray]:
        img_path = self.img_dir / f"{tile_id}.tif"
        mask_path = self.mask_dir / f"{tile_id}.png"
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image: {img_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask: {mask_path}")

        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        mask = np.array(Image.open(mask_path), dtype=np.uint8)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.uint8)
        return img, mask

    def __getitem__(self, idx: int):
        if self.cache and idx in self._cache:
            return self._cache[idx]

        tile_id = self.tile_ids[idx]
        img, mask = self._load_pair(tile_id)
        weight = float(self.loss_weights.get(tile_id, 1.0))

        if self.transform is not None:
            out = self.transform(image=img, mask=mask)
            if isinstance(out, dict):
                out.setdefault("tile_id", tile_id)
                out.setdefault("loss_weight", weight)
        else:
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mask_t = torch.from_numpy(mask).long()
            out = {
                "image": img_t,
                "mask": mask_t,
                "tile_id": tile_id,
                "loss_weight": weight,
            }

        if self.cache:
            self._cache[idx] = out
        return out


def _save_sample_item_figure(
    sample: dict,
    out_path: Path,
) -> None:
    """Save a side-by-side image + mask preview for a single sample."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = sample["image"]
    mask = sample["mask"]

    if isinstance(image, torch.Tensor):
        img_np = (image.permute(1, 2, 0).cpu().numpy() * 255.0).astype("uint8")
    else:
        img_np = np.asarray(image)
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = np.asarray(mask)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_np)
    axes[0].set_title(f"image: {sample.get('tile_id', '?')}")
    axes[0].axis("off")
    axes[1].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"mask (vessel frac={float(mask_np.mean()):.4f})")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _main() -> None:
    from src.utils.paths import ensure_dirs, load_config

    cfg = load_config()
    ensure_dirs(cfg)

    splits_path = cfg.paths.splits
    if not splits_path.exists():
        raise FileNotFoundError(
            f"splits.json not found at {splits_path} -- run main.py first."
        )
    with open(splits_path, "r") as f:
        splits = json.load(f)

    img_dir = cfg.paths.raw / "train"
    mask_dir = cfg.paths.masks

    train_ds = HuBMAPDataset(splits["train"], img_dir, mask_dir, transform=None)
    val_ds = HuBMAPDataset(splits["val"], img_dir, mask_dir, transform=None)

    print(f"[dataset] train length : {len(train_ds)}")
    print(f"[dataset] val   length : {len(val_ds)}")

    sample = train_ds[0]
    img_t = sample["image"]
    mask_t = sample["mask"]
    print(
        f"[dataset] sample image  : shape={tuple(img_t.shape)} dtype={img_t.dtype} "
        f"min={float(img_t.min()):.3f} max={float(img_t.max()):.3f}"
    )
    print(
        f"[dataset] sample mask   : shape={tuple(mask_t.shape)} dtype={mask_t.dtype} "
        f"unique={sorted(set(mask_t.unique().tolist()))}"
    )

    out_path = cfg.paths.results / "sample_dataset_item.png"
    _save_sample_item_figure(sample, out_path)
    print(f"[dataset] preview saved : {out_path}")


if __name__ == "__main__":
    _main()
