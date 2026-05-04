"""Single-run training loop.

`train_one_run` trains one (model, aug) pair, picks best by val Dice,
saves curves + per-tile JSON + a predictions panel under `results/`,
and returns per-tile val/test Dice arrays. Caller is responsible for
seeding (`set_seed`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data.dataset import HuBMAPDataset
from src.data.transforms import (
    get_rgb_aug_transforms,
    get_stain_aware_hed_only_transforms,
    get_stain_aware_transforms,
)
from src.models import build_model
from src.training.evaluate import per_tile_dice
from src.training.losses import DiceBCELoss
from src.utils.paths import Config


_AUG_BUILDERS = {
    "rgb_aug": get_rgb_aug_transforms,
    "stain_aware_hed_only": get_stain_aware_hed_only_transforms,
    "stain_aware": get_stain_aware_transforms,
}


@dataclass
class TrainResult:
    train_loss: List[float]
    val_loss: List[float]
    val_dice: List[float]
    best_epoch: int
    best_val_dice: float
    per_tile_val_dice_at_best: np.ndarray
    per_tile_test_dice_at_best: np.ndarray
    weights_path: Path
    val_tile_ids: List[str] = field(default_factory=list)
    test_tile_ids: List[str] = field(default_factory=list)


def _resolve_device(cfg: Config) -> torch.device:
    requested = cfg.device.lower()
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_aug(aug_name: str, image_size: int, train: bool):
    if aug_name not in _AUG_BUILDERS:
        raise ValueError(
            f"unknown aug '{aug_name}'. expected one of {sorted(_AUG_BUILDERS)}"
        )
    return _AUG_BUILDERS[aug_name](image_size=image_size, train=train)


def _load_split_ids(cfg: Config) -> dict:
    with open(cfg.paths.splits, "r") as f:
        return json.load(f)


def _make_loader(
    ds: HuBMAPDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    persistent: bool = False,
) -> DataLoader:
    use_persistent = persistent and num_workers > 0
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        persistent_workers=use_persistent,
    )


@torch.no_grad()
def _eval_loss_and_dice(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, List[str]]:
    """Mean loss + per-tile Dice over a loader. Leaves model in eval()."""
    model.eval()
    losses: List[float] = []
    dice_chunks: List[np.ndarray] = []
    ids: List[str] = []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=False).float()
        mask = batch["mask"].to(device, non_blocking=False).long()
        logits = model(image)
        target = mask.unsqueeze(1).float()
        loss = criterion(logits, target)
        losses.append(float(loss.item()))

        pred = (torch.sigmoid(logits) > 0.5).squeeze(1).to("cpu").numpy().astype(np.uint8)
        gt = mask.to("cpu").numpy().astype(np.uint8)
        dice_chunks.append(per_tile_dice(pred, gt))

        bids = batch.get("tile_id", [])
        if isinstance(bids, (list, tuple)):
            ids.extend(list(bids))
        else:
            ids.extend([str(t) for t in bids])

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    per_tile = np.concatenate(dice_chunks) if dice_chunks else np.array([], dtype=np.float64)
    return mean_loss, per_tile, ids


def _save_loss_curve(
    train_loss: List[float],
    val_loss: List[float],
    out_path: Path,
    title: str,
) -> None:
    epochs = np.arange(1, len(train_loss) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_loss, label="train")
    ax.plot(epochs, val_loss, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_val_dice_curve(
    val_dice: List[float],
    out_path: Path,
    title: str,
) -> None:
    epochs = np.arange(1, len(val_dice) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, val_dice, marker="o", label="val Dice")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Dice")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _pick_qualitative_indices(per_tile: np.ndarray, n: int) -> List[int]:
    """Return up to n indices: half worst Dice, half best Dice."""
    if per_tile.size == 0 or n <= 0:
        return []
    n = min(int(n), int(per_tile.size))
    if n <= 1:
        return [int(np.argmax(per_tile))]
    n_best = n // 2
    n_worst = n - n_best
    order = np.argsort(per_tile)
    worst = order[:n_worst].tolist()
    best = order[-n_best:][::-1].tolist()
    seen, picked = set(), []
    for i in worst + best:
        if i not in seen:
            picked.append(int(i))
            seen.add(i)
    return picked


@torch.no_grad()
def _save_qualitative_predictions(
    model: torch.nn.Module,
    val_ds: HuBMAPDataset,
    val_per_tile: np.ndarray,
    device: torch.device,
    n_examples: int,
    out_path: Path,
    title: str,
) -> None:
    """Save best+worst val tiles as image / GT / prediction triplets."""
    indices = _pick_qualitative_indices(val_per_tile, n_examples)
    if not indices:
        return

    rows = len(indices)
    fig, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    model.eval()
    img_dir = val_ds.img_dir
    mask_dir = val_ds.mask_dir
    for r, idx in enumerate(indices):
        sample = val_ds[idx]
        tile_id = sample["tile_id"]
        image_t = sample["image"].unsqueeze(0).to(device).float()
        logits = model(image_t)
        pred = (torch.sigmoid(logits) > 0.5).squeeze().to("cpu").numpy().astype(np.uint8)

        raw_img = np.array(Image.open(img_dir / f"{tile_id}.tif").convert("RGB"))
        gt = np.array(Image.open(mask_dir / f"{tile_id}.png"))
        if gt.ndim == 3:
            gt = gt[..., 0]
        gt = (gt > 0).astype(np.uint8)
        dice = float(val_per_tile[idx])

        axes[r, 0].imshow(raw_img)
        axes[r, 0].set_title(f"{tile_id} (Dice={dice:.3f})")
        axes[r, 0].axis("off")
        axes[r, 1].imshow(raw_img)
        axes[r, 1].imshow(gt, cmap="Reds", alpha=0.45, vmin=0, vmax=1)
        axes[r, 1].set_title("ground truth")
        axes[r, 1].axis("off")
        axes[r, 2].imshow(raw_img)
        axes[r, 2].imshow(pred, cmap="Greens", alpha=0.45, vmin=0, vmax=1)
        axes[r, 2].set_title("prediction")
        axes[r, 2].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def train_one_run(
    cfg: Config,
    model_name: str,
    aug_name: str,
    epochs: int,
    *,
    run_tag: str,
    log_every: int = 1,
) -> TrainResult:
    """Train one (model, aug) configuration end-to-end."""
    device = _resolve_device(cfg)
    print(f"[train:{run_tag}] device={device.type} model={model_name} aug={aug_name}")

    image_size = int(cfg.image_size)
    batch_size = int(cfg.batch_size)
    num_workers = int(cfg.num_workers)
    lr = float(cfg.raw.get("lr", 1e-4))
    weight_decay = float(cfg.raw.get("weight_decay", 1e-4))

    splits = _load_split_ids(cfg)
    img_dir = cfg.paths.raw / "train"
    mask_dir = cfg.paths.masks

    train_tf = _build_aug(aug_name, image_size, train=True)
    eval_tf = _build_aug(aug_name, image_size, train=False)

    # cache=True on eval avoids recomputing Macenko per epoch
    train_ds = HuBMAPDataset(splits["train"], img_dir, mask_dir, transform=train_tf)
    val_ds = HuBMAPDataset(
        splits["val"], img_dir, mask_dir, transform=eval_tf, cache=True
    )
    test_ds = HuBMAPDataset(
        splits.get("test", []), img_dir, mask_dir, transform=eval_tf, cache=True
    )

    # eval loaders stay single-process so the cache survives across epochs
    train_loader = _make_loader(
        train_ds, batch_size, shuffle=True, num_workers=num_workers, persistent=True
    )
    val_loader = _make_loader(val_ds, batch_size, shuffle=False, num_workers=0)
    test_loader = _make_loader(test_ds, batch_size, shuffle=False, num_workers=0)
    print(
        f"[train:{run_tag}] tiles  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"
    )

    model = build_model(model_name).to(device)
    criterion = DiceBCELoss(dice_weight=0.5).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=0.0)

    ckpt_dir = cfg.paths.results / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights_path = ckpt_dir / f"{run_tag}.pth"

    train_loss_hist: List[float] = []
    val_loss_hist: List[float] = []
    val_dice_hist: List[float] = []
    best_epoch = 0
    best_val_dice = -1.0
    best_state: Optional[dict] = None
    patience = int(cfg.raw.get("early_stop_patience", 0) or 0)
    epochs_since_best = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=False).float()
            mask = batch["mask"].to(device, non_blocking=False).long()
            target = mask.unsqueeze(1).float()

            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        scheduler.step()
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        val_loss, val_per_tile, _ = _eval_loss_and_dice(model, val_loader, criterion, device)
        val_dice = float(val_per_tile.mean()) if val_per_tile.size else float("nan")

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        val_dice_hist.append(val_dice)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            best_state = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(
                f"[train:{run_tag}] epoch {epoch:3d}/{epochs}"
                f"  train_loss={train_loss:.4f}"
                f"  val_loss={val_loss:.4f}"
                f"  val_dice={val_dice:.4f}"
                f"  best@{best_epoch}={best_val_dice:.4f}"
                f"  no_improve={epochs_since_best}"
            )

        if patience > 0 and epochs_since_best >= patience:
            print(
                f"[train:{run_tag}] early stop at epoch {epoch} "
                f"(no val Dice improvement for {patience} epochs; best @ {best_epoch})."
            )
            break

    if best_state is None:
        raise RuntimeError(f"[{run_tag}] training produced no best state (epochs={epochs}).")
    torch.save(best_state, weights_path)
    model.load_state_dict(best_state)

    _, val_per_tile, val_ids = _eval_loss_and_dice(model, val_loader, criterion, device)
    _, test_per_tile, test_ids = _eval_loss_and_dice(model, test_loader, criterion, device)
    print(
        f"[train:{run_tag}] best epoch {best_epoch}"
        f"  val_dice={float(val_per_tile.mean()):.4f}"
        f"  test_dice={float(test_per_tile.mean()):.4f}"
    )

    loss_png = cfg.paths.results / f"{run_tag}_loss.png"
    dice_png = cfg.paths.results / f"{run_tag}_val_dice.png"
    pred_png = cfg.paths.results / f"{run_tag}_predictions.png"
    _save_loss_curve(train_loss_hist, val_loss_hist, loss_png, title=f"{run_tag} loss")
    _save_val_dice_curve(val_dice_hist, dice_png, title=f"{run_tag} val Dice")

    n_pred = int(cfg.raw.get("n_pred_examples", 6))
    _save_qualitative_predictions(
        model, val_ds, val_per_tile, device, n_pred, pred_png,
        title=f"{run_tag} val predictions (best + worst by Dice)",
    )

    per_tile_path = cfg.paths.results / f"{run_tag}_per_tile.json"
    with open(per_tile_path, "w") as f:
        json.dump(
            {
                "run_tag": run_tag,
                "model": model_name,
                "aug": aug_name,
                "best_epoch": best_epoch,
                "best_val_dice": float(best_val_dice),
                "val_tile_ids": val_ids,
                "val_per_tile_dice": val_per_tile.tolist(),
                "test_tile_ids": test_ids,
                "test_per_tile_dice": test_per_tile.tolist(),
            },
            f,
            indent=2,
        )

    return TrainResult(
        train_loss=train_loss_hist,
        val_loss=val_loss_hist,
        val_dice=val_dice_hist,
        best_epoch=best_epoch,
        best_val_dice=float(best_val_dice),
        per_tile_val_dice_at_best=val_per_tile,
        per_tile_test_dice_at_best=test_per_tile,
        weights_path=weights_path,
        val_tile_ids=val_ids,
        test_tile_ids=test_ids,
    )
