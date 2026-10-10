"""Model loading and single-tile inference for the demo.

Plain functions (no Streamlit) so `demo/build_space.py` can reuse them;
`demo/app.py` wraps them in `st.cache_resource`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torch import nn

from src.data.transforms import (
    MacenkoNormalize,
    get_full_stain_aware_transforms,
    get_hed_only_transforms,
    get_macenko_only_transforms,
    get_rgb_aug_transforms,
)
from src.training.evaluate import per_tile_dice, per_tile_iou

IMAGE_SIZE = 512

# eval-time transform per arm; must match _AUG_BUILDERS in src/training/train.py
EVAL_BUILDERS: Dict[str, Callable] = {
    "rgb_aug": get_rgb_aug_transforms,
    "hed_only": get_hed_only_transforms,
    "macenko_only": get_macenko_only_transforms,
    "full_stain_aware": get_full_stain_aware_transforms,
}

# arms whose eval transform already applies Macenko
MACENKO_AT_EVAL = ("macenko_only", "full_stain_aware")


def resolve_device() -> torch.device:
    """cuda -> mps -> cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_unet() -> nn.Module:
    """U-Net + ResNet-34 with no pretrained download.

    Mirrors `_BUILDERS["unet_resnet34"]` in src/models/__init__.py except
    `encoder_weights=None`: the checkpoint overwrites every weight anyway,
    and skipping the ImageNet fetch keeps cold starts offline.
    """
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )


def load_model(ckpt_path: Path, device: torch.device) -> nn.Module:
    """Build the U-Net and load a checkpoint strictly, in eval mode."""
    model = build_unet()
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def eval_transform(arm: str):
    """The arm's eval-time transform (same as used for the reported scores)."""
    if arm not in EVAL_BUILDERS:
        raise ValueError(f"no eval transform for arm '{arm}'; add it to EVAL_BUILDERS")
    return EVAL_BUILDERS[arm](image_size=IMAGE_SIZE, train=False)


@torch.no_grad()
def predict(model: nn.Module, arm: str, img: np.ndarray, device: torch.device) -> np.ndarray:
    """Binary vessel mask (uint8 HW, 0/1): sigmoid > 0.5, as in training eval."""
    dummy = np.zeros(img.shape[:2], dtype=np.uint8)
    x = eval_transform(arm)(image=img, mask=dummy)["image"].unsqueeze(0).to(device)
    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return (prob > 0.5).astype(np.uint8)


def scores(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    """(Dice, IoU) for one tile, using the training code's metric functions."""
    d = per_tile_dice(pred[None], gt[None])[0]
    i = per_tile_iou(pred[None], gt[None])[0]
    return float(d), float(i)


def macenko(img: np.ndarray) -> np.ndarray:
    """Apply the fixed-reference Macenko normalisation used at eval."""
    return MacenkoNormalize(p=1.0).apply(img)


def resize_upload(img: Image.Image) -> np.ndarray:
    """Uploaded image -> RGB uint8 resized to 512x512 (no Resize in eval transforms)."""
    rgb = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(rgb, dtype=np.uint8)
