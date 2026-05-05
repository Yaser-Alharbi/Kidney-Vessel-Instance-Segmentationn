"""Augmentation pipelines (RGB / HED-only / Macenko-only / HED+Macenko).

Each builder returns an Albumentations-shaped callable:
    transform(image=uint8_hwc, mask=uint8_hw)
        -> {"image": float32 [3,H,W], "mask": int64 [H,W], ...}
"""

from __future__ import annotations

from typing import Any, Dict

import albumentations as A
import numpy as np
import torch
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# Ruifrok & Johnston (2001) RGB<-HED. DAB row is unused for PAS but
# keeps the matrix invertible.
_RGB_FROM_HED = np.array(
    [
        [0.65, 0.70, 0.29],
        [0.07, 0.99, 0.11],
        [0.27, 0.57, 0.78],
    ],
    dtype=np.float64,
)
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED)

# Macenko (2009) reference: stain matrix + 99th-pct concentrations.
_MACENKO_REF_STAIN = np.array(
    [
        [0.5626, 0.2159],
        [0.7201, 0.8012],
        [0.4062, 0.5581],
    ],
    dtype=np.float64,
)
_MACENKO_REF_MAX_C = np.array([1.9705, 1.0308], dtype=np.float64)
_MACENKO_IO = 240.0  # transmitted-light baseline


def _rgb_to_od(rgb: np.ndarray) -> np.ndarray:
    """OD = -log10((I + 1) / Io)."""
    rgb = rgb.astype(np.float64) + 1.0
    return -np.log10(np.maximum(rgb / _MACENKO_IO, 1e-6))


def _od_to_rgb(od: np.ndarray) -> np.ndarray:
    rgb = (10.0 ** (-od)) * _MACENKO_IO
    return np.clip(rgb, 0.0, 255.0)


class HEDJitter(ImageOnlyTransform):
    """Multiplicative + additive jitter in HED stain space (image only)."""

    def __init__(self, sigma: float = 0.05, p: float = 0.5):
        super().__init__(p=p)
        self.sigma = float(sigma)

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        if img.ndim != 3 or img.shape[-1] != 3:
            return img
        h, w, _ = img.shape
        flat = img.reshape(-1, 3)
        od = _rgb_to_od(flat)
        hed = od @ _HED_FROM_RGB
        alpha = 1.0 + np.random.uniform(-self.sigma, self.sigma, size=(1, 3))
        beta = np.random.uniform(-self.sigma, self.sigma, size=(1, 3))
        hed = hed * alpha + beta
        od_new = hed @ _RGB_FROM_HED
        rgb = _od_to_rgb(od_new).reshape(h, w, 3)
        return rgb.astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("sigma",)


class MacenkoNormalize(ImageOnlyTransform):
    """Macenko stain normalization to a fixed reference (image only).

    Falls back to the input image on degenerate tiles (SVD/lstsq fail).
    """

    def __init__(self, beta: float = 0.15, alpha_pct: float = 1.0, p: float = 1.0):
        super().__init__(p=p)
        self.beta = float(beta)
        self.alpha_pct = float(alpha_pct)

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        if img.ndim != 3 or img.shape[-1] != 3:
            return img
        h, w, _ = img.shape
        flat = img.reshape(-1, 3)
        od = _rgb_to_od(flat)

        # drop near-white pixels (glass/background)
        fg = od[(od > self.beta).any(axis=1)]
        if fg.shape[0] < 50:
            return img

        try:
            _, _, V = np.linalg.svd(
                fg - fg.mean(axis=0, keepdims=True), full_matrices=False
            )
        except np.linalg.LinAlgError:
            return img
        plane = V[:2].T  # top-2 singular vectors as 3x2 basis

        proj = fg @ plane
        phi = np.arctan2(proj[:, 1], proj[:, 0])
        lo, hi = np.percentile(phi, [self.alpha_pct, 100.0 - self.alpha_pct])
        v_lo = plane @ np.array([np.cos(lo), np.sin(lo)])
        v_hi = plane @ np.array([np.cos(hi), np.sin(hi)])

        # H-like (more red-absorbing) stain in column 0
        if v_lo[0] > v_hi[0]:
            stain = np.column_stack([v_lo, v_hi])
        else:
            stain = np.column_stack([v_hi, v_lo])
        norms = np.linalg.norm(stain, axis=0, keepdims=True)
        stain = stain / (norms + 1e-8)

        try:
            conc, *_ = np.linalg.lstsq(stain, od.T, rcond=None)
        except np.linalg.LinAlgError:
            return img

        max_c = np.maximum(np.percentile(conc, 99.0, axis=1), 1e-6)
        conc = conc * (_MACENKO_REF_MAX_C[:, None] / max_c[:, None])
        od_norm = (_MACENKO_REF_STAIN @ conc).T
        return _od_to_rgb(od_norm).reshape(h, w, 3).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("beta", "alpha_pct")


class _MaskedCompose:
    """Wrap a Compose; coerce mask to int64 (0, 1)."""

    def __init__(self, compose: A.Compose) -> None:
        self.compose = compose

    def __call__(self, *, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        out = self.compose(image=image, mask=mask)
        m = out["mask"]
        if isinstance(m, torch.Tensor):
            out["mask"] = (m > 0).long()
        else:
            out["mask"] = (np.asarray(m) > 0).astype(np.int64)
        return out


def get_rgb_aug_transforms(image_size: int = 512, train: bool = True) -> _MaskedCompose:
    """RGB control: flips + rotate90 + brightness/contrast."""
    if train:
        ops = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    else:
        ops = [
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    return _MaskedCompose(A.Compose(ops))


def get_hed_only_transforms(
    image_size: int = 512, train: bool = True
) -> _MaskedCompose:
    """Ablation: HED jitter on train, no Macenko on eval."""
    if train:
        ops = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            HEDJitter(sigma=0.05, p=0.8),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    else:
        ops = [
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    return _MaskedCompose(A.Compose(ops))


def get_macenko_only_transforms(
    image_size: int = 512, train: bool = True
) -> _MaskedCompose:
    """Isolates Macenko: stain normalization on train + eval, no HED jitter.

    Reuses the same fixed reference matrix as full_stain_aware via
    `MacenkoNormalize` (defined above with `_MACENKO_REF_STAIN` /
    `_MACENKO_REF_MAX_C`).
    """
    if train:
        ops = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            MacenkoNormalize(p=1.0),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    else:
        ops = [
            MacenkoNormalize(p=1.0),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    return _MaskedCompose(A.Compose(ops))


def get_full_stain_aware_transforms(
    image_size: int = 512, train: bool = True
) -> _MaskedCompose:
    """HED jitter on train + Macenko on eval (combined arm)."""
    if train:
        ops = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            HEDJitter(sigma=0.05, p=0.8),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    else:
        ops = [
            MacenkoNormalize(p=1.0),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    return _MaskedCompose(A.Compose(ops))


# back-compat aliases (older code paths used the longer names)
get_stain_aware_hed_only_transforms = get_hed_only_transforms
get_stain_aware_transforms = get_full_stain_aware_transforms
