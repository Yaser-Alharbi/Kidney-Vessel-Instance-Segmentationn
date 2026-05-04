"""Model factory. Forward output is logits [B, 1, H, W] (no activation).

    unet_resnet34       smp.Unet + ResNet-34 (ImageNet)
    segformer_mit_b0    smp.Segformer + MiT-B0 (ImageNet)
"""

from __future__ import annotations

from typing import Dict

import segmentation_models_pytorch as smp
import torch.nn as nn


_BUILDERS: Dict[str, callable] = {
    "unet_resnet34": lambda in_channels, out_classes: smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=out_classes,
        activation=None,
    ),
    "segformer_mit_b0": lambda in_channels, out_classes: smp.Segformer(
        encoder_name="mit_b0",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=out_classes,
        activation=None,
    ),
}


def build_model(name: str, in_channels: int = 3, out_classes: int = 1) -> nn.Module:
    """Build a model by name."""
    if name not in _BUILDERS:
        raise ValueError(
            f"unknown model name '{name}'. expected one of {sorted(_BUILDERS)}"
        )
    return _BUILDERS[name](in_channels, out_classes)
