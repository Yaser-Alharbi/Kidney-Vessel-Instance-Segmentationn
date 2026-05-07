"""Segmentation losses for Phase 3.

Default is a Dice + soft BCE combo. Vessels cover ~4.5% of pixels,
so plain BCE underperforms; Dice aligns with the val metric, BCE
keeps gradients smooth early on.

    criterion = DiceBCELoss(dice_weight=0.5)
    logits = model(image)              # [B, 1, H, W]
    loss   = criterion(logits, mask)   # mask: [B, H, W] in {0, 1}
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    """w * Dice + (1 - w) * soft BCE on logits."""

    def __init__(self, dice_weight: float = 0.5):
        super().__init__()
        if not 0.0 <= dice_weight <= 1.0:
            raise ValueError(f"dice_weight must be in [0, 1], got {dice_weight}")
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.bce = smp.losses.SoftBCEWithLogitsLoss()
        self.w = dice_weight

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return self.w * self.dice(logits, target) + (1.0 - self.w) * self.bce(
            logits, target
        )
