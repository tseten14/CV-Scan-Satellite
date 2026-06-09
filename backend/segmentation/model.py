"""SMP U-Net model factory and combined Dice + BCE loss."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

BackboneName = Literal["efficientnet-b0", "resnet34"]


class DiceBCELoss(nn.Module):
    """Combined Dice and binary cross-entropy loss for imbalanced building masks."""

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1.0) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model output (B, 1, H, W), unnormalized.
            targets: Binary mask (B, 1, H, W) in {0, 1}.
        """
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice_den = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice = 1.0 - ((2.0 * intersection + self.smooth) / (dice_den + self.smooth)).mean()
        return self.dice_weight * dice + self.bce_weight * bce


def create_model(
    backbone: BackboneName = "efficientnet-b0",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
) -> nn.Module:
    """Initialize a U-Net with the requested SMP encoder backbone."""
    return smp.Unet(
        encoder_name=backbone,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )
