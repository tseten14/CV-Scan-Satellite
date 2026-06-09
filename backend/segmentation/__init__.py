"""Lightweight building segmentation (SMP U-Net) — training and inference."""

from segmentation.config import SegmentationConfig
from segmentation.inference import predict_mask
from segmentation.model import DiceBCELoss, create_model

__all__ = [
    "SegmentationConfig",
    "predict_mask",
    "DiceBCELoss",
    "create_model",
]
