"""Default hyperparameters and paths for building segmentation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_WEIGHTS = _BACKEND_ROOT / "segmentation" / "weights" / "building_unet_best.pt"


@dataclass(frozen=True)
class SegmentationConfig:
    """Configuration for SMP U-Net building segmentation."""

    backbone: str = "efficientnet-b0"
    encoder_weights: str = "imagenet"
    in_channels: int = 3
    classes: int = 1
    image_size: int = 512
    threshold: float = 0.5
    model_path: Path = field(default_factory=lambda: _DEFAULT_WEIGHTS)

    @classmethod
    def from_env(cls) -> SegmentationConfig:
        """Build config from environment overrides."""
        backbone = (os.environ.get("SMP_BACKBONE") or "efficientnet-b0").strip()
        if backbone not in {"efficientnet-b0", "resnet34"}:
            backbone = "efficientnet-b0"

        try:
            image_size = int((os.environ.get("SMP_IMAGE_SIZE") or "512").strip())
        except ValueError:
            image_size = 512
        image_size = max(256, min(1024, image_size))

        try:
            threshold = float((os.environ.get("SMP_MASK_THRESHOLD") or "0.5").strip())
        except ValueError:
            threshold = 0.5
        threshold = max(0.1, min(0.9, threshold))

        weights_raw = (os.environ.get("SMP_MODEL_PATH") or "").strip()
        model_path = Path(weights_raw) if weights_raw else _DEFAULT_WEIGHTS

        return cls(
            backbone=backbone,
            image_size=image_size,
            threshold=threshold,
            model_path=model_path,
        )
