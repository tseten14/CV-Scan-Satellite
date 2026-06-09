"""Fast inference hook for building segmentation masks."""

from __future__ import annotations

import logging
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from segmentation.config import SegmentationConfig
from segmentation.model import create_model

logger = logging.getLogger(__name__)

_loaded_models: dict[str, tuple[torch.nn.Module, str]] = {}


def _build_inference_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def _resolve_device() -> str:
    raw = (__import__("os").environ.get("SMP_DEVICE") or "auto").strip().lower()
    if raw == "cpu":
        return "cpu"
    if raw == "cuda" and torch.cuda.is_available():
        return "cuda"
    if raw == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if raw == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def load_model(model_path: str | Path, config: SegmentationConfig | None = None) -> torch.nn.Module:
    """Load a trained U-Net checkpoint onto the configured device."""
    cfg = config or SegmentationConfig.from_env()
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Segmentation weights not found at {path}. "
            "Train with scripts/train_building_segmentation.py or set SMP_MODEL_PATH."
        )

    cache_key = str(path.resolve())
    if cache_key in _loaded_models:
        return _loaded_models[cache_key][0]

    device = _resolve_device()
    model = create_model(
        backbone=cfg.backbone,  # type: ignore[arg-type]
        encoder_weights=None,
        in_channels=cfg.in_channels,
        classes=cfg.classes,
    )
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    _loaded_models[cache_key] = (model, device)
    logger.info("Loaded segmentation model from %s on %s", path, device)
    return model


def predict_mask(
    image_array: np.ndarray,
    model_path: str | Path,
    *,
    config: SegmentationConfig | None = None,
    threshold: float | None = None,
) -> np.ndarray:
    """
    Run fast building segmentation inference on an RGB image array.

    Args:
        image_array: HxWx3 uint8 RGB image.
        model_path: Path to a `.pt` checkpoint saved by the training script.
        config: Optional segmentation config overrides.
        threshold: Probability threshold for binarization (default from config).

    Returns:
        HxW uint8 binary mask with values {0, 255}. Empty (all zeros) when no
        buildings are detected above threshold.
    """
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("image_array must be HxWx3 RGB")

    cfg = config or SegmentationConfig.from_env()
    model = load_model(model_path, cfg)
    _, device = _loaded_models[str(Path(model_path).resolve())]

    orig_h, orig_w = image_array.shape[:2]
    transform = _build_inference_transform(cfg.image_size)
    tensor = transform(image=image_array)["image"].unsqueeze(0).to(device)

    prob_threshold = cfg.threshold if threshold is None else threshold

    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mask_small = (probs >= prob_threshold).astype(np.uint8) * 255
    if mask_small.max() == 0:
        return np.zeros((orig_h, orig_w), dtype=np.uint8)

    mask = cv2.resize(mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return mask
