"""PyTorch Dataset for satellite image chips and binary building masks."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def default_train_transforms(image_size: int = 512) -> A.Compose:
    """Augmentations for robust training on satellite imagery."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )


def default_val_transforms(image_size: int = 512) -> A.Compose:
    """Deterministic transforms for validation and inference-style loading."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )


class SatelliteBuildingDataset(Dataset):
    """
    Loads paired satellite chips and binary building masks.

    Expected layout::

        images_dir/
            chip_001.png
        masks_dir/
            chip_001.png   # same stem, white=building, black=background

    Mask values are thresholded to {0, 1}.
    """

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        transform: Callable | None = None,
        image_size: int = 512,
        is_train: bool = True,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform or (
            default_train_transforms(image_size) if is_train else default_val_transforms(image_size)
        )

        self.samples: list[tuple[Path, Path]] = []
        for image_path in sorted(self.images_dir.iterdir()):
            if image_path.suffix.lower() not in self.IMAGE_EXTENSIONS:
                continue
            mask_path = self._resolve_mask_path(image_path)
            if mask_path is not None:
                self.samples.append((image_path, mask_path))

        if not self.samples:
            raise ValueError(
                f"No paired image/mask samples found in {self.images_dir} and {self.masks_dir}"
            )

    def _resolve_mask_path(self, image_path: Path) -> Path | None:
        for ext in self.IMAGE_EXTENSIONS:
            candidate = self.masks_dir / f"{image_path.stem}{ext}"
            if candidate.exists():
                return candidate
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image_path, mask_path = self.samples[index]

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        mask = (mask > 127).astype(np.float32)

        augmented = self.transform(image=image, mask=mask)
        image_tensor: torch.Tensor = augmented["image"]
        mask_tensor = augmented["mask"].unsqueeze(0).float()
        return {"image": image_tensor, "mask": mask_tensor, "path": str(image_path)}
