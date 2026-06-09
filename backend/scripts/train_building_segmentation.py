#!/usr/bin/env python3
"""
Train a lightweight U-Net building segmentation model.

Usage::

    python scripts/train_building_segmentation.py \\
        --images-dir /path/to/images \\
        --masks-dir /path/to/masks \\
        --epochs 40 \\
        --backbone efficientnet-b0

Expects paired chips::

    images/chip_001.png
    masks/chip_001.png   # white=building, black=background
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from segmentation.config import SegmentationConfig
from segmentation.dataset import SatelliteBuildingDataset
from segmentation.model import DiceBCELoss, create_model

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SMP U-Net building segmentation")
    parser.add_argument("--images-dir", required=True, help="Directory of RGB satellite chips")
    parser.add_argument("--masks-dir", required=True, help="Directory of binary building masks")
    parser.add_argument("--output-dir", default=str(BACKEND_ROOT / "segmentation" / "weights"))
    parser.add_argument("--backbone", choices=["efficientnet-b0", "resnet34"], default="efficientnet-b0")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def resolve_device(name: str) -> str:
    if name == "cpu":
        return "cpu"
    if name == "cuda" and torch.cuda.is_available():
        return "cuda"
    if name == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: DiceBCELoss,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    count = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        count += 1
    return total_loss / max(count, 1)


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: DiceBCELoss,
    device: str,
) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += float(loss.item())
        count += 1
    return total_loss / max(count, 1)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    device = resolve_device(args.device)
    logger.info("Training on device: %s", device)

    dataset = SatelliteBuildingDataset(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        image_size=args.image_size,
        is_train=True,
    )
    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = create_model(backbone=args.backbone).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "building_unet_best.pt"

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        logger.info("Epoch %d/%d  train=%.4f  val=%.4f", epoch, args.epochs, train_loss, val_loss)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "image_size": args.image_size,
                    "val_loss": val_loss,
                },
                best_path,
            )
            logger.info("Saved best checkpoint to %s", best_path)

    logger.info("Training complete. Best val loss: %.4f", best_val)


if __name__ == "__main__":
    main()
