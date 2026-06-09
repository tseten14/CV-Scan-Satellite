"""FastAPI service wrapper for SMP building segmentation (satellite mode)."""

from __future__ import annotations

import io
import logging

from PIL import Image

from pipeline import run_building_pipeline

logger = logging.getLogger("uvicorn.error")


def run_smp_building_detection(image_bytes: bytes, mode: str = "satellite") -> dict:
    """
    Run lightweight U-Net building detection compatible with the frontend API.

    Satellite mode uses the full infer → vectorize pipeline. Streetview mode is
    unsupported and returns an empty result with a warning in logs.
    """
    if mode != "satellite":
        logger.warning("SMP building engine only supports satellite mode; returning empty detections.")
        image = Image.open(io.BytesIO(image_bytes))
        w, h = image.size
        return {
            "image_width": w,
            "image_height": h,
            "detections": [],
            "processing_time_s": 0.0,
            "engine": "smp",
        }

    try:
        result = run_building_pipeline(image_bytes)
        return result["detection_result"]
    except FileNotFoundError as exc:
        raise RuntimeError(str(exc)) from exc
