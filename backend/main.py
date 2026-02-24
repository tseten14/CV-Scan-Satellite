"""
FastAPI backend for scene + entrance detection.
Exposes /detect for uploaded images — returns COCO objects + door detections.
"""
import os
import logging

if os.environ.get("TFHUB_INSECURE") or os.environ.get("PYTHONHTTPSVERIFY") == "0":
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from door_detection import detect_doors
from inference import run_detection
from model_loader import load_model
from scene_segmentation import segment_scene

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Loading COCO detection model (first time may download)…")
        load_model()
        logger.info("COCO model ready.")
    except Exception as e:
        logger.warning(f"COCO model failed to load: {e}  — will use door-only mode")
    yield


app = FastAPI(
    title="CV-SCAN-GEOAI Detection API",
    description="Scene + entrance detection for facade / street-view images",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8080", "http://127.0.0.1:5173", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


def _iou_overlap(bx: dict, kx: dict, threshold: float = 0.5) -> bool:
    ix1, iy1 = max(bx["xmin"], kx["xmin"]), max(bx["ymin"], kx["ymin"])
    ix2, iy2 = min(bx["xmax"], kx["xmax"]), min(bx["ymax"], kx["ymax"])
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    inter = (ix2 - ix1) * (iy2 - iy1)
    area = (bx["xmax"] - bx["xmin"]) * (bx["ymax"] - bx["ymin"])
    return area > 0 and inter / area > threshold


def _merge_all(
    door_result: dict,
    coco_result: dict | None,
    scene_dets: list,
) -> dict:
    """Merge door + COCO + scene detections, deduplicating overlaps."""
    all_dets: list = []
    idx = 0

    for d in door_result.get("detections", []):
        d["id"] = f"det_{idx}"
        all_dets.append(d)
        idx += 1

    existing_boxes = [d["bbox"] for d in all_dets]

    for d in (coco_result or {}).get("detections", []):
        if not any(_iou_overlap(d["bbox"], eb) for eb in existing_boxes):
            d["id"] = f"det_{idx}"
            all_dets.append(d)
            existing_boxes.append(d["bbox"])
            idx += 1

    for d in scene_dets:
        if not any(_iou_overlap(d["bbox"], eb, 0.6) for eb in existing_boxes):
            d["id"] = f"det_{idx}"
            all_dets.append(d)
            existing_boxes.append(d["bbox"])
            idx += 1

    total_ms = door_result["processing_time_ms"]
    if coco_result:
        total_ms += coco_result.get("processing_time_ms", 0)

    return {
        "image_width": door_result["image_width"],
        "image_height": door_result["image_height"],
        "detections": all_dets,
        "processing_time_ms": total_ms,
    }


@app.post("/detect")
async def detect_entrances(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (jpeg, png, webp)")

    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(400, f"Failed to read file: {e}")

    if len(image_bytes) == 0:
        raise HTTPException(400, "Empty file")

    try:
        door_result = detect_doors(image_bytes)

        coco_result = None
        try:
            coco_result = run_detection(image_bytes)
        except Exception as e:
            logger.warning(f"COCO detection skipped: {e}")

        scene_dets: list = []
        try:
            scene_dets = segment_scene(image_bytes)
        except Exception as e:
            logger.warning(f"Scene segmentation skipped: {e}")

        return _merge_all(door_result, coco_result, scene_dets)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Detection failed: {str(e)}")
