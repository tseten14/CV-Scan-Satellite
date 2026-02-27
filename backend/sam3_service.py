
"""
SAM 3 (Segment Anything Model 3) detection service.
Uses Hugging Face transformers for promptable concept segmentation.
"""
import os
import time
import logging
from typing import Any

from PIL import Image

logger = logging.getLogger("uvicorn.error")

# Prompts for urban accessibility / infrastructure mapping (road first for priority)
DEFAULT_PROMPTS = [
    "road",
    "sidewalk",
    "building",
    "door",
    "car",
    "person",
    "bicycle",
    "truck",
    "bus",
    "motorcycle",
    "traffic light",
    "trash can",
    "pole",
    "bench",
    "sign",
    "fire hydrant",
    "street light",
    "mailbox",
    "vegetation",
    "grass",
    "tree",
]

_model: Any = None
_processor: Any = None
_device: str = "cpu"


def _get_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
    except Exception:
        pass
    return "cpu"


def load_sam3() -> bool:
    """Load SAM 3 model and processor. Returns True on success."""
    global _model, _processor, _device
    if _model is not None:
        return True

    try:
        import torch
        from transformers import Sam3Model, Sam3Processor

        _device = _get_device()
        logger.info(f"Loading SAM 3 on {_device}…")

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token

        _processor = Sam3Processor.from_pretrained(
            "facebook/sam3",
            token=token,
        )
        _model = Sam3Model.from_pretrained(
            "facebook/sam3",
            token=token,
        ).to(_device)

        logger.info("SAM 3 ready.")
        return True
    except Exception as e:
        logger.error(f"SAM 3 failed to load: {e}")
        return False


def _iou(box_a: dict, box_b: dict) -> float:
    """Compute IoU between two boxes (xmin, ymin, xmax, ymax)."""
    ix1 = max(box_a["xmin"], box_b["xmin"])
    iy1 = max(box_a["ymin"], box_b["ymin"])
    ix2 = min(box_a["xmax"], box_b["xmax"])
    iy2 = min(box_a["ymax"], box_b["ymax"])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (box_a["xmax"] - box_a["xmin"]) * (box_a["ymax"] - box_a["ymin"])
    area_b = (box_b["xmax"] - box_b["xmin"]) * (box_b["ymax"] - box_b["ymin"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms(detections: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """Non-maximum suppression by confidence. Road/sidewalk can coexist (adjacent)."""
    if not detections:
        return []
    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    keep: list[dict] = []
    for det in sorted_dets:
        keep_it = True
        for k in keep:
            iou = _iou(det["bbox"], k["bbox"])
            thresh = iou_threshold
            # Road and sidewalk are adjacent - require higher overlap to suppress
            if {det["label"], k["label"]} <= {"road", "sidewalk"}:
                thresh = 0.85
            if iou > thresh:
                keep_it = False
                break
        if keep_it:
            keep.append(det)
    return keep


def _overlap_ratio(box_a: dict, box_b: dict) -> float:
    """Intersection over box_a area (how much of A is covered by B)."""
    ix1 = max(box_a["xmin"], box_b["xmin"])
    iy1 = max(box_a["ymin"], box_b["ymin"])
    ix2 = min(box_a["xmax"], box_b["xmax"])
    iy2 = min(box_a["ymax"], box_b["ymax"])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (box_a["xmax"] - box_a["xmin"]) * (box_a["ymax"] - box_a["ymin"])
    return inter / area_a if area_a > 0 else 0.0


def _filter_person_building_overlap(detections: list[dict], img_w: int, img_h: int) -> list[dict]:
    """Remove person detections that incorrectly include building (large overlap with building)."""
    buildings = [d for d in detections if d["label"] == "building"]
    persons = [d for d in detections if d["label"] == "person"]
    others = [d for d in detections if d["label"] not in ("building", "person")]

    img_area = img_w * img_h
    filtered_persons: list[dict] = []
    for p in persons:
        p_area = (p["bbox"]["xmax"] - p["bbox"]["xmin"]) * (p["bbox"]["ymax"] - p["bbox"]["ymin"])
        # Remove if person bbox is unrealistically large (>18% of image - likely building)
        if p_area > 0.18 * img_area:
            continue
        # Remove if person overlaps significantly with any building (building bleeding into person)
        overlap_any = any(_overlap_ratio(p["bbox"], b["bbox"]) > 0.25 for b in buildings)
        if overlap_any:
            continue
        filtered_persons.append(p)

    return buildings + filtered_persons + others


def _filter_google_map_signs(detections: list[dict]) -> list[dict]:
    """Remove sign detections that are Google Street View nav arrows (on road surface)."""
    roads = [d for d in detections if d["label"] == "road"]
    signs = [d for d in detections if d["label"] == "sign"]
    others = [d for d in detections if d["label"] not in ("sign", "road")]

    filtered_signs: list[dict] = []
    for s in signs:
        # Skip signs on road surface - Google nav arrows are overlaid on pavement
        if any(_overlap_ratio(s["bbox"], r["bbox"]) > 0.25 for r in roads):
            continue
        filtered_signs.append(s)

    return roads + filtered_signs + others


def _filter_car_doors(detections: list[dict]) -> list[dict]:
    """Remove door detections that overlap cars or trucks (car doors, not building entrances)."""
    vehicles = [d for d in detections if d["label"] in ("car", "truck")]
    doors = [d for d in detections if d["label"] == "door"]
    others = [d for d in detections if d["label"] not in ("door", "car", "truck")]

    filtered_doors: list[dict] = []
    for door in doors:
        # Skip doors that overlap significantly with a vehicle (car doors)
        if any(_overlap_ratio(door["bbox"], v["bbox"]) > 0.4 for v in vehicles):
            continue
        filtered_doors.append(door)

    return others + vehicles + filtered_doors


def _merge_sidewalk_detections(detections: list[dict], img_h: int) -> list[dict]:
    """Keep at most 2 sidewalk detections (largest by area)."""
    sidewalks = [d for d in detections if d["label"] == "sidewalk"]
    others = [d for d in detections if d["label"] != "sidewalk"]

    if len(sidewalks) <= 2:
        return detections

    def bbox_area(d: dict) -> float:
        b = d["bbox"]
        return (b["xmax"] - b["xmin"]) * (b["ymax"] - b["ymin"])

    # Keep only the 2 largest sidewalk regions
    sorted_sw = sorted(sidewalks, key=bbox_area, reverse=True)
    return others + sorted_sw[:2]


def _min_area(bbox: dict, min_pixels: int = 800) -> bool:
    w = bbox["xmax"] - bbox["xmin"]
    h = bbox["ymax"] - bbox["ymin"]
    return w * h >= min_pixels


def _clip_polygon_to_bounds(pts: list[list[float]], img_w: int, img_h: int) -> list[list[float]] | None:
    """Clip polygon points to image bounds so outlines stay inside the frame."""
    if not pts:
        return None
    clipped = []
    for x, y in pts:
        cx = max(0.0, min(float(img_w), x))
        cy = max(0.0, min(float(img_h), y))
        clipped.append([cx, cy])
    # Remove consecutive duplicates
    deduped = [clipped[0]]
    for p in clipped[1:]:
        if abs(p[0] - deduped[-1][0]) > 1e-6 or abs(p[1] - deduped[-1][1]) > 1e-6:
            deduped.append(p)
    if len(deduped) < 3:
        return None
    return deduped


def _mask_to_polygon(mask, img_w: int, img_h: int) -> list[list[float]] | None:
    """Extract smooth, well-aligned polygon contour from binary mask (penguin-quality)."""
    import cv2
    import numpy as np

    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.ndim > 2:
        arr = arr.squeeze()
    if arr.ndim != 2:
        return None
    if arr.dtype != np.uint8:
        arr = (arr > 0.5).astype(np.uint8)
    # Crop mask to image dimensions so contours never extend outside
    h_lim, w_lim = min(arr.shape[0], img_h), min(arr.shape[1], img_w)
    arr = arr[:h_lim, :w_lim]
    # Upsample mask 2x for smoother contour extraction (reduces jagged pixel edges)
    mh, mw = arr.shape[0], arr.shape[1]
    if max(mw, mh) < 512:
        scale = 2
        arr = cv2.resize(arr, (mw * scale, mh * scale), interpolation=cv2.INTER_LINEAR)
        arr = (arr > 0.5).astype(np.uint8)
        mh, mw = arr.shape[0], arr.shape[1]
    # Light morphological closing (3x3) to close small gaps without blurring edges
    kernel = np.ones((3, 3), np.uint8)
    arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel)
    # Use CHAIN_APPROX_NONE to get full contour for smooth outlines
    contours, _ = cv2.findContours(
        arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 3:
        return None
    # Finer simplification (0.08% of perimeter) - more points for accurate placement
    peri = cv2.arcLength(cnt, True)
    epsilon = max(0.5, peri * 0.0008)
    cnt = cv2.approxPolyDP(cnt, epsilon, True)
    if len(cnt) < 3:
        return None
    pts = cnt.reshape(-1, 2).tolist()
    pts = [[float(x), float(y)] for x, y in pts]
    # Scale to image coords (mask may be upsampled or different size)
    sx = img_w / max(1, mw)
    sy = img_h / max(1, mh)
    pts = [[x * sx, y * sy] for x, y in pts]
    # Clip to image bounds so polygons never extend outside the frame
    return _clip_polygon_to_bounds(pts, img_w, img_h)


def run_detection(image_bytes: bytes) -> dict:
    """
    Run SAM 3 detection on image. Returns dict compatible with DetectionResult:
    { image_width, image_height, detections, processing_time_ms }
    """
    if not load_sam3():
        raise RuntimeError("SAM 3 model not loaded")

    import torch

    start = time.perf_counter()
    image = Image.open(__import__("io").BytesIO(image_bytes)).convert("RGB")
    w, h = image.size

    all_dets: list[dict] = []
    confidence_threshold = 0.5

    for prompt in DEFAULT_PROMPTS:
        try:
            inputs = _processor(
                images=image,
                text=prompt,
                return_tensors="pt",
            ).to(_device)

            with torch.no_grad():
                outputs = _model(**inputs)

            target_sizes = inputs.get("original_sizes")
            if target_sizes is not None and hasattr(target_sizes, "tolist"):
                target_sizes = target_sizes.tolist()
            else:
                target_sizes = [[h, w]]

            results = _processor.post_process_instance_segmentation(
                outputs,
                threshold=confidence_threshold,
                mask_threshold=0.5,
                target_sizes=target_sizes,
            )[0]

            boxes = results.get("boxes", [])
            scores = results.get("scores", [])
            masks = results.get("masks", [])

            for i, (box, score) in enumerate(zip(boxes, scores)):
                if score < confidence_threshold:
                    continue
                x1, y1, x2, y2 = box.tolist()
                bbox = {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2}
                if not _min_area(bbox):
                    continue
                polygon = None
                if i < len(masks):
                    mask = masks[i]
                    if hasattr(mask, "cpu"):
                        mask = mask.cpu().numpy()
                    polygon = _mask_to_polygon(mask, w, h)
                all_dets.append({
                    "label": prompt,
                    "confidence": float(score),
                    "bbox": bbox,
                    "polygon": polygon,
                })
        except Exception as e:
            logger.debug(f"Prompt '{prompt}' failed: {e}")
            continue

    all_dets = _nms(all_dets, iou_threshold=0.6)
    all_dets = _filter_person_building_overlap(all_dets, w, h)
    all_dets = _filter_google_map_signs(all_dets)
    all_dets = _filter_car_doors(all_dets)
    all_dets = _merge_sidewalk_detections(all_dets, h)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    detections = []
    for i, d in enumerate(all_dets):
        det = {
            "id": f"det_{i}",
            "label": d["label"],
            "confidence": d["confidence"],
            "bbox": d["bbox"],
        }
        if d.get("polygon"):
            det["polygon"] = d["polygon"]
        detections.append(det)

    return {
        "image_width": w,
        "image_height": h,
        "detections": detections,
        "processing_time_ms": elapsed_ms,
    }
