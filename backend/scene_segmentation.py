"""
Scene segmentation using HSV color analysis.
Identifies vegetation, roads/sidewalks, sky, and grass in outdoor images.
Returns bounding boxes for each detected region.
"""
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

# HSV range definitions for scene classes
SCENE_CLASSES: List[Dict[str, Any]] = [
    {
        "label": "Vegetation",
        "hsv_ranges": [
            ((30, 40, 30), (90, 255, 255)),   # green foliage
        ],
        "min_area_ratio": 0.02,
        "position_filter": None,
    },
    {
        "label": "Sky",
        "hsv_ranges": [
            ((90, 20, 160), (135, 255, 255)),  # blue sky
            ((0, 0, 200), (180, 30, 255)),      # white/overcast sky
        ],
        "min_area_ratio": 0.03,
        "position_filter": "top_half",
    },
    {
        "label": "Road",
        "hsv_ranges": [
            ((0, 0, 60), (180, 35, 150)),  # gray asphalt (tight saturation)
        ],
        "min_area_ratio": 0.04,
        "position_filter": "bottom_40",
    },
    {
        "label": "Grass",
        "hsv_ranges": [
            ((30, 30, 25), (85, 255, 200)),  # green ground-level
        ],
        "min_area_ratio": 0.02,
        "position_filter": "bottom_60",
    },
    {
        "label": "Sidewalk",
        "hsv_ranges": [
            ((0, 0, 130), (180, 40, 230)),  # light gray concrete
        ],
        "min_area_ratio": 0.015,
        "position_filter": "bottom_half",
    },
]


def _build_mask(
    hsv: np.ndarray,
    ranges: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]],
    h: int,
    w: int,
    position_filter: str | None,
) -> np.ndarray:
    """Build a combined binary mask from HSV ranges + position filter."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))

    if position_filter == "top_half":
        mask[h // 2 :, :] = 0
    elif position_filter == "bottom_half":
        mask[: h // 3, :] = 0
    elif position_filter == "bottom_40":
        mask[: int(h * 0.6), :] = 0
    elif position_filter == "bottom_60":
        mask[: int(h * 0.4), :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _mask_to_boxes(
    mask: np.ndarray, label: str, min_area: float, w: int, h: int
) -> List[Dict[str, Any]]:
    """Convert a binary mask into bounding-box detections."""
    num_cc, cc_labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes: List[Dict[str, Any]] = []

    for cc_id in range(1, num_cc):
        area = stats[cc_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        x = stats[cc_id, cv2.CC_STAT_LEFT]
        y = stats[cc_id, cv2.CC_STAT_TOP]
        bw = stats[cc_id, cv2.CC_STAT_WIDTH]
        bh = stats[cc_id, cv2.CC_STAT_HEIGHT]

        coverage = area / (bw * bh) if (bw * bh) > 0 else 0
        if coverage < 0.15:
            continue

        # Road and Sidewalk must be wider than tall (horizontal surfaces)
        if label in ("Road", "Sidewalk") and bh > bw * 0.8:
            continue

        confidence = min(0.95, 0.50 + (area / (w * h)) * 2.0)

        boxes.append({
            "id": "",
            "label": label,
            "confidence": round(confidence, 3),
            "bbox": {
                "xmin": int(x),
                "ymin": int(y),
                "xmax": int(x + bw),
                "ymax": int(y + bh),
            },
        })

    boxes.sort(key=lambda b: b["confidence"], reverse=True)
    return boxes[:3]


def _resolve_overlapping_classes(all_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Grass vs Vegetation and Road vs Sidewalk can overlap.
    Keep the higher-confidence one when IoU > 0.5.
    """
    kept: List[Dict[str, Any]] = []
    for det in all_detections:
        dominated = False
        for k in kept:
            bx = det["bbox"]
            kx = k["bbox"]
            ix1 = max(bx["xmin"], kx["xmin"])
            iy1 = max(bx["ymin"], kx["ymin"])
            ix2 = min(bx["xmax"], kx["xmax"])
            iy2 = min(bx["ymax"], kx["ymax"])
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                area_det = (bx["xmax"] - bx["xmin"]) * (bx["ymax"] - bx["ymin"])
                area_k = (kx["xmax"] - kx["xmin"]) * (kx["ymax"] - kx["ymin"])
                iou = inter / (area_det + area_k - inter) if (area_det + area_k - inter) > 0 else 0
                if iou > 0.5:
                    dominated = True
                    break
        if not dominated:
            kept.append(det)
    return kept


def segment_scene(image_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Run HSV-based scene segmentation on an image.
    Returns a list of detection dicts (same format as COCO detections).
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    h, w = img.shape[:2]
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    all_detections: List[Dict[str, Any]] = []

    for sc in SCENE_CLASSES:
        mask = _build_mask(hsv, sc["hsv_ranges"], h, w, sc["position_filter"])
        min_area = w * h * sc["min_area_ratio"]
        boxes = _mask_to_boxes(mask, sc["label"], min_area, w, h)
        all_detections.extend(boxes)

    all_detections.sort(key=lambda d: d["confidence"], reverse=True)
    return _resolve_overlapping_classes(all_detections)
