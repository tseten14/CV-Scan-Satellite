"""Non-maximum suppression for building detections."""

from __future__ import annotations


def _bbox_iou(a: dict, b: dict) -> float:
    ax0, ay0, ax1, ay1 = a["xmin"], a["ymin"], a["xmax"], a["ymax"]
    bx0, by0, bx1, by1 = b["xmin"], b["ymin"], b["xmax"], b["ymax"]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_detections(detections: list[dict], iou_threshold: float = 0.55) -> list[dict]:
    """Suppress overlapping building detections, keeping highest confidence boxes."""
    if not detections:
        return []
    sorted_dets = sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True)
    kept: list[dict] = []
    for det in sorted_dets:
        bbox = det.get("bbox")
        if not bbox:
            kept.append(det)
            continue
        suppress = False
        for existing in kept:
            existing_bbox = existing.get("bbox")
            if existing_bbox and _bbox_iou(bbox, existing_bbox) >= iou_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(det)
    return kept
