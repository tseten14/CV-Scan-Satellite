"""
Door/entrance detection using OpenCV color segmentation.
Segments the image into color regions via k-means, then finds
tall rectangular regions that differ from the dominant facade color.
"""
import time
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple


def _decode_image(image_bytes: bytes) -> Tuple[np.ndarray, int, int]:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    h, w = img.shape[:2]
    return img, w, h


def _score_candidate(
    ox: int, oy: int, ocw: int, och: int,
    real_area: float, solidity: float, color_dist: float,
    w: int, h: int,
) -> float:
    bottom_edge = oy + och
    top_edge = oy
    center_x = ox + ocw / 2
    aspect = och / ocw if ocw > 0 else 0

    # Hard filters
    bottom_ratio = bottom_edge / h
    top_ratio = top_edge / h
    if bottom_ratio < 0.45:
        return 0.0  # top-half = windows
    if top_ratio < 0.08:
        return 0.0  # spans from top = whole facade

    # --- Door-like shape ---
    # Single doors: aspect 1.5-3.0 (ideal ~2.0)
    # Double doors: aspect 0.6-1.5
    # Bushes/landscaping: aspect < 0.8, wide and short
    if aspect >= 1.3:
        shape_score = 1.0  # tall rectangle = very door-like
    elif aspect >= 0.8:
        shape_score = 0.7  # could be double doors
    elif aspect >= 0.5:
        shape_score = 0.4  # wide, less door-like
    else:
        shape_score = 0.1  # very wide = likely not a door

    # --- Position ---
    # Horizontal center is strong signal (doors are usually centered)
    horiz_score = 1.0 - abs(center_x - w / 2) / (w / 2)

    # Vertical: door should be in the middle-to-lower portion
    center_y = oy + och / 2
    vert_center_ratio = center_y / h
    if 0.35 <= vert_center_ratio <= 0.70:
        vert_score = 1.0  # sweet spot
    elif vert_center_ratio > 0.70:
        vert_score = 0.6  # low in image, could be ground-level stuff
    else:
        vert_score = 0.3

    size_score = min(real_area / (w * h * 0.03), 1.0)

    return (
        shape_score * 0.30
        + horiz_score * 0.25
        + vert_score * 0.15
        + size_score * 0.10
        + color_dist * 0.10
        + solidity * 0.10
    )


def _find_door_regions(img: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
    """
    Segment image by color (k-means), then score each segment
    as a potential door based on shape, position, and contrast.
    Also uses connected-components within each cluster for finer separation.
    Returns list of (x, y, w, h, score).
    """
    h, w = img.shape[:2]
    min_area = w * h * 0.01  # doors can be small (1% of image)
    max_area = w * h * 0.50

    # Resize for faster k-means
    scale = min(1.0, 300.0 / max(h, w))
    sh, sw = int(h * scale), int(w * scale)
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    small_blur = cv2.GaussianBlur(small, (5, 5), 0)

    candidates: List[Tuple[int, int, int, int, float]] = []

    # Run k-means at multiple k values for robustness
    for k in [4, 6, 8]:
        pixels = small_blur.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, k, None, criteria, 2, cv2.KMEANS_PP_CENTERS
        )
        labels = labels.reshape(sh, sw)

        unique, counts = np.unique(labels, return_counts=True)
        dominant_label = unique[np.argmax(counts)]
        dominant_color = centers[dominant_label]

        for label_id in range(k):
            if label_id == dominant_label:
                continue

            mask = (labels == label_id).astype(np.uint8) * 255

            # Use connected components for fine-grained region separation
            kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
            mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel_small)

            num_cc, cc_labels, stats, _ = cv2.connectedComponentsWithStats(
                mask_clean, connectivity=8
            )

            for cc_id in range(1, num_cc):  # skip background (0)
                x = stats[cc_id, cv2.CC_STAT_LEFT]
                y = stats[cc_id, cv2.CC_STAT_TOP]
                cw = stats[cc_id, cv2.CC_STAT_WIDTH]
                ch = stats[cc_id, cv2.CC_STAT_HEIGHT]
                cc_area = stats[cc_id, cv2.CC_STAT_AREA]

                real_area = cc_area / (scale * scale)
                if real_area < min_area or real_area > max_area:
                    continue

                ox = int(x / scale)
                oy = int(y / scale)
                ocw = int(cw / scale)
                och = int(ch / scale)

                if ocw < 10 or och < 15:
                    continue

                aspect = och / ocw if ocw > 0 else 0
                bottom_edge = (oy + och) / h

                # At ground level (bottom half), allow wider doors (double/triple doors)
                if bottom_edge > 0.50:
                    if aspect < 0.4 or aspect > 5.0:
                        continue
                else:
                    if aspect < 1.1 or aspect > 5.0:
                        continue

                solidity = cc_area / (cw * ch) if (cw * ch) > 0 else 0
                if solidity < 0.35:
                    continue

                cluster_color = centers[label_id]
                color_dist = np.linalg.norm(cluster_color - dominant_color) / 255.0

                # Reject regions that are too similar to the dominant color
                # (noise/texture artifacts, not real features)
                if color_dist < 0.10:
                    continue

                score = _score_candidate(
                    ox, oy, ocw, och, real_area, solidity, color_dist, w, h
                )
                candidates.append((ox, oy, ocw, och, score))

    # Also try line-based detection as backup
    _add_line_candidates(img, candidates, w, h, min_area)

    # Sort by score
    candidates.sort(key=lambda c: c[4], reverse=True)

    # NMS
    kept: List[Tuple[int, int, int, int, float]] = []
    for x, y, cw, ch, score in candidates:
        overlap = False
        for kx, ky, kw, kh, _ in kept:
            ix1, iy1 = max(x, kx), max(y, ky)
            ix2, iy2 = min(x + cw, kx + kw), min(y + ch, ky + kh)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                smaller = min(cw * ch, kw * kh)
                if inter / smaller > 0.4:
                    overlap = True
                    break
        if not overlap:
            kept.append((x, y, cw, ch, score))

    # Return only the single best detection
    return kept[:1]


def _add_line_candidates(
    img: np.ndarray,
    candidates: List[Tuple[int, int, int, int, float]],
    w: int,
    h: int,
    min_area: float,
) -> None:
    """Find vertical line pairs that could be door edges."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=40,
        minLineLength=int(h * 0.3), maxLineGap=int(h * 0.1),
    )
    if lines is None:
        return

    # Collect near-vertical lines
    verticals = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(x2 - x1, y2 - y1)))
        if angle < 15:
            cx = (x1 + x2) / 2
            top = min(y1, y2)
            bot = max(y1, y2)
            verticals.append((cx, top, bot))

    verticals.sort(key=lambda v: v[0])

    # Pair vertical lines that could be door edges
    for i in range(len(verticals)):
        for j in range(i + 1, len(verticals)):
            lx, lt, lb = verticals[i]
            rx, rt, rb = verticals[j]
            gap = rx - lx
            if gap < w * 0.08 or gap > w * 0.45:
                continue
            top = min(lt, rt)
            bot = max(lb, rb)
            height = bot - top
            if height < h * 0.25:
                continue
            aspect = height / gap
            if aspect < 1.2 or aspect > 4.5:
                continue
            area = gap * height
            if area < min_area:
                continue

            cx = (lx + rx) / 2
            bot_ratio = bot / h
            if bot_ratio < 0.5:
                continue  # upper-half = windows
            ground_score = bot_ratio
            horiz_score = 1.0 - abs(cx - w / 2) / (w / 2)
            score = ground_score * 0.45 + horiz_score * 0.35 + 0.2

            candidates.append((int(lx), int(top), int(gap), int(height), score))


def detect_doors(image_bytes: bytes) -> Dict[str, Any]:
    """
    Detect door/entrance regions in a facade image.
    Returns only "Entrance" detections when doors are found;
    empty list when no doors are detected.
    """
    start = time.perf_counter()

    img, orig_w, orig_h = _decode_image(image_bytes)
    regions = _find_door_regions(img)

    detections = []
    for i, (x, y, cw, ch, score) in enumerate(regions):
        if score < 0.62:
            continue
        x = max(0, x)
        y = max(0, y)
        cw = min(cw, orig_w - x)
        ch = min(ch, orig_h - y)

        area_ratio = (cw * ch) / (orig_w * orig_h)
        height_ratio = ch / orig_h
        if area_ratio > 0.30 or height_ratio > 0.55:
            continue

        detections.append({
            "id": f"det_{i}",
            "label": "Entrance",
            "confidence": round(min(score * 1.2, 0.99), 2),
            "bbox": {
                "xmin": x,
                "ymin": y,
                "xmax": x + cw,
                "ymax": y + ch,
            },
        })

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return {
        "image_width": orig_w,
        "image_height": orig_h,
        "detections": detections,
        "processing_time_ms": elapsed_ms,
    }
