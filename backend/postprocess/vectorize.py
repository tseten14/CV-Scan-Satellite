"""High-precision vectorization and architectural regularization of building masks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid


@dataclass(frozen=True)
class RegularizationConfig:
    """Tuning knobs for contour extraction and polygon cleanup."""

    min_contour_area: float = 80.0
    douglas_peucker_ratio: float = 0.002
    angle_snap_threshold_deg: float = 12.0
    rectangularity_threshold: float = 0.82
    morph_kernel_size: int = 3


def _prepare_binary_mask(mask: np.ndarray) -> np.ndarray | None:
    """Ensure a clean uint8 binary mask."""
    if mask is None or mask.size == 0:
        return None
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.dtype != np.uint8:
        arr = (arr > 0).astype(np.uint8) * 255
    elif arr.max() <= 1:
        arr = (arr > 0).astype(np.uint8) * 255
    if arr.max() == 0:
        return None
    k = max(1, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel, iterations=1)
    return arr


def _ring_from_contour(contour: np.ndarray) -> list[tuple[float, float]] | None:
    if contour is None or len(contour) < 3:
        return None
    pts = contour.reshape(-1, 2)
    return [(float(x), float(y)) for x, y in pts]


def _snap_angle_deg(angle_deg: float, threshold_deg: float) -> float:
    """Snap an edge angle to the nearest cardinal direction when close enough."""
    for cardinal in (0.0, 90.0, 180.0, 270.0, 360.0):
        delta = abs((angle_deg - cardinal + 180.0) % 360.0 - 180.0)
        if delta <= threshold_deg:
            return cardinal % 360.0
    return angle_deg


def _orthogonalize_ring(
    ring: list[tuple[float, float]],
    angle_threshold_deg: float,
) -> list[tuple[float, float]]:
    """
    Snap consecutive edges toward horizontal/vertical axes to straighten walls.

    Each edge is rotated to a cardinal direction when its bearing is within
    ``angle_threshold_deg`` of 0°, 90°, 180°, or 270°.
    """
    if len(ring) < 3:
        return ring

    snapped: list[tuple[float, float]] = [ring[0]]
    for idx in range(1, len(ring)):
        x0, y0 = snapped[-1]
        x1, y1 = ring[idx]
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        snapped_angle = _snap_angle_deg(angle, angle_threshold_deg)
        rad = math.radians(snapped_angle)
        snapped.append((x0 + length * math.cos(rad), y0 + length * math.sin(rad)))

    if len(snapped) < 3:
        return ring

    # Close small gaps between first and last vertex.
    x0, y0 = snapped[0]
    x1, y1 = snapped[-1]
    if math.hypot(x1 - x0, y1 - y0) > 2.0:
        snapped.append(snapped[0])
    else:
        snapped[-1] = snapped[0]
    return snapped


def regularize_polygon(
    polygon: Polygon,
    config: RegularizationConfig | None = None,
) -> Polygon | None:
    """
    Simplify and orthogonalize a Shapely polygon for architectural footprints.

    Applies Douglas-Peucker simplification followed by edge snapping. When the
    input is sufficiently rectangular, replaces it with the minimum rotated
    rectangle for crisp 90° corners.
    """
    cfg = config or RegularizationConfig()
    if polygon.is_empty or polygon.area <= 0:
        return None

    poly = make_valid(polygon)
    if poly.is_empty:
        return None
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    if poly.geom_type != "Polygon":
        return None

    min_rect = poly.minimum_rotated_rectangle
    if min_rect.area > 0 and (poly.area / min_rect.area) >= cfg.rectangularity_threshold:
        poly = min_rect

    tolerance = max(0.5, poly.length * cfg.douglas_peucker_ratio)
    simplified = poly.simplify(tolerance, preserve_topology=True)
    if simplified.is_empty or simplified.geom_type != "Polygon":
        return None

    ring = list(simplified.exterior.coords)[:-1]
    ortho_ring = _orthogonalize_ring(ring, cfg.angle_snap_threshold_deg)
    if len(ortho_ring) < 3:
        return None

    result = Polygon(ortho_ring)
    result = make_valid(result)
    if result.is_empty or result.geom_type != "Polygon" or result.area < cfg.min_contour_area:
        return None
    return result


def mask_to_polygons(
    mask: np.ndarray,
    config: RegularizationConfig | None = None,
) -> list[Polygon]:
    """
    Extract and regularize building polygons from a binary inference mask.

    Returns an empty list when the mask contains no building pixels.
    """
    cfg = config or RegularizationConfig()
    binary = _prepare_binary_mask(mask)
    if binary is None:
        return []

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polygons: list[Polygon] = []
    for contour in contours:
        if cv2.contourArea(contour) < cfg.min_contour_area:
            continue
        ring = _ring_from_contour(contour)
        if not ring or len(ring) < 3:
            continue
        raw = Polygon(ring)
        regularized = regularize_polygon(raw, cfg)
        if regularized is not None and regularized.is_valid and regularized.area >= cfg.min_contour_area:
            polygons.append(regularized)
    return polygons


def _polygon_to_detection_dict(
    polygon: Polygon,
    index: int,
    confidence: float = 1.0,
) -> dict:
    """Convert a pixel-space Shapely polygon to the API detection schema."""
    minx, miny, maxx, maxy = polygon.bounds
    exterior = list(polygon.exterior.coords)
    poly_ring = [[float(x), float(y)] for x, y in exterior]
    return {
        "id": f"building_{index}",
        "label": "building",
        "confidence": confidence,
        "bbox": {
            "xmin": float(minx),
            "ymin": float(miny),
            "xmax": float(maxx),
            "ymax": float(maxy),
        },
        "polygon": poly_ring,
    }


def mask_to_building_detections(
    mask: np.ndarray,
    config: RegularizationConfig | None = None,
    *,
    default_confidence: float = 1.0,
) -> list[dict]:
    """
    Vectorize a binary mask into detection dicts compatible with the frontend API.

    Handles empty masks gracefully by returning ``[]``.
    """
    polygons = mask_to_polygons(mask, config)
    return [
        _polygon_to_detection_dict(poly, idx, confidence=default_confidence)
        for idx, poly in enumerate(polygons)
    ]
