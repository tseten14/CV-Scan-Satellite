"""End-to-end building segmentation pipeline orchestration."""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from gis.export import export_gis_bundle
from gis.georef import GeorefBounds, bounds_to_geodataframe, pixel_polygons_to_geodataframe, read_raster_georef
from postprocess.nms import nms_detections
from postprocess.vectorize import RegularizationConfig, mask_to_building_detections
from segmentation.config import SegmentationConfig
from segmentation.inference import predict_mask

logger = logging.getLogger(__name__)


def run_building_pipeline(
    image_bytes: bytes,
    *,
    model_path: str | Path | None = None,
    segmentation_config: SegmentationConfig | None = None,
    regularization_config: RegularizationConfig | None = None,
    georef_bounds: GeorefBounds | None = None,
    raster_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Run the full satellite building pipeline: infer → vectorize → georeference.

    Returns a dict with keys:
        - detection_result: API-compatible DetectionResult payload
        - polygons_px: list of Shapely polygons in pixel space
        - geodataframe: WGS84 GeoDataFrame (empty when no buildings or no georef)
        - mask: binary uint8 mask
    """
    cfg = segmentation_config or SegmentationConfig.from_env()
    weights = Path(model_path) if model_path else cfg.model_path

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_array = np.asarray(image)
    width, height = image.size

    start = time.perf_counter()
    mask = predict_mask(image_array, weights, config=cfg)
    detections = mask_to_building_detections(mask, regularization_config)
    detections = nms_detections(detections, iou_threshold=0.55)

    polygons_px: list[Polygon] = []
    props: list[dict] = []
    for det in detections:
        poly_coords = det.get("polygon")
        if not poly_coords or len(poly_coords) < 3:
            continue
        poly = Polygon(poly_coords)
        if poly.is_empty or not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        polygons_px.append(poly)
        props.append(
            {
                "id": det["id"],
                "label": det["label"],
                "confidence": det["confidence"],
            }
        )

    gdf = _build_geodataframe(
        polygons_px,
        props,
        width=width,
        height=height,
        georef_bounds=georef_bounds,
        raster_path=raster_path,
    )

    elapsed = round(time.perf_counter() - start, 3)
    detection_result = {
        "image_width": width,
        "image_height": height,
        "detections": detections,
        "processing_time_s": elapsed,
        "engine": "smp",
    }

    return {
        "detection_result": detection_result,
        "polygons_px": polygons_px,
        "geodataframe": gdf,
        "mask": mask,
    }


def _build_geodataframe(
    polygons: list[Polygon],
    properties: list[dict],
    *,
    width: int,
    height: int,
    georef_bounds: GeorefBounds | None,
    raster_path: str | Path | None,
) -> gpd.GeoDataFrame:
    if not polygons:
        return gpd.GeoDataFrame(columns=["id", "label", "confidence", "geometry"], crs="EPSG:4326")

    if raster_path is not None:
        transform, source_crs = read_raster_georef(str(raster_path))
        return pixel_polygons_to_geodataframe(
            polygons,
            pixel_transform=transform,
            source_crs=source_crs,
            properties=properties,
        )

    if georef_bounds is not None:
        bounds = GeorefBounds(
            west=georef_bounds.west,
            south=georef_bounds.south,
            east=georef_bounds.east,
            north=georef_bounds.north,
            image_width=width,
            image_height=height,
        )
        return bounds_to_geodataframe(polygons, bounds, properties)

    logger.info("No georef supplied; returning empty GeoDataFrame (pixel polygons only).")
    return gpd.GeoDataFrame(columns=["id", "label", "confidence", "geometry"], crs="EPSG:4326")


def export_pipeline_gis(
    pipeline_result: dict[str, Any],
    output_dir: str | Path,
    *,
    stem: str = "buildings",
) -> dict[str, Path]:
    """Export GIS artifacts from a pipeline result dict."""
    gdf: gpd.GeoDataFrame = pipeline_result["geodataframe"]
    return export_gis_bundle(gdf, output_dir, stem=stem)
