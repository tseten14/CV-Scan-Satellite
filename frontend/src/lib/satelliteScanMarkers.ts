import type { DetectionResult } from "@/types/detection";
import type { MergedSatelliteBuilding } from "@/lib/satelliteBuildingDedupe";

/** Geographic bounds of the map viewport at the moment of html2canvas capture. */
export interface MapScanBounds {
  west: number;
  east: number;
  north: number;
  south: number;
}

/**
 * Map merged building centers (pixel space) to lat/lng.
 * Assumes the analyzed image matches the captured viewport (top = north, left = west).
 */
export function mergedBuildingCentersToMapPoints(
  merged: MergedSatelliteBuilding[],
  bounds: MapScanBounds,
  imageWidth: number,
  imageHeight: number,
): Array<{ lat: number; lng: number }> {
  const iw = Math.max(1, Math.abs(Number(imageWidth)) || 1);
  const ih = Math.max(1, Math.abs(Number(imageHeight)) || 1);

  const latSpan = bounds.north - bounds.south;
  const lngSpan = bounds.east - bounds.west;

  return merged.map((m) => {
    const cx = Number(m.center_px.x);
    const cy = Number(m.center_px.y);
    const fy = cy / ih;
    const fx = cx / iw;
    const lat = bounds.north - fy * latSpan;
    const lng = bounds.west + fx * lngSpan;
    return { lat, lng };
  });
}

/**
 * @deprecated Prefer mergeSatelliteDetectionsOnePerBuilding + mergedBuildingCentersToMapPoints
 */
export function detectionCentersToMapPoints(
  result: DetectionResult,
  bounds: MapScanBounds,
): Array<{ lat: number; lng: number }> {
  const { image_width: iw, image_height: ih, detections } = result;
  if (iw <= 0 || ih <= 0) return [];

  const latSpan = bounds.north - bounds.south;
  const lngSpan = bounds.east - bounds.west;

  return detections.map((d) => {
    const cx = (d.bbox.xmin + d.bbox.xmax) / 2;
    const cy = (d.bbox.ymin + d.bbox.ymax) / 2;
    const fy = cy / ih;
    const fx = cx / iw;
    const lat = bounds.north - fy * latSpan;
    const lng = bounds.west + fx * lngSpan;
    return { lat, lng };
  });
}
