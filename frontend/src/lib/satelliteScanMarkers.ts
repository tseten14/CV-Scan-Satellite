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
/** Map one image pixel (origin top-left) to WGS84 using the same linear viewport mapping as scan capture. */
export function pixelToLatLng(
  x: number,
  y: number,
  bounds: MapScanBounds,
  imageWidth: number,
  imageHeight: number,
): { lat: number; lng: number } {
  const iw = Math.max(1, Math.abs(Number(imageWidth)) || 1);
  const ih = Math.max(1, Math.abs(Number(imageHeight)) || 1);
  const latSpan = bounds.north - bounds.south;
  const lngSpan = bounds.east - bounds.west;
  const fy = y / ih;
  const fx = x / iw;
  const lat = bounds.north - fy * latSpan;
  const lng = bounds.west + fx * lngSpan;
  return { lat, lng };
}

/** Closed GeoJSON exterior ring [lng, lat][] from a segmentation polygon in pixel space. */
export function polygonPxToWgs84Ring(
  polygon_px: [number, number][],
  bounds: MapScanBounds,
  imageWidth: number,
  imageHeight: number,
): [number, number][] | null {
  if (!polygon_px || polygon_px.length < 3) return null;
  const ring: [number, number][] = polygon_px.map(([x, y]) => {
    const p = pixelToLatLng(x, y, bounds, imageWidth, imageHeight);
    return [p.lng, p.lat];
  });
  const first = ring[0];
  const last = ring[ring.length - 1];
  if (first[0] !== last[0] || first[1] !== last[1]) {
    ring.push([first[0], first[1]]);
  }
  return ring;
}

/** Rectangle footprint from bbox corners in pixel space → closed [lng, lat][] ring. */
export function bboxPxToWgs84Ring(
  bbox: { xmin: number; ymin: number; xmax: number; ymax: number },
  bounds: MapScanBounds,
  imageWidth: number,
  imageHeight: number,
): [number, number][] {
  const corners: [number, number][] = [
    [bbox.xmin, bbox.ymin],
    [bbox.xmax, bbox.ymin],
    [bbox.xmax, bbox.ymax],
    [bbox.xmin, bbox.ymax],
    [bbox.xmin, bbox.ymin],
  ];
  return corners.map(([x, y]) => {
    const p = pixelToLatLng(x, y, bounds, imageWidth, imageHeight);
    return [p.lng, p.lat] as [number, number];
  });
}

export function mergedBuildingCentersToMapPoints(
  merged: MergedSatelliteBuilding[],
  bounds: MapScanBounds,
  imageWidth: number,
  imageHeight: number,
): Array<{ lat: number; lng: number }> {
  return merged.map((m) =>
    pixelToLatLng(m.center_px.x, m.center_px.y, bounds, imageWidth, imageHeight),
  );
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
