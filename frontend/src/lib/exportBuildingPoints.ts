import type { DetectionResult } from "@/types/detection";
import type { MergedSatelliteBuilding } from "@/lib/satelliteBuildingDedupe";

/** GeoJSON FeatureCollection (WGS84 lon/lat) for GIS tools. */
export type BuildingsGeoJSON = {
  type: "FeatureCollection";
  metadata: {
    generator: string;
    buildings_count: number;
    /** e.g. EPSG:4326 — helps PostGIS / spatial apps */
    crs?: string;
    note?: string;
    image_width: number;
    image_height: number;
    processing_time_s: number;
    generated_at: string;
  };
  features: Array<{
    type: "Feature";
    id?: string;
    geometry: {
      type: "Point";
      coordinates: [number, number]; // [lng, lat]
    };
    properties: {
      id: string;
      label: string;
      confidence: number;
      bbox_px: { xmin: number; ymin: number; xmax: number; ymax: number };
      source_detection_ids: string[];
      polygon_px?: [number, number][];
    };
  }>;
};

/** Flat JSON list of map-ready points with detection metadata. */
export type BuildingsPointsJSON = {
  schema: "cv-scan-satellite/building-map-points/v1";
  crs: "EPSG:4326";
  generated_at: string;
  image_width: number;
  image_height: number;
  processing_time_s: number;
  points: Array<{
    lat: number;
    lng: number;
    id: string;
    label: string;
    confidence: number;
    bbox_px: { xmin: number; ymin: number; xmax: number; ymax: number };
    source_detection_ids: string[];
    polygon_px?: [number, number][];
  }>;
};

function zipMergedWithMapPoints(
  merged: MergedSatelliteBuilding[],
  mapPoints: Array<{ lat: number; lng: number }>,
) {
  const n = Math.min(mapPoints.length, merged.length);
  const pairs: Array<{
    building: MergedSatelliteBuilding;
    point: { lat: number; lng: number };
  }> = [];
  for (let i = 0; i < n; i++) {
    pairs.push({ building: merged[i], point: mapPoints[i] });
  }
  return pairs;
}

export function buildBuildingsGeoJSON(
  merged: MergedSatelliteBuilding[],
  mapPoints: Array<{ lat: number; lng: number }>,
  meta: Pick<DetectionResult, "image_width" | "image_height" | "processing_time_s">,
  options?: { note?: string; crs?: string },
): BuildingsGeoJSON {
  const pairs = zipMergedWithMapPoints(merged, mapPoints);
  const generated_at = new Date().toISOString();

  return {
    type: "FeatureCollection",
    metadata: {
      generator: "CV-Scan-Satellite",
      buildings_count: pairs.length,
      crs: options?.crs ?? "EPSG:4326",
      note:
        options?.note ??
        "One point per merged building (overlapping detections combined). Geometry is WGS84 (lon, lat).",
      image_width: meta.image_width,
      image_height: meta.image_height,
      processing_time_s: meta.processing_time_s,
      generated_at,
    },
    features: pairs.map(({ building: b, point: p }) => ({
      type: "Feature" as const,
      id: b.id,
      geometry: {
        type: "Point" as const,
        coordinates: [p.lng, p.lat] as [number, number],
      },
      properties: {
        id: b.id,
        label: b.label,
        confidence: b.confidence,
        bbox_px: { ...b.bbox_px },
        source_detection_ids: b.source_detection_ids,
        ...(b.polygon_px && b.polygon_px.length >= 3 ? { polygon_px: b.polygon_px } : {}),
      },
    })),
  };
}

export function buildBuildingsPointsJSON(
  merged: MergedSatelliteBuilding[],
  mapPoints: Array<{ lat: number; lng: number }>,
  meta: Pick<DetectionResult, "image_width" | "image_height" | "processing_time_s">,
): BuildingsPointsJSON {
  const pairs = zipMergedWithMapPoints(merged, mapPoints);
  const generated_at = new Date().toISOString();

  return {
    schema: "cv-scan-satellite/building-map-points/v1",
    crs: "EPSG:4326",
    generated_at,
    image_width: meta.image_width,
    image_height: meta.image_height,
    processing_time_s: meta.processing_time_s,
    points: pairs.map(({ building: b, point: p }) => ({
      lat: p.lat,
      lng: p.lng,
      id: b.id,
      label: b.label,
      confidence: b.confidence,
      bbox_px: { ...b.bbox_px },
      source_detection_ids: b.source_detection_ids,
      ...(b.polygon_px && b.polygon_px.length >= 3 ? { polygon_px: b.polygon_px } : {}),
    })),
  };
}

/** GeoJSON with geometry: null — point centers are in properties (image pixel space). */
export type BuildingsGeoJSONPixels = {
  type: "FeatureCollection";
  metadata: {
    generator: string;
    coordinate_space: "image_pixels";
    buildings_count: number;
    note: string;
    image_width: number;
    image_height: number;
    processing_time_s: number;
    generated_at: string;
  };
  features: Array<{
    type: "Feature";
    id?: string;
    geometry: null;
    properties: {
      id: string;
      label: string;
      confidence: number;
      center_px: { x: number; y: number };
      bbox_px: { xmin: number; ymin: number; xmax: number; ymax: number };
      source_detection_ids: string[];
      polygon_px?: [number, number][];
    };
  }>;
};

export type BuildingsPointsPixelsJSON = {
  schema: "cv-scan-satellite/building-detection-points/v1";
  coordinate_space: "image_pixels";
  note: string;
  generated_at: string;
  image_width: number;
  image_height: number;
  processing_time_s: number;
  points: Array<{
    id: string;
    label: string;
    confidence: number;
    center_px: { x: number; y: number };
    bbox_px: { xmin: number; ymin: number; xmax: number; ymax: number };
    source_detection_ids: string[];
    polygon_px?: [number, number][];
  }>;
};

/** Export when no map scan: one merged building per point, pixel coordinates. */
export function buildBuildingsGeoJSONPixels(
  merged: MergedSatelliteBuilding[],
  meta: Pick<DetectionResult, "image_width" | "image_height" | "processing_time_s">,
): BuildingsGeoJSONPixels {
  const generated_at = new Date().toISOString();
  return {
    type: "FeatureCollection",
    metadata: {
      generator: "CV-Scan-Satellite",
      coordinate_space: "image_pixels",
      buildings_count: merged.length,
      note:
        "One point per merged building. Coordinates are image pixels (origin top-left). Use Scan Map in satellite view for WGS84 lat/lng, or georeference this image in GIS software.",
      image_width: meta.image_width,
      image_height: meta.image_height,
      processing_time_s: meta.processing_time_s,
      generated_at,
    },
    features: merged.map((b) => ({
      type: "Feature" as const,
      id: b.id,
      geometry: null,
      properties: {
        id: b.id,
        label: b.label,
        confidence: b.confidence,
        center_px: { ...b.center_px },
        bbox_px: { ...b.bbox_px },
        source_detection_ids: b.source_detection_ids,
        ...(b.polygon_px && b.polygon_px.length >= 3 ? { polygon_px: b.polygon_px } : {}),
      },
    })),
  };
}

export function buildBuildingsPointsJSONPixels(
  merged: MergedSatelliteBuilding[],
  meta: Pick<DetectionResult, "image_width" | "image_height" | "processing_time_s">,
): BuildingsPointsPixelsJSON {
  const generated_at = new Date().toISOString();
  return {
    schema: "cv-scan-satellite/building-detection-points/v1",
    coordinate_space: "image_pixels",
    note:
      "One point per merged building. center_px is in image coordinates. For lat/lng, use Scan Map on the satellite basemap or georeference the image.",
    generated_at,
    image_width: meta.image_width,
    image_height: meta.image_height,
    processing_time_s: meta.processing_time_s,
    points: merged.map((b) => ({
      id: b.id,
      label: b.label,
      confidence: b.confidence,
      center_px: { ...b.center_px },
      bbox_px: { ...b.bbox_px },
      source_detection_ids: b.source_detection_ids,
      ...(b.polygon_px && b.polygon_px.length >= 3 ? { polygon_px: b.polygon_px } : {}),
    })),
  };
}

export function downloadJsonFile(filename: string, data: unknown, mimeType = "application/json") {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
