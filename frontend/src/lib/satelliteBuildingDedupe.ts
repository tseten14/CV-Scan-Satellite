import type { Detection } from "@/types/detection";

/** One logical building after merging overlapping model detections. */
export type MergedSatelliteBuilding = {
  id: string;
  label: string;
  confidence: number;
  center_px: { x: number; y: number };
  bbox_px: { xmin: number; ymin: number; xmax: number; ymax: number };
  source_detection_ids: string[];
  polygon_px?: [number, number][];
};

function unionBbox(detections: Detection[]) {
  return detections.reduce(
    (acc, d) => ({
      xmin: Math.min(acc.xmin, d.bbox.xmin),
      ymin: Math.min(acc.ymin, d.bbox.ymin),
      xmax: Math.max(acc.xmax, d.bbox.xmax),
      ymax: Math.max(acc.ymax, d.bbox.ymax),
    }),
    { ...detections[0].bbox },
  );
}

function bboxIoU(
  a: { xmin: number; ymin: number; xmax: number; ymax: number },
  b: { xmin: number; ymin: number; xmax: number; ymax: number },
): number {
  const xi1 = Math.max(a.xmin, b.xmin);
  const yi1 = Math.max(a.ymin, b.ymin);
  const xi2 = Math.min(a.xmax, b.xmax);
  const yi2 = Math.min(a.ymax, b.ymax);
  const iw = Math.max(0, xi2 - xi1);
  const ih = Math.max(0, yi2 - yi1);
  const inter = iw * ih;
  const areaA = Math.max(0, a.xmax - a.xmin) * Math.max(0, a.ymax - a.ymin);
  const areaB = Math.max(0, b.xmax - b.xmin) * Math.max(0, b.ymax - b.ymin);
  const union = areaA + areaB - inter;
  return union > 0 ? inter / union : 0;
}

function bboxCenter(b: { xmin: number; ymin: number; xmax: number; ymax: number }) {
  return { x: (b.xmin + b.xmax) / 2, y: (b.ymin + b.ymax) / 2 };
}

/**
 * Group overlapping / duplicate detections so each real-world building yields a single point
 * (union bbox center). Tuned for satellite footprints that may be split into multiple masks.
 */
export function mergeSatelliteDetectionsOnePerBuilding(
  detections: Detection[],
  imageWidth: number,
  imageHeight: number,
): MergedSatelliteBuilding[] {
  if (detections.length === 0) return [];

  const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
  /**
   * Merge only when a detection overlaps the cluster's combined footprint (IoU with union).
   * Avoid distance-to-union-center — it incorrectly chains separate buildings into one cluster.
   */
  const iouMergeThreshold = 0.06;

  const clusters: Detection[][] = [];

  for (const d of sorted) {
    let placed = false;
    for (let i = 0; i < clusters.length; i++) {
      const u = unionBbox(clusters[i]);
      const iou = bboxIoU(d.bbox, u);
      if (iou >= iouMergeThreshold) {
        clusters[i].push(d);
        placed = true;
        break;
      }
    }
    if (!placed) {
      clusters.push([d]);
    }
  }

  return clusters.map((members, index) => {
    const bbox_px = unionBbox(members);
    const center_px = bboxCenter(bbox_px);
    const best = members.reduce((a, b) => (a.confidence >= b.confidence ? a : b));
    return {
      id: `building-${index + 1}`,
      label: best.label,
      confidence: Math.max(...members.map((m) => m.confidence)),
      center_px,
      bbox_px,
      source_detection_ids: members.map((m) => m.id),
      ...(best.polygon && best.polygon.length >= 3 ? { polygon_px: best.polygon } : {}),
    };
  });
}
