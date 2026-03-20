import * as cocoSsd from "@tensorflow-models/coco-ssd";
import type { Detection, DetectionResult } from "@/types/detection";

/**
 * Fallback entrance detection using COCO-SSD.
 *
 * Why this exists:
 * - The primary entrance extraction uses SAM 3 on the backend.
 * - If the backend is unavailable (or fails), the UI falls back to a lightweight
 *   browser model so the app still demonstrates the workflow.
 *
 * COCO-SSD does not include a generic "door" class, so we infer “entrance-like”
 * regions by looking for objects that often co-occur at building entrances:
 *   - person (arriving/leaving)
 *   - seating (chairs/couches)
 *   - outdoor items (potted plants/couches) that often surround doors
 */
const ENTRANCE_RELEVANT_CLASSES = new Set([
  "person",
  "chair",
  "dining table",
  "potted plant",
  "couch",
  "car",
]);

function toTitleCase(str: string): string {
  // Turn COCO classes like "dining table" into "Dining Table" for UI readability.
  return str.replace(/\b\w/g, (c) => c.toUpperCase());
}

export async function runEntranceDetection(imageFile: File): Promise<DetectionResult> {
  // We return the same DetectionResult shape as the backend so the overlay UI can
  // render both SAM3 and fallback detections without special-casing.
  const startTime = performance.now();

  const img = new Image();
  const url = URL.createObjectURL(imageFile);

  const dims = await new Promise<{ w: number; h: number }>((resolve, reject) => {
    // Load the image into an HTMLImageElement to get dimensions.
    img.onload = () => {
      resolve({ w: img.naturalWidth, h: img.naturalHeight });
      URL.revokeObjectURL(url);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Failed to load image"));
    };
    img.src = url;
  });

  const model = await cocoSsd.load();
  const predictions = await model.detect(img);

  const detections: Detection[] = predictions
    .filter((p) => ENTRANCE_RELEVANT_CLASSES.has(p.class))
    .map((p, i) => {
      const [x, y, width, height] = p.bbox;
      return {
        id: `det_${i}`,
        label: toTitleCase(p.class),
        confidence: p.score,
        bbox: {
          xmin: Math.round(x),
          ymin: Math.round(y),
          xmax: Math.round(x + width),
          ymax: Math.round(y + height),
        },
      };
    })
    .sort((a, b) => b.confidence - a.confidence);

  const processing_time_s = Math.round((performance.now() - startTime) / 10) / 100;

  return {
    image_width: dims.w,
    image_height: dims.h,
    detections,
    processing_time_s,
  };
}
