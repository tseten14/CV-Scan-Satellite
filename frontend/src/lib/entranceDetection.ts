import * as cocoSsd from "@tensorflow-models/coco-ssd";
import type { Detection, DetectionResult } from "@/types/detection";

/**
 * COCO-SSD classes that often appear at/near building entrances.
 * "door" is not in COCO, so we use person (entering/leaving), chair (outdoor seating),
 * dining table, potted plant, couch as entrance-area indicators.
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
  return str.replace(/\b\w/g, (c) => c.toUpperCase());
}

export async function runEntranceDetection(imageFile: File): Promise<DetectionResult> {
  const startTime = performance.now();

  const img = new Image();
  const url = URL.createObjectURL(imageFile);

  const dims = await new Promise<{ w: number; h: number }>((resolve, reject) => {
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

  const processing_time_ms = Math.round(performance.now() - startTime);

  return {
    image_width: dims.w,
    image_height: dims.h,
    detections,
    processing_time_ms,
  };
}
