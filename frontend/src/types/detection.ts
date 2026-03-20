export interface Detection {
  // Unique id for a single detected instance.
  id: string;

  // Semantic label used by the UI (e.g., "entrance" or "building").
  label: string;

  // Confidence score between 0 and 1.
  confidence: number;
  bbox: {
    // Bounding box in image coordinates (x increases to the right, y increases downward).
    xmin: number;
    ymin: number;
    xmax: number;
    ymax: number;
  };

  // Polygon outline in image coords for mask-based rendering (optional for some detections).
  /** Polygon outline in image coords [[x,y],...] for mask-based rendering */
  polygon?: [number, number][];
}

/** Backend inference stack (SAM 3 vs YOLO). */
export type DetectionEngineId = "sam3" | "yolo";

/** YOLO-World (text prompts) vs YOLOv8 COCO — set by API when engine is yolo. */
export type YoloVariantId = "world" | "coco";

export interface DetectionResult {
  // Width/height of the analyzed image so the UI can set the correct SVG viewBox.
  image_width: number;
  image_height: number;
  detections: Detection[];

  // End-to-end model processing time in seconds (from backend or measured client-side).
  processing_time_s: number;

  /** Set by API: which model produced this result. */
  engine?: DetectionEngineId;

  /** When engine is yolo: open-vocabulary World vs COCO detector. */
  yolo_variant?: YoloVariantId;

  /** True when the frontend used the offline mock (backend error). */
  mock?: boolean;
}

export interface MapPin {
  lat: number;
  lng: number;
  label?: string;
}
