export interface Detection {
  id: string;
  label: string;
  confidence: number;
  bbox: {
    xmin: number;
    ymin: number;
    xmax: number;
    ymax: number;
  };
  /** Polygon outline in image coords [[x,y],...] for mask-based rendering */
  polygon?: [number, number][];
}

/** Backend inference: SAM 3 or self-trained YOLOv9 (reported as engine \"yolo\"). */
export type DetectionEngineId = "sam3" | "yolo";

export interface DetectionResult {
  image_width: number;
  image_height: number;
  detections: Detection[];
  processing_time_s: number;

  /** Which model produced this result. */
  engine?: DetectionEngineId;

  /** True when the frontend used the offline mock (backend error). */
  mock?: boolean;
}

export interface MapPin {
  lat: number;
  lng: number;
  label?: string;
}
