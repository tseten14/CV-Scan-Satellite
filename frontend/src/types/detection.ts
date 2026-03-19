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

export interface DetectionResult {
  // Width/height of the analyzed image so the UI can set the correct SVG viewBox.
  image_width: number;
  image_height: number;
  detections: Detection[];

  // End-to-end model processing time in milliseconds (sent from the backend).
  processing_time_ms: number;
}

export interface MapPin {
  lat: number;
  lng: number;
  label?: string;
}
