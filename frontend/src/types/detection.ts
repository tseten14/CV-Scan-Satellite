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

export interface DetectionResult {
  image_width: number;
  image_height: number;
  detections: Detection[];
  processing_time_ms: number;
}

export interface MapPin {
  lat: number;
  lng: number;
  label?: string;
}
