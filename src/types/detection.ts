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
