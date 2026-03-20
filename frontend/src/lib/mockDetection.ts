import type { Detection, DetectionEngineId, DetectionResult } from "@/types/detection";

function bboxToPolygon(bbox: Detection["bbox"]): [number, number][] {
  const { xmin, ymin, xmax, ymax } = bbox;
  return [
    [xmin, ymin],
    [xmax, ymin],
    [xmax, ymax],
    [xmin, ymax],
  ];
}

// Labels shown by the fallback mock detector when the backend is unavailable.
// These are intentionally “human friendly” so the UI still looks useful even
// without SAM 3 model access.
const MOCK_LABELS = [
  "Main Entrance",
  "Side Entrance",
  "Emergency Exit",
  "Service Entrance",
  "Revolving Entrance",
];

function randomBbox(imgW: number, imgH: number) {
  // Create a semi-realistic bbox placement: mostly in the upper half of the image,
  // and with a width/height ratio that produces visible overlays.
  const w = imgW * (0.08 + Math.random() * 0.15);
  const h = imgH * (0.15 + Math.random() * 0.25);
  const xmin = Math.random() * (imgW - w);
  const ymin = imgH * 0.3 + Math.random() * (imgH * 0.5 - h);
  return {
    xmin: Math.round(xmin),
    ymin: Math.round(ymin),
    xmax: Math.round(xmin + w),
    ymax: Math.round(ymin + h),
  };
}

export type MockDetectionOptions = {
  mode?: "streetview" | "satellite";
  engine?: DetectionEngineId;
};

export async function runMockDetection(
  imageFile: File,
  options: MockDetectionOptions = {},
): Promise<DetectionResult> {
  const mode = options.mode ?? "streetview";
  const engine = options.engine ?? "sam3";
  // Simulate network + inference latency so the UI’s loading states behave
  // like the real SAM 3 backend pipeline.
  // Simulate network + inference latency
  await new Promise((r) => setTimeout(r, 1200 + Math.random() * 800));

  const img = new Image();
  const url = URL.createObjectURL(imageFile);

  const dims = await new Promise<{ w: number; h: number }>((resolve) => {
    img.onload = () => {
      resolve({ w: img.naturalWidth, h: img.naturalHeight });
      URL.revokeObjectURL(url);
    };
    img.src = url;
  });

  const numDetections = 2 + Math.floor(Math.random() * 3);
  const detections: Detection[] = [];

  for (let i = 0; i < numDetections; i++) {
    const bbox = randomBbox(dims.w, dims.h);
    const label = mode === "satellite" ? "building" : MOCK_LABELS[i % MOCK_LABELS.length];
    detections.push({
      id: `det_${i}`,
      label,
      confidence: 0.72 + Math.random() * 0.26,
      bbox,
      polygon: bboxToPolygon(bbox),
    });
  }

  // Sort by confidence descending
  detections.sort((a, b) => b.confidence - a.confidence);

  return {
    image_width: dims.w,
    image_height: dims.h,
    detections,
    processing_time_s: Math.round((1200 + Math.random() * 800) / 10) / 100,
    engine,
    mock: true,
  };
}
