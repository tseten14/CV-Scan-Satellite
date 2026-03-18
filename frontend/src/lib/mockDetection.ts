import { Detection, DetectionResult } from "@/types/detection";

const MOCK_LABELS = ["Main Entrance", "Side Door", "Emergency Exit", "Service Entrance", "Revolving Door"];

function randomBbox(imgW: number, imgH: number) {
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

export async function runMockDetection(imageFile: File): Promise<DetectionResult> {
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
    detections.push({
      id: `det_${i}`,
      label: MOCK_LABELS[i % MOCK_LABELS.length],
      confidence: 0.72 + Math.random() * 0.26,
      bbox: randomBbox(dims.w, dims.h),
    });
  }

  // Sort by confidence descending
  detections.sort((a, b) => b.confidence - a.confidence);

  return {
    image_width: dims.w,
    image_height: dims.h,
    detections,
    processing_time_ms: Math.round(1200 + Math.random() * 800),
  };
}
