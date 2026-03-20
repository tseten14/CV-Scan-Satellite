import type { DetectionEngineId, DetectionResult } from "@/types/detection";

const API_BASE = import.meta.env.VITE_API_URL ?? "/api";

// Detection can take a while because SAM 3 is heavy and depends on model download,
// input image size, and available hardware acceleration.
// We set a generous timeout so the UI can surface a readable error instead of
// failing immediately for long-running scans.
const DETECTION_TIMEOUT_MS = 8 * 60 * 1000; // 8 minutes

export async function runBackendDetection(
  imageFile: File,
  mode: "streetview" | "satellite" = "streetview",
  engine: DetectionEngineId = "sam3",
): Promise<DetectionResult> {
  // The backend expects a multipart/form-data upload with the image bytes.
  const formData = new FormData();
  formData.append("file", imageFile);

  // Use an AbortController so we can stop the request if it runs longer than the timeout.
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), DETECTION_TIMEOUT_MS);

  // POST /detect?mode=streetview|satellite&engine=sam3|yolo
  const response = await fetch(`${API_BASE}/detect?mode=${mode}&engine=${engine}`, {
    method: "POST",
    body: formData,
    signal: controller.signal,
  });
  clearTimeout(timeoutId);

  if (!response.ok) {
    // Surface server errors (including validation errors) as a human-readable string.
    const err = await response.text();
    throw new Error(err || `Detection failed: ${response.status}`);
  }

  // The server returns a DetectionResult JSON payload that the UI renders as polygons.
  const result = (await response.json()) as DetectionResult;
  return result;
}
