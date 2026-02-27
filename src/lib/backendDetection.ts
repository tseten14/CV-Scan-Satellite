import type { DetectionResult } from "@/types/detection";

const API_BASE = import.meta.env.VITE_API_URL ?? "/api";

const DETECTION_TIMEOUT_MS = 8 * 60 * 1000; // 8 minutes for SAM 3 inference

export async function runBackendDetection(imageFile: File): Promise<DetectionResult> {
  const formData = new FormData();
  formData.append("file", imageFile);

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), DETECTION_TIMEOUT_MS);

  const response = await fetch(`${API_BASE}/detect`, {
    method: "POST",
    body: formData,
    signal: controller.signal,
  });
  clearTimeout(timeoutId);

  if (!response.ok) {
    const err = await response.text();
    throw new Error(err || `Detection failed: ${response.status}`);
  }

  const result = (await response.json()) as DetectionResult;
  return result;
}
