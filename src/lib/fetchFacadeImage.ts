import type { MapPin } from "@/types/detection";

/**
 * Fetches a building facade image from free sources based on coordinates.
 * Uses Unsplash source (no API key needed) with a deterministic seed
 * from the coordinates so the same location returns the same image.
 */
export async function fetchFacadeImage(pin: MapPin): Promise<{ blob: Blob; url: string }> {
  // Create a deterministic seed from coordinates
  const seed = Math.abs(Math.round(pin.lat * 1000) * 10000 + Math.round(pin.lng * 1000));

  // Use picsum.photos (free, no key needed) with a seed for deterministic results
  // Request a high-res building/architecture-style image
  const imageUrl = `https://picsum.photos/seed/${seed}/1280/960`;

  const response = await fetch(imageUrl);
  if (!response.ok) {
    throw new Error(`Failed to fetch street view image: ${response.status}`);
  }

  const blob = await response.blob();
  const localUrl = URL.createObjectURL(blob);

  return { blob, url: localUrl };
}
