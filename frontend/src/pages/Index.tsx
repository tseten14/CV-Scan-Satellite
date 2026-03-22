import { useState, useCallback, useEffect, useLayoutEffect, useRef } from "react";
import { motion } from "framer-motion";
import {
  MapPin,
  Eye,
  Upload,
  Building2,
  DoorOpen,
  ScanSearch,
  Boxes,
  Sparkles,
  ArrowUp,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
} from "lucide-react";
import { GeoAiMark } from "@/components/GeoAiMark";
import html2canvas from "html2canvas";
import MapPanel from "@/components/MapPanel";
import type { MapPanelHandle } from "@/components/MapPanel";
import DetectionOverlay from "@/components/DetectionOverlay";
import { runBackendDetection } from "@/lib/backendDetection";
import { runMockDetection } from "@/lib/mockDetection";
import type { MapPin as MapPinType, DetectionResult, DetectionEngineId } from "@/types/detection";
import type { MapScanBounds } from "@/lib/satelliteScanMarkers";
import { mergedBuildingCentersToMapPoints } from "@/lib/satelliteScanMarkers";
import { mergeSatelliteDetectionsOnePerBuilding } from "@/lib/satelliteBuildingDedupe";
import {
  buildBuildingsGeoJSON,
  buildBuildingsGeoJSONPixels,
  downloadJsonFile,
} from "@/lib/exportBuildingPoints";

type DetectionMode = "streetview" | "satellite";

const SATELLITE_BOX_SIZE_M = 220;
const SATELLITE_IMAGE_WIDTH = 1200;
const SATELLITE_IMAGE_HEIGHT = 1200;
const SATELLITE_PAN_STEP_M = 70;
const EARTH_RADIUS_M = 6378137;

function offsetLatLngMeters(lat: number, lng: number, northMeters: number, eastMeters: number) {
  const dLat = (northMeters / EARTH_RADIUS_M) * (180 / Math.PI);
  const dLng = (eastMeters / (EARTH_RADIUS_M * Math.cos((lat * Math.PI) / 180))) * (180 / Math.PI);
  return {
    lat: Math.max(-90, Math.min(90, lat + dLat)),
    lng: Math.max(-180, Math.min(180, lng + dLng)),
  };
}

const Index = () => {
  const [selectedPin, setSelectedPin] = useState<MapPinType | null>(null);
  const [buildingMapMarkers, setBuildingMapMarkers] = useState<Array<{ lat: number; lng: number }>>(
    [],
  );
  const [isProcessing, setIsProcessing] = useState(false);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null);
  const [statusMessage, setStatusMessage] = useState<string>("");
  const [detectionMode, setDetectionMode] = useState<DetectionMode>("streetview");
  const [detectionEngine, setDetectionEngine] = useState<DetectionEngineId>("sam3");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const mapPanelRef = useRef<MapPanelHandle>(null);
  /** Seconds remaining for long-running work; 0 = show "OOPS!"; null = idle */
  const [scanCountdown, setScanCountdown] = useState<number | null>(null);
  const countdownIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useLayoutEffect(() => {
    if (!isProcessing) {
      if (countdownIntervalRef.current) {
        clearInterval(countdownIntervalRef.current);
        countdownIntervalRef.current = null;
      }
      setScanCountdown(null);
      return;
    }
    setScanCountdown(20);
    if (countdownIntervalRef.current) clearInterval(countdownIntervalRef.current);
    countdownIntervalRef.current = setInterval(() => {
      setScanCountdown((c) => {
        if (c === null || c <= 0) return 0;
        return c - 1;
      });
    }, 1000);
    return () => {
      if (countdownIntervalRef.current) {
        clearInterval(countdownIntervalRef.current);
        countdownIntervalRef.current = null;
      }
    };
  }, [isProcessing]);

  const runDetectionOnFile = useCallback(
    async (file: File, mode?: DetectionMode, scanMapBounds?: MapScanBounds | null) => {
    const activeMode = mode ?? detectionMode;
    // Clear previous building dots; satellite map scans will repopulate after inference.
    setBuildingMapMarkers([]);
    setIsProcessing(true);
    setImageUrl((prev) => {
      if (prev?.startsWith("blob:")) URL.revokeObjectURL(prev);
      return null;
    });
    setDetectionResult(null);
    setStatusMessage("");

    try {
      const url = URL.createObjectURL(file);
      setImageUrl(url);
      let result: DetectionResult | null = null;
      try {
        result = await runBackendDetection(file, activeMode, detectionEngine);
      } catch (backendErr) {
        console.warn("Backend detection failed:", backendErr);
        const hint =
          backendErr instanceof Error ? backendErr.message : String(backendErr);
        // YOLO must hit the real API — mock labels look like "Main Entrance" and are not YOLO output.
        if (detectionEngine === "yolo") {
          const sslExtra =
            /certificate|CERTIFICATE_VERIFY|ssl.*verify/i.test(hint)
              ? " SSL (YOLO-World/CLIP): try YOLO_INSECURE_SSL=1 or SSL_CERT_FILE=… (README). "
              : "";
          setStatusMessage(
            `YOLO backend error: ${hint.slice(0, 260)}.${sslExtra}Run backend on :8000 (pip install -r requirements.txt), ` +
              `Vite proxy /api or VITE_API_URL.`,
          );
          result = null;
        } else {
          setStatusMessage(`SAM 3 unavailable — mock preview. ${hint.slice(0, 160)}`);
          result = await runMockDetection(file, { mode: activeMode, engine: "sam3" });
        }
      }
      if (result) {
        setDetectionResult(result);
        if (!result.mock) setStatusMessage("");
        if (activeMode === "satellite" && scanMapBounds) {
          const merged = mergeSatelliteDetectionsOnePerBuilding(
            result.detections,
            result.image_width,
            result.image_height,
          );
          setBuildingMapMarkers(
            mergedBuildingCentersToMapPoints(merged, scanMapBounds, result.image_width, result.image_height),
          );
        }
      } else {
        setDetectionResult(null);
      }
    } catch (err) {
      console.error("Detection failed:", err);
      const msg = err instanceof Error ? err.message : "Detection failed";
      setStatusMessage(msg);
      setDetectionResult(null);
    } finally {
      setIsProcessing(false);
    }
  },
  [detectionMode, detectionEngine],
  );

  const handleReset = useCallback(() => {
    setImageUrl((prev) => {
      if (prev?.startsWith("blob:")) URL.revokeObjectURL(prev);
      return null;
    });
    setDetectionResult(null);
    setStatusMessage("");
    setBuildingMapMarkers([]);
  }, []);

  const handleDownloadBuildingExport = useCallback(() => {
    if (!detectionResult || detectionMode !== "satellite") return;
    const merged = mergeSatelliteDetectionsOnePerBuilding(
      detectionResult.detections,
      detectionResult.image_width,
      detectionResult.image_height,
    );
    const meta = {
      image_width: detectionResult.image_width,
      image_height: detectionResult.image_height,
      processing_time_s: detectionResult.processing_time_s,
    };
    const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    const hasMapPoints =
      buildingMapMarkers.length > 0 && buildingMapMarkers.length === merged.length;
    const data = hasMapPoints
      ? buildBuildingsGeoJSON(merged, buildingMapMarkers, meta, {
          crs: "EPSG:4326",
          note: "WGS84 from Scan Map capture. One Point per merged building; coordinates are [longitude, latitude].",
        })
      : buildBuildingsGeoJSONPixels(merged, meta);
    downloadJsonFile(
      hasMapPoints ? `building-points-wgs84-${stamp}.geojson` : `building-points-pixels-${stamp}.geojson`,
      data,
      "application/geo+json",
    );
  }, [detectionResult, buildingMapMarkers, detectionMode]);

  /**
   * For uploaded images: spatial DB / map apps need real lon/lat. Pixel exports have geometry:null.
   * User aligns the left Leaflet map with the image, then downloads WGS84 GeoJSON.
   */
  const handleDownloadWgs84FromMapExtent = useCallback(() => {
      if (!detectionResult || detectionMode !== "satellite") return;
      if (mapPanelRef.current?.isStreetView()) {
        setStatusMessage(
          "Switch the left panel to Map or Satellite, align it with your image, then export again.",
        );
        setTimeout(() => setStatusMessage(""), 6500);
        return;
      }
      const bounds = mapPanelRef.current?.getVisibleMapBounds();
      if (!bounds) {
        setStatusMessage("Map not ready — wait a moment and try again.");
        setTimeout(() => setStatusMessage(""), 4000);
        return;
      }
      const merged = mergeSatelliteDetectionsOnePerBuilding(
        detectionResult.detections,
        detectionResult.image_width,
        detectionResult.image_height,
      );
      const mapPoints = mergedBuildingCentersToMapPoints(
        merged,
        bounds,
        detectionResult.image_width,
        detectionResult.image_height,
      );
      const paired = merged
        .map((b, i) => ({ b, p: mapPoints[i] }))
        .filter((x) => Number.isFinite(x.p.lat) && Number.isFinite(x.p.lng));
      if (paired.length === 0) {
        setStatusMessage("Could not compute coordinates — check map and image dimensions.");
        setTimeout(() => setStatusMessage(""), 5000);
        return;
      }
      const mergedOk = paired.map((x) => x.b);
      const pointsOk = paired.map((x) => x.p);
      const meta = {
        image_width: detectionResult.image_width,
        image_height: detectionResult.image_height,
        processing_time_s: detectionResult.processing_time_s,
      };
      const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
      const note =
        "WGS84 EPSG:4326. Each feature is a Point [longitude, latitude]. Pixel centers were projected using the visible bounds of the LEFT map at export time — align that map with your uploaded image first, or locations will be wrong. Use this file for PostGIS, spatial SQL, and map viewers.";
      const data = buildBuildingsGeoJSON(mergedOk, pointsOk, meta, { crs: "EPSG:4326", note });
      downloadJsonFile(
        `building-points-wgs84-epsg4326-${stamp}.geojson`,
        data,
        "application/geo+json",
      );
  }, [detectionResult, detectionMode]);

  const handlePaste = useCallback(
    (e: ClipboardEvent) => {
      if (isProcessing) return;
      const item = Array.from(e.clipboardData?.items ?? []).find((i) =>
        i.type.startsWith("image/")
      );
      if (!item) return;
      e.preventDefault();
      const file = item.getAsFile();
      if (!file) return;
      // Match pasted screenshots to the active map mode so Street View pastes use door prompts.
      const pasteMode: DetectionMode = mapPanelRef.current?.isStreetView()
        ? "streetview"
        : mapPanelRef.current?.isSatelliteView()
          ? "satellite"
          : detectionMode;
      runDetectionOnFile(file, pasteMode);
    },
    [runDetectionOnFile, isProcessing, detectionMode]
  );

  useEffect(() => {
    document.addEventListener("paste", handlePaste);
    return () => document.removeEventListener("paste", handlePaste);
  }, [handlePaste]);

  const API_BASE = import.meta.env.VITE_API_URL ?? "/api";

  const fetchSatelliteImageFile = useCallback(
    async (pin: { lat: number; lng: number }) => {
      const query = new URLSearchParams({
        lat: String(pin.lat),
        lng: String(pin.lng),
        box_size_m: String(SATELLITE_BOX_SIZE_M),
        width: String(SATELLITE_IMAGE_WIDTH),
        height: String(SATELLITE_IMAGE_HEIGHT),
      });
      const res = await fetch(`${API_BASE}/satellite-image?${query.toString()}`);
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(errText || "Failed to fetch satellite image");
      }

      const blob = await res.blob();
      const bounds: MapScanBounds = {
        west: Number(res.headers.get("X-Satellite-Bbox-West") ?? NaN),
        east: Number(res.headers.get("X-Satellite-Bbox-East") ?? NaN),
        north: Number(res.headers.get("X-Satellite-Bbox-North") ?? NaN),
        south: Number(res.headers.get("X-Satellite-Bbox-South") ?? NaN),
      };
      if (!Number.isFinite(bounds.west) || !Number.isFinite(bounds.east) || !Number.isFinite(bounds.north) || !Number.isFinite(bounds.south)) {
        throw new Error("Satellite API response is missing bounding-box metadata");
      }

      const file = new File([blob], "satellite-here.png", { type: blob.type || "image/png" });
      return { file, bounds };
    },
    [API_BASE],
  );

  const runSatelliteScanAtPin = useCallback(
    async (pin: { lat: number; lng: number }) => {
      setIsProcessing(true);
      setStatusMessage("Fetching satellite image...");
      try {
        const { file, bounds } = await fetchSatelliteImageFile(pin);
        setIsProcessing(false);
        runDetectionOnFile(file, "satellite", bounds);
      } catch (err) {
        console.error("Satellite fetch failed:", err);
        const msg = err instanceof Error ? err.message : "Could not fetch satellite image";
        setStatusMessage(msg);
        setIsProcessing(false);
        setTimeout(() => setStatusMessage(""), 4500);
      }
    },
    [fetchSatelliteImageFile, runDetectionOnFile],
  );

  const handleScanMap = useCallback(async () => {
    if (isProcessing) return;

    if (mapPanelRef.current?.isStreetView()) {
      const pin = mapPanelRef.current.getPin();
      if (!pin) {
        setStatusMessage("Drop a pin on the map first");
        setTimeout(() => setStatusMessage(""), 3000);
        return;
      }
      setIsProcessing(true);
      setStatusMessage("Fetching street view image...");
      try {
        const heading = mapPanelRef.current.getHeading();
        const res = await fetch(
          `${API_BASE}/streetview-image?lat=${pin.lat}&lng=${pin.lng}&heading=${heading}`
        );
        if (!res.ok) throw new Error("Failed to fetch street view image");
        const blob = await res.blob();
        const file = new File([blob], "streetview.jpg", { type: blob.type || "image/jpeg" });
        setIsProcessing(false);
        runDetectionOnFile(file, "streetview");
      } catch (err) {
        console.error("Street view fetch failed:", err);
        setStatusMessage("Could not fetch street view — try pasting a screenshot (⌘V)");
        setIsProcessing(false);
        setTimeout(() => setStatusMessage(""), 4000);
      }
      return;
    }

    // Auto-select mode based on active map view
    const scanMode: DetectionMode = mapPanelRef.current?.isSatelliteView()
      ? "satellite"
      : detectionMode;

    if (scanMode === "satellite") {
      const pin = mapPanelRef.current?.getPin() ?? selectedPin;
      if (!pin) {
        setStatusMessage("Drop a pin on the map first");
        setTimeout(() => setStatusMessage(""), 3000);
        return;
      }
      await runSatelliteScanAtPin(pin);
      return;
    }

    const el = mapPanelRef.current?.getContainerEl();
    if (!el) return;

    setIsProcessing(true);
    setStatusMessage("Capturing map view...");
    try {
      const scanBounds: MapScanBounds | null =
        scanMode === "satellite" ? mapPanelRef.current?.getVisibleMapBounds() ?? null : null;
      const canvas = await html2canvas(el, {
        useCORS: true,
        allowTaint: true,
        backgroundColor: null,
        // scale 1 produced ~200–300px wide captures; YOLO/SAM need more pixels.
        scale: Math.min(3, Math.max(2, typeof window !== "undefined" ? window.devicePixelRatio || 2 : 2)),
      });
      const blob = await new Promise<Blob>((resolve, reject) =>
        canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("Canvas capture failed"))), "image/png")
      );
      const file = new File([blob], "map-capture.png", { type: "image/png" });
      setIsProcessing(false);
      runDetectionOnFile(file, scanMode, scanBounds);
    } catch (err) {
      console.error("Map capture failed:", err);
      setStatusMessage("Map capture failed");
      setIsProcessing(false);
    }
  }, [
    isProcessing,
    runDetectionOnFile,
    API_BASE,
    detectionMode,
    runSatelliteScanAtPin,
    selectedPin,
  ]);

  const handleMoveSatellite = useCallback(
    async (direction: "north" | "south" | "west" | "east") => {
      if (isProcessing) return;
      const pin = mapPanelRef.current?.getPin() ?? selectedPin;
      if (!pin) {
        setStatusMessage("Drop a pin on the map first");
        setTimeout(() => setStatusMessage(""), 3000);
        return;
      }

      const north = direction === "north" ? SATELLITE_PAN_STEP_M : direction === "south" ? -SATELLITE_PAN_STEP_M : 0;
      const east = direction === "east" ? SATELLITE_PAN_STEP_M : direction === "west" ? -SATELLITE_PAN_STEP_M : 0;
      const moved = offsetLatLngMeters(pin.lat, pin.lng, north, east);
      const movedPin: MapPinType = {
        lat: moved.lat,
        lng: moved.lng,
        label: `${moved.lat.toFixed(5)}, ${moved.lng.toFixed(5)}`,
      };

      setSelectedPin(movedPin);
      await runSatelliteScanAtPin(movedPin);
    },
    [isProcessing, selectedPin, runSatelliteScanAtPin],
  );

  return (
    <div className="relative flex h-full min-h-0 w-full flex-col overflow-hidden bg-background">
      {/* Ambient background (purely visual) */}
      <div className="pointer-events-none absolute inset-0 opacity-[0.55] grid-bg" />
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(1100px_circle_at_18%_12%,hsl(var(--primary)/0.16),transparent_45%),radial-gradient(900px_circle_at_85%_22%,hsl(150_70%_45%/0.10),transparent_45%),radial-gradient(1200px_circle_at_50%_85%,hsl(40_90%_55%/0.08),transparent_55%)]" />
      <div className="pointer-events-none absolute inset-x-0 top-0 h-24 scanline opacity-70" />

      {/* Top bar */}
      <header className="relative z-30 flex items-center justify-between border-b border-border/70 bg-card/70 px-6 py-3 backdrop-blur-md">
        <div className="flex items-center gap-3.5">
          <div className="relative flex h-10 w-10 items-center justify-center rounded-xl border border-border/60 bg-gradient-to-br from-primary/20 via-background/20 to-background/10 shadow-[0_0_0_1px_hsl(var(--primary)/0.12),0_16px_34px_-22px_hsl(var(--primary)/0.55)]">
            <div className="pointer-events-none absolute inset-0 rounded-xl bg-[radial-gradient(14px_circle_at_30%_30%,hsl(var(--primary)/0.35),transparent_60%)]" />
            <GeoAiMark className="relative h-7 w-7 shrink-0 drop-shadow-[0_0_14px_hsl(var(--primary)/0.4)]" />
          </div>
          <div>
            <h1 className="font-brand text-[19px] leading-none sm:text-[23px]">
              <span className="text-foreground">CV-</span>
              <span className="text-[#81e6d9] drop-shadow-[0_0_14px_hsl(173_80%_50%/0.35)]">
                SCAN-Satellite
              </span>
            </h1>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <StatusIndicator
            icon={<MapPin className="h-3 w-3" />}
            label="Location"
            active={!!selectedPin}
          />
          <StatusIndicator
            icon={<Eye className="h-3 w-3" />}
            label="Detection"
            active={!!detectionResult}
          />
          <div className="ml-2 hidden rounded-md border border-border/60 bg-background/30 px-2.5 py-1 font-mono text-[10px] tracking-wide text-muted-foreground sm:block">
            {detectionMode === "satellite" ? "Building detection" : "Entrance detection"}
          </div>
        </div>
      </header>

      {/* Split panes */}
      <div className="relative z-20 flex min-h-0 min-w-0 flex-1 overflow-hidden">
        {/* Left: Map */}
        <div className="min-w-0 w-1/2 shrink-0 overflow-hidden border-r border-border/70 bg-card/20">
          <div className="h-full w-full p-3">
            <div className="h-full w-full overflow-hidden rounded-xl border border-border/70 bg-card/40 shadow-[0_10px_30px_-18px_hsl(var(--primary)/0.22)]">
              <MapPanel
                ref={mapPanelRef}
                onPinDrop={setSelectedPin}
                selectedPin={selectedPin}
                buildingMarkers={buildingMapMarkers}
              />
            </div>
          </div>
        </div>

        {/* Right: Image analysis */}
        <div className="relative z-20 flex min-h-0 min-w-0 w-1/2 shrink-0 flex-col overflow-hidden bg-card/20">
          {/* Panel header — flex-wrap + min-w-0 so controls are never clipped horizontally */}
          <div className="relative flex shrink-0 flex-wrap items-center gap-x-2 gap-y-2 border-b border-border/70 bg-card/70 px-3 py-2.5 backdrop-blur-md sm:gap-x-3 sm:px-4 sm:py-3 md:px-5">
            <div className="flex shrink-0 items-center gap-2">
              <div className="h-2 w-2 shrink-0 rounded-full bg-primary shadow-[0_0_18px_hsl(var(--primary)/0.55)] animate-pulse-glow" />
              <span
                className="font-mono text-[10px] font-semibold uppercase tracking-[0.18em] text-primary sm:text-[11px] sm:tracking-[0.22em]"
                title="Inference pipeline"
              >
                Inference
              </span>
            </div>
            <div className="flex shrink-0 overflow-hidden rounded-lg border border-border/70 bg-background/20">
              <button
                type="button"
                onClick={() => setDetectionMode("streetview")}
                disabled={isProcessing}
                className={`flex shrink-0 items-center gap-1.5 whitespace-nowrap px-2.5 py-1.5 font-mono text-[10px] tracking-wide transition-colors sm:px-3 ${
                  detectionMode === "streetview"
                    ? "bg-primary/20 text-primary"
                    : "text-muted-foreground hover:bg-primary/10 hover:text-primary"
                } ${isProcessing ? "opacity-50 pointer-events-none" : ""}`}
              >
                <DoorOpen className="h-3 w-3 shrink-0" />
                Entrances
              </button>
              <button
                type="button"
                onClick={() => setDetectionMode("satellite")}
                disabled={isProcessing}
                className={`flex shrink-0 items-center gap-1.5 whitespace-nowrap border-l border-border/70 px-2.5 py-1.5 font-mono text-[10px] tracking-wide transition-colors sm:px-3 ${
                  detectionMode === "satellite"
                    ? "bg-primary/20 text-primary"
                    : "text-muted-foreground hover:bg-primary/10 hover:text-primary"
                } ${isProcessing ? "opacity-50 pointer-events-none" : ""}`}
              >
                <Building2 className="h-3 w-3 shrink-0" />
                Buildings
              </button>
            </div>
            <div className="flex shrink-0 overflow-hidden rounded-lg border border-border/70 bg-background/20">
              <button
                type="button"
                onClick={() => setDetectionEngine("sam3")}
                disabled={isProcessing}
                title="Meta SAM 3 — promptable segmentation (mask polygons)"
                className={`flex shrink-0 items-center gap-1.5 whitespace-nowrap px-2.5 py-1.5 font-mono text-[10px] tracking-wide transition-colors ${
                  detectionEngine === "sam3"
                    ? "bg-violet-500/20 text-violet-200"
                    : "text-muted-foreground hover:bg-violet-500/10 hover:text-violet-200/90"
                } ${isProcessing ? "opacity-50 pointer-events-none" : ""}`}
              >
                <Sparkles className="h-3 w-3 shrink-0" />
                SAM 3
              </button>
              <button
                type="button"
                onClick={() => setDetectionEngine("yolo")}
                disabled={isProcessing}
                title={
                  detectionMode === "streetview"
                    ? "YOLO-World (local weights) or YOLOv8 COCO — bounding boxes"
                    : "YOLO — YOLO-World for building prompts when world weights exist; else COCO (coarse)"
                }
                className={`flex shrink-0 items-center gap-1.5 whitespace-nowrap border-l border-border/70 px-2.5 py-1.5 font-mono text-[10px] tracking-wide transition-colors ${
                  detectionEngine === "yolo"
                    ? "bg-amber-500/15 text-amber-200"
                    : "text-muted-foreground hover:bg-amber-500/10 hover:text-amber-200/90"
                } ${isProcessing ? "opacity-50 pointer-events-none" : ""}`}
              >
                <Boxes className="h-3 w-3 shrink-0" />
                YOLO
              </button>
            </div>
            <label
              className={`group flex shrink-0 cursor-pointer items-center gap-2 whitespace-nowrap rounded-lg border border-primary/40 bg-primary/10 px-2.5 py-1.5 font-mono text-[11px] text-primary transition-all hover:bg-primary/15 hover:border-primary/55 hover:shadow-[0_0_0_1px_hsl(var(--primary)/0.18),0_10px_22px_-16px_hsl(var(--primary)/0.35)] sm:px-3 sm:text-xs ${
                isProcessing ? "pointer-events-none opacity-50" : ""
              }`}
            >
              <Upload className="h-3.5 w-3.5 shrink-0 transition-transform group-hover:-translate-y-[1px]" />
              <input
                id="facade-file-input"
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/png,image/webp,image/*"
                disabled={isProcessing}
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) runDetectionOnFile(file);
                  e.target.value = "";
                }}
                className="hidden"
              />
              Upload image
            </label>
            <button
              type="button"
              onClick={handleScanMap}
              disabled={isProcessing}
              className={`group flex shrink-0 items-center gap-2 whitespace-nowrap rounded-lg border border-primary/40 bg-primary/10 px-2.5 py-1.5 font-mono text-[11px] text-primary transition-all hover:bg-primary/15 hover:border-primary/55 hover:shadow-[0_0_0_1px_hsl(var(--primary)/0.18),0_10px_22px_-16px_hsl(var(--primary)/0.35)] sm:px-3 sm:text-xs ${
                isProcessing ? "pointer-events-none opacity-50" : ""
              }`}
            >
              <ScanSearch className="h-3.5 w-3.5 shrink-0 transition-transform group-hover:-translate-y-[1px]" />
              Scan Map
            </button>
            {detectionMode === "satellite" && (
              <div className="ml-auto flex shrink-0 items-center gap-1">
                <button
                  type="button"
                  onClick={() => void handleMoveSatellite("west")}
                  disabled={isProcessing}
                  title="Move west and refetch satellite image"
                  className={`rounded-md border border-border/70 bg-background/25 p-1.5 text-muted-foreground transition-colors hover:bg-primary/10 hover:text-primary ${isProcessing ? "pointer-events-none opacity-50" : ""}`}
                >
                  <ArrowLeft className="h-3.5 w-3.5" />
                </button>
                <button
                  type="button"
                  onClick={() => void handleMoveSatellite("north")}
                  disabled={isProcessing}
                  title="Move north and refetch satellite image"
                  className={`rounded-md border border-border/70 bg-background/25 p-1.5 text-muted-foreground transition-colors hover:bg-primary/10 hover:text-primary ${isProcessing ? "pointer-events-none opacity-50" : ""}`}
                >
                  <ArrowUp className="h-3.5 w-3.5" />
                </button>
                <button
                  type="button"
                  onClick={() => void handleMoveSatellite("south")}
                  disabled={isProcessing}
                  title="Move south and refetch satellite image"
                  className={`rounded-md border border-border/70 bg-background/25 p-1.5 text-muted-foreground transition-colors hover:bg-primary/10 hover:text-primary ${isProcessing ? "pointer-events-none opacity-50" : ""}`}
                >
                  <ArrowDown className="h-3.5 w-3.5" />
                </button>
                <button
                  type="button"
                  onClick={() => void handleMoveSatellite("east")}
                  disabled={isProcessing}
                  title="Move east and refetch satellite image"
                  className={`rounded-md border border-border/70 bg-background/25 p-1.5 text-muted-foreground transition-colors hover:bg-primary/10 hover:text-primary ${isProcessing ? "pointer-events-none opacity-50" : ""}`}
                >
                  <ArrowRight className="h-3.5 w-3.5" />
                </button>
              </div>
            )}
          </div>

          <div className="flex flex-1 flex-col overflow-auto">
            {detectionResult && imageUrl ? (
              <DetectionOverlay
                imageUrl={imageUrl}
                result={detectionResult}
                onReset={handleReset}
                onUploadClick={() => document.getElementById("facade-file-input")?.click()}
                isProcessing={isProcessing}
                satelliteMode={detectionMode === "satellite"}
                hasMapLinkedPoints={buildingMapMarkers.length > 0}
                onDownloadBuildingExport={handleDownloadBuildingExport}
                onDownloadWgs84FromMapExtent={handleDownloadWgs84FromMapExtent}
              />
            ) : imageUrl && statusMessage && !isProcessing ? (
              <div className="flex flex-1 flex-col items-center justify-center gap-5 p-10">
                <img
                  src={imageUrl}
                  alt="Uploaded"
                  className="max-h-64 rounded-xl border border-border/70 bg-background/20 object-contain shadow-[0_14px_30px_-22px_rgba(0,0,0,0.75)]"
                />
                <p className="font-mono text-sm text-destructive">{statusMessage}</p>
                <div className="flex gap-3">
                  <button
                    type="button"
                    onClick={() => document.getElementById("facade-file-input")?.click()}
                    className="rounded-lg border border-primary/40 bg-primary/10 px-3 py-1.5 font-mono text-xs text-primary transition-all hover:bg-primary/15 hover:border-primary/55"
                  >
                    Try again
                  </button>
                  <button
                    type="button"
                    onClick={handleReset}
                    className="rounded-lg border border-border/70 bg-background/10 px-3 py-1.5 font-mono text-xs text-muted-foreground transition-colors hover:bg-muted/40"
                  >
                    Upload different
                  </button>
                </div>
              </div>
            ) : isProcessing ? (
              <ProcessingCountdownPanel scanCountdown={scanCountdown} />
            ) : (
              <div
                className="flex min-h-0 flex-1 flex-col items-center justify-center gap-7 p-10 text-center"
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  const file = e.dataTransfer.files[0];
                  if (file?.type.startsWith("image/") && !isProcessing)
                    runDetectionOnFile(file);
                }}
              >
                <p className="font-mono text-xs font-semibold uppercase tracking-widest text-muted-foreground">
                  Upload an image
                </p>
                {/* Primary upload: visible native file input - most reliable */}
                <label className="flex cursor-pointer flex-col items-center gap-3">
                  <input
                    type="file"
                    accept="image/jpeg,image/png,image/webp,image/*"
                    disabled={isProcessing}
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) runDetectionOnFile(file);
                      e.target.value = "";
                    }}
                    className="block w-full max-w-xs font-mono text-xs file:mr-3 file:cursor-pointer file:rounded-lg file:border-0 file:bg-primary file:px-4 file:py-2.5 file:font-semibold file:text-primary-foreground hover:file:bg-primary/90"
                  />
                  <span className="font-mono text-[10px] text-muted-foreground">
                    or drag & drop, paste (⌘V)
                  </span>
                </label>
                <p className="font-mono text-[10px] text-muted-foreground/60">
                  Click map to get coordinates
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

/** Same centered panel + giant type as pre-refactor processing UI (no dimmed image / gray scrim). */
function ProcessingCountdownPanel({ scanCountdown }: { scanCountdown: number | null }) {
  return (
    <div
      className="flex h-full min-h-0 flex-1 flex-col items-center justify-center gap-5 p-10"
      aria-busy="true"
    >
      <div className="text-center">
        {scanCountdown !== null && scanCountdown > 0 ? (
          <div
            className="font-mono text-[10.5rem] font-extrabold leading-none text-primary drop-shadow-[0_0_18px_hsl(var(--primary)/0.35)] tabular-nums"
            aria-live="polite"
          >
            {scanCountdown}
          </div>
        ) : scanCountdown === 0 ? (
          <div
            className="mt-1 font-mono text-7xl font-extrabold leading-none text-primary"
            aria-live="assertive"
          >
            OOPS!
          </div>
        ) : null}
      </div>
    </div>
  );
}

function StatusIndicator({
  icon,
  label,
  active,
}: {
  icon: React.ReactNode;
  label: string;
  active: boolean;
}) {
  return (
    <div className="flex items-center gap-1.5 font-mono text-[11px]">
      <motion.div
        animate={{ opacity: active ? 1 : 0.3 }}
        className={active ? "text-primary" : "text-muted-foreground"}
      >
        {icon}
      </motion.div>
      <span className={active ? "text-primary" : "text-muted-foreground"}>
        {label}
      </span>
      <div
        className={`h-1.5 w-1.5 rounded-full ${
          active ? "bg-success" : "bg-muted-foreground/30"
        }`}
      />
    </div>
  );
}

export default Index;
