import { useState, useCallback, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { Radar, MapPin, Eye, Loader2, Upload, Building2, DoorOpen, ScanSearch } from "lucide-react";
import html2canvas from "html2canvas";
import MapPanel from "@/components/MapPanel";
import type { MapPanelHandle } from "@/components/MapPanel";
import DetectionOverlay from "@/components/DetectionOverlay";
import { runBackendDetection } from "@/lib/backendDetection";
import { runMockDetection } from "@/lib/mockDetection";
import { captureScreenLeftHalfAsPngFile } from "@/lib/captureScreenLeftHalf";
import type { MapPin as MapPinType, DetectionResult } from "@/types/detection";

type DetectionMode = "streetview" | "satellite";

const Index = () => {
  const [selectedPin, setSelectedPin] = useState<MapPinType | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null);
  const [statusMessage, setStatusMessage] = useState<string>("");
  const [detectionMode, setDetectionMode] = useState<DetectionMode>("streetview");
  const [scanCountdown, setScanCountdown] = useState<number | null>(null);
  const [ughh, setUghh] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const mapPanelRef = useRef<MapPanelHandle>(null);
  const countdownIntervalRef = useRef<number | null>(null);

  const runDetectionOnFile = useCallback(async (file: File, mode?: DetectionMode) => {
    const activeMode = mode ?? detectionMode;
    setIsProcessing(true);
    // Start the "still running" countdown when the image scan begins.
    setScanCountdown(20);
    setUghh(false);
    if (countdownIntervalRef.current) {
      window.clearInterval(countdownIntervalRef.current);
    }
    const startMs = Date.now();
    countdownIntervalRef.current = window.setInterval(() => {
      const elapsedSec = Math.floor((Date.now() - startMs) / 1000);
      const remaining = 20 - elapsedSec;
      if (remaining > 0) {
        setScanCountdown(remaining);
      } else {
        setScanCountdown(null);
        setUghh(true);
        if (countdownIntervalRef.current) window.clearInterval(countdownIntervalRef.current);
        countdownIntervalRef.current = null;
      }
    }, 250);
    setImageUrl((prev) => {
      if (prev?.startsWith("blob:")) URL.revokeObjectURL(prev);
      return null;
    });
    setDetectionResult(null);
    const modeLabel = activeMode === "satellite" ? "Scanning buildings" : "Running detection";
    setStatusMessage(modeLabel);

    try {
      const url = URL.createObjectURL(file);
      setImageUrl(url);
      let result: DetectionResult;
      try {
        result = await runBackendDetection(file, activeMode);
      } catch (backendErr) {
        console.warn("Backend detection failed, using mock:", backendErr);
        setStatusMessage("Backend unavailable, using mock detection...");
        result = await runMockDetection(file);
      }
      setDetectionResult(result);
      setStatusMessage("");
    } catch (err) {
      console.error("Detection failed:", err);
      const msg = err instanceof Error ? err.message : "Detection failed";
      setStatusMessage(msg);
      setDetectionResult(null);
    } finally {
      setIsProcessing(false);
      // Stop countdown UI when processing ends.
      if (countdownIntervalRef.current) {
        window.clearInterval(countdownIntervalRef.current);
        countdownIntervalRef.current = null;
      }
      setScanCountdown(null);
      setUghh(false);
    }
  }, [detectionMode]);

  const handleReset = useCallback(() => {
    setImageUrl((prev) => {
      if (prev?.startsWith("blob:")) URL.revokeObjectURL(prev);
      return null;
    });
    setDetectionResult(null);
    setStatusMessage("");
  }, []);

  const handlePaste = useCallback(
    (e: ClipboardEvent) => {
      if (isProcessing) return;
      const item = Array.from(e.clipboardData?.items ?? []).find((i) =>
        i.type.startsWith("image/")
      );
      if (!item) return;
      e.preventDefault();
      const file = item.getAsFile();
      if (file) runDetectionOnFile(file);
    },
    [runDetectionOnFile, isProcessing]
  );

  useEffect(() => {
    document.addEventListener("paste", handlePaste);
    return () => document.removeEventListener("paste", handlePaste);
  }, [handlePaste]);

  const API_BASE = import.meta.env.VITE_API_URL ?? "/api";

  const handleLeftMapScreenScan = useCallback(async () => {
    if (isProcessing) return;
    setIsProcessing(true);
    setStatusMessage(
      "Choose this browser window or your screen — we use the left half of the capture."
    );
    let file: File;
    try {
      file = await captureScreenLeftHalfAsPngFile();
      setStatusMessage("");
    } catch (e: unknown) {
      console.error("Screen capture failed:", e);
      if (e instanceof DOMException && e.name === "NotAllowedError") {
        setStatusMessage("Screen capture canceled");
      } else {
        setStatusMessage(e instanceof Error ? e.message : "Screen capture failed");
      }
      setTimeout(() => setStatusMessage(""), 4000);
      setIsProcessing(false);
      return;
    }
    const scanMode: DetectionMode = mapPanelRef.current?.isSatelliteView()
      ? "satellite"
      : "streetview";
    await runDetectionOnFile(file, scanMode);
  }, [isProcessing, runDetectionOnFile]);

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

    const el = mapPanelRef.current?.getContainerEl();
    if (!el) return;

    // Auto-select mode based on active map view
    const scanMode: DetectionMode = mapPanelRef.current?.isSatelliteView()
      ? "satellite"
      : detectionMode;

    setIsProcessing(true);
    setStatusMessage("Capturing map view...");
    try {
      const canvas = await html2canvas(el, {
        useCORS: true,
        allowTaint: true,
        backgroundColor: null,
        scale: 1,
      });
      const blob = await new Promise<Blob>((resolve, reject) =>
        canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("Canvas capture failed"))), "image/png")
      );
      const file = new File([blob], "map-capture.png", { type: "image/png" });
      setIsProcessing(false);
      runDetectionOnFile(file, scanMode);
    } catch (err) {
      console.error("Map capture failed:", err);
      setStatusMessage("Map capture failed");
      setIsProcessing(false);
    }
  }, [isProcessing, runDetectionOnFile, API_BASE, detectionMode]);

  return (
    <div className="relative flex h-screen w-screen flex-col overflow-hidden bg-background">
      {/* Ambient background (purely visual) */}
      <div className="pointer-events-none absolute inset-0 opacity-[0.55] grid-bg" />
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(1100px_circle_at_18%_12%,hsl(var(--primary)/0.16),transparent_45%),radial-gradient(900px_circle_at_85%_22%,hsl(150_70%_45%/0.10),transparent_45%),radial-gradient(1200px_circle_at_50%_85%,hsl(40_90%_55%/0.08),transparent_55%)]" />
      <div className="pointer-events-none absolute inset-x-0 top-0 h-24 scanline opacity-70" />

      {/* Top bar */}
      <header className="relative z-30 flex items-center justify-between border-b border-border/70 bg-card/70 px-6 py-3 backdrop-blur-md">
        <div className="flex items-center gap-3.5">
          <div className="relative flex h-10 w-10 items-center justify-center rounded-xl border border-border/60 bg-gradient-to-br from-primary/20 via-background/20 to-background/10 shadow-[0_0_0_1px_hsl(var(--primary)/0.12),0_16px_34px_-22px_hsl(var(--primary)/0.55)]">
            <div className="pointer-events-none absolute inset-0 rounded-xl bg-[radial-gradient(14px_circle_at_30%_30%,hsl(var(--primary)/0.35),transparent_60%)]" />
            <Radar className="relative h-4.5 w-4.5 text-primary drop-shadow-[0_0_18px_hsl(var(--primary)/0.35)]" />
          </div>
          <div>
            <h1 className="font-display text-[20px] font-extrabold tracking-tight sm:text-[24px]">
              <span className="bg-gradient-to-r from-sky-400 via-cyan-300 to-violet-400 bg-clip-text text-transparent drop-shadow-[0_10px_30px_rgba(0,0,0,0.55)]">
                CV-SCAN-SATELLITE
              </span>
            </h1>
            <p className="font-mono text-[10px] tracking-[0.18em] text-muted-foreground/85">
              Accessibility & Infrastructure Mapping
            </p>
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
      <div className="relative z-20 flex flex-1 overflow-hidden">
        {/* Left: Map */}
        <div className="w-1/2 shrink-0 overflow-hidden border-r border-border/70 bg-card/20">
          <div className="h-full w-full p-3">
            <div className="h-full w-full overflow-hidden rounded-xl border border-border/70 bg-card/40 shadow-[0_10px_30px_-18px_hsl(var(--primary)/0.22)]">
              <MapPanel
                ref={mapPanelRef}
                onPinDrop={setSelectedPin}
                selectedPin={selectedPin}
                onScanClick={handleLeftMapScreenScan}
                scanDisabled={isProcessing}
              />
            </div>
          </div>
        </div>

        {/* Right: Image analysis */}
        <div className="relative z-20 flex w-1/2 shrink-0 flex-col overflow-hidden bg-card/20">
          {/* Panel header */}
          <div className="relative flex shrink-0 items-center gap-3 border-b border-border/70 bg-card/70 px-5 py-3 backdrop-blur-md">
            <div className="h-2 w-2 rounded-full bg-primary shadow-[0_0_18px_hsl(var(--primary)/0.55)] animate-pulse-glow" />
            <span className="font-mono text-[11px] font-semibold uppercase tracking-[0.26em] text-primary">
              Inference Pipeline
            </span>
            <div className="flex rounded-lg border border-border/70 overflow-hidden bg-background/20">
              <button
                type="button"
                onClick={() => setDetectionMode("streetview")}
                disabled={isProcessing}
                className={`flex items-center gap-1.5 px-3 py-1.5 font-mono text-[10px] tracking-wide transition-colors ${
                  detectionMode === "streetview"
                    ? "bg-primary/20 text-primary"
                    : "text-muted-foreground hover:bg-primary/10 hover:text-primary"
                } ${isProcessing ? "opacity-50 pointer-events-none" : ""}`}
              >
                <DoorOpen className="h-3 w-3" />
                Entrances
              </button>
              <button
                type="button"
                onClick={() => setDetectionMode("satellite")}
                disabled={isProcessing}
                className={`flex items-center gap-1.5 border-l border-border/70 px-3 py-1.5 font-mono text-[10px] tracking-wide transition-colors ${
                  detectionMode === "satellite"
                    ? "bg-primary/20 text-primary"
                    : "text-muted-foreground hover:bg-primary/10 hover:text-primary"
                } ${isProcessing ? "opacity-50 pointer-events-none" : ""}`}
              >
                <Building2 className="h-3 w-3" />
                Buildings
              </button>
            </div>
            <label
              className={`group flex cursor-pointer items-center gap-2 rounded-lg border border-primary/40 bg-primary/10 px-3 py-1.5 font-mono text-xs text-primary transition-all hover:bg-primary/15 hover:border-primary/55 hover:shadow-[0_0_0_1px_hsl(var(--primary)/0.18),0_10px_22px_-16px_hsl(var(--primary)/0.35)] ${
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
              className={`group flex items-center gap-2 rounded-lg border border-primary/40 bg-primary/10 px-3 py-1.5 font-mono text-xs text-primary transition-all hover:bg-primary/15 hover:border-primary/55 hover:shadow-[0_0_0_1px_hsl(var(--primary)/0.18),0_10px_22px_-16px_hsl(var(--primary)/0.35)] ${
                isProcessing ? "pointer-events-none opacity-50" : ""
              }`}
            >
              <ScanSearch className="h-3.5 w-3.5 shrink-0 transition-transform group-hover:-translate-y-[1px]" />
              Scan Map
            </button>
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
              <div className="flex h-full flex-col items-center justify-center gap-5 p-10">
                <Loader2 className="h-10 w-10 animate-spin text-primary drop-shadow-[0_0_18px_hsl(var(--primary)/0.35)]" />
                <div className="text-center">
                  <p className="font-mono text-sm font-semibold text-primary">
                    {statusMessage || "Processing..."}
                  </p>
                  {scanCountdown !== null && !ughh && (
                    <div className="mt-3 text-9xl font-extrabold leading-none text-primary">
                      {scanCountdown}
                    </div>
                  )}
                  {ughh && (
                    <div className="mt-3 text-7xl font-extrabold leading-none text-primary">
                      OOPS!
                    </div>
                  )}
                </div>
              </div>
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
