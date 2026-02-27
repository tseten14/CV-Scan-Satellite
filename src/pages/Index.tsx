import { useState, useCallback, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { Scan, MapPin, Eye, Loader2, Upload } from "lucide-react";
import MapPanel from "@/components/MapPanel";
import DetectionOverlay from "@/components/DetectionOverlay";
import { runBackendDetection } from "@/lib/backendDetection";
import { runMockDetection } from "@/lib/mockDetection";
import type { MapPin as MapPinType, DetectionResult } from "@/types/detection";

const Index = () => {
  const [selectedPin, setSelectedPin] = useState<MapPinType | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null);
  const [statusMessage, setStatusMessage] = useState<string>("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const runDetectionOnFile = useCallback(async (file: File) => {
    setIsProcessing(true);
    setImageUrl((prev) => {
      if (prev?.startsWith("blob:")) URL.revokeObjectURL(prev);
      return null;
    });
    setDetectionResult(null);
    setStatusMessage("Running detection... (may take 2–4 min)");

    try {
      const url = URL.createObjectURL(file);
      setImageUrl(url);
      let result: DetectionResult;
      try {
        result = await runBackendDetection(file);
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
    }
  }, []);

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

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-background">
      {/* Top bar */}
      <header className="flex items-center justify-between border-b border-border bg-card px-5 py-2.5">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-md bg-primary/10 glow-border">
            <Scan className="h-4 w-4 text-primary" />
          </div>
          <div>
            <h1 className="font-mono text-sm font-bold uppercase tracking-wider text-foreground glow-text">
              CV-SCAN-GEOAI
            </h1>
            <p className="font-mono text-[10px] text-muted-foreground">
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
          <div className="ml-2 rounded-sm border border-border px-2 py-1 font-mono text-[10px] text-muted-foreground">
            Door / entrance detection
          </div>
        </div>
      </header>

      {/* Split panes */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left: Map */}
        <div className="w-1/2 shrink-0 overflow-hidden border-r border-border">
          <MapPanel onPinDrop={setSelectedPin} selectedPin={selectedPin} />
        </div>

        {/* Right: Image analysis */}
        <div className="relative z-20 flex w-1/2 shrink-0 flex-col overflow-hidden bg-card/30">
          {/* Panel header */}
          <div className="relative flex shrink-0 items-center gap-3 border-b border-border bg-card/80 px-4 py-2.5 backdrop-blur-sm">
            <div className="h-2 w-2 rounded-full bg-primary animate-pulse-glow" />
            <span className="font-mono text-xs font-semibold uppercase tracking-widest text-primary">
              Inference Pipeline
            </span>
            <label
              className={`flex cursor-pointer items-center gap-2 rounded border border-primary/50 bg-primary/10 px-3 py-1.5 font-mono text-xs text-primary transition-colors hover:bg-primary/20 ${
                isProcessing ? "pointer-events-none opacity-50" : ""
              }`}
            >
              <Upload className="h-3.5 w-3.5 shrink-0" />
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
          </div>

          <div className="flex flex-1 flex-col overflow-auto">
            {detectionResult && imageUrl ? (
              <DetectionOverlay
                imageUrl={imageUrl}
                result={detectionResult}
                onReset={handleReset}
                onUploadClick={() => document.getElementById("facade-file-input")?.click()}
                isProcessing={isProcessing}
              />
            ) : imageUrl && statusMessage && !isProcessing ? (
              <div className="flex flex-1 flex-col items-center justify-center gap-4 p-8">
                <img
                  src={imageUrl}
                  alt="Uploaded"
                  className="max-h-64 rounded-md border border-border object-contain"
                />
                <p className="font-mono text-sm text-destructive">{statusMessage}</p>
                <div className="flex gap-3">
                  <button
                    type="button"
                    onClick={() => document.getElementById("facade-file-input")?.click()}
                    className="rounded border border-primary bg-primary/10 px-3 py-1.5 font-mono text-xs text-primary hover:bg-primary/20"
                  >
                    Try again
                  </button>
                  <button
                    type="button"
                    onClick={handleReset}
                    className="rounded border border-border px-3 py-1.5 font-mono text-xs text-muted-foreground hover:bg-muted"
                  >
                    Upload different
                  </button>
                </div>
              </div>
            ) : isProcessing ? (
              <div className="flex h-full flex-col items-center justify-center gap-4 p-8">
                <Loader2 className="h-10 w-10 animate-spin text-primary" />
                <div className="text-center">
                  <p className="font-mono text-sm font-semibold text-primary">
                    {statusMessage || "Processing..."}
                  </p>
                </div>
              </div>
            ) : (
              <div
                className="flex min-h-0 flex-1 flex-col items-center justify-center gap-6 p-8 text-center"
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
                    className="block w-full max-w-xs font-mono text-xs file:mr-3 file:cursor-pointer file:rounded-md file:border-0 file:bg-primary file:px-4 file:py-2 file:font-semibold file:text-primary-foreground hover:file:bg-primary/90"
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
