import { useState, useCallback, useEffect } from "react";
import { motion } from "framer-motion";
import { Scan, MapPin, Eye, Loader2 } from "lucide-react";
import MapPanel from "@/components/MapPanel";
import DetectionOverlay from "@/components/DetectionOverlay";
import { runMockDetection } from "@/lib/mockDetection";
import { fetchFacadeImage } from "@/lib/fetchFacadeImage";
import type { MapPin as MapPinType, DetectionResult } from "@/types/detection";

const Index = () => {
  const [selectedPin, setSelectedPin] = useState<MapPinType | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null);
  const [statusMessage, setStatusMessage] = useState<string>("");

  // Auto-fetch facade image and run detection when a pin is dropped
  useEffect(() => {
    if (!selectedPin) return;

    let cancelled = false;

    const runPipeline = async () => {
      setIsProcessing(true);
      setImageUrl(null);
      setDetectionResult(null);

      try {
        // Step 1: Fetch street-level image
        setStatusMessage("Fetching street-level imagery...");
        const { blob, url } = await fetchFacadeImage(selectedPin);
        if (cancelled) return;

        setImageUrl(url);

        // Step 2: Run detection on fetched image
        setStatusMessage("Running inference pipeline...");
        const file = new File([blob], "facade.jpg", { type: "image/jpeg" });
        const result = await runMockDetection(file);
        if (cancelled) return;

        setDetectionResult(result);
        setStatusMessage("");
      } catch (err) {
        console.error("Pipeline failed:", err);
        setStatusMessage("Failed to process location");
      } finally {
        if (!cancelled) setIsProcessing(false);
      }
    };

    runPipeline();
    return () => { cancelled = true; };
  }, [selectedPin]);

  const handleReset = useCallback(() => {
    setImageUrl(null);
    setDetectionResult(null);
    setStatusMessage("");
  }, []);

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
              UrbanVision
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
            Model: YOLOv8-Door-v2 (mock)
          </div>
        </div>
      </header>

      {/* Split panes */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left: Map */}
        <div className="w-1/2 border-r border-border">
          <MapPanel onPinDrop={setSelectedPin} selectedPin={selectedPin} />
        </div>

        {/* Right: Image analysis */}
        <div className="flex w-1/2 flex-col bg-card/30">
          {/* Panel header */}
          <div className="flex items-center gap-3 border-b border-border bg-card/80 px-4 py-2.5 backdrop-blur-sm">
            <div className="h-2 w-2 rounded-full bg-primary animate-pulse-glow" />
            <span className="font-mono text-xs font-semibold uppercase tracking-widest text-primary">
              Inference Pipeline
            </span>
          </div>

          <div className="flex-1 overflow-hidden">
            {detectionResult && imageUrl ? (
              <DetectionOverlay
                imageUrl={imageUrl}
                result={detectionResult}
                onReset={handleReset}
              />
            ) : isProcessing ? (
              <div className="flex h-full flex-col items-center justify-center gap-4 p-8">
                <Loader2 className="h-10 w-10 animate-spin text-primary" />
                <div className="text-center">
                  <p className="font-mono text-sm font-semibold text-primary">
                    {statusMessage || "Processing..."}
                  </p>
                  <p className="mt-1 font-mono text-xs text-muted-foreground">
                    {selectedPin
                      ? `${selectedPin.lat.toFixed(4)}, ${selectedPin.lng.toFixed(4)}`
                      : ""}
                  </p>
                </div>
                {imageUrl && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="mt-2 overflow-hidden rounded-md border border-border"
                  >
                    <img
                      src={imageUrl}
                      alt="Fetched facade"
                      className="h-40 w-auto object-cover opacity-60"
                    />
                  </motion.div>
                )}
              </div>
            ) : (
              <div className="flex h-full flex-col items-center justify-center p-8 text-center">
                <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full border border-border bg-secondary">
                  <MapPin className="h-7 w-7 text-muted-foreground" />
                </div>
                <p className="font-mono text-sm text-muted-foreground">
                  Click a building on the map to begin analysis
                </p>
                <p className="mt-1 font-mono text-[10px] text-muted-foreground/60">
                  Street-level imagery will be fetched automatically
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
