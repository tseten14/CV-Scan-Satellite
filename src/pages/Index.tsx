import { useState, useCallback } from "react";
import { motion } from "framer-motion";
import { Scan, MapPin, Eye } from "lucide-react";
import MapPanel from "@/components/MapPanel";
import ImageUpload from "@/components/ImageUpload";
import DetectionOverlay from "@/components/DetectionOverlay";
import { runMockDetection } from "@/lib/mockDetection";
import type { MapPin as MapPinType, DetectionResult } from "@/types/detection";

const Index = () => {
  const [selectedPin, setSelectedPin] = useState<MapPinType | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null);

  const handleImageSelect = useCallback(async (file: File) => {
    setIsProcessing(true);
    setImageUrl(URL.createObjectURL(file));
    setDetectionResult(null);

    try {
      const result = await runMockDetection(file);
      setDetectionResult(result);
    } catch {
      console.error("Detection failed");
    } finally {
      setIsProcessing(false);
    }
  }, []);

  const handleReset = useCallback(() => {
    setImageUrl(null);
    setDetectionResult(null);
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
            ) : (
              <ImageUpload
                onImageSelect={handleImageSelect}
                isProcessing={isProcessing}
                hasPin={!!selectedPin}
              />
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
