import { useRef, useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Detection, DetectionResult } from "@/types/detection";

interface DetectionOverlayProps {
  imageUrl: string;
  result: DetectionResult;
  onReset: () => void;
  onUploadClick?: () => void;
  isProcessing?: boolean;
}

const DetectionOverlay = ({ imageUrl, result, onReset, onUploadClick, isProcessing }: DetectionOverlayProps) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState({ x: 1, y: 1 });
  const [activeTooltip, setActiveTooltip] = useState<string | null>(null);
  const [imageLoaded, setImageLoaded] = useState(false);

  useEffect(() => {
    const updateScale = () => {
      if (!containerRef.current) return;
      const img = containerRef.current.querySelector("img");
      if (!img) return;
      setScale({
        x: img.clientWidth / result.image_width,
        y: img.clientHeight / result.image_height,
      });
    };

    window.addEventListener("resize", updateScale);
    return () => window.removeEventListener("resize", updateScale);
  }, [result, imageLoaded]);

  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    setImageLoaded(true);
    const img = e.currentTarget;
    setScale({
      x: img.clientWidth / result.image_width,
      y: img.clientHeight / result.image_height,
    });
  };

  const getConfidenceColor = (conf: number) => {
    if (conf >= 0.9) return "hsl(150 70% 45%)";
    if (conf >= 0.8) return "hsl(185 80% 50%)";
    return "hsl(40 90% 55%)";
  };

  return (
    <div className="flex h-full flex-col">
      {/* Stats bar */}
      <div className="flex items-center gap-6 border-b border-border bg-card/80 px-4 py-2 backdrop-blur-sm">
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-success" />
          <span className="font-mono text-xs text-muted-foreground">
            {result.detections.length} detections
          </span>
        </div>
        <span className="font-mono text-xs text-muted-foreground">
          {result.processing_time_ms}ms
        </span>
        <span className="font-mono text-xs text-muted-foreground">
          {result.image_width}×{result.image_height}px
        </span>
        <div className="ml-auto flex gap-3">
          {onUploadClick && (
            <button
              type="button"
              onClick={onUploadClick}
              disabled={isProcessing}
              className="font-mono text-xs text-primary hover:text-primary/80 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Upload different
            </button>
          )}
          <button
            onClick={onReset}
            className="font-mono text-xs text-primary hover:text-primary/80 transition-colors"
          >
            ← New image
          </button>
        </div>
      </div>

      {/* Image with overlays */}
      <div className="relative flex flex-1 items-center justify-center overflow-auto bg-background p-4">
        <div ref={containerRef} className="relative inline-block max-h-full max-w-full">
          <img
            src={imageUrl}
            alt="Analyzed facade"
            className="max-h-[calc(100vh-200px)] max-w-full object-contain"
            onLoad={handleImageLoad}
          />

          {imageLoaded &&
            result.detections.map((det, i) => (
              <BoundingBox
                key={det.id}
                detection={det}
                scale={scale}
                index={i}
                color={getConfidenceColor(det.confidence)}
                isActive={activeTooltip === det.id}
                onToggle={() =>
                  setActiveTooltip(activeTooltip === det.id ? null : det.id)
                }
              />
            ))}
        </div>
      </div>

      {/* Detection list */}
      <div className="border-t border-border bg-card/50 px-4 py-2">
        <div className="flex flex-wrap gap-3">
          {result.detections.map((det) => (
            <button
              key={det.id}
              onClick={() => setActiveTooltip(activeTooltip === det.id ? null : det.id)}
              className={`flex items-center gap-2 rounded-sm border px-2 py-1 font-mono text-[11px] transition-all ${
                activeTooltip === det.id
                  ? "border-primary bg-primary/10 text-primary"
                  : "border-border text-muted-foreground hover:border-primary/40"
              }`}
            >
              <div
                className="h-1.5 w-1.5 rounded-full"
                style={{ background: getConfidenceColor(det.confidence) }}
              />
              {det.label}
              <span className="text-[10px] opacity-60">
                {(det.confidence * 100).toFixed(1)}%
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

function BoundingBox({
  detection,
  scale,
  index,
  color,
  isActive,
  onToggle,
}: {
  detection: Detection;
  scale: { x: number; y: number };
  index: number;
  color: string;
  isActive: boolean;
  onToggle: () => void;
}) {
  const { bbox } = detection;
  const left = bbox.xmin * scale.x;
  const top = bbox.ymin * scale.y;
  const width = (bbox.xmax - bbox.xmin) * scale.x;
  const height = (bbox.ymax - bbox.ymin) * scale.y;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: index * 0.15, duration: 0.3 }}
      className="absolute cursor-pointer"
      style={{ left, top, width, height }}
      onClick={onToggle}
    >
      {/* Bbox border */}
      <div
        className="absolute inset-0 rounded-[2px]"
        style={{
          border: `2px solid ${color}`,
          boxShadow: `0 0 8px ${color}40, inset 0 0 8px ${color}10`,
        }}
      />

      {/* Label */}
      <div
        className="absolute -top-6 left-0 flex items-center gap-1.5 rounded-sm px-1.5 py-0.5 font-mono text-[10px] font-semibold"
        style={{ background: color, color: "hsl(220 20% 6%)" }}
      >
        {detection.label}
      </div>

      {/* Tooltip */}
      {isActive && (
        <motion.div
          initial={{ opacity: 0, y: 5 }}
          animate={{ opacity: 1, y: 0 }}
          className="absolute left-1/2 -bottom-20 z-50 -translate-x-1/2 rounded-md border border-border bg-popover p-3 shadow-lg"
          style={{ minWidth: 180 }}
        >
          <div className="font-mono text-xs">
            <div className="mb-1 font-semibold text-foreground">{detection.label}</div>
            <div className="flex justify-between text-muted-foreground">
              <span>Confidence</span>
              <span style={{ color }}>{(detection.confidence * 100).toFixed(1)}%</span>
            </div>
            <div className="mt-1 flex justify-between text-muted-foreground">
              <span>Position</span>
              <span className="text-foreground">
                [{detection.bbox.xmin}, {detection.bbox.ymin}]
              </span>
            </div>
            <div className="flex justify-between text-muted-foreground">
              <span>Size</span>
              <span className="text-foreground">
                {detection.bbox.xmax - detection.bbox.xmin}×
                {detection.bbox.ymax - detection.bbox.ymin}px
              </span>
            </div>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}

export default DetectionOverlay;
