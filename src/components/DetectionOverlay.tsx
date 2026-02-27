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

  const LABEL_COLORS: Record<string, string> = {
    Door: "hsl(150 80% 45%)",
    Entrance: "hsl(150 80% 45%)",
    Building: "hsl(200 70% 50%)",
    Person: "hsl(210 90% 60%)",
    Car: "hsl(25 95% 55%)",
    Truck: "hsl(30 85% 50%)",
    Bus: "hsl(35 80% 50%)",
    Bicycle: "hsl(180 70% 50%)",
    "Traffic light": "hsl(55 90% 50%)",
    "Trash can": "hsl(280 50% 55%)",
    Pole: "hsl(40 30% 55%)",
    Bench: "hsl(270 50% 55%)",
    Sign: "hsl(50 70% 55%)",
    "Fire hydrant": "hsl(0 70% 50%)",
    "Street light": "hsl(45 60% 55%)",
    Mailbox: "hsl(200 60% 50%)",
    Vegetation: "hsl(130 70% 40%)",
    Grass: "hsl(95 60% 45%)",
    Tree: "hsl(120 60% 45%)",
    Motorcycle: "hsl(20 90% 55%)",
    Bicycle: "hsl(180 70% 50%)",
    "Traffic Light": "hsl(55 90% 50%)",
    "Stop Sign": "hsl(0 80% 55%)",
    "Fire Hydrant": "hsl(0 70% 50%)",
    "Parking Meter": "hsl(45 60% 50%)",
    Bench: "hsl(270 50% 55%)",
    Chair: "hsl(280 60% 55%)",
    Couch: "hsl(290 50% 50%)",
    "Dining Table": "hsl(300 45% 50%)",
    "Potted Plant": "hsl(120 60% 45%)",
    Dog: "hsl(30 70% 55%)",
    Cat: "hsl(340 60% 55%)",
    Bird: "hsl(190 70% 50%)",
    Backpack: "hsl(230 50% 55%)",
    Umbrella: "hsl(320 60% 55%)",
    Tv: "hsl(200 70% 50%)",
    Laptop: "hsl(205 65% 50%)",
    Clock: "hsl(50 70% 50%)",
    Vase: "hsl(160 50% 50%)",
    Bottle: "hsl(170 50% 50%)",
    Cup: "hsl(15 60% 55%)",
    Boat: "hsl(195 80% 50%)",
    Vegetation: "hsl(130 70% 40%)",
    Sky: "hsl(210 80% 65%)",
    Road: "hsl(0 0% 55%)",
    Grass: "hsl(95 60% 45%)",
    Sidewalk: "hsl(35 30% 60%)",
  };

  const getLabelColor = (label: string) => {
    const key = label.charAt(0).toUpperCase() + label.slice(1).toLowerCase();
    return LABEL_COLORS[key] ?? LABEL_COLORS[label] ?? "hsl(185 80% 50%)";
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

      {/* Image with overlays - fills entire panel */}
      <div className="relative flex min-h-0 flex-1 overflow-hidden bg-background">
        <div ref={containerRef} className="relative h-full w-full">
          <img
            src={imageUrl}
            alt="Analyzed facade"
            className="h-full w-full object-contain"
            onLoad={handleImageLoad}
          />

          {imageLoaded &&
            result.detections.map((det, i) => (
              <DetectionOutline
                key={det.id}
                detection={det}
                scale={scale}
                index={i}
                color={getLabelColor(det.label)}
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
                style={{ background: getLabelColor(det.label) }}
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

function DetectionOutline({
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
  const { bbox, polygon } = detection;

  const tooltipContent = (
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
  );

  if (polygon && polygon.length >= 3) {
    const pointsStr = polygon
      .map(([x, y]) => `${x * scale.x},${y * scale.y}`)
      .join(" ");
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: index * 0.15, duration: 0.3 }}
        className="absolute left-0 top-0 cursor-pointer"
        style={{ width: "100%", height: "100%" }}
        onClick={onToggle}
      >
        <svg
          className="pointer-events-none absolute left-0 top-0 overflow-visible"
          style={{ width: "100%", height: "100%" }}
        >
          <polygon
            points={pointsStr}
            fill="none"
            stroke={color}
            strokeWidth={2}
            style={{
              filter: `drop-shadow(0 0 6px ${color}66)`,
            }}
          />
        </svg>
        <div
          className="absolute flex items-center gap-1.5 rounded-sm px-1.5 py-0.5 font-mono text-[10px] font-semibold"
          style={{
            background: color,
            color: "hsl(220 20% 6%)",
            left: bbox.xmin * scale.x,
            top: bbox.ymin * scale.y - 24,
          }}
        >
          {detection.label}
        </div>
        {isActive && tooltipContent}
      </motion.div>
    );
  }

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
      <div
        className="absolute inset-0 rounded-[2px]"
        style={{
          border: `2px solid ${color}`,
          boxShadow: `0 0 8px ${color}40, inset 0 0 8px ${color}10`,
        }}
      />
      <div
        className="absolute -top-6 left-0 flex items-center gap-1.5 rounded-sm px-1.5 py-0.5 font-mono text-[10px] font-semibold"
        style={{ background: color, color: "hsl(220 20% 6%)" }}
      >
        {detection.label}
      </div>
      {isActive && tooltipContent}
    </motion.div>
  );
}

export default DetectionOverlay;
