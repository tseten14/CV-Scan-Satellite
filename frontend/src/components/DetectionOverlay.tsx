import { useRef, useState } from "react";
import { Download } from "lucide-react";
import type { Detection, DetectionEngineId, DetectionResult } from "@/types/detection";

interface DetectionOverlayProps {
  imageUrl: string;
  result: DetectionResult;
  onReset: () => void;
  onUploadClick?: () => void;
  isProcessing?: boolean;
  satelliteMode?: boolean;
  /** True after Scan Map — exports include WGS84 lat/lng. */
  hasMapLinkedPoints?: boolean;
  onDownloadBuildingExport?: () => void;
  /** WGS84 GeoJSON export using current left-map bounds (for uploads / PostGIS — pixel exports lack geometry). */
  onDownloadWgs84FromMapExtent?: () => void;
}

const DetectionOverlay = ({
  imageUrl,
  result,
  onReset,
  onUploadClick,
  isProcessing,
  satelliteMode,
  hasMapLinkedPoints: _hasMapLinkedPoints,
  onDownloadBuildingExport,
  onDownloadWgs84FromMapExtent: _onDownloadWgs84FromMapExtent,
}: DetectionOverlayProps) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [activeTooltip, setActiveTooltip] = useState<string | null>(null);

  const resultEngine: DetectionEngineId = result.engine ?? "sam3";
  const yoloVariant = result.yolo_variant;

  const filteredDetections = result.detections.filter((det) => {
    const lbl = det.label.trim().toLowerCase();
    return lbl !== "car" && lbl !== "truck";
  });

  const LABEL_COLORS: Record<string, string> = {
    Entrance: "hsl(150 80% 45%)",
    Building: "hsl(200 70% 50%)",
    Person: "hsl(210 90% 60%)",
    Car: "hsl(25 95% 55%)",
    Truck: "hsl(30 85% 50%)",
    Bus: "hsl(35 80% 50%)",
    Bicycle: "hsl(180 70% 50%)",
    Motorcycle: "hsl(20 90% 55%)",
    Road: "hsl(0 0% 55%)",
    Sidewalk: "hsl(35 30% 60%)",
    Grass: "hsl(95 60% 45%)",
    Tree: "hsl(120 60% 45%)",
    Vegetation: "hsl(130 70% 40%)",
    Sky: "hsl(210 80% 65%)",

    Pole: "hsl(40 30% 55%)",
    Bench: "hsl(270 50% 55%)",
    Sign: "hsl(50 70% 55%)",
    Chair: "hsl(280 60% 55%)",
    Couch: "hsl(290 50% 50%)",
    "Dining Table": "hsl(300 45% 50%)",
    "Potted Plant": "hsl(120 60% 45%)",
    "Trash can": "hsl(280 50% 55%)",
    Mailbox: "hsl(200 60% 50%)",
    Umbrella: "hsl(320 60% 55%)",
    Backpack: "hsl(230 50% 55%)",
    Boat: "hsl(195 80% 50%)",
    Bottle: "hsl(170 50% 50%)",
    Cup: "hsl(15 60% 55%)",
    Vase: "hsl(160 50% 50%)",
    Clock: "hsl(50 70% 50%)",
    Tv: "hsl(200 70% 50%)",
    Laptop: "hsl(205 65% 50%)",

    "Traffic light": "hsl(55 90% 50%)",
    "Traffic Light": "hsl(55 90% 50%)",
    "Stop Sign": "hsl(0 80% 55%)",
    "Fire hydrant": "hsl(0 70% 50%)",
    "Fire Hydrant": "hsl(0 70% 50%)",
    "Street light": "hsl(45 60% 55%)",
    "Parking Meter": "hsl(45 60% 50%)",
    Dog: "hsl(30 70% 55%)",
    Cat: "hsl(340 60% 55%)",
    Bird: "hsl(190 70% 50%)",
  };

  const getLabelColor = (label: string) => {
    const key = label.charAt(0).toUpperCase() + label.slice(1).toLowerCase();
    return LABEL_COLORS[key] ?? LABEL_COLORS[label] ?? "hsl(185 80% 50%)";
  };

  return (
    <div className="flex h-full flex-col">
      {/* Stats bar */}
      <div className="flex min-w-0 flex-wrap items-center gap-x-6 gap-y-2 border-b border-border/70 bg-card/65 px-5 py-3 backdrop-blur-md">
        <div className="flex min-w-0 flex-wrap items-center gap-x-6 gap-y-1">
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-success shadow-[0_0_16px_hsl(var(--success)/0.35)]" />
            <span className="font-mono text-[11px] tracking-wide text-muted-foreground/90">
              {filteredDetections.length} detections
            </span>
            <span
              className={`rounded border px-1.5 py-0.5 font-mono text-[9px] font-medium uppercase tracking-wider ${
                resultEngine === "yolo"
                  ? "border-amber-500/40 bg-amber-500/10 text-amber-200/95"
                  : "border-violet-500/40 bg-violet-500/10 text-violet-200/95"
              }`}
              title={
                resultEngine === "yolo"
                  ? yoloVariant === "world"
                    ? "YOLO-World — text prompts (doors / entrances)"
                    : "YOLOv8 COCO (Ultralytics)"
                  : "SAM 3 (segmentation)"
              }
            >
              {resultEngine === "yolo"
                ? yoloVariant === "world"
                  ? "YOLO-World"
                  : "YOLOv8"
                : "SAM 3"}
            </span>
            {result.mock && (
              <span
                className="rounded border border-rose-500/45 bg-rose-500/15 px-1.5 py-0.5 font-mono text-[9px] font-medium uppercase tracking-wider text-rose-200/95"
                title="Backend request failed — placeholder shapes only"
              >
                Mock
              </span>
            )}
          </div>
          <span className="font-mono text-[11px] tracking-wide text-muted-foreground/80">
            {result.processing_time_s.toFixed(3)}s
          </span>
          <span className="font-mono text-[11px] tracking-wide text-muted-foreground/80">
            {result.image_width}×{result.image_height}px
          </span>
        </div>
        <div className="ml-auto flex flex-shrink-0 flex-wrap items-center justify-end gap-2 sm:gap-3">
          {onUploadClick && (
            <button
              type="button"
              onClick={onUploadClick}
              disabled={isProcessing}
              className="font-mono text-[11px] tracking-wide text-primary transition-colors hover:text-primary/80 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Upload different
            </button>
          )}
          <button
            onClick={onReset}
            className="font-mono text-[11px] tracking-wide text-primary transition-colors hover:text-primary/80"
          >
            ← New image
          </button>
        </div>
      </div>

      {resultEngine === "yolo" && !satelliteMode && !result.mock && yoloVariant === "coco" && (
        <div className="border-b border-sky-500/35 bg-sky-950/50 px-4 py-2 font-mono text-[10px] leading-snug text-sky-100/95">
          <strong>YOLOv8 (COCO)</strong> has <strong>no door/entrance</strong> class — add{" "}
          <code className="rounded bg-background/50 px-1">yolov8s-worldv2.pt</code> in{" "}
          <code className="rounded bg-background/50 px-1">backend/</code> for fast{" "}
          <strong>YOLO-World</strong> entrance prompts, or use <strong>SAM 3</strong> for masks.
        </div>
      )}

      {/* Image with overlays - SVG viewBox matches image so polygons stay within bounds */}
      <div className="relative flex min-h-0 flex-1 overflow-hidden bg-background">
        <div ref={containerRef} className="relative h-full w-full">
          <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(900px_circle_at_15%_15%,hsl(var(--primary)/0.10),transparent_45%),radial-gradient(900px_circle_at_80%_10%,hsl(40_90%_55%/0.06),transparent_55%)]" />
          <img
            src={imageUrl}
            alt="Analyzed facade"
            className="absolute inset-0 h-full w-full object-contain"
            onLoad={() => setImageLoaded(true)}
          />
          {imageLoaded && (
            <svg
              className="absolute inset-0 h-full w-full pointer-events-none [&>*]:pointer-events-auto"
              viewBox={`0 0 ${result.image_width} ${result.image_height}`}
              preserveAspectRatio="xMidYMid meet"
            >
              {satelliteMode
                ? filteredDetections.map((det) => {
                    if (!det.polygon || det.polygon.length < 3) return null;
                    const pts = det.polygon.map(([x, y]) => `${x},${y}`).join(" ");
                    return (
                      <polygon
                        key={det.id}
                        points={pts}
                        fill="hsla(50, 90%, 55%, 0.45)"
                        stroke="hsl(50, 90%, 55%)"
                        strokeWidth={1.5}
                        className="cursor-pointer"
                        onClick={() => setActiveTooltip(activeTooltip === det.id ? null : det.id)}
                      >
                        <title>{`${det.label} ${(det.confidence * 100).toFixed(1)}%`}</title>
                      </polygon>
                    );
                  })
                : filteredDetections.map((det) => (
                    <DetectionOutline
                      key={det.id}
                      detection={det}
                      index={0}
                      color={getLabelColor(det.label)}
                      isActive={activeTooltip === det.id}
                      onToggle={() =>
                        setActiveTooltip(activeTooltip === det.id ? null : det.id)
                      }
                    />
                  ))}
            </svg>
          )}
        </div>
      </div>

      {/* Detection list — compact summary for satellite, scrollable list for street view */}
      <div className="shrink-0 border-t border-border/70 bg-card/55 px-4 py-2 sm:px-5 sm:py-2.5">
        {satelliteMode ? (
          result.detections.length > 0 && onDownloadBuildingExport ? (
            <button
              type="button"
              onClick={() => onDownloadBuildingExport()}
              disabled={isProcessing}
              className="inline-flex items-center gap-1.5 rounded-md border border-primary/50 bg-primary/10 px-3 py-1.5 font-mono text-[11px] font-medium text-primary transition-colors hover:bg-primary/15 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <Download className="h-3.5 w-3.5 shrink-0" />
              Download: GeoJSON
            </button>
          ) : null
        ) : (
          <div className="flex max-h-28 flex-wrap gap-2.5 overflow-y-auto pr-1">
            {filteredDetections.map((det) => (
              <button
                key={det.id}
                onClick={() => setActiveTooltip(activeTooltip === det.id ? null : det.id)}
                className={`group flex items-center gap-2 rounded-lg border px-2.5 py-1.5 font-mono text-[11px] tracking-wide transition-all ${
                  activeTooltip === det.id
                    ? "border-primary/60 bg-primary/10 text-primary shadow-[0_0_0_1px_hsl(var(--primary)/0.18),0_12px_26px_-22px_hsl(var(--primary)/0.35)]"
                    : "border-border/70 bg-background/10 text-muted-foreground hover:border-primary/40 hover:bg-muted/30"
                }`}
              >
                <div
                  className="h-1.5 w-1.5 rounded-full shadow-[0_0_14px_rgba(0,0,0,0.25)]"
                  style={{ background: getLabelColor(det.label) }}
                />
                {det.label}
                <span className="text-[10px] opacity-70">
                  {(det.confidence * 100).toFixed(1)}%
                </span>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

function DetectionOutline({
  detection,
  color,
  isActive,
  onToggle,
}: {
  detection: Detection;
  index: number;
  color: string;
  isActive: boolean;
  onToggle: () => void;
}) {
  const { bbox, polygon } = detection;
  const tooltipText = `${detection.label} ${(detection.confidence * 100).toFixed(1)}%`;

  if (polygon && polygon.length >= 3) {
    const pointsStr = polygon.map(([x, y]) => `${x},${y}`).join(" ");
    return (
      <g
        className="cursor-pointer"
        onClick={onToggle}
        style={{ opacity: isActive ? 1 : 0.9 }}
      >
        <title>{tooltipText}</title>
        <polygon
          points={pointsStr}
          fill="none"
          stroke={color}
          strokeWidth={2}
          style={{
            filter: `drop-shadow(0 0 6px ${color}66)`,
          }}
        />
        <foreignObject
          x={bbox.xmin}
          y={Math.max(0, bbox.ymin - 24)}
          width={120}
          height={24}
          className="overflow-visible"
        >
          <div
            xmlns="http://www.w3.org/1999/xhtml"
            className="flex items-center gap-1.5 rounded-sm px-1.5 py-0.5 font-mono text-[10px] font-semibold pointer-events-none"
            style={{
              background: color,
              color: "hsl(220 20% 6%)",
            }}
          >
            {detection.label}
          </div>
        </foreignObject>
      </g>
    );
  }

  const x = bbox.xmin;
  const y = bbox.ymin;
  const width = bbox.xmax - bbox.xmin;
  const height = bbox.ymax - bbox.ymin;

  return (
    <g className="cursor-pointer" onClick={onToggle}>
      <title>{tooltipText}</title>
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        fill="none"
        stroke={color}
        strokeWidth={2}
        rx={2}
        style={{
          filter: `drop-shadow(0 0 8px ${color}40)`,
        }}
      />
      <foreignObject x={x} y={Math.max(0, y - 24)} width={120} height={24} className="overflow-visible">
        <div
          xmlns="http://www.w3.org/1999/xhtml"
          className="flex items-center gap-1.5 rounded-sm px-1.5 py-0.5 font-mono text-[10px] font-semibold pointer-events-none"
          style={{ background: color, color: "hsl(220 20% 6%)" }}
        >
          {detection.label}
        </div>
      </foreignObject>
    </g>
  );
}

export default DetectionOverlay;
