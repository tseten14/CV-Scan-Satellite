import { useState } from "react";
import type { VenueDetection, EntranceMarker } from "@/data/venues";
import satelliteMichiganStadium from "@/assets/satellite-michigan-stadium.jpg";

const markerColors: Record<EntranceMarker["type"], string> = {
  main: "bg-primary",
  emergency: "bg-geo-danger",
  service: "bg-geo-amber",
  vip: "bg-geo-success",
};

const markerLabels: Record<EntranceMarker["type"], string> = {
  main: "Main",
  emergency: "Emergency",
  service: "Service",
  vip: "VIP",
};

const SatelliteViewer = ({
  venue,
  onEntranceSelect,
}: {
  venue: VenueDetection;
  onEntranceSelect: (entrance: EntranceMarker) => void;
}) => {
  const [hoveredEntrance, setHoveredEntrance] = useState<string | null>(null);
  const [isScanning, setIsScanning] = useState(false);

  const handleScan = () => {
    setIsScanning(true);
    setTimeout(() => setIsScanning(false), 3000);
  };

  return (
    <div className="relative rounded-lg overflow-hidden border border-border bg-card">
      {/* Header bar */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-secondary/50">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-geo-success animate-pulse" />
          <span className="font-mono text-sm text-foreground">
            SAT-VIEW // {venue.name.toUpperCase()}
          </span>
        </div>
        <div className="flex items-center gap-4">
          <span className="font-mono text-xs text-muted-foreground">
            {venue.coordinates.lat.toFixed(4)}°N, {Math.abs(venue.coordinates.lng).toFixed(4)}°W
          </span>
          <button
            onClick={handleScan}
            className="px-3 py-1 text-xs font-mono bg-primary/10 text-primary border border-primary/30 rounded hover:bg-primary/20 transition-colors"
          >
            RE-SCAN
          </button>
        </div>
      </div>

      {/* Image container */}
      <div className="relative aspect-square max-h-[500px] overflow-hidden">
        <img
          src={satelliteMichiganStadium}
          alt={`Satellite view of ${venue.name}`}
          className="w-full h-full object-cover"
        />

        {/* Grid overlay */}
        <div className="absolute inset-0 grid-overlay opacity-20 pointer-events-none" />

        {/* Scan animation */}
        {isScanning && (
          <div className="absolute inset-0 pointer-events-none overflow-hidden">
            <div className="w-full h-1/4 scanline animate-scan" />
            <div className="absolute inset-0 border-2 border-primary/30 animate-pulse" />
          </div>
        )}

        {/* Entrance markers */}
        {venue.entrances.map((entrance) => (
          <div
            key={entrance.id}
            className="absolute cursor-pointer group"
            style={{ left: `${entrance.x}%`, top: `${entrance.y}%`, transform: "translate(-50%, -50%)" }}
            onMouseEnter={() => setHoveredEntrance(entrance.id)}
            onMouseLeave={() => setHoveredEntrance(null)}
            onClick={() => onEntranceSelect(entrance)}
          >
            {/* Pulse ring */}
            <div
              className={`absolute w-8 h-8 -left-2 -top-2 rounded-full ${markerColors[entrance.type]} opacity-30 animate-pulse-marker`}
            />
            {/* Marker dot */}
            <div
              className={`relative w-4 h-4 rounded-full ${markerColors[entrance.type]} border-2 border-background shadow-lg z-10`}
            />

            {/* Tooltip */}
            {hoveredEntrance === entrance.id && (
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-3 px-3 py-2 bg-card border border-border rounded-md shadow-xl z-20 whitespace-nowrap">
                <div className="text-xs font-semibold text-foreground">{entrance.label}</div>
                <div className="flex items-center gap-2 mt-1">
                  <span className={`inline-block w-2 h-2 rounded-full ${markerColors[entrance.type]}`} />
                  <span className="text-xs text-muted-foreground">{markerLabels[entrance.type]}</span>
                  <span className="text-xs font-mono text-primary">{entrance.confidence}%</span>
                </div>
                <div className="absolute top-full left-1/2 -translate-x-1/2 w-2 h-2 bg-card border-r border-b border-border rotate-45 -mt-1" />
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Legend */}
      <div className="flex items-center gap-6 px-4 py-3 border-t border-border bg-secondary/30">
        {Object.entries(markerLabels).map(([type, label]) => (
          <div key={type} className="flex items-center gap-2">
            <div className={`w-2.5 h-2.5 rounded-full ${markerColors[type as EntranceMarker["type"]]}`} />
            <span className="text-xs text-muted-foreground">{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default SatelliteViewer;
