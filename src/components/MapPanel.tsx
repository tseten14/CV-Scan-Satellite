import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { MapPin } from "@/types/detection";

interface MapPanelProps {
  onPinDrop: (pin: MapPin) => void;
  selectedPin: MapPin | null;
}

const MapPanel = ({ onPinDrop, selectedPin }: MapPanelProps) => {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const markerRef = useRef<L.Marker | null>(null);

  useEffect(() => {
    if (!mapRef.current || mapInstanceRef.current) return;

    const map = L.map(mapRef.current, {
      center: [40.7128, -74.006],
      zoom: 15,
      zoomControl: true,
    });

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>',
    }).addTo(map);

    const cyberIcon = L.divIcon({
      className: "custom-pin",
      html: `<div style="
        width: 20px; height: 20px;
        background: hsl(185 80% 50%);
        border: 2px solid hsl(185 80% 70%);
        border-radius: 50%;
        box-shadow: 0 0 15px hsl(185 80% 50% / 0.6), 0 0 30px hsl(185 80% 50% / 0.3);
        position: relative;
      "><div style="
        position: absolute; top: 50%; left: 50%;
        width: 6px; height: 6px;
        background: hsl(220 20% 6%);
        border-radius: 50%;
        transform: translate(-50%, -50%);
      "></div></div>`,
      iconSize: [20, 20],
      iconAnchor: [10, 10],
    });

    map.on("click", (e: L.LeafletMouseEvent) => {
      const { lat, lng } = e.latlng;

      if (markerRef.current) {
        markerRef.current.setLatLng([lat, lng]);
      } else {
        markerRef.current = L.marker([lat, lng], { icon: cyberIcon }).addTo(map);
      }

      onPinDrop({ lat, lng, label: `${lat.toFixed(5)}, ${lng.toFixed(5)}` });
    });

    mapInstanceRef.current = map;

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  return (
    <div className="relative h-full w-full">
      {/* Header bar */}
      <div className="absolute top-0 left-0 right-0 z-[1000] flex items-center gap-3 border-b border-border bg-card/90 px-4 py-2.5 backdrop-blur-sm">
        <div className="h-2 w-2 rounded-full bg-primary animate-pulse-glow" />
        <span className="font-mono text-xs font-semibold uppercase tracking-widest text-primary">
          Spatial Selection
        </span>
        <span className="ml-auto font-mono text-[10px] text-muted-foreground">
          Click to place pin
        </span>
      </div>

      {/* Coordinates bar */}
      {selectedPin && (
        <div className="absolute bottom-0 left-0 right-0 z-[1000] border-t border-border bg-card/90 px-4 py-2 backdrop-blur-sm">
          <div className="flex items-center gap-4 font-mono text-xs">
            <span className="text-muted-foreground">LAT</span>
            <span className="text-primary">{selectedPin.lat.toFixed(6)}</span>
            <span className="text-muted-foreground">LNG</span>
            <span className="text-primary">{selectedPin.lng.toFixed(6)}</span>
          </div>
        </div>
      )}

      <div ref={mapRef} className="h-full w-full" />
    </div>
  );
};

export default MapPanel;
