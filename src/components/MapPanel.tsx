import { useEffect, useRef, useState, useCallback } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { MapPin } from "@/types/detection";
import { Copy, Check, Map, Search } from "lucide-react";

interface MapPanelProps {
  onPinDrop: (pin: MapPin) => void;
  selectedPin: MapPin | null;
}

const MapPanel = ({
  onPinDrop,
  selectedPin,
}: MapPanelProps) => {
  const [copied, setCopied] = useState(false);
  const [showStreetView, setShowStreetView] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);

  const copyCoords = useCallback(() => {
    if (!selectedPin) return;
    const text = `${selectedPin.lat.toFixed(6)}, ${selectedPin.lng.toFixed(6)}`;
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [selectedPin]);

  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const markerRef = useRef<L.Marker | null>(null);

  useEffect(() => {
    if (!mapRef.current || mapInstanceRef.current) return;

    const map = L.map(mapRef.current, {
      center: [42.2808, -83.743],
      zoom: 15,
      zoomControl: true,
    });

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>',
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
        markerRef.current = L.marker([lat, lng], { icon: cyberIcon }).addTo(
          map
        );
      }
      onPinDrop({ lat, lng, label: `${lat.toFixed(5)}, ${lng.toFixed(5)}` });
    });

    mapInstanceRef.current = map;

    return () => {
      markerRef.current = null;
      map.remove();
      mapInstanceRef.current = null;
    };
  }, [onPinDrop]);

  useEffect(() => {
    if (selectedPin) setShowStreetView(true);
  }, [selectedPin]);

  const searchAddress = useCallback(async () => {
    const query = searchQuery.trim();
    if (!query || !mapInstanceRef.current) return;

    setSearching(true);
    setSearchError(null);
    try {
      const res = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=1`,
        {
          headers: {
            Accept: "application/json",
            "User-Agent": "CV-Scan-Satellite/1.0 (urban accessibility mapping)",
          },
        }
      );
      const data = await res.json();
      if (!data || data.length === 0) {
        setSearchError("Address not found");
        return;
      }
      const { lat, lon } = data[0];
      const latNum = parseFloat(lat);
      const lngNum = parseFloat(lon);

      const map = mapInstanceRef.current;
      map.setView([latNum, lngNum], 17);

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

      if (markerRef.current) {
        markerRef.current.setLatLng([latNum, lngNum]);
      } else {
        markerRef.current = L.marker([latNum, lngNum], { icon: cyberIcon }).addTo(map);
      }

      onPinDrop({ lat: latNum, lng: lngNum, label: data[0].display_name || query });
    } catch (err) {
      setSearchError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setSearching(false);
    }
  }, [searchQuery, onPinDrop]);

  const streetViewEmbedUrl = selectedPin
    ? `https://maps.google.com/maps?layer=c&cbll=${selectedPin.lat},${selectedPin.lng}&cbp=12,0,,0,0&output=svembed`
    : null;

  return (
    <div className="relative h-full w-full">
      {/* Header bar */}
      <div className="absolute top-0 left-0 right-0 z-[1000] flex flex-col gap-2 border-b border-border bg-card/90 px-4 py-2.5 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <div className="h-2 w-2 rounded-full bg-primary animate-pulse-glow" />
          <span className="font-mono text-xs font-semibold uppercase tracking-widest text-primary">
            {showStreetView && selectedPin ? "Street View" : "Spatial Selection"}
          </span>

          <div className="ml-auto flex items-center gap-2">
          {selectedPin && (
            <button
              type="button"
              onClick={() => setShowStreetView(!showStreetView)}
              className="flex items-center gap-1.5 rounded border border-primary/40 bg-primary/10 px-2 py-1 font-mono text-[10px] text-primary transition-colors hover:bg-primary/20"
            >
              {showStreetView ? (
                <>
                  <Map className="h-3 w-3" />
                  Map
                </>
              ) : (
                <span>Street View</span>
              )}
            </button>
          )}
          {!showStreetView && (
            <span className="font-mono text-[10px] text-muted-foreground">
              Click to place pin
            </span>
          )}
        </div>
        </div>

        {/* Search bar - visible when map is shown */}
        {!showStreetView && (
          <form
            onSubmit={(e) => {
              e.preventDefault();
              searchAddress();
            }}
            className="flex items-center gap-2"
          >
            <div className="relative flex-1">
              <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => {
                  setSearchQuery(e.target.value);
                  setSearchError(null);
                }}
                placeholder="Search address (e.g. 722 Spring St, Ann Arbor)"
                className="w-full rounded border border-border bg-background/80 py-1.5 pl-8 pr-3 font-mono text-xs text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/30"
                disabled={searching}
              />
            </div>
            <button
              type="submit"
              disabled={searching || !searchQuery.trim()}
              className="flex items-center gap-1.5 rounded border border-primary/40 bg-primary/10 px-3 py-1.5 font-mono text-[10px] text-primary transition-colors hover:bg-primary/20 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {searching ? (
                <span className="animate-pulse">Searching…</span>
              ) : (
                <>
                  <Search className="h-3 w-3" />
                  Go
                </>
              )}
            </button>
            {searchError && (
              <span className="font-mono text-[10px] text-destructive">{searchError}</span>
            )}
          </form>
        )}
      </div>

      {/* Coordinates bar */}
      {selectedPin && (
        <div className="absolute bottom-0 left-0 right-0 z-[1000] border-t border-border bg-card/90 px-4 py-2 backdrop-blur-sm">
          <div className="flex items-center gap-4 font-mono text-xs">
            <span className="text-muted-foreground">LAT</span>
            <span className="text-primary select-all">
              {selectedPin.lat.toFixed(6)}
            </span>
            <span className="text-muted-foreground">LNG</span>
            <span className="text-primary select-all">
              {selectedPin.lng.toFixed(6)}
            </span>
            <button
              type="button"
              onClick={copyCoords}
              className="ml-auto flex items-center gap-1.5 rounded border border-primary/40 bg-primary/10 px-2 py-1 text-primary transition-colors hover:bg-primary/20"
            >
              {copied ? (
                <>
                  <Check className="h-3 w-3" />
                  Copied
                </>
              ) : (
                <>
                  <Copy className="h-3 w-3" />
                  Copy
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* Map */}
      <div
        ref={mapRef}
        className="h-full w-full"
        style={{ display: showStreetView && selectedPin ? "none" : "block" }}
      />

      {/* Street View embed */}
      {showStreetView && selectedPin && streetViewEmbedUrl && (
        <div className="flex h-full w-full flex-col">
          <div className="flex-1 pt-10 pb-9 overflow-hidden">
            <iframe
              src={streetViewEmbedUrl}
              className="h-full w-full border-0"
              allowFullScreen
              loading="lazy"
              referrerPolicy="no-referrer-when-downgrade"
              title="Google Street View"
            />
          </div>
          <div className="absolute bottom-9 left-1/2 z-[1001] -translate-x-1/2">
            <p className="rounded bg-card/90 px-3 py-1.5 font-mono text-[10px] text-muted-foreground backdrop-blur-sm text-center">
              Screenshot this view → paste ⌘V in detection panel
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default MapPanel;
