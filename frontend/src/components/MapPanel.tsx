import { useEffect, useRef, useState, useCallback, forwardRef, useImperativeHandle } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { MapPin } from "@/types/detection";
import type { MapScanBounds } from "@/lib/satelliteScanMarkers";
import { Map, Search, Satellite } from "lucide-react";

type MapView = "map" | "satellite" | "streetview";

interface MapPanelProps {
  onPinDrop: (pin: MapPin) => void;
  selectedPin: MapPin | null;
  /** Building centroids from a satellite map capture (distinct from the user pin). */
  buildingMarkers?: Array<{ lat: number; lng: number }>;
}

export interface MapPanelHandle {
  getContainerEl: () => HTMLDivElement | null;
  isStreetView: () => boolean;
  isSatelliteView: () => boolean;
  getPin: () => { lat: number; lng: number } | null;
  getHeading: () => number;
  /** Current visible map bounds (for correlating a screenshot with lat/lng). */
  getVisibleMapBounds: () => MapScanBounds | null;
}

const MapPanel = forwardRef<MapPanelHandle, MapPanelProps>(({
  onPinDrop,
  selectedPin,
  buildingMarkers = [],
}, ref) => {
  const rootRef = useRef<HTMLDivElement>(null);
  const [activeView, setActiveView] = useState<MapView>("map");

  useImperativeHandle(ref, () => ({
    getContainerEl: () => mapRef.current,
    isStreetView: () => activeView === "streetview",
    isSatelliteView: () => activeView === "satellite",
    getPin: () => selectedPin ? { lat: selectedPin.lat, lng: selectedPin.lng } : null,
    getHeading: () => 0,
    getVisibleMapBounds: () => {
      const map = mapInstanceRef.current;
      if (!map) return null;
      const b = map.getBounds();
      return {
        west: b.getWest(),
        east: b.getEast(),
        north: b.getNorth(),
        south: b.getSouth(),
      };
    },
  }), [activeView, selectedPin]);
  const [searchQuery, setSearchQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  /** Bumped when the Leaflet map is created/destroyed so building markers re-sync (Strict Mode / remount). */
  const [mapVersion, setMapVersion] = useState(0);

  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const markerRef = useRef<L.Marker | null>(null);
  const buildingMarkersLayerRef = useRef<L.LayerGroup | null>(null);
  const osmLayerRef = useRef<L.TileLayer | null>(null);
  const satLayerRef = useRef<L.TileLayer | null>(null);
  const annArborBoundaryLayerRef = useRef<L.LayerGroup | null>(null);

  /**
   * Load and render the Ann Arbor city boundary outline.
   *
   * We store the boundary polyline data under `frontend/public/` so it can be loaded
   * via the browser without any API keys or backend services.
   *
   * The source file is a CSV containing `lat,lon` pairs per line, with blank lines
   * separating multiple polyline segments (rings).
   */
  const loadAnnArborBoundary = useCallback(async (map: L.Map) => {
    try {
      const res = await fetch("/ann_arbor_city_boundary.csv");
      if (!res.ok) throw new Error(`Failed to load boundary csv: ${res.status}`);
      const text = await res.text();

      // Parse: each non-empty row is one point; blank lines split into separate rings/segments.
      const segments: Array<Array<[number, number]>> = [];
      let current: Array<[number, number]> = [];

      for (const rawLine of text.split(/\r?\n/)) {
        const line = rawLine.trim();
        if (!line) {
          if (current.length > 1) segments.push(current);
          current = [];
          continue;
        }
        const [latStr, lonStr] = line.split(",");
        const lat = Number(latStr);
        const lng = Number(lonStr);
        if (Number.isFinite(lat) && Number.isFinite(lng)) {
          current.push([lat, lng]);
        }
      }
      if (current.length > 1) segments.push(current);

      if (annArborBoundaryLayerRef.current) {
        annArborBoundaryLayerRef.current.remove();
        annArborBoundaryLayerRef.current = null;
      }

      // The boundary dataset may include multiple rings (e.g., township islands).
      // For this research view we only want the outer border, so keep only the
      // longest ring(s) by approximate perimeter length.
      const segScore = (seg: Array<[number, number]>) => {
        let sum = 0;
        for (let i = 1; i < seg.length; i++) {
          const [latA, lngA] = seg[i - 1];
          const [latB, lngB] = seg[i];
          const dx = latB - latA;
          const dy = lngB - lngA;
          sum += Math.sqrt(dx * dx + dy * dy);
        }
        return sum;
      };

      const scored = segments
        .map((seg) => ({ seg, score: segScore(seg) }))
        .sort((a, b) => b.score - a.score);

      const maxScore = scored[0]?.score ?? 0;
      const keepThreshold = maxScore * 0.25; // keep outer ring parts, drop smaller islands
      const kept = scored.filter((x) => x.score >= keepThreshold).map((x) => x.seg);

      const layer = L.layerGroup();
      for (const seg of kept) {
        L.polyline(seg, {
          color: "#b94a00", // dark orange
          weight: 2,
          opacity: 0.95,
          fill: false,
          interactive: false,
        }).addTo(layer);
      }

      layer.addTo(map);
      annArborBoundaryLayerRef.current = layer;
    } catch (e) {
      // Non-fatal: map still works without the boundary overlay.
      console.warn("Ann Arbor boundary load failed:", e);
    }
  }, []);

  useEffect(() => {
    if (!mapRef.current || mapInstanceRef.current) return;

    const map = L.map(mapRef.current, {
      center: [42.2808, -83.743],
      zoom: 15,
      zoomControl: true,
    });

    const osmLayer = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>',
      maxZoom: 19,
    });

    const satLayer = L.tileLayer(
      "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
      {
        attribution: '&copy; Google',
        maxZoom: 21,
      }
    );

    osmLayerRef.current = osmLayer;
    satLayerRef.current = satLayer;
    osmLayer.addTo(map);

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
    // Draw Ann Arbor boundary outline for focused research.
    loadAnnArborBoundary(map);

    const bumpMapVersion = () => setMapVersion((v) => v + 1);
    map.whenReady(bumpMapVersion);

    return () => {
      markerRef.current = null;
      if (buildingMarkersLayerRef.current) {
        buildingMarkersLayerRef.current.remove();
        buildingMarkersLayerRef.current = null;
      }
      osmLayerRef.current = null;
      satLayerRef.current = null;
      if (annArborBoundaryLayerRef.current) {
        annArborBoundaryLayerRef.current.remove();
        annArborBoundaryLayerRef.current = null;
      }
      map.remove();
      mapInstanceRef.current = null;
      bumpMapVersion();
    };
  }, [onPinDrop]);

  // Satellite scan: one marker per detected building (distinct styling from user pin).
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    if (buildingMarkersLayerRef.current) {
      buildingMarkersLayerRef.current.clearLayers();
    } else {
      buildingMarkersLayerRef.current = L.layerGroup().addTo(map);
    }

    const layer = buildingMarkersLayerRef.current;
    if (!layer || buildingMarkers.length === 0) return;

    const buildingIcon = L.divIcon({
      className: "building-scan-marker",
      html: `<div style="
        width: 12px; height: 12px;
        background: hsl(38 92% 50%);
        border: 2px solid hsl(45 100% 70%);
        border-radius: 50%;
        box-shadow: 0 0 10px hsl(38 92% 50% / 0.55), 0 0 20px hsl(38 92% 50% / 0.25);
      "></div>`,
      iconSize: [12, 12],
      iconAnchor: [6, 6],
    });

    for (const m of buildingMarkers) {
      if (!Number.isFinite(m.lat) || !Number.isFinite(m.lng)) continue;
      L.marker([m.lat, m.lng], { icon: buildingIcon }).addTo(layer);
    }
  }, [buildingMarkers, mapVersion]);

  useEffect(() => {
    if (selectedPin) setActiveView("streetview");
  }, [selectedPin]);

  useEffect(() => {
    const map = mapInstanceRef.current;
    const osm = osmLayerRef.current;
    const sat = satLayerRef.current;
    if (!map || !osm || !sat) return;
    if (activeView === "satellite") {
      if (map.hasLayer(osm)) map.removeLayer(osm);
      if (!map.hasLayer(sat)) map.addLayer(sat);
    } else {
      if (map.hasLayer(sat)) map.removeLayer(sat);
      if (!map.hasLayer(osm)) map.addLayer(osm);
    }
  }, [activeView]);

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
    <div ref={rootRef} className="relative h-full w-full">
      {/* Soft vignette + highlight (visual only) */}
      <div className="pointer-events-none absolute inset-0 z-[5] bg-[radial-gradient(900px_circle_at_18%_12%,hsl(var(--primary)/0.10),transparent_40%),radial-gradient(700px_circle_at_90%_18%,hsl(40_90%_55%/0.06),transparent_45%)]" />
      {/* Header bar */}
      <div className="absolute top-0 left-0 right-0 z-[1000] flex flex-col gap-2.5 border-b border-border/70 bg-card/65 px-4 py-3 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="h-2 w-2 rounded-full bg-primary shadow-[0_0_18px_hsl(var(--primary)/0.55)] animate-pulse-glow" />
          <span className="font-mono text-[11px] font-semibold uppercase tracking-[0.26em] text-primary">
            {activeView === "streetview" && selectedPin
              ? "Street View"
              : activeView === "satellite"
              ? "Satellite View"
              : "Spatial Selection"}
          </span>

          <div className="ml-auto flex items-center gap-2">
            <div className="flex rounded-lg border border-border/70 overflow-hidden bg-background/20">
              <button
                type="button"
                onClick={() => setActiveView("map")}
                className={`flex items-center gap-1.5 px-3 py-1.5 font-mono text-[10px] tracking-wide transition-colors ${
                  activeView === "map"
                    ? "bg-primary/20 text-primary"
                    : "text-muted-foreground hover:bg-primary/10 hover:text-primary"
                }`}
              >
                <Map className="h-3 w-3" />
                Map
              </button>
              <button
                type="button"
                onClick={() => setActiveView("satellite")}
                className={`flex items-center gap-1.5 border-l border-border/70 px-3 py-1.5 font-mono text-[10px] tracking-wide transition-colors ${
                  activeView === "satellite"
                    ? "bg-primary/20 text-primary"
                    : "text-muted-foreground hover:bg-primary/10 hover:text-primary"
                }`}
              >
                <Satellite className="h-3 w-3" />
                Satellite
              </button>
              {selectedPin && (
                <button
                  type="button"
                  onClick={() => setActiveView("streetview")}
                  className={`flex items-center gap-1.5 border-l border-border/70 px-3 py-1.5 font-mono text-[10px] tracking-wide transition-colors ${
                    activeView === "streetview"
                      ? "bg-primary/20 text-primary"
                      : "text-muted-foreground hover:bg-primary/10 hover:text-primary"
                  }`}
                >
                  Street View
                </button>
              )}
            </div>
            {activeView !== "streetview" && (
              <span className="font-mono text-[10px] tracking-wide text-muted-foreground/90">
                Click to place pin
              </span>
            )}
          </div>
        </div>

        {/* Search bar - visible when map is shown */}
        {activeView !== "streetview" && (
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
              className="group flex items-center gap-1.5 rounded-lg border border-primary/40 bg-primary/10 px-3 py-1.5 font-mono text-[10px] tracking-wide text-primary transition-all hover:bg-primary/15 hover:border-primary/55 hover:shadow-[0_0_0_1px_hsl(var(--primary)/0.16),0_10px_22px_-16px_hsl(var(--primary)/0.35)] disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {searching ? (
                <span className="animate-pulse">Searching…</span>
              ) : (
                <>
                  <Search className="h-3 w-3 transition-transform group-hover:-translate-y-[1px]" />
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

      {/* Coordinates bar removed per UX request */}

      {/* Map */}
      <div
        ref={mapRef}
        className="h-full w-full"
        style={{ display: activeView === "streetview" && selectedPin ? "none" : "block" }}
      />

      {/* Street View embed */}
      {activeView === "streetview" && selectedPin && streetViewEmbedUrl && (
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
        </div>
      )}
    </div>
  );
});

MapPanel.displayName = "MapPanel";

export default MapPanel;
