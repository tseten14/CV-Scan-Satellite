import { useState, useEffect } from "react";
import Navbar from "@/components/Navbar";
import VenueMap from "@/components/VenueMap";
import VenueCard from "@/components/VenueCard";
import EntranceDetail from "@/components/EntranceDetail";
import { MOCK_VENUES, type EntranceMarker } from "@/data/venues";
import { type TransitEntrance, loadTransitData } from "@/lib/transit-data";
import { ScanLine, Target, Layers, Train, Loader2, MapPin } from "lucide-react";

const VenuesPage = () => {
  const [selectedVenueId, setSelectedVenueId] = useState<string>(MOCK_VENUES[0].id);
  const [selectedEntrance, setSelectedEntrance] = useState<string | null>(null);
  const [cityEntrances, setCityEntrances] = useState<TransitEntrance[]>([]);
  const [extraLayers, setExtraLayers] = useState<{ entrances: TransitEntrance[]; color: string; label: string }[]>([]);
  const [cityLoading, setCityLoading] = useState(false);
  const [cityError, setCityError] = useState<string | null>(null);

  const activeVenue = MOCK_VENUES.find((v) => v.id === selectedVenueId) ?? MOCK_VENUES[0];
  const hasDataFile = !!activeVenue.dataFile;

  const handleVenueSelect = (venueId: string) => {
    setSelectedVenueId(venueId);
    setSelectedEntrance(null);
    setCityEntrances([]);
    setExtraLayers([]);
    setCityError(null);
  };

  const loadCityData = async () => {
    if (!activeVenue.dataFile || !activeVenue.sourceLabel) return;
    setCityLoading(true);
    setCityError(null);
    try {
      const entrances = await loadTransitData(activeVenue.dataFile, activeVenue.sourceLabel);
      setCityEntrances(entrances);

      // Load extra data files (e.g. Metra for Chicago)
      if (activeVenue.extraDataFiles) {
        const extras = await Promise.all(
          activeVenue.extraDataFiles.map(async (extra) => {
            const data = await loadTransitData(extra.file, extra.label);
            return { entrances: data, color: extra.color, label: extra.label };
          })
        );
        setExtraLayers(extras);
      } else {
        setExtraLayers([]);
      }
    } catch (e) {
      setCityError(e instanceof Error ? e.message : "Failed to load data");
      setCityEntrances([]);
      setExtraLayers([]);
    } finally {
      setCityLoading(false);
    }
  };

  // Auto-load transit data when a venue with a data file is selected
  useEffect(() => {
    if (!activeVenue.dataFile) return;
    if (cityEntrances.length === 0 && !cityLoading) loadCityData();
  }, [selectedVenueId]);

  const handleEntranceSelect = (entrance: EntranceMarker) => {
    setSelectedEntrance(entrance.id);
  };

  return (
    <div className="h-full min-h-0 overflow-y-auto bg-background">
      <Navbar />

      {/* Dashboard Section */}
      <section className="relative py-12 px-6 pt-8">
        <div className="max-w-7xl mx-auto">
          {/* Section header */}
          <div className="mb-10">
            <div className="flex items-center gap-2 mb-3">
              <ScanLine className="w-5 h-5 text-primary" />
              <span className="font-mono text-sm text-primary uppercase tracking-wider">
                Analysis Dashboard
              </span>
            </div>
            <h2 className="text-3xl md:text-4xl font-bold text-foreground">
              Detected Venues &amp; Entrances
            </h2>
            <p className="text-muted-foreground mt-2 max-w-xl">
              Select a city to view transit station entrances plotted on satellite imagery.
            </p>
          </div>

          {/* Stats bar */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            {(() => {
              const allEntrances = [...cityEntrances, ...extraLayers.flatMap((l) => l.entrances)];
              const totalLoaded = allEntrances.length;
              return [
                { icon: Target, label: "Cities", value: `${MOCK_VENUES.length}`, color: "text-primary" },
                { icon: Layers, label: "Loaded Entrances", value: totalLoaded > 0 ? `${totalLoaded}` : "—", color: "text-geo-amber" },
              ];
            })().map((stat) => (
              <div
                key={stat.label}
                className="p-4 rounded-lg border border-border bg-card"
              >
                <div className="flex items-center gap-2 mb-2">
                  <stat.icon className={`w-4 h-4 ${stat.color}`} />
                  <span className="text-xs text-muted-foreground uppercase tracking-wider">
                    {stat.label}
                  </span>
                </div>
                <span className="text-xl font-bold font-mono text-foreground">
                  {stat.value}
                </span>
              </div>
            ))}
          </div>

          {/* Main layout */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Venue list */}
            <div className="lg:col-span-3 space-y-3">
              <h3 className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-3">
                Select City
              </h3>
              <div className="space-y-2 max-h-[600px] overflow-y-auto pr-1">
                {MOCK_VENUES.map((venue) => (
                  <VenueCard
                    key={venue.id}
                    venue={venue}
                    isActive={venue.id === selectedVenueId}
                    onClick={() => handleVenueSelect(venue.id)}
                  />
                ))}
              </div>
            </div>

            {/* Viewer area */}
            <div className="lg:col-span-6 space-y-4">
              <VenueMap
                venue={activeVenue}
                selectedEntranceId={selectedEntrance}
                onEntranceSelect={handleEntranceSelect}
                transitEntrances={[]}
                cityEntrances={cityEntrances}
                cityColor={activeVenue.markerColor ?? "#ea580c"}
                cityZoom={activeVenue.zoom ?? 12}
                extraLayers={extraLayers}
              />
            </div>

            {/* Entrance details + Transit data */}
            <div className="lg:col-span-3 space-y-6">
              <div className="space-y-3">
                <h3 className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-3">
                  Landmark Points ({activeVenue.entrances.length})
                </h3>
                {activeVenue.entrances.map((entrance) => (
                  <EntranceDetail
                    key={entrance.id}
                    entrance={entrance}
                    isSelected={selectedEntrance === entrance.id}
                    onClick={() => handleEntranceSelect(entrance)}
                  />
                ))}
              </div>

              {/* Transit entrances from data file */}
              <div className="space-y-3 pt-4 border-t border-border">
                <h3 className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-3 flex items-center gap-2">
                  <MapPin className="w-3.5 h-3.5" />
                  {activeVenue.sourceLabel ?? activeVenue.name} Entrances
                </h3>
                <p className="text-xs text-muted-foreground">
                  All locations from <code className="bg-muted px-1 rounded">data/entrances/{activeVenue.dataFile}</code>. Shown as colored dots on the map.
                </p>
                {hasDataFile && (
                  <>
                    <button
                      type="button"
                      onClick={loadCityData}
                      disabled={cityLoading}
                      className="w-full px-3 py-2 text-xs font-mono rounded border border-primary/50 bg-primary/10 text-primary hover:bg-primary/20 disabled:opacity-50 flex items-center justify-center gap-1.5"
                      style={{
                        borderColor: activeVenue.markerColor ? `${activeVenue.markerColor}80` : undefined,
                        backgroundColor: activeVenue.markerColor ? `${activeVenue.markerColor}15` : undefined,
                        color: activeVenue.markerColor ?? undefined,
                      }}
                    >
                      {cityLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Train className="w-3.5 h-3.5" />}
                      {cityLoading ? "Loading…" : "Reload data"}
                    </button>
                    {cityError && (
                      <p className="text-xs text-amber-600 dark:text-amber-400">{cityError}</p>
                    )}
                    {cityEntrances.length > 0 && (
                      <div className="space-y-1.5 max-h-52 overflow-y-auto">
                        <p className="text-xs text-muted-foreground">
                          {cityEntrances.length} entrance point(s) — unique stations:{" "}
                          {[...new Set(cityEntrances.map((e) => e.stationName))].length}
                        </p>
                        <ul className="text-xs font-mono space-y-1">
                          {[...new Set(cityEntrances.map((e) => e.stationName))].slice(0, 25).map((name) => {
                            const count = cityEntrances.filter((e) => e.stationName === name).length;
                            return (
                              <li key={name} className="flex justify-between py-0.5 border-b border-border/50 last:border-0">
                                <span className="text-foreground truncate">{name}</span>
                                <span className="text-muted-foreground shrink-0 ml-2">{count}</span>
                              </li>
                            );
                          })}
                        </ul>
                        {[...new Set(cityEntrances.map((e) => e.stationName))].length > 25 && (
                          <p className="text-xs text-muted-foreground">+ more stations on map</p>
                        )}
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border py-8 px-6">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <span className="text-xs text-muted-foreground font-mono">
            GeoAI Entrance Detection System © 2026
          </span>
          <span className="text-xs text-muted-foreground font-mono">
            GTFS data • Satellite + map views
          </span>
        </div>
      </footer>
    </div>
  );
};

export default VenuesPage;
