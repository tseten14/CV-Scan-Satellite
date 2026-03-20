export interface VenueDetection {
  id: string;
  name: string;
  type: "stadium" | "hospital" | "airport" | "district";
  confidence: number;
  entrances: EntranceMarker[];
  coordinates: { lat: number; lng: number };
  status: "detected" | "scanning" | "verified";
  /** Data file name in /data/entrances/ */
  dataFile?: string;
  /** Label shown in tooltips, e.g. "CTA" */
  sourceLabel?: string;
  /** Hex color for transit markers */
  markerColor?: string;
  /** Initial map zoom level */
  zoom?: number;
  /** Additional transit data files to overlay on the same map */
  extraDataFiles?: { file: string; label: string; color: string }[];
}

export interface EntranceMarker {
  id: string;
  label: string;
  x: number; // percentage position on satellite image
  y: number;
  lat: number; // real GPS coordinates
  lng: number;
  type: "main" | "emergency" | "service" | "vip";
  confidence: number;
}

// ── City venue definitions ──────────────────────────────────────────────────

export const MOCK_VENUES: VenueDetection[] = [
  {
    id: "v-chicago",
    name: "Chicago",
    type: "district",
    confidence: 94.8,
    coordinates: { lat: 41.8826, lng: -87.6226 },
    status: "verified",
    dataFile: "cta.txt",
    sourceLabel: "CTA",
    markerColor: "#ea580c",
    zoom: 11,
    extraDataFiles: [
      { file: "metra.txt", label: "Metra", color: "#a855f7" },
    ],
    entrances: [
      { id: "ch-1", label: "Millennium Park (Cloud Gate)", x: 50, y: 45, lat: 41.8827, lng: -87.6233, type: "main", confidence: 98.0 },
      { id: "ch-2", label: "Art Institute of Chicago", x: 55, y: 55, lat: 41.8796, lng: -87.6275, type: "main", confidence: 96.5 },
      { id: "ch-3", label: "Willis Tower", x: 45, y: 58, lat: 41.8789, lng: -87.6359, type: "main", confidence: 95.8 },
      { id: "ch-4", label: "Union Station", x: 40, y: 52, lat: 41.8787, lng: -87.6402, type: "main", confidence: 95.2 },
      { id: "ch-5", label: "Chicago Theatre", x: 52, y: 42, lat: 41.8853, lng: -87.6278, type: "vip", confidence: 94.5 },
      { id: "ch-6", label: "CTA State/Lake", x: 54, y: 40, lat: 41.8857, lng: -87.6278, type: "main", confidence: 93.8 },
      { id: "ch-7", label: "Daley Center", x: 48, y: 48, lat: 41.8842, lng: -87.6299, type: "service", confidence: 92.1 },
    ],
  },
  {
    id: "v-sf",
    name: "San Francisco Bay Area",
    type: "district",
    confidence: 93.5,
    coordinates: { lat: 37.7749, lng: -122.4194 },
    status: "verified",
    dataFile: "sfmta.txt",
    sourceLabel: "SFMTA",
    markerColor: "#3b82f6",
    zoom: 10,
    entrances: [
      { id: "sf-1", label: "Embarcadero Station", x: 50, y: 50, lat: 37.7929, lng: -122.3970, type: "main", confidence: 97.0 },
      { id: "sf-2", label: "Powell Street Station", x: 48, y: 52, lat: 37.7844, lng: -122.4077, type: "main", confidence: 95.5 },
      { id: "sf-3", label: "Caltrain 4th & King", x: 52, y: 54, lat: 37.7765, lng: -122.3947, type: "main", confidence: 96.0 },
    ],
  },
  {
    id: "v-la",
    name: "Los Angeles",
    type: "district",
    confidence: 93.0,
    coordinates: { lat: 34.0522, lng: -118.2437 },
    status: "verified",
    dataFile: "lametro.txt",
    sourceLabel: "LA METRO",
    markerColor: "#ef4444",
    zoom: 10,
    entrances: [
      { id: "la-1", label: "Union Station", x: 50, y: 50, lat: 34.0555, lng: -118.2348, type: "main", confidence: 97.0 },
      { id: "la-2", label: "7th Street / Metro Center", x: 48, y: 52, lat: 34.0489, lng: -118.2584, type: "main", confidence: 95.0 },
    ],
  },
  {
    id: "v-boston",
    name: "Boston",
    type: "district",
    confidence: 92.5,
    coordinates: { lat: 42.3601, lng: -71.0589 },
    status: "verified",
    dataFile: "mbta.txt",
    sourceLabel: "MBTA",
    markerColor: "#22c55e",
    zoom: 12,
    entrances: [
      { id: "bos-1", label: "Park Street Station", x: 50, y: 50, lat: 42.3562, lng: -71.0624, type: "main", confidence: 96.5 },
      { id: "bos-2", label: "South Station", x: 52, y: 54, lat: 42.3519, lng: -71.0552, type: "main", confidence: 95.0 },
    ],
  },
  {
    id: "v-nyc",
    name: "New York City",
    type: "district",
    confidence: 95.0,
    coordinates: { lat: 40.7580, lng: -73.9855 },
    status: "verified",
    dataFile: "mta.txt",
    sourceLabel: "MTA",
    markerColor: "#0ea5e9",
    zoom: 11,
    entrances: [
      { id: "nyc-1", label: "Times Sq-42 St", x: 50, y: 50, lat: 40.7560, lng: -73.9870, type: "main", confidence: 98.0 },
      { id: "nyc-2", label: "Grand Central-42 St", x: 54, y: 48, lat: 40.7527, lng: -73.9772, type: "main", confidence: 97.0 },
    ],
  },
  {
    id: "v-paris",
    name: "Paris",
    type: "district",
    confidence: 94.0,
    coordinates: { lat: 48.8566, lng: 2.3522 },
    status: "verified",
    dataFile: "parismetro.txt",
    sourceLabel: "PARIS METRO",
    markerColor: "#ec4899",
    zoom: 12,
    entrances: [
      { id: "par-1", label: "Châtelet-Les Halles", x: 50, y: 50, lat: 48.8622, lng: 2.3469, type: "main", confidence: 97.5 },
      { id: "par-2", label: "Gare du Nord", x: 52, y: 44, lat: 48.8809, lng: 2.3553, type: "main", confidence: 96.0 },
    ],
  },
  {
    id: "v-london",
    name: "London",
    type: "district",
    confidence: 93.5,
    coordinates: { lat: 51.5074, lng: -0.1278 },
    status: "verified",
    dataFile: "tfl.txt",
    sourceLabel: "TFL",
    markerColor: "#6366f1",
    zoom: 11,
    entrances: [
      { id: "lon-1", label: "King's Cross St. Pancras", x: 50, y: 50, lat: 51.5302, lng: -0.1240, type: "main", confidence: 97.0 },
      { id: "lon-2", label: "Westminster", x: 48, y: 54, lat: 51.5013, lng: -0.1246, type: "main", confidence: 96.0 },
    ],
  },
  {
    id: "v-dc",
    name: "Washington D.C.",
    type: "district",
    confidence: 93.0,
    coordinates: { lat: 38.9072, lng: -77.0369 },
    status: "verified",
    dataFile: "wmata.txt",
    sourceLabel: "WMATA",
    markerColor: "#14b8a6",
    zoom: 11,
    entrances: [
      { id: "dc-1", label: "Metro Center", x: 50, y: 50, lat: 38.8981, lng: -77.0268, type: "main", confidence: 97.0 },
      { id: "dc-2", label: "Union Station", x: 54, y: 46, lat: 38.8983, lng: -77.0072, type: "main", confidence: 96.0 },
    ],
  },
];
