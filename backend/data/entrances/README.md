# Transit Entrance Dataset

GPS-located transit station entrance points derived from GTFS (General Transit Feed Specification) data. This directory contains the core datasets consumed by the Venue Finder AI application's frontend map, backend search API, and Python analysis scripts.

---

## Overview

The dataset contains **15,525 entrance records** across **10 transit agencies** spanning **2,399 unique station names** in **8 cities** across **North America and Europe**. Each row represents a single physical entrance point — a door, stairway, elevator, or accessibility ramp — at a transit station, identified by its station name, route/line ID, and precise GPS coordinates.

The data was extracted from publicly available GTFS feeds published by each transit agency. GTFS (General Transit Feed Specification) is the industry-standard format used by Google Maps, Apple Maps, and other transit applications to describe transit routes, schedules, and stop locations.

---

## Data Format

All files use **CSV format** with the following columns:

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| *(index)* | integer | `0` | Optional row index. Present in most files; some files (e.g., `parismetro.txt`) omit it. The system handles both formats automatically. |
| `stationName` | string | `"Times Sq-42 St"` | Human-readable station name as published by the transit agency. May include line information in some datasets (e.g., CTA entries like `"18thPink Line"`). |
| `uniqueId` | string | `"R16"` | Internal route, line, or stop identifier from the GTFS feed. Format varies by agency. |
| `lat` | float | `40.755983` | Latitude in decimal degrees (WGS 84 coordinate reference system). Precision varies from 5 to 14 decimal places depending on the source. |
| `lon` | float | `-73.986229` | Longitude in decimal degrees (WGS 84). Negative values indicate west of the Prime Meridian. |

### Important Notes

- **Multiple rows per station**: A single station commonly has 2–10 entrance rows. Major hubs (e.g., Châtelet–Les Halles in Paris, Times Sq-42 St in NYC) can have 50–250+ entrance points.
- **Quoted fields**: Some station names contain commas and are wrapped in double quotes (e.g., `"Washington/WabashBrown, Purple, Orange, Pink, Green Lines"`). The frontend CSV parser handles this.
- **Coordinate precision**: Coordinates are rounded to 6 decimal places (~0.11m precision) when consumed by the application.

### Example (`cta.txt`)

```csv
,stationName,uniqueId,lat,lon
0,18th,18thPink Line,41.857849,-87.669144
1,35th/Archer,35th/ArcherOrange Line,41.829274,-87.680632
2,Adams/Wabash,"Adams/WabashBrown, Purple, Orange, Pink, Green Lines",41.879504,-87.625823
```

### Example (`parismetro.txt` — no index column)

```csv
stationName,uniqueId,lat,lon
Abbesses,22059,48.88248016274083,2.3373144088037865
Abbesses,22059,48.882285642979824,2.3379168705488924
```

---

## Files

### Transit Entrance Data

| File | Agency | City | Country | Records | Unique Stations | Avg Entrances/Station | File Size |
|------|--------|------|---------|---------:|----------------:|----------------------:|----------:|
| `bart.txt` | BART (Bay Area Rapid Transit) | San Francisco Bay Area | USA | 132 | 50 | 2.6 | 6.7 KB |
| `cta.txt` | CTA (Chicago Transit Authority) | Chicago | USA | 381 | 109 | 3.5 | 22.6 KB |
| `lametro.txt` | LA Metro | Los Angeles | USA | 229 | 101 | 2.3 | 12.6 KB |
| `mbta.txt` | MBTA (Massachusetts Bay Transportation Authority) | Boston | USA | 323 | 98 | 3.3 | 16.0 KB |
| `metra.txt` | Metra (Chicago commuter rail) | Chicago suburbs | USA | 247 | 90 | 2.7 | 15.2 KB |
| `mta.txt` | MTA (Metropolitan Transportation Authority) | New York City | USA | 2,120 | 372 | 5.7 | 95.8 KB |
| `parismetro.txt` | Paris Métro (RATP) | Paris | France | 11,111 | 1,193 | 9.3 | 670.2 KB |
| `sfmta.txt` | SFMTA (San Francisco Municipal Transportation Agency) | San Francisco | USA | 230 | 73 | 3.2 | 14.4 KB |
| `tfl.txt` | TfL (Transport for London) | London | UK | 513 | 221 | 2.3 | 34.4 KB |
| `wmata.txt` | WMATA (Washington Metropolitan Area Transit Authority) | Washington D.C. | USA | 239 | 98 | 2.4 | 12.4 KB |
| **Total** | **10 agencies** | **8 cities** | **3 countries** | **15,525** | **2,399** | — | **900.4 KB** |

### Bounding Box Index

`bounding.txt` contains the geographic bounding box for each agency's service area. The backend API uses this file to determine which source CSVs to search when a bounding-box query parameter is provided.

| Column | Type | Description |
|--------|------|-------------|
| `file` | string | CSV filename (e.g., `"cta.txt"`) |
| `latMin` | float | Southern boundary latitude |
| `latMax` | float | Northern boundary latitude |
| `lonMin` | float | Western boundary longitude |
| `lonMax` | float | Eastern boundary longitude |

Example:
```csv
,file,latMin,latMax,lonMin,lonMax
cta.txt,cta.txt,41.721558,42.073623,-87.904004,-87.605799
mta.txt,mta.txt,40.5121599,40.9036489,-74.2529363,-73.754178
```

---

## How the Data Is Consumed

### 1. Frontend — Static CSV Loading

**File**: `src/lib/transit-data.ts`

The frontend fetches CSV files directly from `/data/entrances/*.txt` via the Vite static file server. The `loadTransitData()` function:

1. Fetches the file via `fetch()`
2. Splits into lines and detects headers
3. Finds `stationName`, `lat`, `lon` column indices
4. Parses each row into a `TransitEntrance` object
5. Rounds coordinates to 6 decimal places
6. Returns the array for rendering as Leaflet map markers

The `VenueMap` component renders each entrance as a custom SVG circle marker, color-coded by the agency's configured marker color.

### 2. Backend — Fuzzy Search API

**File**: `backend/entrances.py`

The FastAPI backend loads CSV files with pandas and provides fuzzy station name search:

1. Reads `bounding.txt` to identify which agency files overlap the requested bounding box
2. Loads matching CSV files with `pd.read_csv()`
3. Filters rows by bounding box coordinates
4. Uses `rapidfuzz.process.extract()` with `fuzz.token_sort_ratio` to match station names
5. Returns up to 15 matches per agency with a default score cutoff of 45

### 3. Analysis Scripts — Statistical Visualization

**File**: `scripts/report_visualization.py`

The Python analysis script loads all CSV files, computes per-agency statistics, and generates 10 charts plus a written report. See `scripts/README.md` for details.

---

## Geographic Coverage

### North America (8 agencies)

| Region | Agencies | Latitude Range | Longitude Range |
|--------|----------|----------------|-----------------|
| West Coast | BART, SFMTA, LA Metro | 33.8°–38.4°N | 122.7°–117.9°W |
| Midwest | CTA, Metra | 41.6°–42.1°N | 87.9°–87.5°W |
| East Coast | MTA, MBTA, WMATA | 38.8°–42.4°N | 77.5°–71.0°W |

### Europe (2 agencies)

| Region | Agencies | Latitude Range | Longitude Range |
|--------|----------|----------------|-----------------|
| Western Europe | Paris Métro, TfL | 48.0°–51.7°N | 0.6°W–3.5°E |

### Key Geographic Facts

- **Furthest south**: LA Metro (33.8°N — Los Angeles)
- **Furthest north**: TfL (51.7°N — London)
- **Furthest west**: SFMTA/BART (122.7°W — San Francisco Bay Area)
- **Furthest east**: Paris Métro (3.5°E — suburban Paris)
- **No geographic overlap** exists between any two agency datasets — each system occupies a distinct geographic region, making cross-system comparison purely structural rather than spatial.

---

## Data Quality Notes

- All coordinates have been verified to fall within each agency's published service area boundaries
- Station names follow each agency's official naming convention and may include line/route suffixes
- Duplicate entrance coordinates may exist where multiple routes share a physical entrance point
- The `uniqueId` field format varies by agency: some use route codes (MTA: `R16`), stop IDs (BART: `12TH`), or internal identifiers (Paris Métro: `22059`)
