/**
 * Load and parse CTA entrance data from data/entrances/cta.txt (served as /data/entrances/cta.txt).
 * No backend — static file only.
 */
export interface CtaEntrance {
  stationName: string;
  source: string;
  lat: number;
  lon: number;
}

/** Parse a CSV line handling quoted fields (e.g. "Adams/WabashBrown, Purple, Orange, Pink, Green Lines") */
function parseCSVLine(line: string): string[] {
  const result: string[] = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"') inQuotes = !inQuotes;
    else if (c === "," && !inQuotes) {
      result.push(current.trim());
      current = "";
    } else current += c;
  }
  result.push(current.trim());
  return result;
}

/**
 * Fetch and parse all CTA entrances from /data/entrances/cta.txt.
 * Columns: index, stationName, uniqueId, lat, lon
 */
export async function loadCtaFromData(): Promise<CtaEntrance[]> {
  const res = await fetch("/data/entrances/cta.txt");
  if (!res.ok) throw new Error(`Failed to load CTA data: ${res.status}`);
  const text = await res.text();
  const lines = text.split(/\r?\n/).filter((line) => line.trim());
  if (lines.length < 2) return [];
  const out: CtaEntrance[] = [];
  for (let i = 1; i < lines.length; i++) {
    const row = parseCSVLine(lines[i]);
    const stationName = row[1] ?? "";
    const lat = parseFloat(row[3]);
    const lon = parseFloat(row[4]);
    if (!stationName || Number.isNaN(lat) || Number.isNaN(lon)) continue;
    out.push({
      stationName,
      source: "CTA",
      lat: Math.round(lat * 1e6) / 1e6,
      lon: Math.round(lon * 1e6) / 1e6,
    });
  }
  return out;
}
