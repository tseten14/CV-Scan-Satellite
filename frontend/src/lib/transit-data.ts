/**
 * Generic loader for transit entrance data files (CSV format).
 * Handles both formats:
 *  - With leading index column: ,stationName,uniqueId,lat,lon
 *  - Without leading index column: stationName,uniqueId,lat,lon  (e.g. parismetro.txt)
 */

export interface TransitEntrance {
  stationName: string;
  source: string;
  lat: number;
  lon: number;
}

/** Parse a CSV line handling quoted fields */
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
 * Fetch and parse a transit entrance data file from /data/entrances/<filename>.
 * @param filename  e.g. "cta.txt", "bart.txt", "parismetro.txt"
 * @param sourceLabel  e.g. "CTA", "BART", "PARIS METRO"
 */
export async function loadTransitData(
  filename: string,
  sourceLabel: string
): Promise<TransitEntrance[]> {
  const res = await fetch(`/data/entrances/${filename}`);
  if (!res.ok)
    throw new Error(`Failed to load ${filename}: ${res.status}`);
  const text = await res.text();
  const lines = text.split(/\r?\n/).filter((line) => line.trim());
  if (lines.length < 2) return [];

  // Detect column layout from header
  const header = parseCSVLine(lines[0]);
  const stationIdx = header.indexOf("stationName");
  const latIdx = header.indexOf("lat");
  const lonIdx = header.indexOf("lon");
  if (stationIdx === -1 || latIdx === -1 || lonIdx === -1) {
    throw new Error(`Invalid header in ${filename}: ${lines[0]}`);
  }

  const out: TransitEntrance[] = [];
  for (let i = 1; i < lines.length; i++) {
    const row = parseCSVLine(lines[i]);
    const stationName = row[stationIdx] ?? "";
    const lat = parseFloat(row[latIdx]);
    const lon = parseFloat(row[lonIdx]);
    if (!stationName || Number.isNaN(lat) || Number.isNaN(lon)) continue;
    out.push({
      stationName,
      source: sourceLabel,
      lat: Math.round(lat * 1e6) / 1e6,
      lon: Math.round(lon * 1e6) / 1e6,
    });
  }
  return out;
}
