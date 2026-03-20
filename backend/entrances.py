"""
Locate transit entrances from GTFS-derived data (heretech_sampledata).
Used by GET /api/entrances in the web app.
"""
from pathlib import Path
import pandas as pd
from rapidfuzz import process, fuzz

DATA_DIR = Path(__file__).resolve().parent / "data" / "entrances"
BOUNDING_FILE = DATA_DIR / "bounding.txt"

# Default: no bbox (search all sources). Pass floats: lat_min, lat_max, lon_min, lon_max.
DEFAULT_BBOX = (float("-inf"), float("inf"), float("-inf"), float("inf"))


def get_entrances(
    query: str,
    lat_min: float | None = None,
    lat_max: float | None = None,
    lon_min: float | None = None,
    lon_max: float | None = None,
    score_cutoff: int = 45,
) -> list[dict]:
    """
    Return list of entrance records: { "stationName", "source", "lat", "lon" }.
    """
    if not query or not query.strip():
        return []

    bounding_box = (
        float(lat_min) if lat_min is not None else DEFAULT_BBOX[0],
        float(lat_max) if lat_max is not None else DEFAULT_BBOX[1],
        float(lon_min) if lon_min is not None else DEFAULT_BBOX[2],
        float(lon_max) if lon_max is not None else DEFAULT_BBOX[3],
    )

    if not BOUNDING_FILE.exists():
        return []

    sources_df = pd.read_csv(BOUNDING_FILE)
    # Handle optional leading empty column in bounding.txt
    if "file" not in sources_df.columns and len(sources_df.columns) >= 2:
        sources_df = pd.read_csv(BOUNDING_FILE, index_col=0)
    if "lonMin" in sources_df.columns and "lonMax" in sources_df.columns:
        lat_min_col, lat_max_col = "latMin", "latMax"
        lon_min_col, lon_max_col = "lonMin", "lonMax"
    else:
        lat_min_col, lat_max_col = "latMin", "latMax"
        lon_min_col, lon_max_col = "lonMin", "lonMax"

    # Sources whose bounding box overlaps the request bbox (any overlap)
    source_matches = sources_df[
        (sources_df[lat_max_col] >= bounding_box[0])
        & (sources_df[lat_min_col] <= bounding_box[1])
        & (sources_df[lon_max_col] >= bounding_box[2])
        & (sources_df[lon_min_col] <= bounding_box[3])
    ]
    if source_matches.empty and bounding_box != DEFAULT_BBOX:
        source_matches = sources_df  # fallback: search all if no overlap
    elif source_matches.empty:
        source_matches = sources_df

    results: list[dict] = []
    for _, row in source_matches.iterrows():
        file_name = row["file"]
        if not isinstance(file_name, str) or not file_name.endswith(".txt"):
            continue
        csv_path = DATA_DIR / file_name
        if not csv_path.exists():
            continue
        try:
            source_csv = pd.read_csv(csv_path)
        except Exception:
            continue
        if "stationName" not in source_csv.columns or "lat" not in source_csv.columns or "lon" not in source_csv.columns:
            continue
        filtered = source_csv[
            (source_csv["lat"] >= bounding_box[0])
            & (source_csv["lat"] <= bounding_box[1])
            & (source_csv["lon"] >= bounding_box[2])
            & (source_csv["lon"] <= bounding_box[3])
        ]
        if filtered.empty:
            continue
        name_matches = process.extract(
            query.strip(),
            filtered["stationName"].dropna().astype(str).str.strip().unique().tolist(),
            scorer=fuzz.token_sort_ratio,
            limit=15,
            score_cutoff=score_cutoff,
        )
        if not name_matches:
            continue
        for match_name, score, _ in name_matches:
            rows = filtered[filtered["stationName"] == match_name]
            source_label = file_name.replace(".txt", "").upper()
            for _, r in rows.iterrows():
                results.append({
                    "stationName": match_name,
                    "source": source_label,
                    "lat": round(float(r["lat"]), 6),
                    "lon": round(float(r["lon"]), 6),
                })

    return results


# CTA (Chicago Transit Authority) data file - same format as other sources
CTA_FILE = DATA_DIR / "cta.txt"
# Chicago CTA bounding box (from bounding.txt)
CTA_BBOX = (41.721558, 42.073623, -87.904004, -87.605799)


def get_cta_entrances(
    lat_min: float | None = None,
    lat_max: float | None = None,
    lon_min: float | None = None,
    lon_max: float | None = None,
) -> list[dict]:
    """
    Parse entrance data from data/entrances/cta.txt and return records in the same
    shape as get_entrances: { "stationName", "source", "lat", "lon" }.
    Optionally filter by bounding box (default: full CTA area).
    """
    if not CTA_FILE.exists():
        return []
    bbox = (
        float(lat_min) if lat_min is not None else CTA_BBOX[0],
        float(lat_max) if lat_max is not None else CTA_BBOX[1],
        float(lon_min) if lon_min is not None else CTA_BBOX[2],
        float(lon_max) if lon_max is not None else CTA_BBOX[3],
    )
    try:
        df = pd.read_csv(CTA_FILE)
    except Exception:
        return []
    if "stationName" not in df.columns or "lat" not in df.columns or "lon" not in df.columns:
        return []
    filtered = df[
        (df["lat"] >= bbox[0])
        & (df["lat"] <= bbox[1])
        & (df["lon"] >= bbox[2])
        & (df["lon"] <= bbox[3])
    ]
    results = []
    for _, r in filtered.iterrows():
        results.append({
            "stationName": str(r["stationName"]).strip(),
            "source": "CTA",
            "lat": round(float(r["lat"]), 6),
            "lon": round(float(r["lon"]), 6),
        })
    return results
