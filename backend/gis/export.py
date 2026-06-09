"""Reliable GeoJSON, KML, and zipped Shapefile export for researchers."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
from xml.sax.saxutils import escape

import geopandas as gpd

from gis.georef import geodataframe_to_geojson_dict

ExportFormat = Literal["geojson", "kml", "shapefile", "all"]


def export_geojson(gdf: gpd.GeoDataFrame, output_path: str | Path) -> Path:
    """
    Write a standard GeoJSON file from a GeoDataFrame.

    Empty GeoDataFrames produce a valid empty FeatureCollection.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = geodataframe_to_geojson_dict(gdf)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _polygon_coords_to_kml(coords: list) -> str:
    parts = []
    for lon, lat, *rest in coords:
        alt = rest[0] if rest else 0.0
        parts.append(f"{lon},{lat},{alt}")
    return " ".join(parts)


def export_kml(gdf: gpd.GeoDataFrame, output_path: str | Path, *, name_field: str = "id") -> Path:
    """
    Write a KML file suitable for Google Earth.

    Handles empty datasets with a minimal valid document.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    placemarks: list[str] = []
    if not gdf.empty:
        working = gdf.to_crs(epsg=4326) if gdf.crs else gdf.set_crs(epsg=4326)
        for _, row in working.iterrows():
            geom = row.geometry
            name = escape(str(row.get(name_field, "building")))
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == "Polygon":
                outer = _polygon_coords_to_kml(list(geom.exterior.coords))
                placemarks.append(
                    f"<Placemark><name>{name}</name>"
                    f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{outer}"
                    f"</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>"
                )
            elif geom.geom_type == "MultiPolygon":
                for poly in geom.geoms:
                    outer = _polygon_coords_to_kml(list(poly.exterior.coords))
                    placemarks.append(
                        f"<Placemark><name>{name}</name>"
                        f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{outer}"
                        f"</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>"
                    )

    body = "\n".join(placemarks)
    kml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<kml xmlns="http://www.opengis.net/kml/2.2">'
        "<Document>"
        "<name>CV-Scan-Satellite Buildings</name>"
        f"{body}"
        "</Document></kml>"
    )
    path.write_text(kml, encoding="utf-8")
    return path


def export_shapefile_zip(
    gdf: gpd.GeoDataFrame,
    output_zip_path: str | Path,
    *,
    layer_name: str = "buildings",
) -> Path:
    """
    Export a Shapefile with all mandatory sidecar files packaged into a zip.

    Generates ``.shp``, ``.shx``, ``.dbf``, and ``.prj`` (and any other files
    Fiona emits) so QGIS/ArcGIS imports work out of the box.
    """
    zip_path = Path(output_zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    working = gdf.copy()
    if working.empty:
        working = gpd.GeoDataFrame(
            {"id": [], "label": [], "confidence": []},
            geometry=[],
            crs="EPSG:4326",
        )
    elif working.crs is None:
        working = working.set_crs(epsg=4326)
    else:
        working = working.to_crs(epsg=4326)

    with TemporaryDirectory() as tmpdir:
        shp_base = Path(tmpdir) / layer_name
        working.to_file(shp_base.with_suffix(".shp"), driver="ESRI Shapefile", encoding="UTF-8")

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for sidecar in Path(tmpdir).glob(f"{layer_name}.*"):
                zf.write(sidecar, arcname=sidecar.name)

    return zip_path


def export_gis_bundle(
    gdf: gpd.GeoDataFrame,
    output_dir: str | Path,
    *,
    stem: str = "buildings",
) -> dict[str, Path]:
    """
    Export GeoJSON, KML, and zipped Shapefile for a GeoDataFrame.

    Returns a mapping of format name to written file path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "geojson": export_geojson(gdf, out / f"{stem}.geojson"),
        "kml": export_kml(gdf, out / f"{stem}.kml"),
        "shapefile": export_shapefile_zip(gdf, out / f"{stem}.zip"),
    }
