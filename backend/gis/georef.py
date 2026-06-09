"""Project pixel-space polygons into geographic coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import geopandas as gpd
import numpy as np
from rasterio.transform import Affine, xy
from shapely.geometry import Polygon, mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform


@dataclass(frozen=True)
class GeorefBounds:
    """
    Linear georeferencing from image pixels to WGS84 using map bounds.

    Matches the frontend ``satelliteScanMarkers`` affine model: image top = north,
    left = west, image fills the bounding box.
    """

    west: float
    south: float
    east: float
    north: float
    image_width: int
    image_height: int

    def to_affine(self) -> Affine:
        """Return an Affine transform from pixel (col, row) to lon/lat."""
        w = max(self.image_width, 1)
        h = max(self.image_height, 1)
        lon_scale = (self.east - self.west) / w
        lat_scale = (self.north - self.south) / h
        return Affine.translation(self.west, self.north) * Affine.scale(lon_scale, -lat_scale)


def read_raster_georef(image_path: str) -> tuple[Affine, str | None]:
    """
    Read the affine transform and CRS from a georeferenced raster.

    Returns:
        (transform, crs_wkt_or_epsg_string). CRS may be None for non-georeferenced files.
    """
    import rasterio

    with rasterio.open(image_path) as dataset:
        return dataset.transform, dataset.crs.to_string() if dataset.crs else None


def _transform_polygon_pixels_to_wgs84(
    polygon: Polygon,
    pixel_transform: Affine,
    source_crs: str | None,
) -> Polygon:
    """Map a pixel polygon through affine transform and reproject to EPSG:4326."""

    def _pixel_to_geo(x: float, y: float, z: float | None = None) -> tuple[float, float]:
        lon, lat = xy(pixel_transform, y, x, offset="center")
        return float(lon), float(lat)

    geo_poly = transform(_pixel_to_geo, polygon)
    if source_crs and source_crs.upper() not in {"EPSG:4326", "OGC:CRS84"}:
        gdf = gpd.GeoDataFrame(geometry=[geo_poly], crs=source_crs)
        gdf = gdf.to_crs(epsg=4326)
        geom = gdf.geometry.iloc[0]
        if isinstance(geom, BaseGeometry):
            return geom  # type: ignore[return-value]
    return geo_poly


def pixel_polygons_to_geodataframe(
    polygons: Iterable[Polygon],
    *,
    pixel_transform: Affine,
    source_crs: str | None = None,
    properties: Iterable[dict] | None = None,
) -> gpd.GeoDataFrame:
    """
    Convert pixel-space Shapely polygons to a WGS84 GeoDataFrame.

    Args:
        polygons: Footprints in image pixel coordinates (x=col, y=row).
        pixel_transform: Rasterio-style affine from pixel to source CRS coordinates.
        source_crs: CRS of the affine target (defaults to EPSG:4326 when omitted).
        properties: Optional per-polygon attribute dicts.

    Returns:
        GeoDataFrame in EPSG:4326. Empty when no polygons are supplied.
    """
    poly_list = list(polygons)
    if not poly_list:
        return gpd.GeoDataFrame(
            columns=["id", "label", "confidence", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    props = list(properties) if properties is not None else [{} for _ in poly_list]
    if len(props) != len(poly_list):
        raise ValueError("properties length must match polygons length")

    src_crs = source_crs or "EPSG:4326"
    geoms = [
        _transform_polygon_pixels_to_wgs84(poly, pixel_transform, src_crs)
        for poly in poly_list
    ]
    gdf = gpd.GeoDataFrame(props, geometry=geoms, crs="EPSG:4326")
    return gdf


def bounds_to_geodataframe(
    polygons: Iterable[Polygon],
    bounds: GeorefBounds,
    properties: Iterable[dict] | None = None,
) -> gpd.GeoDataFrame:
    """Convenience wrapper using linear map-bounds georeferencing."""
    return pixel_polygons_to_geodataframe(
        polygons,
        pixel_transform=bounds.to_affine(),
        source_crs="EPSG:4326",
        properties=properties,
    )


def geodataframe_to_geojson_dict(gdf: gpd.GeoDataFrame) -> dict:
    """Serialize a GeoDataFrame to a GeoJSON FeatureCollection dict."""
    if gdf.empty:
        return {"type": "FeatureCollection", "features": []}
    working = gdf.to_crs(epsg=4326) if gdf.crs else gdf.set_crs(epsg=4326)
    features = []
    for idx, row in working.iterrows():
        props = {k: v for k, v in row.items() if k != "geometry"}
        for key, value in list(props.items()):
            if isinstance(value, (np.floating, np.integer)):
                props[key] = value.item()
        features.append(
            {
                "type": "Feature",
                "id": str(props.get("id", idx)),
                "geometry": mapping(row.geometry),
                "properties": props,
            }
        )
    return {"type": "FeatureCollection", "features": features}
