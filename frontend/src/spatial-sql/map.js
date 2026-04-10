import maplibregl from 'maplibre-gl';

let map = null;
/** DOM markers — reliable on top of raster basemap (some MapLibre + iframe setups hide circle layers). */
const pointMarkers = [];
const SOURCE_ID = 'query-results';
const FILL_LAYER = 'query-fill';
const LINE_LAYER = 'query-line';
const POINT_LAYER = 'query-points';

const PREVIEW_NOTE =
  'Preview layout only (pixel space → small map window). For real WGS84, use Scan Map + Download GeoJSON in CV-Scan-Satellite.';

/**
 * CV-Scan "pixel" GeoJSON uses geometry: null with center_px / polygon_px in properties.
 * MapLibre only draws real geometries, so we map pixels into a fixed preview WGS84 bbox.
 */
export function normalizeGeoJSONForMap(geojson) {
  const rawFeatures = geojson?.features;
  if (!Array.isArray(rawFeatures) || rawFeatures.length === 0) {
    return { type: 'FeatureCollection', features: [] };
  }

  const needsNormalize = rawFeatures.some((f) => {
    const g = f?.geometry;
    const noGeom = g === null || g === undefined;
    const hasPixel =
      f?.properties?.center_px &&
      typeof f.properties.center_px.x === 'number' &&
      typeof f.properties.center_px.y === 'number';
    const hasPoly =
      Array.isArray(f?.properties?.polygon_px) && f.properties.polygon_px.length >= 3;
    const hasFootprintRing =
      Array.isArray(f?.properties?.footprint_ring_px) && f.properties.footprint_ring_px.length >= 3;
    return noGeom && (hasPixel || hasPoly || hasFootprintRing);
  });

  if (!needsNormalize) {
    return {
      type: 'FeatureCollection',
      features: rawFeatures,
    };
  }

  const meta = geojson.metadata || {};
  const iw = typeof meta.image_width === 'number' ? meta.image_width : null;
  const ih = typeof meta.image_height === 'number' ? meta.image_height : null;

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  const expand = (x, y) => {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  };

  for (const f of rawFeatures) {
    const c = f.properties?.center_px;
    if (c) expand(c.x, c.y);
    const poly = f.properties?.polygon_px ?? f.properties?.footprint_ring_px;
    if (Array.isArray(poly)) {
      for (const pt of poly) {
        if (Array.isArray(pt) && pt.length >= 2) expand(pt[0], pt[1]);
      }
    }
  }

  if (iw != null && ih != null) {
    expand(0, 0);
    expand(iw, ih);
  }

  if (!Number.isFinite(minX) || !Number.isFinite(maxX)) {
    return { type: 'FeatureCollection', features: rawFeatures };
  }

  if (maxX === minX) {
    minX -= 1;
    maxX += 1;
  }
  if (maxY === minY) {
    minY -= 1;
    maxY += 1;
  }

  const padX = (maxX - minX) * 0.06 || 1;
  const padY = (maxY - minY) * 0.06 || 1;
  minX -= padX;
  maxX += padX;
  minY -= padY;
  maxY += padY;

  /** ~10×12 km preview over US — relative shape only */
  const LNG_MIN = -98.52;
  const LNG_MAX = -98.38;
  const LAT_MIN = 39.02;
  const LAT_MAX = 39.14;

  const pxToLngLat = (x, y) => {
    const nx = (x - minX) / (maxX - minX);
    const ny = (y - minY) / (maxY - minY);
    const lng = LNG_MIN + nx * (LNG_MAX - LNG_MIN);
    const lat = LAT_MAX - ny * (LAT_MAX - LAT_MIN);
    return [lng, lat];
  };

  const outFeatures = [];

  for (const f of rawFeatures) {
    const g = f.geometry;
    if (g && g.type) {
      outFeatures.push({
        type: 'Feature',
        geometry: g,
        properties: { ...(f.properties || {}) },
      });
      continue;
    }

    const props = { ...(f.properties || {}) };
    const poly = props.polygon_px ?? props.footprint_ring_px;
    let added = false;

    if (Array.isArray(poly) && poly.length >= 3) {
      const ring = poly.map((pt) => {
        if (!Array.isArray(pt) || pt.length < 2) return null;
        return pxToLngLat(pt[0], pt[1]);
      }).filter(Boolean);

      if (ring.length >= 3) {
        const first = ring[0];
        const last = ring[ring.length - 1];
        if (first[0] !== last[0] || first[1] !== last[1]) ring.push([...first]);

        outFeatures.push({
          type: 'Feature',
          geometry: { type: 'Polygon', coordinates: [ring] },
          properties: { ...props, _mapPreview: PREVIEW_NOTE },
        });
        added = true;
      }
    }

    const c = props.center_px;
    if (!added && c && typeof c.x === 'number' && typeof c.y === 'number') {
      const [lng, lat] = pxToLngLat(c.x, c.y);
      outFeatures.push({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [lng, lat] },
        properties: {
          ...props,
          _mapPreview: PREVIEW_NOTE,
        },
      });
      added = true;
    }

    if (!added) {
      outFeatures.push({
        type: 'Feature',
        geometry: g ?? null,
        properties: props,
      });
    }
  }

  // eslint-disable-next-line no-console
  console.info(
    '[spatial-sql] GeoJSON had null geometries (pixel export). Mapped to preview WGS84 for map display.',
  );

  return { type: 'FeatureCollection', features: outFeatures };
}

export function initMap(container) {
  map = new maplibregl.Map({
    container,
    style: {
      version: 8,
      name: 'Satellite',
      sources: {
        'esri-world-imagery': {
          type: 'raster',
          tiles: [
            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
          ],
          tileSize: 256,
          attribution:
            '&copy; <a href="https://www.esri.com/">Esri</a> &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community',
        },
      },
      layers: [
        {
          id: 'satellite-base',
          type: 'raster',
          source: 'esri-world-imagery',
          minzoom: 0,
          maxzoom: 22,
        },
      ],
    },
    center: [-98.5, 39.8],
    zoom: 3,
  });

  map.addControl(new maplibregl.NavigationControl(), 'top-right');

  // iframe / flex layouts often report 0×0 until after layout — vector layers need a resize
  const ro = new ResizeObserver(() => {
    map.resize();
  });
  ro.observe(container);

  return map;
}

/**
 * GeoJSON from DuckDB / files sometimes has coordinate components as strings.
 * That breaks fitBounds (string iteration) and can prevent MapLibre from drawing fills.
 */
export function coerceGeometryCoordinates(geom) {
  if (!geom || typeof geom !== 'object' || !geom.type) return geom;

  const mapPair = (pair) => {
    if (!Array.isArray(pair) || pair.length < 2) return pair;
    const lng = Number(pair[0]);
    const lat = Number(pair[1]);
    if (!Number.isFinite(lng) || !Number.isFinite(lat)) return pair;
    return pair.length > 2 ? [lng, lat, ...pair.slice(2).map(Number)] : [lng, lat];
  };

  const mapCoordsDeep = (coords) => {
    if (!Array.isArray(coords) || coords.length === 0) return coords;
    const a = coords[0];
    const b = coords[1];
    const isPositionPair =
      coords.length >= 2 &&
      (typeof a === 'number' || typeof a === 'string') &&
      (typeof b === 'number' || typeof b === 'string') &&
      typeof a !== 'object' &&
      typeof b !== 'object';
    if (isPositionPair && Number.isFinite(Number(a)) && Number.isFinite(Number(b))) {
      return mapPair(coords);
    }
    return coords.map(mapCoordsDeep);
  };

  if (geom.type === 'GeometryCollection' && Array.isArray(geom.geometries)) {
    return {
      ...geom,
      geometries: geom.geometries.map((g) => coerceGeometryCoordinates(g)),
    };
  }

  if (geom.coordinates !== undefined) {
    return { ...geom, coordinates: mapCoordsDeep(geom.coordinates) };
  }
  return geom;
}

/**
 * MapLibre's vector-tile serialization silently drops features whose properties contain
 * non-primitive values (nested objects, arrays). Flatten them to JSON strings.
 */
function flattenProperties(props) {
  if (!props || typeof props !== 'object') return props || {};
  const out = {};
  for (const [k, v] of Object.entries(props)) {
    if (v === null || v === undefined) {
      out[k] = null;
    } else if (typeof v === 'object') {
      out[k] = JSON.stringify(v);
    } else {
      out[k] = v;
    }
  }
  return out;
}

/** DuckDB rows / some tools store geometry as JSON strings — MapLibre needs objects. */
function sanitizeFeatures(features) {
  return features.map((f) => {
    let g = f.geometry;
    if (typeof g === 'string') {
      try {
        g = JSON.parse(g);
      } catch {
        g = null;
      }
    }
    if (!g || typeof g !== 'object' || !g.type) {
      return { ...f, geometry: null, properties: flattenProperties(f.properties) };
    }
    return {
      ...f,
      geometry: coerceGeometryCoordinates(g),
      properties: flattenProperties(f.properties),
    };
  });
}

function clearPointMarkers() {
  for (const m of pointMarkers) {
    try {
      m.remove();
    } catch {
      /* ignore */
    }
  }
  pointMarkers.length = 0;
}

function addPointMarkersFromFeatures(features) {
  if (!map) return;
  for (const f of features) {
    const g = f.geometry;
    if (!g) continue;
    if (g.type === 'Point') {
      const c = g.coordinates;
      if (!Array.isArray(c) || c.length < 2) continue;
      const lng = Number(c[0]);
      const lat = Number(c[1]);
      if (!Number.isFinite(lng) || !Number.isFinite(lat)) continue;
      pushPointMarker(lng, lat);
    } else if (g.type === 'MultiPoint') {
      for (const pt of g.coordinates || []) {
        if (!Array.isArray(pt) || pt.length < 2) continue;
        const lng = Number(pt[0]);
        const lat = Number(pt[1]);
        if (!Number.isFinite(lng) || !Number.isFinite(lat)) continue;
        pushPointMarker(lng, lat);
      }
    }
  }
}

function pushPointMarker(lng, lat) {
  const el = document.createElement('div');
  el.setAttribute('aria-hidden', 'true');
  el.style.cssText =
    'width:14px;height:14px;border-radius:50%;background:#f472b6;border:2px solid #fff;box-shadow:0 0 8px rgba(0,0,0,0.55);pointer-events:none;';
  const marker = new maplibregl.Marker({ element: el, anchor: 'center' })
    .setLngLat([lng, lat])
    .addTo(map);
  pointMarkers.push(marker);
}

/** Style must be ready before addSource; iframe can race — retry a few times. */
function runWhenMapReady(fn) {
  if (!map) return;
  let ran = false;
  const run = () => {
    if (ran) return;
    if (!map?.isStyleLoaded?.()) return;
    ran = true;
    try {
      fn();
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error('[spatial-sql] map draw failed', e);
    }
  };
  if (map.isStyleLoaded?.()) {
    queueMicrotask(() => requestAnimationFrame(run));
  } else {
    map.once('load', () => {
      queueMicrotask(() => requestAnimationFrame(run));
    });
  }
  setTimeout(run, 80);
  setTimeout(run, 350);
}

/** Merge several GeoJSON layers for map display; adds `_layer` (source file) to properties. */
export function displayMergedGeoJSON(tableInfos) {
  if (!tableInfos.length) return;

  const features = [];
  for (const t of tableInfos) {
    const label = t.fileName || t.name;
    for (const f of t.geojson.features || []) {
      features.push({
        type: 'Feature',
        geometry: f.geometry,
        properties: { ...(f.properties || {}), _layer: label },
      });
    }
  }

  return displayGeoJSON({ type: 'FeatureCollection', features });
}

export function displayGeoJSON(geojson) {
  if (!map) return 0;
  runWhenMapReady(() => displayGeoJSONInternal(geojson));
  // Return count synchronously only when already applied — callers use upload totals for UI
  const normalized = normalizeGeoJSONForMap(geojson);
  const sanitized = sanitizeFeatures(normalized.features);
  return sanitized.filter((f) => f.geometry && f.geometry.type).length;
}

function displayGeoJSONInternal(geojson) {
  if (!map) return 0;

  clearMap();
  map.resize();

  const normalized = normalizeGeoJSONForMap(geojson);
  const sanitized = sanitizeFeatures(normalized.features);
  const forMap = {
    type: 'FeatureCollection',
    features: sanitized.filter((f) => f.geometry && f.geometry.type),
  };

  if (!forMap.features.length) {
    // eslint-disable-next-line no-console
    console.warn('[spatial-sql] No drawable geometries in GeoJSON after normalize.');
    return 0;
  }

  const hasPolygons = forMap.features.some((f) =>
    f.geometry?.type?.includes('Polygon')
  );
  const hasLines = forMap.features.some((f) =>
    f.geometry?.type?.includes('Line')
  );
  const hasPoints = forMap.features.some((f) =>
    f.geometry?.type === 'Point' || f.geometry?.type === 'MultiPoint'
  );

  /** GeoJSON source is only needed for fill/line layers; points use DOM markers. */
  const needsGeoJsonSource = hasPolygons || hasLines;

  if (needsGeoJsonSource) {
    map.addSource(SOURCE_ID, {
      type: 'geojson',
      data: forMap,
    });
  }

  if (hasPolygons) {
    map.addLayer({
      id: FILL_LAYER,
      type: 'fill',
      source: SOURCE_ID,
      filter: ['any',
        ['==', ['geometry-type'], 'Polygon'],
        ['==', ['geometry-type'], 'MultiPolygon'],
      ],
      paint: {
        /* High-contrast amber fill — matches CV-Scan footprint overlays on satellite */
        'fill-color': '#eab308',
        'fill-opacity': 0.42,
      },
    });

    map.addLayer({
      id: LINE_LAYER,
      type: 'line',
      source: SOURCE_ID,
      filter: ['any',
        ['==', ['geometry-type'], 'Polygon'],
        ['==', ['geometry-type'], 'MultiPolygon'],
      ],
      paint: {
        'line-color': '#ca8a04',
        'line-width': 2,
      },
    });
  }

  if (hasLines) {
    map.addLayer({
      id: 'query-lines',
      type: 'line',
      source: SOURCE_ID,
      filter: ['any',
        ['==', ['geometry-type'], 'LineString'],
        ['==', ['geometry-type'], 'MultiLineString'],
      ],
      paint: {
        'line-color': '#22d3ee',
        'line-width': 2,
      },
    });
  }

  if (hasPoints) {
    // HTML markers render above the raster basemap; circle layers can fail to appear in iframe/WebGL stacks.
    addPointMarkersFromFeatures(forMap.features);
  }

  /* Ensure vector overlays draw above the Esri raster (some GL / iframe stacks order incorrectly). */
  try {
    if (map.getLayer(LINE_LAYER)) map.moveLayer(LINE_LAYER);
    if (map.getLayer(FILL_LAYER)) map.moveLayer(FILL_LAYER, LINE_LAYER);
    if (map.getLayer('query-lines')) map.moveLayer('query-lines');
  } catch (e) {
    // eslint-disable-next-line no-console
    console.warn('[spatial-sql] moveLayer for overlay order:', e);
  }

  const popup = new maplibregl.Popup({
    closeButton: false,
    closeOnClick: false,
  });

  const interactiveLayers = [FILL_LAYER, 'query-lines'].filter((id) => map.getLayer(id));

  for (const layerId of interactiveLayers) {
    map.on('mouseenter', layerId, (e) => {
      map.getCanvas().style.cursor = 'pointer';
      if (e.features?.length) {
        const props = e.features[0].properties;
        const html = Object.entries(props)
          .filter(([k]) => k !== 'geometry')
          .map(([k, v]) => `<strong>${k}:</strong> ${v}`)
          .join('<br/>');
        popup.setLngLat(e.lngLat).setHTML(html).addTo(map);
      }
    });

    map.on('mouseleave', layerId, () => {
      map.getCanvas().style.cursor = '';
      popup.remove();
    });
  }

  fitToFeatures(forMap);

  map.resize();
  return forMap.features.length;
}

function extendBoundsWithFiniteCoords(coords, bounds) {
  let n = 0;
  if (typeof coords[0] === 'number') {
    const lng = coords[0];
    const lat = coords[1];
    if (Number.isFinite(lng) && Number.isFinite(lat)) {
      bounds.extend([lng, lat]);
      return 1;
    }
    return 0;
  }
  for (const c of coords) {
    n += extendBoundsWithFiniteCoords(c, bounds);
  }
  return n;
}

function fitToFeatures(geojson) {
  if (!geojson.features.length || !map) return;

  const bounds = new maplibregl.LngLatBounds();
  let total = 0;

  for (const feature of geojson.features) {
    if (!feature.geometry?.coordinates) continue;
    total += extendBoundsWithFiniteCoords(feature.geometry.coordinates, bounds);
  }

  if (total === 0) {
    // eslint-disable-next-line no-console
    console.warn('[spatial-sql] No finite lng/lat in features — check GeoJSON coordinate order [lng, lat].');
    return;
  }

  map.resize();

  try {
    if (total === 1) {
      const c = bounds.getCenter();
      map.jumpTo({ center: [c.lng, c.lat], zoom: 17 });
    } else {
      map.fitBounds(bounds, { padding: 72, maxZoom: 18, duration: 0 });
    }
  } catch (e) {
    // eslint-disable-next-line no-console
    console.warn('[spatial-sql] fitBounds failed, using center fallback', e);
    try {
      const c = bounds.getCenter();
      map.jumpTo({ center: [c.lng, c.lat], zoom: 15 });
    } catch {
      /* ignore */
    }
  }
}

export function clearMap() {
  if (!map) return;

  clearPointMarkers();

  const layerIds = [FILL_LAYER, LINE_LAYER, POINT_LAYER, 'query-lines'];
  for (const id of layerIds) {
    if (map.getLayer(id)) map.removeLayer(id);
  }

  if (map.getSource(SOURCE_ID)) map.removeSource(SOURCE_ID);
}
