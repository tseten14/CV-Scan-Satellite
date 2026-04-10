import { runQuery } from './duckdb.js';

const loadedTables = [];

export function getLoadedTables() {
  return loadedTables;
}

function sanitizeTableName(filename) {
  return filename
    .replace(/\.(geo)?json$/i, '')
    .replace(/[^a-zA-Z0-9_]/g, '_')
    .replace(/^(\d)/, '_$1')
    .toLowerCase();
}

export async function handleGeoJSONUpload(file, onStatus) {
  const tableName = sanitizeTableName(file.name);
  onStatus('loading', `Loading ${file.name}...`);

  const text = await file.text();
  let geojson;

  try {
    geojson = JSON.parse(text);
  } catch {
    throw new Error(`Invalid JSON in ${file.name}`);
  }

  if (geojson.type !== 'FeatureCollection' && geojson.type !== 'Feature') {
    if (geojson.type && geojson.coordinates) {
      geojson = {
        type: 'FeatureCollection',
        features: [{ type: 'Feature', geometry: geojson, properties: {} }],
      };
    } else if (Array.isArray(geojson)) {
      const features = geojson.map((item) => {
        if (item && item.type === 'Feature') return item;
        const geomKey = Object.keys(item).find((k) => k.toLowerCase() === 'geometry' || k.toLowerCase() === 'geom');
        let geometry = null;
        let properties = { ...item };
        if (geomKey && item[geomKey]) {
          geometry = item[geomKey];
          delete properties[geomKey];
          if (typeof geometry === 'string') {
            try { geometry = JSON.parse(geometry); } catch (e) {}
          }
        }
        return { type: 'Feature', geometry, properties };
      });
      geojson = { type: 'FeatureCollection', features };
    } else if (typeof geojson === 'object' && geojson !== null) {
      const geomKey = Object.keys(geojson).find((k) => k.toLowerCase() === 'geometry' || k.toLowerCase() === 'geom');
      if (geomKey) {
        let geometry = geojson[geomKey];
        const properties = { ...geojson };
        delete properties[geomKey];
        if (typeof geometry === 'string') {
          try { geometry = JSON.parse(geometry); } catch (e) {}
        }
        geojson = {
          type: 'FeatureCollection',
          features: [{ type: 'Feature', geometry, properties }]
        };
      } else {
        throw new Error(`Not a valid GeoJSON file: missing "type" field`);
      }
    } else {
      throw new Error(`Not a valid GeoJSON file: missing "type" field`);
    }
  }

  if (geojson.type === 'Feature') {
    geojson = { type: 'FeatureCollection', features: [geojson] };
  }

  const features = geojson.features || [];
  if (!features.length) {
    throw new Error('GeoJSON contains no features');
  }

  onStatus('loading', `Registering ${features.length} features as "${tableName}"...`);

  const allProps = new Set();
  for (const f of features) {
    if (f.properties) {
      Object.keys(f.properties).forEach((k) => allProps.add(k));
    }
  }

  const rows = features.map((f) => {
    const props = {};
    for (const key of allProps) {
      const val = f.properties?.[key];
      if (val === undefined || val === null) {
        props[key] = null;
      } else if (typeof val === 'object') {
        props[key] = JSON.stringify(val);
      } else {
        props[key] = val;
      }
    }
    return {
      ...props,
      geometry: JSON.stringify(f.geometry),
    };
  });

  const columns = [...allProps];
  const allColumns = [...columns, 'geometry'];

  const colDefs = allColumns.map((col) => {
    if (col === 'geometry') return '"geometry" VARCHAR';
    const sampleVal = rows.find((r) => r[col] !== null)?.[col];
    if (typeof sampleVal === 'number') {
      return Number.isInteger(sampleVal)
        ? `"${col}" BIGINT`
        : `"${col}" DOUBLE`;
    }
    if (typeof sampleVal === 'boolean') return `"${col}" BOOLEAN`;
    return `"${col}" VARCHAR`;
  });

  await runQuery(`DROP TABLE IF EXISTS "${tableName}"`);
  await runQuery(`CREATE TABLE "${tableName}" (${colDefs.join(', ')})`);

  const BATCH_SIZE = 500;
  for (let i = 0; i < rows.length; i += BATCH_SIZE) {
    const batch = rows.slice(i, i + BATCH_SIZE);
    const valuesList = batch.map((row) => {
      const vals = allColumns.map((col) => {
        const v = row[col];
        if (v === null || v === undefined) return 'NULL';
        if (typeof v === 'number' || typeof v === 'boolean') return String(v);
        return `'${String(v).replace(/'/g, "''")}'`;
      });
      return `(${vals.join(', ')})`;
    });

    await runQuery(
      `INSERT INTO "${tableName}" VALUES ${valuesList.join(', ')}`
    );
  }

  const tableInfo = {
    name: tableName,
    fileName: file.name,
    featureCount: features.length,
    columns: allColumns,
    geojson,
  };

  loadedTables.push(tableInfo);
  onStatus('ready', 'Ready');

  return tableInfo;
}

/** Columns that are often huge JSON blobs in CV-Scan exports — omit from default SELECT. */
const SKIP_DEFAULT_SELECT = new Set(['bbox_px', 'polygon_px', 'source_detection_ids']);

export function buildStarterQuery(tableName, columns) {
  const propCols = columns.filter((c) => c !== 'geometry');

  if (propCols.length === 0) {
    return `-- Uploaded table "${tableName}" (geometry only)
-- "geometry" is already GeoJSON text — keep as-is so the map can draw polygons/points.

SELECT
  geometry AS geometry
FROM "${tableName}"
LIMIT 500;`;
  }

  const preferred = ['id', 'label', 'confidence', 'footprint_source', 'name', 'type'];
  const ordered = [
    ...preferred.filter((c) => propCols.includes(c)),
    ...propCols.filter((c) => !preferred.includes(c) && !SKIP_DEFAULT_SELECT.has(c)),
  ];
  const pick = ordered.slice(0, 8);
  const selectCols = pick.map((c) => `"${c}"`).join(',\n  ');

  return `-- Uploaded GeoJSON as "${tableName}" — map reads "geometry" as GeoJSON text
-- Use ST_GeomFromGeoJSON(geometry) when you need spatial functions (area, buffer, …).

SELECT
  ${selectCols},
  geometry AS geometry
FROM "${tableName}"
LIMIT 500;`;
}

export function buildMultiTableStarterQuery(tableInfos) {
  if (tableInfos.length === 1) {
    return buildStarterQuery(tableInfos[0].name, tableInfos[0].columns);
  }

  const header = tableInfos
    .map(
      (t, i) =>
        `-- ${i + 1}. "${t.name}" ← ${t.fileName} (${t.featureCount} features)`
    )
    .join('\n');

  const first = tableInfos[0];
  const body = buildStarterQuery(first.name, first.columns);
  const bodyLines = body.split('\n');
  const withoutFirstCommentBlock = bodyLines.slice(2).join('\n');

  return `${header}\n\n${withoutFirstCommentBlock}`;
}
