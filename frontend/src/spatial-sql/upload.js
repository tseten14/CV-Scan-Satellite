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
      props[key] = val === undefined ? null : val;
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

export function buildStarterQuery(tableName, columns) {
  const propCols = columns.filter((c) => c !== 'geometry');
  const selectCols = propCols.length > 0
    ? propCols.slice(0, 5).map((c) => `"${c}"`).join(',\n  ')
    : '*';

  return `-- Uploaded data is available as table "${tableName}"
-- The geometry column contains GeoJSON geometry strings

SELECT
  ${selectCols},
  geometry
FROM "${tableName}"
LIMIT 100;`;
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
