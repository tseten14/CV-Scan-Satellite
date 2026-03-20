import './style.css';
import { initDuckDB, runQuery, arrowToObjects } from './duckdb.js';
import { initMap, displayGeoJSON, displayMergedGeoJSON, clearMap } from './map.js';
import { initEditor, getQuery, setQuery } from './editor.js';
import { renderResults, renderError, clearResults } from './results.js';
import { EXAMPLE_QUERIES } from './examples.js';
import {
  handleGeoJSONUpload,
  buildStarterQuery,
  buildMultiTableStarterQuery,
  getLoadedTables,
} from './upload.js';

const MAX_FILES_PER_UPLOAD = 3;

const statusEl = document.getElementById('status');
const runBtn = document.getElementById('run-btn');
const clearBtn = document.getElementById('clear-btn');
const editorContainer = document.getElementById('editor-container');
const resultsContainer = document.getElementById('results-container');
const mapContainer = document.getElementById('map-container');
const resultInfo = document.getElementById('result-info');
const featureCount = document.getElementById('feature-count');
const exampleSelect = document.getElementById('example-queries');
const fileInput = document.getElementById('file-input');
const dropOverlay = document.getElementById('drop-overlay');
const loadedLayersEl = document.getElementById('loaded-layers');

function setStatus(type, message) {
  statusEl.className = `status ${type}`;
  if (type === 'loading') {
    statusEl.innerHTML = `<span class="spinner"></span>${message}`;
  } else {
    statusEl.textContent = message;
  }
}

const map = initMap(mapContainer);

const editor = initEditor(editorContainer, executeQuery);

populateExamples();

exampleSelect.addEventListener('change', (e) => {
  const query = EXAMPLE_QUERIES.find((q) => q.name === e.target.value);
  if (query) {
    setQuery(query.query);
    exampleSelect.value = '';
  }
});

runBtn.addEventListener('click', executeQuery);
clearBtn.addEventListener('click', () => {
  clearResults(resultsContainer);
  clearMap();
  resultInfo.textContent = '';
  featureCount.textContent = '';
});

// File upload handling (up to 3 GeoJSON/JSON files per selection)
fileInput.addEventListener('change', async (e) => {
  const files = Array.from(e.target.files);
  await processUploadedFiles(files);
  fileInput.value = '';
});

async function processUploadedFiles(files) {
  const jsonFiles = files.filter(
    (f) => f.name.endsWith('.geojson') || f.name.endsWith('.json')
  );

  if (!jsonFiles.length) {
    setStatus('error', 'Choose .geojson or .json files');
    return;
  }

  if (jsonFiles.length > MAX_FILES_PER_UPLOAD) {
    setStatus(
      'error',
      `Select at most ${MAX_FILES_PER_UPLOAD} files at once (you picked ${jsonFiles.length})`
    );
    return;
  }

  const tableInfos = [];
  try {
    for (const file of jsonFiles) {
      const tableInfo = await handleGeoJSONUpload(file, setStatus);
      tableInfos.push(tableInfo);
    }

    displayMergedGeoJSON(tableInfos);
    const totalFeatures = tableInfos.reduce((s, t) => s + t.featureCount, 0);
    featureCount.textContent =
      tableInfos.length > 1
        ? `${tableInfos.length} layers · ${totalFeatures} features`
        : `${totalFeatures} features`;

    setQuery(buildMultiTableStarterQuery(tableInfos));
    renderLayerBadges();
  } catch (err) {
    setStatus('error', err.message);
    renderError(resultsContainer, err);
    console.error(err);
  }
}

function renderLayerBadges() {
  const tables = getLoadedTables();
  loadedLayersEl.innerHTML = '';

  for (const table of tables) {
    const badge = document.createElement('div');
    badge.className = 'layer-badge';
    badge.innerHTML = `
      <span class="layer-dot"></span>
      <span>${table.fileName}</span>
      <span class="layer-count">${table.featureCount}</span>
    `;
    badge.addEventListener('click', () => {
      const query = buildStarterQuery(table.name, table.columns);
      setQuery(query);
      displayGeoJSON(table.geojson);
      featureCount.textContent = `${table.featureCount} features`;
    });
    loadedLayersEl.appendChild(badge);
  }
}

// Drag-and-drop support on the whole page
let dragCounter = 0;

document.addEventListener('dragenter', (e) => {
  e.preventDefault();
  dragCounter++;
  if (dragCounter === 1) dropOverlay.classList.remove('hidden');
});

document.addEventListener('dragleave', (e) => {
  e.preventDefault();
  dragCounter--;
  if (dragCounter === 0) dropOverlay.classList.add('hidden');
});

document.addEventListener('dragover', (e) => {
  e.preventDefault();
});

document.addEventListener('drop', async (e) => {
  e.preventDefault();
  dragCounter = 0;
  dropOverlay.classList.add('hidden');

  const files = Array.from(e.dataTransfer.files).filter(
    (f) => f.name.endsWith('.geojson') || f.name.endsWith('.json')
  );

  if (!files.length) {
    setStatus('error', 'Please drop .geojson or .json files (up to 3 at once)');
    return;
  }

  await processUploadedFiles(files);
});

// Init DuckDB
initDuckDB(setStatus)
  .then(() => {
    runBtn.disabled = false;
  })
  .catch((err) => {
    setStatus('error', `Failed: ${err.message}`);
    console.error(err);
  });

async function executeQuery() {
  const sql = getQuery().trim();
  if (!sql) return;

  runBtn.disabled = true;
  setStatus('loading', 'Running query...');
  resultInfo.textContent = '';
  featureCount.textContent = '';

  const startTime = performance.now();

  try {
    const result = await runQuery(sql);
    const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
    const data = arrowToObjects(result);

    renderResults(resultsContainer, data);
    resultInfo.textContent = `${data.rows.length} rows · ${elapsed}s`;
    setStatus('ready', 'Ready');

    const geojson = tryBuildGeoJSON(data);
    if (geojson && geojson.features.length > 0) {
      const count = displayGeoJSON(geojson);
      featureCount.textContent = `${count} features on map`;
    } else {
      // Do not clear the map: SELECT without drawable geometry would wipe an uploaded GeoJSON layer.
      featureCount.textContent =
        'Query has no drawable geometry column · map left as-is (upload still visible)';
    }
  } catch (err) {
    const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
    renderError(resultsContainer, err);
    resultInfo.textContent = `Error · ${elapsed}s`;
    setStatus('error', 'Query failed');
    console.error(err);
  } finally {
    runBtn.disabled = false;
  }
}

/** Normalize geometry values from DuckDB Arrow (string, plain object, or nested). */
function coerceRowGeometry(raw) {
  if (raw == null) return null;
  if (typeof raw === 'string') {
    const s = raw.trim();
    if (!s) return null;
    try {
      return JSON.parse(s);
    } catch {
      return null;
    }
  }
  if (typeof raw === 'object') {
    if (raw.type && raw.coordinates !== undefined) return raw;
    if (typeof raw.toJSON === 'function') {
      try {
        const j = raw.toJSON();
        if (j && j.type && j.coordinates !== undefined) return j;
      } catch {
        /* ignore */
      }
    }
    try {
      const j = JSON.parse(JSON.stringify(raw));
      if (j && j.type && j.coordinates !== undefined) return j;
    } catch {
      /* ignore */
    }
  }
  return null;
}

function tryBuildGeoJSON(data) {
  const { rows, columns } = data;
  if (!rows.length) return null;

  const geomCol = columns.find(
    (c) => c.toLowerCase() === 'geometry' || c.toLowerCase() === 'geom'
  );
  if (!geomCol) return null;

  const features = [];
  for (const row of rows) {
    const geomVal = row[geomCol];
    const geometry = coerceRowGeometry(geomVal);
    if (!geometry || !geometry.type) continue;

    const properties = {};
    for (const col of columns) {
      if (col === geomCol) continue;
      properties[col] = row[col];
    }

    features.push({ type: 'Feature', geometry, properties });
  }

  if (!features.length) return null;

  return { type: 'FeatureCollection', features };
}

function populateExamples() {
  for (const example of EXAMPLE_QUERIES) {
    const option = document.createElement('option');
    option.value = example.name;
    option.textContent = example.name;
    exampleSelect.appendChild(option);
  }
}

// Resizable panels
const resizeHandle = document.getElementById('resize-handle');
const leftPanel = document.getElementById('left-panel');
let isResizing = false;

resizeHandle.addEventListener('mousedown', (e) => {
  isResizing = true;
  document.body.style.cursor = 'col-resize';
  document.body.style.userSelect = 'none';
  e.preventDefault();
});

document.addEventListener('mousemove', (e) => {
  if (!isResizing) return;
  const containerWidth = document.getElementById('main').offsetWidth;
  const newWidth = Math.max(300, Math.min(e.clientX, containerWidth - 300));
  leftPanel.style.width = `${newWidth}px`;
});

document.addEventListener('mouseup', () => {
  if (isResizing) {
    isResizing = false;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  }
});
