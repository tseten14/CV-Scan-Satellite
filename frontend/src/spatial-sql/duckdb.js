import * as duckdb from '@duckdb/duckdb-wasm';
import duckdb_wasm from '@duckdb/duckdb-wasm/dist/duckdb-mvp.wasm?url';
import mvp_worker from '@duckdb/duckdb-wasm/dist/duckdb-browser-mvp.worker.js?url';
import duckdb_wasm_next from '@duckdb/duckdb-wasm/dist/duckdb-eh.wasm?url';
import eh_worker from '@duckdb/duckdb-wasm/dist/duckdb-browser-eh.worker.js?url';

let db = null;
let conn = null;

export async function initDuckDB(onStatus) {
  onStatus('loading', 'Loading DuckDB WASM...');

  const MANUAL_BUNDLES = {
    mvp: { mainModule: duckdb_wasm, mainWorker: mvp_worker },
    eh: { mainModule: duckdb_wasm_next, mainWorker: eh_worker },
  };

  const bundle = await duckdb.selectBundle(MANUAL_BUNDLES);
  const worker = new Worker(bundle.mainWorker);
  const logger = new duckdb.ConsoleLogger();

  db = new duckdb.AsyncDuckDB(logger, worker);
  await db.instantiate(bundle.mainModule, bundle.pthreadWorker);

  onStatus('loading', 'Loading spatial extension...');

  conn = await db.connect();

  try {
    await conn.query('INSTALL spatial; LOAD spatial;');
  } catch (e) {
    console.warn('Spatial extension not available in this WASM build:', e.message);
  }

  onStatus('ready', 'Ready');
  return conn;
}

export async function runQuery(sql) {
  if (!conn) throw new Error('DuckDB not initialized');
  const result = await conn.query(sql);
  return result;
}

export function arrowToObjects(table) {
  const rows = [];
  const schema = table.schema.fields.map((f) => f.name);

  for (let i = 0; i < table.numRows; i++) {
    const row = {};
    for (const col of schema) {
      const val = table.getChild(col)?.get(i);
      row[col] = val;
    }
    rows.push(row);
  }

  return { rows, columns: schema };
}
