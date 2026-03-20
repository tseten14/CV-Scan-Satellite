/**
 * Embedded Spatial Visualizer (DuckDB WASM + MapLibre) — vanilla app from spatial-sql-explorer.
 */
const SpatialSqlPage = () => {
  return (
    <iframe
      title="Spatial Visualizer"
      src="/spatial-sql.html"
      className="h-full w-full min-h-0 flex-1 border-0 bg-background"
    />
  );
};

export default SpatialSqlPage;
