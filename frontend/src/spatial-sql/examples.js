export const EXAMPLE_QUERIES = [
  {
    name: 'US Cities (Points)',
    query: `-- Major US cities as point features
SELECT
  id,
  ST_AsGeoJSON(ST_Point(lon, lat)) as geometry,
  name,
  population
FROM (
  VALUES
    (1, -122.4194, 37.7749, 'San Francisco', 873965),
    (2, -118.2437, 34.0522, 'Los Angeles', 3979576),
    (3, -73.9857, 40.7484, 'New York', 8336817),
    (4, -87.6298, 41.8781, 'Chicago', 2693976),
    (5, -95.3698, 29.7604, 'Houston', 2320268),
    (6, -104.9903, 39.7392, 'Denver', 715522),
    (7, -122.3321, 47.6062, 'Seattle', 737015),
    (8, -84.3880, 33.7490, 'Atlanta', 498715),
    (9, -90.0715, 29.9511, 'New Orleans', 383997),
    (10, -80.1918, 25.7617, 'Miami', 467963)
) AS t(id, lon, lat, name, population);`,
  },
  {
    name: 'Bounding Box (Polygon)',
    query: `-- Create a bounding box polygon around the continental US
SELECT
  ST_AsGeoJSON(
    ST_MakeEnvelope(-124.7, 24.5, -66.9, 49.4)
  ) as geometry,
  'Continental US' as name;`,
  },
  {
    name: 'Buffer Around Points',
    query: `-- Create circular buffers (~100km) around cities
SELECT
  id,
  ST_AsGeoJSON(
    ST_Buffer(ST_Point(lon, lat), 1.0)
  ) as geometry,
  name,
  'buffer' as type
FROM (
  VALUES
    (1, -122.4194, 37.7749, 'San Francisco'),
    (2, -118.2437, 34.0522, 'Los Angeles'),
    (3, -73.9857, 40.7484, 'New York')
) AS t(id, lon, lat, name);`,
  },
  {
    name: 'Distance Calculation',
    query: `-- Calculate distances between cities
WITH cities AS (
  SELECT * FROM (
    VALUES
      ('San Francisco', -122.4194, 37.7749),
      ('Los Angeles', -118.2437, 34.0522),
      ('New York', -73.9857, 40.7484),
      ('Chicago', -87.6298, 41.8781)
  ) AS t(name, lon, lat)
)
SELECT
  a.name as from_city,
  b.name as to_city,
  ROUND(
    ST_Distance(
      ST_Point(a.lon, a.lat)::GEOMETRY,
      ST_Point(b.lon, b.lat)::GEOMETRY
    ), 4
  ) as distance_deg
FROM cities a
CROSS JOIN cities b
WHERE a.name < b.name
ORDER BY distance_deg DESC;`,
  },
  {
    name: 'Random Points',
    query: `-- Generate 50 random points across the US
SELECT
  row_number() OVER () as id,
  ST_AsGeoJSON(
    ST_Point(
      -124.7 + random() * 57.8,
      24.5 + random() * 24.9
    )
  ) as geometry,
  ROUND(random() * 100, 1) as value
FROM generate_series(1, 50);`,
  },
  {
    name: 'GeoParquet (Remote)',
    query: `-- Load GeoParquet data from a remote URL
-- This queries Overture Maps building data
-- Note: requires network access and may take a moment
SELECT
  ST_AsGeoJSON(ST_GeomFromWKB(geometry)) as geometry,
  names.primary as name,
  class
FROM read_parquet(
  'https://data.source.coop/cholmes/overture/geoparquet-country/US.parquet'
)
LIMIT 100;`,
  },
  {
    name: 'Simple Table Query',
    query: `-- Non-spatial query: DuckDB system info
SELECT
  version() as duckdb_version,
  current_date as today,
  current_timestamp as now;`,
  },
];
