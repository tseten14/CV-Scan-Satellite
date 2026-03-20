import { EditorState } from '@codemirror/state';
import { EditorView, keymap, lineNumbers, highlightActiveLine, highlightActiveLineGutter } from '@codemirror/view';
import { sql } from '@codemirror/lang-sql';
import { oneDark } from '@codemirror/theme-one-dark';
import { defaultKeymap, history, historyKeymap } from '@codemirror/commands';
import { syntaxHighlighting, defaultHighlightStyle, bracketMatching } from '@codemirror/language';
import { closeBrackets, closeBracketsKeymap } from '@codemirror/autocomplete';

// Default: CV-Scan-Satellite building-points GeoJSON (WGS84 Point per building).
// Table name = sanitized upload filename, e.g. building-points-wgs84-2026-03-20T04-24-59.geojson → "building_points_wgs84_2026_03_20T04_24_59".
const DEFAULT_QUERY = `-- Spatial Visualizer — CV-Scan building points (GeoJSON upload)
-- 1) Upload your .geojson (e.g. building-points-wgs84-*.geojson).
-- 2) If your filename differs, change the FROM table to match (badges above the map show the layer name).
--    Sanitize rule: drop extension; letters/digits/underscore kept; hyphens etc. → _

SELECT
  id,
  label,
  ROUND(confidence::DOUBLE, 4) AS confidence,
  ST_AsGeoJSON(
    ST_GeomFromGeoJSON(geometry)
  ) AS geometry
FROM "building_points_wgs84_2026_03_20T04_24_59"
LIMIT 500;
`;

let editorView = null;

export function initEditor(container, onRun) {
  const runQueryKeymap = keymap.of([
    {
      key: 'Ctrl-Enter',
      mac: 'Cmd-Enter',
      run: () => {
        onRun();
        return true;
      },
    },
  ]);

  const state = EditorState.create({
    doc: DEFAULT_QUERY,
    extensions: [
      lineNumbers(),
      highlightActiveLine(),
      highlightActiveLineGutter(),
      history(),
      bracketMatching(),
      closeBrackets(),
      sql(),
      oneDark,
      syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
      runQueryKeymap,
      keymap.of([...defaultKeymap, ...historyKeymap, ...closeBracketsKeymap]),
      EditorView.theme({
        '&': { height: '100%' },
        '.cm-scroller': { overflow: 'auto' },
        '.cm-content': { padding: '8px 0' },
      }),
    ],
  });

  editorView = new EditorView({
    state,
    parent: container,
  });

  return editorView;
}

export function getQuery() {
  if (!editorView) return '';
  return editorView.state.doc.toString();
}

export function setQuery(queryText) {
  if (!editorView) return;
  editorView.dispatch({
    changes: {
      from: 0,
      to: editorView.state.doc.length,
      insert: queryText,
    },
  });
}
