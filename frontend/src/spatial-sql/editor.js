import { EditorState } from '@codemirror/state';
import { EditorView, keymap, lineNumbers, highlightActiveLine, highlightActiveLineGutter } from '@codemirror/view';
import { sql } from '@codemirror/lang-sql';
import { oneDark } from '@codemirror/theme-one-dark';
import { defaultKeymap, history, historyKeymap } from '@codemirror/commands';
import { syntaxHighlighting, defaultHighlightStyle, bracketMatching } from '@codemirror/language';
import { closeBrackets, closeBracketsKeymap } from '@codemirror/autocomplete';

const DEFAULT_QUERY = `-- Spatial SQL Explorer
-- Write your SQL query here and press Ctrl+Enter or click Run
-- DuckDB with spatial extension runs entirely in your browser!

-- Example: Create a sample point dataset
SELECT
  id,
  ST_AsGeoJSON(ST_Point(lon, lat)) as geometry,
  name
FROM (
  VALUES
    (1, -122.4194, 37.7749, 'San Francisco'),
    (2, -118.2437, 34.0522, 'Los Angeles'),
    (3, -73.9857, 40.7484, 'New York'),
    (4, -87.6298, 41.8781, 'Chicago'),
    (5, -95.3698, 29.7604, 'Houston')
) AS t(id, lon, lat, name);`;

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
