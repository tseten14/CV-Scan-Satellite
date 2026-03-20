export function renderResults(container, data) {
  const { rows, columns } = data;

  if (!rows.length) {
    container.innerHTML = '<div class="placeholder">Query returned no rows</div>';
    return;
  }

  const wrapper = document.createElement('div');
  wrapper.className = 'results-table-wrapper';

  const table = document.createElement('table');
  table.className = 'results-table';

  const thead = document.createElement('thead');
  const headerRow = document.createElement('tr');
  for (const col of columns) {
    const th = document.createElement('th');
    th.textContent = col;
    headerRow.appendChild(th);
  }
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  const displayCols = columns.filter((c) => c !== 'geometry');
  const showAll = displayCols.length === columns.length;

  if (!showAll) {
    headerRow.innerHTML = '';
    for (const col of displayCols) {
      const th = document.createElement('th');
      th.textContent = col;
      headerRow.appendChild(th);
    }
  }

  for (const row of rows) {
    const tr = document.createElement('tr');
    const cols = showAll ? columns : displayCols;
    for (const col of cols) {
      const td = document.createElement('td');
      const val = row[col];
      td.textContent = formatValue(val);
      td.title = String(val ?? '');
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }

  table.appendChild(tbody);
  wrapper.appendChild(table);

  container.innerHTML = '';
  container.appendChild(wrapper);
}

function formatValue(val) {
  if (val === null || val === undefined) return 'NULL';
  if (typeof val === 'object') {
    try {
      return JSON.stringify(val);
    } catch {
      return String(val);
    }
  }
  return String(val);
}

export function renderError(container, error) {
  container.innerHTML = `<div class="error-message">${escapeHtml(error.message || String(error))}</div>`;
}

export function clearResults(container) {
  container.innerHTML = '<div class="placeholder">Run a query to see results</div>';
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
