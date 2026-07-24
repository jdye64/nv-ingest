/* Config — system configuration tree with impact badges */

function ConfigTree({ value, path, impactBySection }) {
  if (value === null || value === undefined) {
    return React.createElement('span', { className: 'mono', style: { color: 'var(--nv-text-muted)' } }, 'null');
  }
  if (typeof value !== 'object' || Array.isArray(value)) {
    const text = typeof value === 'string' ? JSON.stringify(value) : String(value);
    return React.createElement('span', { className: 'mono' }, text);
  }

  const entries = Object.entries(value);
  if (entries.length === 0) {
    return React.createElement('span', { className: 'mono', style: { color: 'var(--nv-text-muted)' } }, '{}');
  }

  const sectionPath = path.join('.');
  const impacts = impactBySection[sectionPath] || [];

  return React.createElement('div', { className: 'config-tree' },
    impacts.length > 0 && React.createElement('div', { className: 'badge-row', style: { marginBottom: 8 } },
      impacts.map(tag =>
        React.createElement('span', { key: tag, className: 'badge badge-info' }, tag)
      )
    ),
    entries.map(([key, child]) =>
      React.createElement('details', { key: key, open: path.length < 2 },
        React.createElement('summary', { className: 'mono' }, key),
        React.createElement('div', { style: { paddingLeft: 16, marginTop: 4 } },
          React.createElement(ConfigTree, {
            value: child,
            path: [...path, key],
            impactBySection,
          })
        )
      )
    )
  );
}

function ConfigView() {
  const [data, setData] = React.useState(null);
  const [error, setError] = React.useState(null);

  React.useEffect(() => {
    fetch('/v1/dashboard/api/config')
      .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(setData)
      .catch(e => setError(e.message));
  }, []);

  if (error) {
    return React.createElement('div', { className: 'empty-state' },
      'Failed to load configuration: ' + error
    );
  }
  if (!data) {
    return React.createElement('div', { className: 'empty-state' }, 'Loading…');
  }

  const impactBySection = {};
  (data.configured || []).forEach(entry => {
    impactBySection[entry.section] = (entry.justification || []).map(j => j.toUpperCase());
  });

  return React.createElement(React.Fragment, null,
    React.createElement('p', {
      style: { color: 'var(--nv-text-muted)', marginBottom: 16 },
    }, 'Effective system configuration (secrets redacted). Impact badges reflect @configured registry tags.'),
    React.createElement('div', { className: 'card' },
      React.createElement(ConfigTree, {
        value: data.config,
        path: [],
        impactBySection,
      })
    ),
    (data.configured || []).length > 0 && React.createElement('div', { className: 'card', style: { marginTop: 16 } },
      React.createElement('h3', { style: { marginBottom: 12 } }, 'Configured resolvers'),
      React.createElement('table', { className: 'data-table' },
        React.createElement('thead', null,
          React.createElement('tr', null,
            React.createElement('th', null, 'Section'),
            React.createElement('th', null, 'Impact'),
            React.createElement('th', null, 'Rationale'),
          )
        ),
        React.createElement('tbody', null,
          (data.configured || []).map(entry =>
            React.createElement('tr', { key: entry.qualname },
              React.createElement('td', { className: 'mono' }, entry.section),
              React.createElement('td', null,
                (entry.justification || []).map(tag =>
                  React.createElement('span', {
                    key: tag,
                    className: 'badge badge-info',
                    style: { marginRight: 4 },
                  }, tag)
                )
              ),
              React.createElement('td', null, entry.rationale || '—'),
            )
          )
        )
      )
    )
  );
}
