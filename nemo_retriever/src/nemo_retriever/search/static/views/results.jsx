function ResultsTable({ results, expandedRow, setExpandedRow }) {
  if (!results || !results.hits || !results.hits.length) {
    return (
      <div className="card" style={{ padding: 40, textAlign: 'center', color: 'var(--nv-text-muted)' }}>
        No results found.
      </div>
    );
  }

  return (
    <div className="card">
      <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--nv-border)', display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
        <div>
          <span style={{ fontSize: 15, fontWeight: 600, color: '#fff' }}>{results.hit_count} result{results.hit_count !== 1 ? 's' : ''}</span>
          <span style={{ fontSize: 12, color: 'var(--nv-text-dim)', marginLeft: 12 }}>for "{results.query}"</span>
        </div>
        <span className="mono" style={{ fontSize: 11, color: 'var(--nv-text-dim)' }}>search_id: {results.search_id}</span>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="runs-table">
          <thead>
            <tr>
              <th style={{ width: 40 }}>#</th>
              <th>Source</th>
              <th style={{ width: 60 }}>Page</th>
              <th style={{ width: 90 }}>Type</th>
              <th style={{ width: 90, textAlign: 'right' }}>Distance</th>
              <th>Preview</th>
              <th style={{ width: 220 }}>Export</th>
            </tr>
          </thead>
          <tbody>
            {results.hits.map((hit, idx) => {
              const expanded = expandedRow === idx;
              return React.createElement(React.Fragment, { key: hit.hit_id },
                React.createElement('tr', {
                  style: { cursor: 'pointer' },
                  onClick: () => setExpandedRow(expanded ? null : idx),
                },
                  React.createElement('td', { style: { color: 'var(--nv-text-dim)', fontWeight: 600 } }, hit.rank),
                  React.createElement('td', { className: 'mono', style: { fontSize: 12, maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }, title: hit.source }, hit.source || '—'),
                  React.createElement('td', { style: { textAlign: 'center' } }, hit.page_number != null ? hit.page_number : '—'),
                  React.createElement('td', null, hit.content_type || '—'),
                  React.createElement('td', { className: 'mono', style: { textAlign: 'right', color: typeof hit.distance === 'number' ? 'var(--nv-green)' : 'var(--nv-text-muted)' } },
                    typeof hit.distance === 'number' ? hit.distance.toFixed(4) : '—'),
                  React.createElement('td', { style: { maxWidth: 320, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: 'var(--nv-text-muted)' }, title: hit.text }, hit.text_preview || '—'),
                  React.createElement('td', { onClick: e => e.stopPropagation() },
                    React.createElement('div', { style: { display: 'flex', gap: 6, flexWrap: 'wrap' } },
                      React.createElement('a', {
                        className: 'btn btn-secondary btn-sm',
                        href: hit.export.document_url,
                        download: true,
                        title: 'Download original document',
                      }, 'Doc'),
                      React.createElement('a', { className: 'btn btn-secondary btn-sm', href: hit.export.text_url + '&download=1', download: true }, 'Text'),
                      React.createElement('a', { className: 'btn btn-secondary btn-sm', href: hit.export.json_url + '&download=1', download: true }, 'JSON'),
                      React.createElement('button', {
                        className: 'btn btn-secondary btn-sm',
                        onClick: async () => { await copyText(hit.export.text_url); },
                        title: 'Copy text export URL',
                      }, 'URL')
                    )
                  )
                ),
                expanded && React.createElement('tr', null,
                  React.createElement('td', { colSpan: 7, style: { padding: 0, background: 'var(--nv-bg)' } },
                    React.createElement('div', { style: { padding: '16px 20px', borderTop: '1px solid var(--nv-border)' } },
                      React.createElement('div', { style: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 14 } },
                        React.createElement('div', null,
                          React.createElement('div', { style: { fontSize: 10, fontWeight: 600, color: 'var(--nv-text-dim)', textTransform: 'uppercase', marginBottom: 4 } }, 'Source'),
                          React.createElement('div', { className: 'mono', style: { fontSize: 12, wordBreak: 'break-all' } }, hit.source || '—')
                        ),
                        React.createElement('div', null,
                          React.createElement('div', { style: { fontSize: 10, fontWeight: 600, color: 'var(--nv-text-dim)', textTransform: 'uppercase', marginBottom: 4 } }, 'Provenance'),
                          React.createElement('div', { className: 'mono', style: { fontSize: 12 } }, hit.provenance.pdf_page || hit.provenance.source_id || '—')
                        )
                      ),
                      React.createElement('div', { style: { fontSize: 10, fontWeight: 600, color: 'var(--nv-text-dim)', textTransform: 'uppercase', marginBottom: 6 } }, 'Full Text'),
                      React.createElement('div', {
                        style: {
                          fontSize: 13, lineHeight: 1.7, padding: 14, borderRadius: 8,
                          background: 'rgba(255,255,255,0.03)', border: '1px solid var(--nv-border)',
                          maxHeight: 280, overflowY: 'auto', whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                        },
                      }, hit.text || '—'),
                      React.createElement('div', { style: { marginTop: 12, display: 'flex', gap: 8, flexWrap: 'wrap' } },
                        React.createElement('a', {
                          className: 'btn btn-secondary btn-sm',
                          href: hit.export.document_url,
                          download: true,
                        }, 'Download document'),
                        React.createElement('button', {
                          className: 'btn btn-secondary btn-sm',
                          onClick: async () => { await copyText(hit.text || ''); },
                        }, 'Copy text'),
                        React.createElement('button', {
                          className: 'btn btn-secondary btn-sm',
                          onClick: async () => {
                            const res = await fetch(hit.export.summary_url);
                            await copyText(await res.text());
                          },
                        }, 'Copy summary')
                      )
                    )
                  )
                )
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
