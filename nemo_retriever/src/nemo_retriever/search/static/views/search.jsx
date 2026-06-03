function SearchBar({ query, setQuery, searching, onSearch, status, onStatusRefresh, onUploadComplete }) {
  const handleSubmit = (e) => {
    e.preventDefault();
    onSearch();
  };

  const statusLabel = useMemo(() => {
    if (!status) return 'Checking corpus…';
    if (!status.service_reachable) return 'Retriever service offline';
    if (!status.table_exists || status.total_rows === 0) return 'No documents ingested yet — use + to upload';
    return `${status.total_rows.toLocaleString()} chunks ready`;
  }, [status]);

  return (
    <>
      <div className={`search-hero ${query ? 'compact' : ''}`}>
        <div className="brand">
          <h1>NeMo Retriever</h1>
          <p>Search your ingested documents</p>
        </div>
        <form onSubmit={handleSubmit} style={{ width: '100%', maxWidth: 720 }}>
          <div className="search-bar-wrap">
            <UploadControls onUploadComplete={() => { onStatusRefresh && onStatusRefresh(); }} />
            <input
              className="search-input"
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="Search your documents…"
              disabled={searching}
              autoFocus
            />
            <button type="submit" className="btn btn-primary" disabled={searching || !query.trim()} style={{ borderRadius: 999 }}>
              {searching ? <span className="spinner"></span> : <IconSearch />}
            </button>
          </div>
        </form>
        <div className="status-line">
          {status && status.service_reachable && status.total_rows > 0 && (
            <span className="badge badge-pass" style={{ marginRight: 8, fontSize: 10 }}>Ready</span>
          )}
          {statusLabel}
        </div>
      </div>
    </>
  );
}
