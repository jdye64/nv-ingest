function SearchApp() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState('');
  const [status, setStatus] = useState(null);
  const [expandedRow, setExpandedRow] = useState(null);
  const [topK, setTopK] = useState(10);
  const [bootstrapped, setBootstrapped] = useState(false);

  const loadStatus = useCallback(async () => {
    try {
      const res = await fetch('/api/v1/status');
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      setStatus(data);
    } catch (err) {
      setStatus({ service_reachable: false, total_rows: 0, table_exists: false, message: err.message });
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const applied = await bootstrapSettingsFromStorage();
      if (cancelled) return;
      if (applied && applied.default_top_k) {
        setTopK(applied.default_top_k);
      }
      setBootstrapped(true);
      await loadStatus();
    })();
    return () => { cancelled = true; };
  }, [loadStatus]);

  const handleSettingsSaved = useCallback((data) => {
    if (data && data.default_top_k) {
      setTopK(data.default_top_k);
    }
    loadStatus();
  }, [loadStatus]);

  const handleSearch = useCallback(async () => {
    const q = query.trim();
    if (!q) return;
    setSearching(true);
    setError('');
    setExpandedRow(null);
    try {
      const res = await fetch('/api/v1/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, top_k: topK }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      setResults(data);
    } catch (err) {
      setError(err.message);
      setResults(null);
    } finally {
      setSearching(false);
    }
  }, [query, topK]);

  return (
    <div className="search-page">
      <header className="top-bar">
        <SettingsButton onSaved={handleSettingsSaved} />
      </header>
      {!bootstrapped ? (
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--nv-text-muted)' }}>
          <span className="spinner" style={{ width: 28, height: 28 }}></span>
        </div>
      ) : (
        <>
          <SearchBar
            query={query}
            setQuery={setQuery}
            searching={searching}
            onSearch={handleSearch}
            status={status}
            onStatusRefresh={loadStatus}
          />
          {error && <div className="error-banner">{error}</div>}
          {(results || searching) && (
            <div className="results-area">
              {searching ? (
                <div style={{ textAlign: 'center', padding: 40, color: 'var(--nv-text-muted)' }}>
                  <span className="spinner" style={{ width: 28, height: 28, marginBottom: 12 }}></span>
                  <div>Searching…</div>
                </div>
              ) : (
                <ResultsTable results={results} expandedRow={expandedRow} setExpandedRow={setExpandedRow} />
              )}
            </div>
          )}
        </>
      )}
      <div className="footer">
        API: <span className="mono">/api/v1/search</span> · MCP: <span className="mono">/mcp</span> · Docs: <span className="mono">/api/docs</span>
      </div>
    </div>
  );
}

try {
  ReactDOM.createRoot(document.getElementById("root")).render(React.createElement(SearchApp));
} catch (err) {
  const bootError = document.getElementById("boot-error");
  if (bootError) {
    bootError.style.display = "block";
    bootError.textContent = "Failed to mount SearchApp: " + err;
  }
  console.error(err);
}
