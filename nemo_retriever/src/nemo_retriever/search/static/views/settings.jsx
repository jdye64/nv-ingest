const SETTINGS_STORAGE_KEY = 'nemo-search-settings';

function loadStoredSettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (e) {
    return null;
  }
}

function storeSettings(payload) {
  try {
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(payload));
  } catch (e) { /* ignore quota errors */ }
}

function clearStoredSettings() {
  try { localStorage.removeItem(SETTINGS_STORAGE_KEY); } catch (e) { /* ignore */ }
}

async function applySettingsPayload(payload) {
  const res = await fetch('/api/v1/settings', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}

function SettingsMenu({ open, onClose, onSaved }) {
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [form, setForm] = useState({
    service_url: '',
    vectordb_url: '',
    api_token: '',
    default_top_k: 10,
  });
  const [apiTokenSet, setApiTokenSet] = useState(false);
  const [defaults, setDefaults] = useState(null);
  const [placeholders, setPlaceholders] = useState([]);

  useEffect(() => {
    if (!open) return;
    setLoading(true);
    setError('');
    fetch('/api/v1/settings')
      .then(r => r.json().then(data => ({ ok: r.ok, data })))
      .then(({ ok, data }) => {
        if (!ok) throw new Error(data.detail || 'Failed to load settings');
        setForm({
          service_url: data.service_url || '',
          vectordb_url: data.vectordb_url || '',
          api_token: '',
          default_top_k: data.default_top_k || 10,
        });
        setApiTokenSet(!!data.api_token_set);
        setDefaults(data.defaults || null);
        setPlaceholders(data.placeholders || []);
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false));
  }, [open]);

  const setField = (key, value) => setForm(prev => ({ ...prev, [key]: value }));

  const handleSave = async () => {
    setSaving(true);
    setError('');
    const payload = {
      service_url: form.service_url.trim(),
      vectordb_url: form.vectordb_url.trim(),
      default_top_k: Number(form.default_top_k) || 10,
    };
    if (form.api_token.trim()) {
      payload.api_token = form.api_token.trim();
    }
    try {
      const data = await applySettingsPayload(payload);
      storeSettings({
        service_url: payload.service_url,
        vectordb_url: payload.vectordb_url,
        default_top_k: payload.default_top_k,
        ...(payload.api_token ? { api_token: payload.api_token } : {}),
      });
      setApiTokenSet(!!data.api_token_set);
      setForm(prev => ({ ...prev, api_token: '' }));
      onSaved && onSaved(data);
      onClose();
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  const handleReset = async () => {
    setSaving(true);
    setError('');
    try {
      const res = await fetch('/api/v1/settings/reset', { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      clearStoredSettings();
      setForm({
        service_url: data.service_url,
        vectordb_url: data.vectordb_url,
        api_token: '',
        default_top_k: data.default_top_k,
      });
      setApiTokenSet(!!data.api_token_set);
      onSaved && onSaved(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  if (!open) return null;

  const labelStyle = {
    display: 'block', fontSize: 11, fontWeight: 600, color: 'var(--nv-text-dim)',
    textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: 6,
  };

  return (
    <>
      <div className="settings-backdrop" onClick={onClose} />
      <aside className="settings-drawer" aria-label="Settings">
        <div className="settings-drawer-head">
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <IconSettings />
            <span style={{ fontWeight: 600, color: '#fff', fontSize: 15 }}>Settings</span>
          </div>
          <button type="button" className="btn btn-icon" onClick={onClose} aria-label="Close settings">
            <IconX />
          </button>
        </div>
        <div className="settings-drawer-body">
          {loading ? (
            <div style={{ textAlign: 'center', padding: 32, color: 'var(--nv-text-muted)' }}>
              <span className="spinner" style={{ width: 24, height: 24 }}></span>
            </div>
          ) : (
            <>
              <div className="settings-section-title">Connection</div>
              <div style={{ marginBottom: 16 }}>
                <label style={labelStyle}>Retriever service URL</label>
                <input
                  className="input settings-input"
                  value={form.service_url}
                  onChange={e => setField('service_url', e.target.value)}
                  placeholder="http://localhost:7670"
                />
                <div className="settings-hint">Base URL for <span className="mono">/v1/query</span> and ingest.</div>
              </div>
              <div style={{ marginBottom: 16 }}>
                <label style={labelStyle}>VectorDB URL</label>
                <input
                  className="input settings-input"
                  value={form.vectordb_url}
                  onChange={e => setField('vectordb_url', e.target.value)}
                  placeholder="http://localhost:7671"
                />
                <div className="settings-hint">Used for corpus status (row count).</div>
              </div>
              <div style={{ marginBottom: 16 }}>
                <label style={labelStyle}>API token {apiTokenSet && <span className="badge badge-pass" style={{ marginLeft: 8, fontSize: 9 }}>Set</span>}</label>
                <input
                  className="input settings-input mono"
                  type="password"
                  value={form.api_token}
                  onChange={e => setField('api_token', e.target.value)}
                  placeholder={apiTokenSet ? 'Leave blank to keep current token' : 'Optional bearer token'}
                  autoComplete="off"
                />
              </div>

              <div className="settings-section-title">Search</div>
              <div style={{ marginBottom: 20 }}>
                <label style={labelStyle}>Default top K</label>
                <input
                  className="input settings-input"
                  type="number"
                  min="1"
                  max="1000"
                  value={form.default_top_k}
                  onChange={e => setField('default_top_k', e.target.value)}
                />
              </div>

              {placeholders.length > 0 && (
                <>
                  <div className="settings-section-title">Coming soon</div>
                  {placeholders.map(ph => (
                    <div key={ph.id} className="settings-placeholder" style={{ marginBottom: 12 }}>
                      <label style={labelStyle}>{ph.label}</label>
                      <input className="input settings-input" disabled placeholder="Not available yet" />
                      <div className="settings-hint">{ph.description}</div>
                    </div>
                  ))}
                </>
              )}

              {defaults && (
                <div className="settings-hint" style={{ marginTop: 8 }}>
                  Server defaults: <span className="mono">{defaults.service_url}</span>
                </div>
              )}
            </>
          )}
          {error && <div className="settings-error">{error}</div>}
        </div>
        <div className="settings-drawer-foot">
          <button type="button" className="btn btn-secondary" onClick={handleReset} disabled={saving || loading}>
            Reset
          </button>
          <button type="button" className="btn btn-primary" onClick={handleSave} disabled={saving || loading}>
            {saving ? 'Saving…' : 'Save'}
          </button>
        </div>
      </aside>
    </>
  );
}

function SettingsButton({ onSaved }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button
        type="button"
        className="btn btn-icon hamburger-btn"
        onClick={() => setOpen(true)}
        aria-label="Open settings menu"
        title="Settings"
      >
        <IconMenu />
      </button>
      <SettingsMenu open={open} onClose={() => setOpen(false)} onSaved={onSaved} />
    </>
  );
}

async function bootstrapSettingsFromStorage() {
  const stored = loadStoredSettings();
  if (!stored) return null;
  try {
    return await applySettingsPayload(stored);
  } catch (e) {
    return null;
  }
}
