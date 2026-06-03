function UploadModal({ open, title, message, busy, onClose }) {
  if (!open) return null;
  return (
    <div className="modal-overlay" onClick={busy ? undefined : onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <div className="modal-head">
          <div style={{ fontWeight: 600, color: '#fff' }}>{title}</div>
          {!busy && (
            <button className="btn btn-icon" onClick={onClose} aria-label="Close">
              <IconX />
            </button>
          )}
        </div>
        <div className="modal-body" style={{ textAlign: 'center' }}>
          {busy && <span className="spinner" style={{ width: 32, height: 32, marginBottom: 16 }}></span>}
          <div style={{ fontSize: 14, color: 'var(--nv-text-muted)', lineHeight: 1.6 }}>{message}</div>
        </div>
      </div>
    </div>
  );
}

function UploadControls({ onUploadStart, onUploadComplete, onUploadError }) {
  const fileInputRef = useRef(null);
  const dirInputRef = useRef(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [modal, setModal] = useState({ open: false, title: '', message: '' });

  const uploadFiles = useCallback(async (fileList) => {
    if (!fileList || !fileList.length) return;
    setUploading(true);
    setModal({ open: true, title: 'Uploading documents', message: `Ingesting ${fileList.length} file(s)…` });
    onUploadStart && onUploadStart(fileList.length);
    const form = new FormData();
    for (const f of fileList) form.append('files', f, f.name);
    try {
      const res = await fetch('/api/v1/ingest', { method: 'POST', body: form });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      setModal({
        open: true,
        title: 'Upload complete',
        message: `${data.documents_succeeded}/${data.documents_submitted} documents ingested in ${data.elapsed_s}s.`,
      });
      onUploadComplete && onUploadComplete(data);
    } catch (err) {
      setModal({ open: true, title: 'Upload failed', message: err.message });
      onUploadError && onUploadError(err.message);
    } finally {
      setUploading(false);
    }
  }, [onUploadStart, onUploadComplete, onUploadError]);

  return (
    <>
      <div className="upload-menu">
        <button
          type="button"
          className="btn btn-icon"
          title="Upload documents"
          onClick={() => setMenuOpen(v => !v)}
          disabled={uploading}
        >
          <IconPlus />
        </button>
        {menuOpen && (
          <div className="upload-dropdown">
            <button type="button" onClick={() => { setMenuOpen(false); fileInputRef.current && fileInputRef.current.click(); }}>
              Upload file(s)
            </button>
            <button type="button" onClick={() => { setMenuOpen(false); dirInputRef.current && dirInputRef.current.click(); }}>
              Upload folder
            </button>
          </div>
        )}
        <input ref={fileInputRef} type="file" multiple style={{ display: 'none' }}
          onChange={e => { uploadFiles(Array.from(e.target.files || [])); e.target.value = ''; }} />
        <input ref={dirInputRef} type="file" multiple webkitdirectory="" style={{ display: 'none' }}
          onChange={e => { uploadFiles(Array.from(e.target.files || [])); e.target.value = ''; }} />
      </div>
      <UploadModal
        open={modal.open}
        title={modal.title}
        message={modal.message}
        busy={uploading}
        onClose={() => setModal({ open: false, title: '', message: '' })}
      />
    </>
  );
}
