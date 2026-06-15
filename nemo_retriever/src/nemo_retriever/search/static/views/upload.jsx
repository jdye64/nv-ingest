async function readJsonResponse(res) {
  const text = await res.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    throw new Error(text.slice(0, 240).trim() || `HTTP ${res.status}`);
  }
}

function terminalJobStatus(status) {
  return status === 'completed' || status === 'failed' || status === 'partial_success';
}

function jobProgressMessage(data) {
  const counts = data.counts || {};
  const completed = counts.completed || 0;
  const failed = counts.failed || 0;
  const processing = counts.processing || 0;
  const pending = counts.pending || 0;
  const total = data.expected_documents || completed + failed + processing + pending;
  const done = completed + failed;
  if (terminalJobStatus(data.status)) {
    return `${completed}/${total} documents ingested` + (failed ? ` (${failed} failed)` : '');
  }
  return `Processing ${done}/${total} documents…`;
}

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

  const pollIngestJob = useCallback(async (jobId) => {
    const deadline = Date.now() + 30 * 60 * 1000;
    while (Date.now() < deadline) {
      await new Promise(resolve => setTimeout(resolve, 2000));
      const res = await fetch(`/api/v1/ingest/jobs/${encodeURIComponent(jobId)}`);
      const data = await readJsonResponse(res);
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      setModal({
        open: true,
        title: 'Processing documents',
        message: jobProgressMessage(data),
      });
      if (terminalJobStatus(data.status)) {
        return data;
      }
    }
    throw new Error('Ingest timed out after 30 minutes. Check the retriever service for job status.');
  }, []);

  const uploadFiles = useCallback(async (fileList) => {
    if (!fileList || !fileList.length) return;
    setUploading(true);
    setModal({ open: true, title: 'Uploading documents', message: `Uploading ${fileList.length} file(s)…` });
    onUploadStart && onUploadStart(fileList.length);
    const form = new FormData();
    for (const f of fileList) form.append('files', f, f.name);
    try {
      const res = await fetch('/api/v1/ingest', { method: 'POST', body: form });
      const data = await readJsonResponse(res);
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);

      if (data.documents_failed > 0 && !data.job_id) {
        throw new Error((data.errors && data.errors[0]) || 'Upload failed');
      }

      let finalStatus = data;
      if (data.job_id && data.status === 'processing') {
        finalStatus = await pollIngestJob(data.job_id);
      }

      const succeeded = (finalStatus.counts && finalStatus.counts.completed) ?? data.documents_succeeded ?? 0;
      const failed = (finalStatus.counts && finalStatus.counts.failed) ?? data.documents_failed ?? 0;
      const total = finalStatus.expected_documents ?? data.documents_submitted ?? fileList.length;
      const title = failed > 0 && succeeded === 0 ? 'Upload failed' : 'Upload complete';
      const message = failed > 0
        ? `${succeeded}/${total} documents ingested (${failed} failed).`
        : `${succeeded}/${total} documents ingested.`;

      setModal({ open: true, title, message });
      onUploadComplete && onUploadComplete({ ...data, ...finalStatus, documents_succeeded: succeeded, documents_failed: failed });
    } catch (err) {
      setModal({ open: true, title: 'Upload failed', message: err.message });
      onUploadError && onUploadError(err.message);
    } finally {
      setUploading(false);
    }
  }, [onUploadStart, onUploadComplete, onUploadError, pollIngestJob]);

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
