/* ===== Run Detail Modal ===== */

function FormattedCommand({ command }) {
  if (!command) return null;
  const tokens = command.match(/(?:[^\s"']+|"[^"]*"|'[^']*')+/g) || [command];
  const groups = [];
  let i = 0;
  while (i < tokens.length) {
    const t = tokens[i];
    if (t.startsWith("--")) {
      const flag = t;
      const vals = [];
      i++;
      while (i < tokens.length && !tokens[i].startsWith("--")) {
        vals.push(tokens[i]);
        i++;
      }
      groups.push({ flag, value: vals.join(" ") });
    } else {
      groups.push({ flag: null, value: t });
      i++;
    }
  }
  return (
    <div style={{
      padding:'14px 16px',borderRadius:'8px',
      background:'var(--nv-bg)',border:'1px solid var(--nv-border)',
      maxHeight:'260px',overflow:'auto',lineHeight:'1.7',
    }}>
      {groups.map((g, idx) => (
        <div key={idx} style={{display:'flex',gap:'8px',alignItems:'baseline',fontSize:'12px'}}>
          {g.flag ? (
            <>
              <span className="mono" style={{color:'#64b4ff',fontWeight:600,whiteSpace:'nowrap'}}>{g.flag}</span>
              {g.value && <span className="mono" style={{color:'var(--nv-green)',wordBreak:'break-all'}}>{g.value}</span>}
            </>
          ) : (
            <span className="mono" style={{color:'var(--nv-text-muted)',wordBreak:'break-all'}}>{g.value}</span>
          )}
        </div>
      ))}
    </div>
  );
}

function GraphPreview({ runId }) {
  const [graphData, setGraphData] = useState(null);
  const [graphLoading, setGraphLoading] = useState(false);
  const [graphError, setGraphError] = useState(null);
  const [showGraph, setShowGraph] = useState(false);
  const [expandedNodes, setExpandedNodes] = useState({});

  useEffect(() => {
    if (showGraph && graphData === null && !graphLoading) {
      setGraphLoading(true);
      fetch(`/api/runs/${runId}/graph`)
        .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
        .then(data => {
          if (data.graph_json) {
            const gj = typeof data.graph_json === 'string' ? JSON.parse(data.graph_json) : data.graph_json;
            setGraphData(gj);
          } else {
            setGraphError("No graph data associated with this run.");
          }
        })
        .catch(() => setGraphError("No graph data associated with this run."))
        .finally(() => setGraphLoading(false));
    }
  }, [showGraph, graphData, graphLoading, runId]);

  function toggleNode(nodeId) {
    setExpandedNodes(prev => ({ ...prev, [nodeId]: !prev[nodeId] }));
  }

  const catColors = {
    'Document Processing': { bg: 'rgba(0,150,255,0.1)', border: 'rgba(0,150,255,0.3)', text: '#4da6ff' },
    'Detection & OCR': { bg: 'rgba(118,185,0,0.1)', border: 'rgba(118,185,0,0.3)', text: '#76b900' },
    'Embedding': { bg: 'rgba(180,100,255,0.1)', border: 'rgba(180,100,255,0.3)', text: '#b464ff' },
    'Storage': { bg: 'rgba(255,165,0,0.1)', border: 'rgba(255,165,0,0.3)', text: '#ffa500' },
    'Evaluation': { bg: 'rgba(255,100,100,0.1)', border: 'rgba(255,100,100,0.3)', text: '#ff6464' },
  };
  const defaultCat = { bg: 'rgba(150,150,150,0.08)', border: 'rgba(150,150,150,0.2)', text: '#aaa' };

  return (
    <div style={{marginBottom:'28px'}}>
      <div className="section-title"
        style={{cursor:'pointer',userSelect:'none',display:'flex',alignItems:'center',gap:'8px'}}
        onClick={() => setShowGraph(!showGraph)}
      >
        <span style={{transform:showGraph?'rotate(90deg)':'rotate(0deg)',transition:'transform 0.15s',display:'inline-block',fontSize:'10px'}}>{"\u25B6"}</span>
        Pipeline Graph
      </div>
      {showGraph && (
        graphLoading ? (
          <div style={{padding:'12px',color:'var(--nv-text-muted)',fontSize:'13px'}}><span className="spinner" style={{marginRight:'8px'}}></span>Loading…</div>
        ) : graphError ? (
          <div style={{padding:'12px',color:'var(--nv-text-dim)',fontSize:'13px',fontStyle:'italic'}}>{graphError}</div>
        ) : graphData && graphData.nodes ? (
          <div style={{
            padding:'16px',borderRadius:'8px',
            background:'rgba(0,0,0,0.2)',border:'1px solid var(--nv-border)',
            overflowX:'auto',
          }}>
            <div style={{display:'flex',gap:'0',alignItems:'stretch',minWidth:'fit-content'}}>
              {(() => {
                const nodes = graphData.nodes || [];
                const edges = graphData.edges || [];
                const ordered = [];
                const visited = new Set();
                const adjMap = {};
                edges.forEach(e => { adjMap[e.source] = adjMap[e.source] || []; adjMap[e.source].push(e.target); });
                const sources = nodes.filter(n => !edges.some(e => e.target === n.id));
                const queue = sources.length > 0 ? [...sources] : (nodes.length > 0 ? [nodes[0]] : []);
                while (queue.length > 0) {
                  const node = queue.shift();
                  if (visited.has(node.id)) continue;
                  visited.add(node.id);
                  ordered.push(node);
                  (adjMap[node.id] || []).forEach(tid => {
                    const tnode = nodes.find(n => n.id === tid);
                    if (tnode && !visited.has(tnode.id)) queue.push(tnode);
                  });
                }
                nodes.forEach(n => { if (!visited.has(n.id)) ordered.push(n); });

                return ordered.map((node, idx) => {
                  const label = (node.data && node.data.label) || node.type || 'Unknown';
                  const category = (node.data && node.data.category) || '';
                  const compute = (node.data && node.data.compute) || '';
                  const colors = catColors[category] || defaultCat;
                  const kwargs = (node.data && node.data.kwargs) || (node.data && node.data.params) || {};
                  const kwEntries = Object.entries(kwargs).filter(([k]) => k !== 'label' && k !== 'category' && k !== 'compute');
                  const isExpanded = expandedNodes[node.id];
                  const hasNext = idx < ordered.length - 1;

                  return (
                    <React.Fragment key={node.id}>
                      <div style={{
                        minWidth:'180px',maxWidth:'260px',borderRadius:'8px',
                        border:`1px solid ${colors.border}`,background:colors.bg,
                        padding:'12px 14px',display:'flex',flexDirection:'column',gap:'6px',
                        flexShrink:0,
                      }}>
                        <div style={{display:'flex',alignItems:'center',gap:'8px',justifyContent:'space-between'}}>
                          <span style={{fontSize:'13px',fontWeight:700,color:'#fff'}}>{label}</span>
                          {compute && (
                            <span style={{
                              fontSize:'9px',padding:'1px 5px',borderRadius:'3px',textTransform:'uppercase',fontWeight:700,letterSpacing:'0.05em',
                              background: compute === 'gpu' ? 'rgba(118,185,0,0.2)' : 'rgba(100,180,255,0.2)',
                              color: compute === 'gpu' ? '#76b900' : '#64b4ff',
                            }}>{compute}</span>
                          )}
                        </div>
                        {category && <span style={{fontSize:'10px',color:colors.text,fontWeight:500}}>{category}</span>}
                        {kwEntries.length > 0 && (
                          <div>
                            <div style={{
                              fontSize:'10px',color:'var(--nv-text-dim)',cursor:'pointer',userSelect:'none',
                              display:'flex',alignItems:'center',gap:'4px',marginTop:'2px',
                            }} onClick={() => toggleNode(node.id)}>
                              <span style={{transform:isExpanded?'rotate(90deg)':'rotate(0deg)',transition:'transform 0.1s',display:'inline-block'}}>{"\u25B6"}</span>
                              {kwEntries.length} param{kwEntries.length !== 1 ? 's' : ''}
                            </div>
                            {isExpanded && (
                              <div style={{
                                marginTop:'6px',fontSize:'11px',
                                display:'grid',gridTemplateColumns:'auto 1fr',gap:'2px 8px',
                              }}>
                                {kwEntries.map(([k, v]) => (
                                  <React.Fragment key={k}>
                                    <span className="mono" style={{color:'#64b4ff',whiteSpace:'nowrap'}}>{k}</span>
                                    <span className="mono" style={{color:'var(--nv-text-muted)',wordBreak:'break-all'}}>
                                      {typeof v === 'object' ? JSON.stringify(v) : String(v)}
                                    </span>
                                  </React.Fragment>
                                ))}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                      {hasNext && (
                        <div style={{display:'flex',alignItems:'center',flexShrink:0,padding:'0 2px'}}>
                          <svg width="28" height="20" viewBox="0 0 28 20" style={{display:'block'}}>
                            <line x1="0" y1="10" x2="20" y2="10" stroke="rgba(255,255,255,0.2)" strokeWidth="2" />
                            <polygon points="18,5 28,10 18,15" fill="rgba(255,255,255,0.25)" />
                          </svg>
                        </div>
                      )}
                    </React.Fragment>
                  );
                });
              })()}
            </div>
          </div>
        ) : null
      )}
    </div>
  );
}

function RunDetailModal({ run, onClose, onDelete, githubRepoUrl }) {
  if (!run) return null;
  const [showRaw, setShowRaw] = useState(false);
  const [showCommand, setShowCommand] = useState(false);
  const [commandText, setCommandText] = useState(null);
  const [commandLoading, setCommandLoading] = useState(false);
  const [linkCopied, setLinkCopied] = useState(false);
  const [rerunState, setRerunState] = useState(null);
  const [rerunSubmitting, setRerunSubmitting] = useState(false);
  const [rerunSelectedRunner, setRerunSelectedRunner] = useState("");
  const [showLogs, setShowLogs] = useState(false);
  const [logLines, setLogLines] = useState(null);
  const [logLoading, setLogLoading] = useState(false);
  const [showPipList, setShowPipList] = useState(false);
  const [pipListText, setPipListText] = useState(null);
  const [pipListLoading, setPipListLoading] = useState(false);
  const [showPlan, setShowPlan] = useState(false);
  const [showRayStats, setShowRayStats] = useState(false);
  const [zipDownloading, setZipDownloading] = useState(false);
  const logRef = useRef(null);
  const raw = run.raw_json && typeof run.raw_json === 'object' && Object.keys(run.raw_json).length > 0 ? run.raw_json : run;

  useEffect(() => {
    if (showLogs && logLines === null && !logLoading && run.job_id) {
      setLogLoading(true);
      fetch(`/api/jobs/${run.job_id}/logs`)
        .then(r => r.json())
        .then(data => setLogLines(data.log_tail || []))
        .catch(() => setLogLines([]))
        .finally(() => setLogLoading(false));
    }
  }, [showLogs, logLines, logLoading, run.job_id]);

  useEffect(() => {
    if (showLogs && logLines && logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [showLogs, logLines]);

  useEffect(() => {
    if (showPipList && pipListText === null && !pipListLoading && run.job_id) {
      setPipListLoading(true);
      fetch(`/api/jobs/${run.job_id}`)
        .then(r => r.json())
        .then(data => setPipListText(data.pip_list || ""))
        .catch(() => setPipListText(""))
        .finally(() => setPipListLoading(false));
    }
  }, [showPipList, pipListText, pipListLoading, run.job_id]);

  useEffect(() => {
    if (showCommand && commandText === null && !commandLoading) {
      setCommandLoading(true);
      fetch(`/api/runs/${run.id}/command`)
        .then(r => r.json())
        .then(data => setCommandText(data.command || ""))
        .catch(() => setCommandText(""))
        .finally(() => setCommandLoading(false));
    }
  }, [showCommand, commandText, commandLoading, run.id]);

  function copyRunLink() {
    const url = `${window.location.origin}${window.location.pathname}#runs/${run.id}`;
    function onSuccess() { setLinkCopied(true); setTimeout(() => setLinkCopied(false), 2000); }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(url).then(onSuccess).catch(() => fallbackCopy(url, onSuccess));
    } else {
      fallbackCopy(url, onSuccess);
    }
  }
  function fallbackCopy(text, onSuccess) {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.cssText = "position:fixed;left:-9999px;top:-9999px;opacity:0";
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); onSuccess(); } catch {}
    document.body.removeChild(ta);
  }

  async function handleZipDownload() {
    setZipDownloading(true);
    try {
      const res = await fetch(`/api/runs/${run.id}/download/zip`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      const ds = (run.dataset || 'unknown').replace(/[^a-zA-Z0-9_-]/g, '_');
      a.download = `run_${run.id}_${ds}.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      alert("Failed to download artifacts: " + err.message);
    } finally {
      setZipDownloading(false);
    }
  }

  async function handleRerunClick() {
    try {
      const res = await fetch(`/api/runs/${run.id}/rerun-info`);
      const info = await res.json();
      if (info.original_runner) {
        setRerunSelectedRunner(String(info.original_runner.id));
      } else {
        setRerunSelectedRunner("");
      }
      setRerunState(info);
    } catch (err) {
      alert("Failed to fetch re-run info: " + err.message);
    }
  }

  async function submitRerun() {
    setRerunSubmitting(true);
    try {
      const payload = {};
      if (rerunSelectedRunner) payload.runner_id = Number(rerunSelectedRunner);
      const res = await fetch(`/api/runs/${run.id}/rerun`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      if (!res.ok) { const d = await res.json().catch(()=>({})); throw new Error(d.detail || `HTTP ${res.status}`); }
      const data = await res.json();
      alert(`Re-run queued as job ${data.job_id}` + (data.git_commit ? ` (commit ${data.git_commit.slice(0, 12)})` : ""));
      setRerunState(null);
    } catch (err) {
      alert("Re-run failed: " + err.message);
    } finally {
      setRerunSubmitting(false);
    }
  }

  const effectiveCommit = run.execution_commit || run.git_commit;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e=>e.stopPropagation()}>
        <div className="modal-head">
          <div style={{display:'flex',alignItems:'center',gap:'14px'}}>
            <StatusBadge success={run.success} />
            <div>
              <h2 style={{fontSize:'18px',fontWeight:700,color:'#fff'}}>Run #{run.id}</h2>
              <span style={{fontSize:'13px',color:'var(--nv-green)'}}>{run.dataset}</span>
            </div>
          </div>
          <div style={{display:'flex',alignItems:'center',gap:'6px'}}>
            <button className="btn btn-ghost btn-sm" onClick={copyRunLink} title="Copy link to this run" style={{fontSize:'12px',color:linkCopied?'var(--nv-green)':'var(--nv-text-muted)'}}>
              {linkCopied ? <><IconCheck /> Copied</> : <><IconLink /> Copy Link</>}
            </button>
            <button className="btn btn-ghost btn-icon" onClick={onClose} style={{borderRadius:'50%'}}><IconX /></button>
          </div>
        </div>
        <div className="modal-body">
          {/* Failure Banner */}
          {run.failure_reason && (
            <div style={{
              marginBottom:'24px',padding:'16px 20px',borderRadius:'10px',
              background:'rgba(255,60,60,0.08)',border:'1px solid rgba(255,60,60,0.25)',
            }}>
              <div style={{display:'flex',alignItems:'center',gap:'10px',marginBottom:'12px'}}>
                <span style={{fontSize:'20px'}}>&#x26A0;</span>
                <span style={{fontSize:'15px',fontWeight:700,color:'#ff5050'}}>Run Failed</span>
                {run.job_id && (
                  <button className="btn btn-sm" style={{fontSize:'11px',padding:'3px 10px',background:'rgba(255,80,80,0.15)',color:'#ff5050',border:'1px solid rgba(255,80,80,0.3)'}}
                    onClick={()=>setShowLogs(true)}>
                    <IconTerminal /> View Logs
                  </button>
                )}
              </div>
              <div style={{display:'grid',gridTemplateColumns:'140px 1fr',gap:'6px 12px',fontSize:'13px'}}>
                <span style={{color:'rgba(255,255,255,0.5)',fontWeight:500}}>Failure Reason</span>
                <span className="mono" style={{color:'#ff5050',fontWeight:600}}>{run.failure_reason}</span>
                <span style={{color:'rgba(255,255,255,0.5)',fontWeight:500}}>Return Code</span>
                <span className="mono" style={{color:'#fff'}}>{run.return_code ?? "\u2014"}</span>
                {(() => {
                  const rj = run.raw_json || {};
                  const tc = rj.test_config || {};
                  const rm = rj.run_metadata || {};
                  const arts = rj.artifacts || {};
                  return <>
                    <span style={{color:'rgba(255,255,255,0.5)',fontWeight:500}}>Dataset</span>
                    <span style={{color:'#fff'}}>{tc.dataset_label || run.dataset || "\u2014"} ({tc.input_type || "\u2014"})</span>
                    <span style={{color:'rgba(255,255,255,0.5)',fontWeight:500}}>Preset</span>
                    <span style={{color:'#fff'}}>{tc.preset || run.preset || "\u2014"}</span>
                    <span style={{color:'rgba(255,255,255,0.5)',fontWeight:500}}>Host</span>
                    <span style={{color:'#fff'}}>{rm.host || run.hostname || "\u2014"}</span>
                    <span style={{color:'rgba(255,255,255,0.5)',fontWeight:500}}>GPU</span>
                    <span style={{color:'#fff'}}>{rm.gpu_type ? `${rm.gpu_type} (x${rm.gpu_count || '?'})` : "\u2014"}</span>
                    <span style={{color:'rgba(255,255,255,0.5)',fontWeight:500}}>Recall Required</span>
                    <span style={{color:'#fff'}}>{tc.recall_required ? "Yes" : "No"}</span>
                    <span style={{color:'rgba(255,255,255,0.5)',fontWeight:500}}>Embed Model</span>
                    <span className="mono" style={{color:'var(--nv-text-muted)',fontSize:'11px',wordBreak:'break-all'}}>{tc.embed_model_name || "\u2014"}</span>
                    {arts.command_file && <>
                      <span style={{color:'rgba(255,255,255,0.5)',fontWeight:500}}>Command File</span>
                      <span className="mono" style={{color:'var(--nv-text-muted)',fontSize:'11px',wordBreak:'break-all'}}>{arts.command_file}</span>
                    </>}
                    {arts.runtime_metrics_dir && <>
                      <span style={{color:'rgba(255,255,255,0.5)',fontWeight:500}}>Metrics Dir</span>
                      <span className="mono" style={{color:'var(--nv-text-muted)',fontSize:'11px',wordBreak:'break-all'}}>{arts.runtime_metrics_dir}</span>
                    </>}
                  </>;
                })()}
              </div>
              {(() => {
                const detail = (run.raw_json || {}).error_detail;
                if (!detail) return null;
                return (
                  <pre className="mono" style={{
                    fontSize:'11px',color:'#ff8888',marginTop:'12px',
                    whiteSpace:'pre-wrap',wordBreak:'break-all',lineHeight:'1.5',
                    maxHeight:'200px',overflow:'auto',
                    background:'rgba(0,0,0,0.2)',padding:'8px',borderRadius:'4px',
                  }}>{detail}</pre>
                );
              })()}
            </div>
          )}

          {/* Overview */}
          <div style={{marginBottom:'28px'}}>
            <div className="section-title">Overview</div>
            <div className="detail-grid">
              <div className="detail-item">
                <div className="detail-label">Timestamp</div>
                <div className="detail-value">{fmtTs(run.timestamp)}</div>
              </div>
              <div className="detail-item">
                <div className="detail-label">Git Commit</div>
                <div className="detail-value"><CommitLink sha={effectiveCommit} repoUrl={githubRepoUrl} truncate={0} style={{fontSize:'13px'}} /></div>
              </div>
              <div className="detail-item">
                <div className="detail-label">Dataset</div>
                <div className="detail-value">{run.dataset}</div>
              </div>
              <div className="detail-item">
                <div className="detail-label">Preset</div>
                <div className="detail-value">
                  {run.preset ? (
                    <a href={`#presets/${encodeURIComponent(run.preset)}`}
                      onClick={e => e.stopPropagation()}
                      style={{color:'var(--nv-green)',textDecoration:'none',borderBottom:'1px dashed var(--nv-green)',cursor:'pointer',fontSize:'13px'}}>
                      {run.preset}
                    </a>
                  ) : "\u2014"}
                </div>
              </div>
              <div className="detail-item">
                <div className="detail-label">Hostname</div>
                <div className="detail-value">{run.hostname || "\u2014"}</div>
              </div>
              <div className="detail-item">
                <div className="detail-label">GPU Type</div>
                <div className="detail-value">{run.gpu_type || "\u2014"}</div>
              </div>
              <div className="detail-item">
                <div className="detail-label">GPUs</div>
                <div className="detail-value">
                  {run.gpus_used != null ? (
                    <>
                      <span style={{color:'#fff',fontWeight:600}}>{run.gpus_used}</span>
                      <span style={{color:'var(--nv-text-muted)'}}> used</span>
                      {run.num_gpus != null && (
                        <span style={{color:'var(--nv-text-dim)'}}> / {run.num_gpus} available</span>
                      )}
                    </>
                  ) : run.num_gpus != null ? (
                    <span>{run.num_gpus}</span>
                  ) : "\u2014"}
                </div>
              </div>
              <div className="detail-item">
                <div className="detail-label">Trigger Source</div>
                <div className="detail-value"><TriggerSourceBadge source={run.trigger_source} /></div>
              </div>
              {run.schedule_id && (
                <div className="detail-item">
                  <div className="detail-label">Schedule ID</div>
                  <div className="detail-value">#{run.schedule_id}</div>
                </div>
              )}
              <div className="detail-item">
                <div className="detail-label">Ray Cluster</div>
                <div className="detail-value" style={{display:'flex',alignItems:'center',gap:'8px'}}>
                  {run.ray_cluster_mode === "existing" ? (
                    <>
                      <span className="badge" style={{background:'rgba(100,180,255,0.12)',color:'#64b4ff',border:'1px solid rgba(100,180,255,0.25)'}}>Existing Cluster</span>
                      {run.ray_dashboard_url && (
                        <a href={run.ray_dashboard_url} target="_blank" rel="noopener noreferrer"
                          style={{fontSize:'12px',color:'var(--nv-green)',textDecoration:'underline',cursor:'pointer'}}>
                          Dashboard
                        </a>
                      )}
                    </>
                  ) : run.ray_cluster_mode === "local" ? (
                    <span className="badge badge-na">Local (started by run)</span>
                  ) : (
                    <span style={{color:'var(--nv-text-muted)'}}>{"\u2014"}</span>
                  )}
                </div>
              </div>
              {run.ray_dashboard_url && run.ray_cluster_mode === "existing" && (
                <div className="detail-item">
                  <div className="detail-label">Ray Dashboard</div>
                  <div className="detail-value">
                    <a href={run.ray_dashboard_url} target="_blank" rel="noopener noreferrer"
                      className="mono" style={{fontSize:'12px',color:'var(--nv-green)',textDecoration:'underline',wordBreak:'break-all'}}>
                      {run.ray_dashboard_url}
                    </a>
                  </div>
                </div>
              )}
              <div className="detail-item">
                <div className="detail-label">Artifact Directory</div>
                <div className="detail-value mono" style={{fontSize:'11px',wordBreak:'break-all',color:'var(--nv-text-muted)'}}>
                  {run.artifact_dir ? (
                    <>
                      <span style={{color:'var(--nv-green)',fontWeight:600}}>{run.hostname || 'unknown-host'}</span>
                      <span style={{color:'var(--nv-text-dim)'}}>:</span>
                      {run.artifact_dir}
                    </>
                  ) : "\u2014"}
                </div>
              </div>
            </div>
          </div>

          {/* Command */}
          <div style={{marginBottom:'28px'}}>
            <div className="section-title"
              style={{cursor:'pointer',userSelect:'none',display:'flex',alignItems:'center',gap:'8px'}}
              onClick={()=>setShowCommand(!showCommand)}
            >
              <span style={{transform:showCommand?'rotate(90deg)':'rotate(0deg)',transition:'transform 0.15s',display:'inline-block',fontSize:'10px'}}>{"\u25B6"}</span>
              Run Command
            </div>
            {showCommand && (
              commandLoading ? (
                <div style={{padding:'12px',color:'var(--nv-text-muted)',fontSize:'13px'}}><span className="spinner" style={{marginRight:'8px'}}></span>Loading…</div>
              ) : commandText ? (
                <div>
                  <div style={{display:'flex',justifyContent:'flex-end',marginBottom:'6px',gap:'6px'}}>
                    <button className="btn btn-ghost btn-sm"
                      style={{fontSize:'11px',color:'var(--nv-text-dim)'}}
                      onClick={() => { navigator.clipboard.writeText(commandText); }}
                      title="Copy raw command for re-execution">
                      Copy Raw Command
                    </button>
                  </div>
                  <FormattedCommand command={commandText} />
                </div>
              ) : (
                <div style={{padding:'12px',color:'var(--nv-text-dim)',fontSize:'13px',fontStyle:'italic'}}>Command not available (artifact may have been removed)</div>
              )
            )}
          </div>

          {/* Job Logs */}
          {run.job_id && (
            <div style={{marginBottom:'28px'}}>
              <div className="section-title"
                style={{cursor:'pointer',userSelect:'none',display:'flex',alignItems:'center',gap:'8px'}}
                onClick={()=>setShowLogs(!showLogs)}
              >
                <span style={{transform:showLogs?'rotate(90deg)':'rotate(0deg)',transition:'transform 0.15s',display:'inline-block',fontSize:'10px'}}>{"\u25B6"}</span>
                Job Output (last 500 lines)
              </div>
              {showLogs && (
                logLoading ? (
                  <div style={{padding:'12px',color:'var(--nv-text-muted)',fontSize:'13px'}}><span className="spinner" style={{marginRight:'8px'}}></span>Loading…</div>
                ) : logLines && logLines.length > 0 ? (
                  <div style={{position:'relative'}}>
                    <div className="log-viewer" ref={logRef} style={{maxHeight:'400px'}}>
                      {logLines.map((line, i) => <div key={i} className="log-line">{line}</div>)}
                    </div>
                    <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginTop:'6px'}}>
                      <span style={{fontSize:'11px',color:'var(--nv-text-dim)'}}>{logLines.length} line{logLines.length!==1?'s':''}</span>
                      <button className="btn btn-ghost btn-sm"
                        style={{fontSize:'11px',color:'var(--nv-text-dim)'}}
                        onClick={() => { navigator.clipboard.writeText(logLines.join('\n')); }}
                        title="Copy logs">
                        Copy All
                      </button>
                    </div>
                  </div>
                ) : (
                  <div style={{padding:'12px',color:'var(--nv-text-dim)',fontSize:'13px',fontStyle:'italic'}}>No log output available for this run.</div>
                )
              )}
            </div>
          )}

          {/* Installed Packages */}
          {run.job_id && (
            <div style={{marginBottom:'28px'}}>
              <div className="section-title"
                style={{cursor:'pointer',userSelect:'none',display:'flex',alignItems:'center',gap:'8px'}}
                onClick={()=>setShowPipList(!showPipList)}
              >
                <span style={{transform:showPipList?'rotate(90deg)':'rotate(0deg)',transition:'transform 0.15s',display:'inline-block',fontSize:'10px'}}>{"\u25B6"}</span>
                Installed Packages
              </div>
              {showPipList && (
                pipListLoading ? (
                  <div style={{padding:'12px',color:'var(--nv-text-muted)',fontSize:'13px'}}><span className="spinner" style={{marginRight:'8px'}}></span>Loading…</div>
                ) : pipListText ? (
                  <div style={{position:'relative'}}>
                    <button className="btn btn-ghost btn-sm"
                      style={{position:'absolute',top:'6px',right:'6px',fontSize:'10px',zIndex:1,color:'var(--nv-text-dim)'}}
                      onClick={() => navigator.clipboard.writeText(pipListText)}>Copy</button>
                    <pre className="mono" style={{
                      fontSize:'11px',color:'var(--nv-text-muted)',margin:0,
                      whiteSpace:'pre',lineHeight:'1.4',
                      maxHeight:'250px',overflow:'auto',
                      background:'rgba(0,0,0,0.25)',padding:'10px',borderRadius:'6px',
                      border:'1px solid rgba(255,255,255,0.06)',
                    }}>{pipListText}</pre>
                    <span style={{fontSize:'11px',color:'var(--nv-text-dim)',marginTop:'4px',display:'inline-block'}}>{pipListText.split('\n').length} lines</span>
                  </div>
                ) : (
                  <div style={{padding:'12px',color:'var(--nv-text-dim)',fontSize:'13px',fontStyle:'italic'}}>No package list available for this run.</div>
                )
              )}
            </div>
          )}

          {/* Requested Plan */}
          {raw.requested_plan && Array.isArray(raw.requested_plan) && raw.requested_plan.length > 0 && (
            <div style={{marginBottom:'28px'}}>
              <button className="btn btn-secondary" style={{fontSize:'12px',padding:'5px 12px',display:'flex',alignItems:'center',gap:'6px',marginBottom:'10px'}}
                onClick={() => setShowPlan(v => !v)}>
                <span style={{transform: showPlan ? 'rotate(90deg)' : 'rotate(0deg)', transition:'transform 0.15s', display:'inline-block'}}>&#9654;</span>
                Requested Plan ({raw.requested_plan.length} stage{raw.requested_plan.length!==1?'s':''})
              </button>
              {showPlan && (
                <div style={{position:'relative'}}>
                  <button className="btn btn-ghost btn-sm"
                    style={{position:'absolute',top:'6px',right:'6px',fontSize:'10px',zIndex:1,color:'var(--nv-text-dim)'}}
                    onClick={() => navigator.clipboard.writeText(JSON.stringify(raw.requested_plan, null, 2))}>Copy JSON</button>
                  <div style={{
                    background:'rgba(0,0,0,0.25)',padding:'10px',borderRadius:'6px',
                    border:'1px solid rgba(255,255,255,0.06)',maxHeight:'300px',overflow:'auto',
                  }}>
                    <table style={{width:'100%',borderCollapse:'collapse',fontSize:'12px'}}>
                      <thead>
                        <tr style={{borderBottom:'1px solid rgba(255,255,255,0.1)'}}>
                          <th style={{textAlign:'left',padding:'6px 8px',color:'var(--nv-text-muted)',fontWeight:600}}>Stage</th>
                          <th style={{textAlign:'left',padding:'6px 8px',color:'var(--nv-text-muted)',fontWeight:600}}>Type</th>
                          <th style={{textAlign:'center',padding:'6px 8px',color:'var(--nv-text-muted)',fontWeight:600}}>GPUs</th>
                          <th style={{textAlign:'center',padding:'6px 8px',color:'var(--nv-text-muted)',fontWeight:600}}>CPUs</th>
                          <th style={{textAlign:'center',padding:'6px 8px',color:'var(--nv-text-muted)',fontWeight:600}}>Batch</th>
                          <th style={{textAlign:'center',padding:'6px 8px',color:'var(--nv-text-muted)',fontWeight:600}}>Concurrency</th>
                        </tr>
                      </thead>
                      <tbody>
                        {raw.requested_plan.map((s, i) => (
                          <tr key={i} style={{borderBottom:'1px solid rgba(255,255,255,0.04)'}}>
                            <td style={{padding:'6px 8px',color:'#fff'}}>{s.display_name || s.stage}</td>
                            <td style={{padding:'6px 8px'}}>
                              <span style={{
                                fontSize:'10px',padding:'2px 6px',borderRadius:'4px',
                                background: s.type==='gpu' ? 'rgba(118,185,0,0.15)' : s.type==='source' ? 'rgba(0,150,255,0.15)' : s.type==='sink' ? 'rgba(255,165,0,0.15)' : s.type==='evaluator' ? 'rgba(180,100,255,0.15)' : 'rgba(150,150,150,0.15)',
                                color: s.type==='gpu' ? '#76b900' : s.type==='source' ? '#4da6ff' : s.type==='sink' ? '#ffa500' : s.type==='evaluator' ? '#b464ff' : '#aaa',
                              }}>{s.type}</span>
                            </td>
                            <td style={{padding:'6px 8px',textAlign:'center',color:'var(--nv-text-muted)'}}>{s.num_gpus != null ? s.num_gpus : '-'}</td>
                            <td style={{padding:'6px 8px',textAlign:'center',color:'var(--nv-text-muted)'}}>{s.num_cpus != null ? s.num_cpus : '-'}</td>
                            <td style={{padding:'6px 8px',textAlign:'center',color:'var(--nv-text-muted)'}}>{s.batch_size != null ? s.batch_size : '-'}</td>
                            <td style={{padding:'6px 8px',textAlign:'center',color:'var(--nv-text-muted)'}}>{s.concurrency != null ? s.concurrency : '-'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Ray Execution Stats */}
          {raw.ray_stats && (
            <div style={{marginBottom:'28px'}}>
              <button className="btn btn-secondary" style={{fontSize:'12px',padding:'5px 12px',display:'flex',alignItems:'center',gap:'6px',marginBottom:'10px'}}
                onClick={() => setShowRayStats(v => !v)}>
                <span style={{transform: showRayStats ? 'rotate(90deg)' : 'rotate(0deg)', transition:'transform 0.15s', display:'inline-block'}}>&#9654;</span>
                Ray Execution Stats
              </button>
              {showRayStats && (
                <div style={{position:'relative'}}>
                  <button className="btn btn-ghost btn-sm"
                    style={{position:'absolute',top:'6px',right:'6px',fontSize:'10px',zIndex:1,color:'var(--nv-text-dim)'}}
                    onClick={() => navigator.clipboard.writeText(raw.ray_stats)}>Copy</button>
                  <pre className="mono" style={{
                    fontSize:'11px',color:'var(--nv-text-muted)',margin:0,
                    whiteSpace:'pre',lineHeight:'1.4',
                    maxHeight:'300px',overflow:'auto',
                    background:'rgba(0,0,0,0.25)',padding:'10px',borderRadius:'6px',
                    border:'1px solid rgba(255,255,255,0.06)',
                  }}>{raw.ray_stats}</pre>
                </div>
              )}
            </div>
          )}

          {/* Metrics */}
          <div style={{marginBottom:'28px'}}>
            <div className="section-title">Performance Metrics</div>
            <div className="metric-grid">
              <div className="metric-card">
                <div className="metric-value">{fmt(run.pages_per_sec)}</div>
                <div className="metric-label">Pages / sec</div>
              </div>
              <div className="metric-card">
                <div className="metric-value">{fmt(run.recall_1,3)}</div>
                <div className="metric-label">Recall@1</div>
              </div>
              <div className="metric-card">
                <div className="metric-value">{fmt(run.recall_5,3)}</div>
                <div className="metric-label">Recall@5</div>
              </div>
              <div className="metric-card">
                <div className="metric-value">{fmt(run.recall_10,3)}</div>
                <div className="metric-label">Recall@10</div>
              </div>
              <div className="metric-card">
                <div className="metric-value" style={{color:'#fff'}}>{fmt(run.ingest_secs)}s</div>
                <div className="metric-label">Ingest Time</div>
              </div>
              <div className="metric-card">
                <div className="metric-value" style={{color:'#fff'}}>{fmt(run.pages,0)} / {fmt(run.files,0)}</div>
                <div className="metric-label">Pages / Files</div>
              </div>
            </div>
          </div>

          {/* Detection Summary */}
          {(() => {
            const ds = (raw.detection_summary && typeof raw.detection_summary === 'object') ? raw.detection_summary : null;
            if (!ds) return null;
            const byLabel = ds.page_elements_v3_counts_by_label || {};
            const labelEntries = Object.entries(byLabel).sort((a,b) => b[1] - a[1]);
            const totalDetections = (ds.page_elements_v3_total_detections || 0)
              + (ds.ocr_table_total_detections || 0)
              + (ds.ocr_chart_total_detections || 0)
              + (ds.ocr_infographic_total_detections || 0);
            return (
              <div style={{marginBottom:'28px'}}>
                <div className="section-title">Detection Summary</div>
                <div className="metric-grid">
                  <div className="metric-card">
                    <div className="metric-value" style={{color:'#fff'}}>{fmt(ds.pages_seen,0)}</div>
                    <div className="metric-label">Pages Seen</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-value">{fmt(totalDetections,0)}</div>
                    <div className="metric-label">Total Detections</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-value" style={{color:'#fff'}}>{fmt(ds.page_elements_v3_total_detections,0)}</div>
                    <div className="metric-label">Page Elements v3</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-value" style={{color:'#fff'}}>{fmt(ds.ocr_table_total_detections,0)}</div>
                    <div className="metric-label">OCR Tables</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-value" style={{color:'#fff'}}>{fmt(ds.ocr_chart_total_detections,0)}</div>
                    <div className="metric-label">OCR Charts</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-value" style={{color:'#fff'}}>{fmt(ds.ocr_infographic_total_detections,0)}</div>
                    <div className="metric-label">OCR Infographics</div>
                  </div>
                </div>
                {labelEntries.length > 0 && (
                  <div style={{marginTop:'16px'}}>
                    <div style={{fontSize:'13px',fontWeight:600,color:'rgba(255,255,255,0.7)',marginBottom:'10px'}}>Page Elements by Label</div>
                    <div style={{
                      display:'grid',gridTemplateColumns:'1fr auto',gap:'4px 16px',
                      padding:'14px 18px',borderRadius:'10px',
                      background:'rgba(255,255,255,0.02)',border:'1px solid var(--nv-border)',
                    }}>
                      {labelEntries.map(([label, count]) => (
                        <React.Fragment key={label}>
                          <span style={{fontSize:'13px',color:'rgba(255,255,255,0.8)'}}>{label}</span>
                          <span className="mono" style={{fontSize:'13px',color:'var(--nv-green)',fontWeight:600,textAlign:'right'}}>{count.toLocaleString()}</span>
                        </React.Fragment>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })()}

          {/* Tags */}
          {run.tags && run.tags.length > 0 && (
            <div style={{marginBottom:'28px'}}>
              <div className="section-title">Tags</div>
              <div style={{display:'flex',gap:'6px',flexWrap:'wrap'}}>
                {run.tags.map((t,i) => <span key={i} className="tag">{t}</span>)}
              </div>
            </div>
          )}

          {/* Raw JSON */}
          <div style={{marginBottom:'28px'}}>
            <div className="section-title"
              style={{cursor:'pointer',userSelect:'none',display:'flex',alignItems:'center',gap:'8px'}}
              onClick={()=>setShowRaw(!showRaw)}
            >
              <span style={{transform:showRaw?'rotate(90deg)':'rotate(0deg)',transition:'transform 0.15s',display:'inline-block',fontSize:'10px'}}>{"\u25B6"}</span>
              Raw JSON
            </div>
            {showRaw && (
              <pre className="mono" style={{
                fontSize:'12px',padding:'16px',borderRadius:'8px',
                background:'var(--nv-bg)',border:'1px solid var(--nv-border)',
                color:'var(--nv-text-muted)',maxHeight:'300px',overflow:'auto',
                whiteSpace:'pre-wrap',wordBreak:'break-all',
              }}>
                {JSON.stringify(raw,null,2)}
              </pre>
            )}
          </div>

          {/* Pipeline Graph */}
          {run.job_id && <GraphPreview runId={run.id} />}

          {/* Nsys Profiling Status */}
          {raw.nsys_status && raw.nsys_status.requested && (
            <div style={{
              marginTop:'20px',padding:'12px 16px',borderRadius:'8px',
              background: raw.nsys_status.found ? 'rgba(118,185,0,0.08)' : raw.nsys_status.enabled ? 'rgba(255,165,0,0.08)' : 'rgba(255,60,60,0.08)',
              border: `1px solid ${raw.nsys_status.found ? 'rgba(118,185,0,0.2)' : raw.nsys_status.enabled ? 'rgba(255,165,0,0.2)' : 'rgba(255,60,60,0.2)'}`,
            }}>
              <div style={{display:'flex',alignItems:'center',gap:'8px',flexWrap:'wrap'}}>
                <span style={{fontSize:'14px'}}>{raw.nsys_status.found ? '\u2705' : raw.nsys_status.enabled ? '\u26A0\uFE0F' : '\u274C'}</span>
                <span style={{fontSize:'13px',fontWeight:600,color: raw.nsys_status.found ? '#76b900' : raw.nsys_status.enabled ? '#ffa500' : '#ff5050'}}>
                  Nsight Systems Profile: {raw.nsys_status.found ? 'Captured' : raw.nsys_status.enabled ? 'No Report Generated' : 'Not Available'}
                </span>
                {raw.nsys_status.files && raw.nsys_status.files.length > 0 && (
                  <span style={{fontSize:'12px',color:'var(--nv-text-muted)'}}>
                    ({raw.nsys_status.files.map(f => `${f.name}: ${f.size_mb} MB`).join(', ')})
                  </span>
                )}
              </div>
              {raw.nsys_status.error && (
                <div style={{fontSize:'12px',color:'var(--nv-text-muted)',marginTop:'6px'}}>{raw.nsys_status.error}</div>
              )}
            </div>
          )}
        </div>

        <div className="modal-foot">
          <a href={`/api/runs/${run.id}/download/json`} download className="btn btn-secondary">
            <IconFile /> Download JSON
          </a>
          <button className="btn btn-primary" onClick={handleZipDownload} disabled={zipDownloading}>
            {zipDownloading ? <><span className="spinner" style={{marginRight:'6px'}}></span>Downloading…</> : <><IconPackage /> Download Artifacts ZIP</>}
          </button>
          {!!run.nsys_profile && (
            <button className="btn btn-secondary" onClick={() => window.open(`/api/runs/${run.id}/download/nsys-profile`, '_blank')}
              title="Download the Nsight Systems profile captured during this run">
              <IconActivity /> Nsys Profile
            </button>
          )}
          <button className="btn btn-secondary" onClick={handleRerunClick} title="Re-run with the exact same configuration and git commit">
            <IconRefresh /> Re-Run
          </button>
          <div style={{flex:1}}></div>
          {onDelete && (
            <button className="btn btn-sm" onClick={() => {
              if (confirm(`Delete run #${run.id}? This cannot be undone.`)) { onDelete(run.id); }
            }} title="Delete this run"
              style={{background:'rgba(255,80,80,0.1)',color:'#ff5050',border:'1px solid rgba(255,80,80,0.2)'}}>
              <IconTrash /> Delete Run
            </button>
          )}
        </div>

        {rerunState && (
          <div style={{position:'absolute',inset:0,background:'rgba(0,0,0,0.7)',display:'flex',alignItems:'center',justifyContent:'center',zIndex:100,borderRadius:'16px'}} onClick={()=>setRerunState(null)}>
            <div style={{background:'var(--nv-surface)',border:'1px solid var(--nv-border)',borderRadius:'12px',padding:'24px',maxWidth:'500px',width:'90%'}} onClick={e=>e.stopPropagation()}>
              <h3 style={{fontSize:'16px',fontWeight:700,color:'#fff',marginBottom:'16px'}}>Re-Run #{run.id}</h3>

              <div style={{display:'grid',gridTemplateColumns:'140px 1fr',gap:'8px 12px',fontSize:'13px',marginBottom:'16px'}}>
                <span style={{color:'var(--nv-text-muted)',fontWeight:500}}>Dataset</span>
                <span style={{color:'#fff'}}>{run.dataset}</span>
                <span style={{color:'var(--nv-text-muted)',fontWeight:500}}>Git Commit</span>
                <span className="mono" style={{color:'var(--nv-green)',fontSize:'12px'}}>{rerunState.original_commit ? rerunState.original_commit.slice(0,12) : "none (latest)"}</span>
                <span style={{color:'var(--nv-text-muted)',fontWeight:500}}>Original Host</span>
                <span style={{color:'#fff'}}>{rerunState.original_hostname || "unknown"}</span>
              </div>

              {rerunState.original_runner ? (
                <div style={{padding:'12px 16px',borderRadius:'8px',background:'rgba(118,185,0,0.06)',border:'1px solid rgba(118,185,0,0.2)',fontSize:'13px',marginBottom:'16px'}}>
                  <span style={{color:'var(--nv-green)',fontWeight:600}}>Original runner is online:</span>{' '}
                  <span style={{color:'#fff'}}>#{rerunState.original_runner.id} — {rerunState.original_runner.name}</span>
                </div>
              ) : (
                <div style={{padding:'12px 16px',borderRadius:'8px',background:'rgba(255,180,50,0.08)',border:'1px solid rgba(255,180,50,0.25)',fontSize:'13px',marginBottom:'16px'}}>
                  <div style={{color:'#ffb432',fontWeight:600,marginBottom:'6px'}}>Original runner is not available</div>
                  <div style={{color:'var(--nv-text-muted)'}}>The runner on <strong style={{color:'#fff'}}>{rerunState.original_hostname || "unknown"}</strong> is offline. Select another runner below.</div>
                </div>
              )}

              <div style={{marginBottom:'16px'}}>
                <label style={{display:'block',fontSize:'11px',fontWeight:500,color:'var(--nv-text-muted)',marginBottom:'4px',textTransform:'uppercase',letterSpacing:'0.04em'}}>Target Runner</label>
                <select className="input" style={{width:'100%'}} value={rerunSelectedRunner} onChange={e=>setRerunSelectedRunner(e.target.value)}>
                  <option value="">Any available runner</option>
                  {(rerunState.online_runners || []).map(r => (
                    <option key={r.id} value={r.id}>
                      #{r.id} — {r.name}{r.hostname ? ` (${r.hostname})` : ''}{r.gpu_type ? ` [${r.gpu_type}]` : ''}
                      {rerunState.original_runner && r.id === rerunState.original_runner.id ? ' \u2605 original' : ''}
                    </option>
                  ))}
                </select>
              </div>

              <div style={{display:'flex',gap:'8px',justifyContent:'flex-end'}}>
                <button className="btn btn-secondary" onClick={()=>setRerunState(null)}>Cancel</button>
                <button className="btn btn-primary" onClick={submitRerun} disabled={rerunSubmitting}>
                  {rerunSubmitting ? <><span className="spinner" style={{marginRight:'8px'}}></span>Submitting…</> : <><IconPlay /> Queue Re-Run</>}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ===== Runner Status Dot ===== */
function RunnerStatusDot({ status }) {
  const colors = { online:'#76b900', offline:'#ff5050', busy:'#fcd34d', paused:'#ff8c00' };
  const color = colors[status] || colors.offline;
  const isOffline = status === 'offline';
  const isPaused = status === 'paused';
  return (
    <span style={{display:'inline-flex',alignItems:'center',gap:'6px'}}>
      <span style={{width:'8px',height:'8px',borderRadius:'50%',background:color,display:'inline-block',
        boxShadow:status==='online'?'0 0 6px rgba(118,185,0,0.5)':isOffline?'0 0 6px rgba(255,80,80,0.4)':isPaused?'0 0 6px rgba(255,140,0,0.4)':'none',
        animation:isOffline?'pulse-offline 2s ease-in-out infinite':'none'}}></span>
      <span style={{fontSize:'12px',color,textTransform:'capitalize',fontWeight:(isOffline||isPaused)?600:500}}>{status||'unknown'}</span>
    </span>
  );
}

function timeSince(dateStr) {
  if (!dateStr) return null;
  try {
    const d = new Date(dateStr);
    const secs = Math.floor((Date.now() - d.getTime()) / 1000);
    if (secs < 60) return `${secs}s ago`;
    if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
    if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
    return `${Math.floor(secs / 86400)}d ago`;
  } catch { return null; }
}
