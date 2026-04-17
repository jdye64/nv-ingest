/* ===== Settings View ===== */
function SettingsView() {
  const [gitInfo, setGitInfo] = useState(null);
  const [gitLoading, setGitLoading] = useState(true);
  const [deployBranch, setDeployBranch] = useState("main");
  const [deployRemote, setDeployRemote] = useState("origin");
  const [activeAction, setActiveAction] = useState(null);
  const [deployResult, setDeployResult] = useState(null);
  const [deployError, setDeployError] = useState("");
  const [confirmAction, setConfirmAction] = useState(null);
  const [reconnecting, setReconnecting] = useState(false);
  const [portalSettings, setPortalSettings] = useState({});
  const [runCodeRefInput, setRunCodeRefInput] = useState("");
  const [savingRunCodeRef, setSavingRunCodeRef] = useState(false);
  const [runCodeRefSaved, setRunCodeRefSaved] = useState(false);
  const [mcpTools, setMcpTools] = useState([]);
  const [mcpToolsLoading, setMcpToolsLoading] = useState(true);
  const [mcpAuditLog, setMcpAuditLog] = useState([]);
  const [mcpAuditStats, setMcpAuditStats] = useState(null);
  const [mcpAuditLoading, setMcpAuditLoading] = useState(true);
  const [mcpAuditFilter, setMcpAuditFilter] = useState({ tool_name: '', agent_name: '' });
  const [mcpCursorConfig, setMcpCursorConfig] = useState(null);
  const [mcpToolFilter, setMcpToolFilter] = useState('');

  async function fetchPortalSettings() {
    try {
      const res = await fetch("/api/portal-settings");
      const data = await res.json();
      setPortalSettings(data);
      setRunCodeRefInput(data.run_code_ref || "upstream/main");
    } catch (err) {
      console.error("Failed to fetch portal settings:", err);
    }
  }

  async function saveRunCodeRef() {
    setSavingRunCodeRef(true);
    setRunCodeRefSaved(false);
    try {
      const res = await fetch("/api/portal-settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_code_ref: runCodeRefInput.trim() }),
      });
      if (!res.ok) throw new Error("Failed to save");
      const data = await res.json();
      setPortalSettings(data);
      setRunCodeRefSaved(true);
      setTimeout(() => setRunCodeRefSaved(false), 3000);
    } catch (err) {
      console.error("Failed to save run code ref:", err);
    } finally {
      setSavingRunCodeRef(false);
    }
  }

  async function fetchMcpTools() {
    setMcpToolsLoading(true);
    try {
      const res = await fetch("/api/mcp/tools");
      setMcpTools(await res.json());
    } catch (err) { console.error("Failed to fetch MCP tools:", err); }
    finally { setMcpToolsLoading(false); }
  }

  async function fetchMcpAuditLog() {
    setMcpAuditLoading(true);
    try {
      const params = new URLSearchParams({ limit: '100' });
      if (mcpAuditFilter.tool_name) params.set('tool_name', mcpAuditFilter.tool_name);
      if (mcpAuditFilter.agent_name) params.set('agent_name', mcpAuditFilter.agent_name);
      const [logRes, statsRes] = await Promise.all([
        fetch(`/api/mcp/audit-log?${params}`),
        fetch("/api/mcp/audit-log/stats"),
      ]);
      setMcpAuditLog(await logRes.json());
      setMcpAuditStats(await statsRes.json());
    } catch (err) { console.error("Failed to fetch MCP audit log:", err); }
    finally { setMcpAuditLoading(false); }
  }

  async function fetchMcpCursorConfig() {
    try {
      const res = await fetch("/api/mcp/cursor-config");
      setMcpCursorConfig(await res.json());
    } catch (err) { console.error("Failed to fetch Cursor config:", err); }
  }

  async function toggleMcpTool(toolKey, enabled) {
    try {
      await fetch(`/api/mcp/tools/${encodeURIComponent(toolKey)}/toggle`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      });
      fetchMcpTools();
    } catch (err) { console.error("Failed to toggle tool:", err); }
  }

  async function saveMcpSetting(key, value) {
    try {
      await fetch("/api/portal-settings", {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ [key]: value }),
      });
      fetchPortalSettings();
    } catch (err) { console.error("Failed to save MCP setting:", err); }
  }

  async function fetchGitInfo() {
    setGitLoading(true);
    try {
      const res = await fetch("/api/settings/git-info");
      const data = await res.json();
      setGitInfo(data);
      if (data.available && data.remotes?.length) {
        const nvidiaR = data.remotes.find(r => r.url && (r.url.includes("NVIDIA/") || r.url.includes("nvidia/")));
        setDeployRemote(nvidiaR ? nvidiaR.name : data.remotes[0].name);
      }
      if (data.available && data.current_branch) {
        setDeployBranch(data.current_branch);
      }
    } catch (err) {
      setGitInfo({ available: false, error: err.message });
    } finally {
      setGitLoading(false);
    }
  }

  useEffect(() => { fetchGitInfo(); fetchPortalSettings(); fetchMcpTools(); fetchMcpAuditLog(); fetchMcpCursorConfig(); }, []);

  async function handleDeploy(updateRunners) {
    const action = updateRunners ? "deploy-all" : "deploy-portal";
    setConfirmAction(null);
    setActiveAction(action);
    setDeployResult(null);
    setDeployError("");
    try {
      const res = await fetch("/api/settings/deploy", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ branch: deployBranch, remote: deployRemote, update_runners: updateRunners }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setDeployResult(data);
      setReconnecting(true);
      pollForRestart();
    } catch (err) {
      setDeployError(err.message);
    } finally {
      setActiveAction(null);
    }
  }

  async function handleUpdateRunnersOnly() {
    setConfirmAction(null);
    setActiveAction("update-runners");
    setDeployResult(null);
    setDeployError("");
    try {
      const res = await fetch("/api/settings/update-runners", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ branch: deployBranch, remote: deployRemote }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setDeployResult(data);
    } catch (err) {
      setDeployError(err.message);
    } finally {
      setActiveAction(null);
    }
  }

  function pollForRestart() {
    let attempts = 0;
    const maxAttempts = 30;
    const interval = setInterval(async () => {
      attempts++;
      try {
        const res = await fetch("/api/version", { signal: AbortSignal.timeout(3000) });
        if (res.ok) {
          clearInterval(interval);
          setReconnecting(false);
          fetchGitInfo();
        }
      } catch {
        if (attempts >= maxAttempts) {
          clearInterval(interval);
          setReconnecting(false);
          setDeployError("Portal did not come back online within 60 seconds. Check the server manually.");
        }
      }
    }, 2000);
  }

  const labelStyle = {display:'block',fontSize:'11px',fontWeight:600,color:'var(--nv-text-dim)',textTransform:'uppercase',letterSpacing:'0.04em',marginBottom:'6px'};
  const isBusy = !!activeAction;
  const refLabel = `${deployRemote}/${deployBranch}`;

  if (gitLoading) {
    return (
      <div className="card" style={{padding:'60px',textAlign:'center'}}>
        <span className="spinner spinner-lg" style={{display:'block',margin:'0 auto 16px'}}></span>
        <div style={{color:'var(--nv-text-muted)',fontSize:'14px'}}>Loading git information…</div>
      </div>
    );
  }

  if (!gitInfo || !gitInfo.available) {
    return (
      <div className="card" style={{padding:'40px',textAlign:'center'}}>
        <div style={{fontSize:'32px',marginBottom:'12px',opacity:0.3}}>&#x26A0;</div>
        <div style={{fontSize:'15px',fontWeight:600,color:'#ff5050',marginBottom:'8px'}}>Git Not Available</div>
        <div style={{fontSize:'13px',color:'var(--nv-text-muted)'}}>{gitInfo?.error || "Unable to detect a git repository."}</div>
      </div>
    );
  }

  const confirmConfig = {
    "deploy-portal": {
      title: "Deploy Portal Only",
      color: "var(--nv-green)",
      steps: [
        "Stash any uncommitted local changes",
        <>Fetch latest code from <span className="mono" style={{color:'var(--nv-green)'}}>{deployRemote}</span></>,
        <>Checkout branch <span className="mono" style={{color:'var(--nv-green)'}}>{deployBranch}</span></>,
        "Restart the portal web server",
      ],
      note: "Runners will NOT be updated. They will continue running their current code.",
      onConfirm: () => handleDeploy(false),
      btnLabel: <>Deploy Portal Only</>,
    },
    "deploy-all": {
      title: "Deploy Portal + Update Runners",
      color: "var(--nv-green)",
      steps: [
        "Stash any uncommitted local changes",
        <>Fetch latest code from <span className="mono" style={{color:'var(--nv-green)'}}>{deployRemote}</span></>,
        <>Checkout branch <span className="mono" style={{color:'var(--nv-green)'}}>{deployBranch}</span></>,
        "Restart the portal web server",
        "Signal all online runners to pull and restart with the new commit",
      ],
      note: null,
      onConfirm: () => handleDeploy(true),
      btnLabel: <>Deploy All</>,
    },
    "update-runners": {
      title: "Update Runners Only",
      color: "#64b4ff",
      steps: [
        <>Resolve <span className="mono" style={{color:'#64b4ff'}}>{refLabel}</span> to its latest commit</>,
        "Signal all online/paused runners to pull that commit and restart",
      ],
      note: "The portal will NOT restart. Only runners will be updated.",
      onConfirm: handleUpdateRunnersOnly,
      btnLabel: <>Update Runners</>,
    },
  };
  const confirm = confirmAction ? confirmConfig[confirmAction] : null;

  return (
    <>
      {/* Reconnecting overlay */}
      {reconnecting && (
        <div style={{
          position:'fixed',top:0,left:0,right:0,bottom:0,zIndex:9999,
          background:'rgba(0,0,0,0.85)',display:'flex',alignItems:'center',justifyContent:'center',flexDirection:'column',gap:'20px',
        }}>
          <span className="spinner spinner-lg"></span>
          <div style={{fontSize:'18px',fontWeight:600,color:'#fff'}}>Portal is restarting…</div>
          <div style={{fontSize:'13px',color:'var(--nv-text-muted)',maxWidth:'400px',textAlign:'center',lineHeight:'1.6'}}>
            The portal is pulling the latest code and restarting. This page will automatically reconnect when the server is back online.
          </div>
        </div>
      )}

      {/* Current State */}
      <div className="card" style={{padding:'24px',marginBottom:'20px'}}>
        <div className="section-title" style={{marginBottom:'16px'}}>Current Portal State</div>
        <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill, minmax(200px, 1fr))',gap:'16px'}}>
          <div>
            <div style={labelStyle}>Branch</div>
            <div style={{fontSize:'14px',color:'#fff',fontWeight:600}}>{gitInfo.current_branch}</div>
          </div>
          <div>
            <div style={labelStyle}>Commit</div>
            <div className="mono" style={{fontSize:'13px',color:'var(--nv-green)'}}>{gitInfo.current_short_sha}</div>
            <div className="mono" style={{fontSize:'11px',color:'var(--nv-text-dim)',marginTop:'2px',wordBreak:'break-all'}}>{gitInfo.current_sha}</div>
          </div>
          <div>
            <div style={labelStyle}>Working Directory</div>
            <div style={{fontSize:'13px',color:gitInfo.is_dirty ? '#ffb84d' : 'var(--nv-green)',fontWeight:500}}>
              {gitInfo.is_dirty ? "Uncommitted changes" : "Clean"}
            </div>
          </div>
          <div>
            <div style={labelStyle}>Repository</div>
            <div className="mono" style={{fontSize:'11px',color:'var(--nv-text-muted)',wordBreak:'break-all'}}>{gitInfo.repo_root}</div>
          </div>
        </div>
      </div>

      {/* Runner Execution Branch */}
      <div className="card" style={{padding:'24px',marginBottom:'20px'}}>
        <div className="section-title" style={{marginBottom:'6px'}}>Runner Execution Branch</div>
        <div style={{fontSize:'12px',color:'var(--nv-text-dim)',lineHeight:'1.6',marginBottom:'16px'}}>
          The git ref that runners will <code className="mono" style={{background:'rgba(255,255,255,0.06)',padding:'1px 5px',borderRadius:'4px'}}>git checkout</code> before executing each job.
          This separates the runner infrastructure code (synced via Deploy) from the harness pipeline code used for actual runs.
          Runners receive this value on every heartbeat.
        </div>
        {portalSettings._nvidia_remote_name && (
          <div style={{marginBottom:'14px',padding:'10px 14px',borderRadius:'8px',background:'rgba(118,185,0,0.06)',border:'1px solid rgba(118,185,0,0.15)'}}>
            <div style={{fontSize:'12px',color:'var(--nv-green)',fontWeight:600,marginBottom:'4px'}}>Detected NVIDIA Remote</div>
            <div style={{fontSize:'12px',color:'var(--nv-text-muted)'}}>
              <span className="mono" style={{color:'#fff',fontWeight:600}}>{portalSettings._nvidia_remote_name}</span>
              {portalSettings._nvidia_remote_url && (
                <span style={{marginLeft:'8px',color:'var(--nv-text-dim)'}}>→ {portalSettings._nvidia_remote_url}</span>
              )}
            </div>
          </div>
        )}
        <div style={{display:'flex',gap:'10px',alignItems:'flex-end'}}>
          <div style={{flex:1}}>
            <label style={labelStyle}>Git Ref (remote/branch)</label>
            <input className="input" style={{width:'100%'}} value={runCodeRefInput}
              onChange={e => setRunCodeRefInput(e.target.value)}
              placeholder={portalSettings._nvidia_remote_name ? `e.g. ${portalSettings._nvidia_remote_name}/main` : "e.g. nvidia/main"} />
          </div>
          <button className="btn btn-primary" onClick={saveRunCodeRef}
            disabled={savingRunCodeRef || !runCodeRefInput.trim() || runCodeRefInput.trim() === portalSettings.run_code_ref}
            style={{whiteSpace:'nowrap'}}>
            {savingRunCodeRef ? <><span className="spinner" style={{marginRight:'6px'}}></span>Saving…</> : "Save"}
          </button>
        </div>
        {runCodeRefSaved && (
          <div style={{marginTop:'10px',fontSize:'12px',color:'var(--nv-green)',display:'flex',alignItems:'center',gap:'6px'}}>
            <IconCheck /> Saved. Runners will pick up the new ref on their next heartbeat.
          </div>
        )}
        {(() => {
          const nvidiaRemote = portalSettings._nvidia_remote_name;
          const nvBranches = (gitInfo?.remote_branches || []).filter(b => nvidiaRemote ? b.startsWith(nvidiaRemote + "/") : true);
          return nvBranches.length > 0 && (
            <div style={{marginTop:'12px'}}>
              <div style={{...labelStyle,marginBottom:'8px'}}>
                Quick Select{nvidiaRemote ? ` (${nvidiaRemote} branches)` : ""}
              </div>
              <div style={{display:'flex',gap:'6px',flexWrap:'wrap'}}>
                {nvBranches.slice(0, 15).map(b => (
                  <button key={b} className="btn btn-sm"
                    onClick={() => setRunCodeRefInput(b)}
                    style={{
                      fontSize:'10px',padding:'2px 8px',
                      background: b === runCodeRefInput ? 'rgba(118,185,0,0.15)' : 'transparent',
                      color: b === runCodeRefInput ? 'var(--nv-green)' : 'var(--nv-text-dim)',
                      border: `1px solid ${b === runCodeRefInput ? 'rgba(118,185,0,0.3)' : 'var(--nv-border)'}`,
                    }}>
                    {b}
                  </button>
                ))}
              </div>
            </div>
          );
        })()}
      </div>

      {/* Deploy & Update Section */}
      <div className="card" style={{padding:'24px',marginBottom:'20px'}}>
        <div className="section-title" style={{marginBottom:'6px'}}>Deploy & Update</div>
        <div style={{fontSize:'12px',color:'var(--nv-text-dim)',lineHeight:'1.6',marginBottom:'20px'}}>
          Pull the latest code from a remote branch. You can update the portal server, the runners, or both independently.
        </div>

        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'14px',marginBottom:'20px'}}>
          <div>
            <label style={labelStyle}>Remote</label>
            <select className="select" style={{width:'100%'}} value={deployRemote} onChange={e => setDeployRemote(e.target.value)}>
              {(gitInfo?.remotes || []).map(r => (
                <option key={r.name} value={r.name}>{r.name}{r.url ? ` — ${r.url}` : ''}</option>
              ))}
            </select>
          </div>
          <div>
            <label style={labelStyle}>Branch</label>
            <input className="input" style={{width:'100%'}} value={deployBranch}
              onChange={e => setDeployBranch(e.target.value)}
              placeholder="e.g. main" />
          </div>
        </div>

        {gitInfo.remote_branches?.length > 0 && (
          <div style={{marginBottom:'20px'}}>
            <div style={{...labelStyle,marginBottom:'8px'}}>Quick Select Branch</div>
            <div style={{display:'flex',gap:'6px',flexWrap:'wrap'}}>
              {gitInfo.remote_branches.slice(0, 12).map(b => {
                const short = b.replace(/^[^/]+\//, "");
                return (
                  <button key={b} className="btn btn-sm"
                    onClick={() => setDeployBranch(short)}
                    style={{
                      fontSize:'10px',padding:'2px 8px',
                      background: short === deployBranch ? 'rgba(118,185,0,0.15)' : 'transparent',
                      color: short === deployBranch ? 'var(--nv-green)' : 'var(--nv-text-dim)',
                      border: `1px solid ${short === deployBranch ? 'rgba(118,185,0,0.3)' : 'var(--nv-border)'}`,
                    }}>
                    {short}
                  </button>
                );
              })}
            </div>
          </div>
        )}

        <div style={{
          display:'grid',gridTemplateColumns:'1fr 1fr 1fr',gap:'12px',
          padding:'16px',borderRadius:'10px',
          background:'rgba(255,255,255,0.015)',border:'1px solid var(--nv-border)',
        }}>
          <div style={{display:'flex',flexDirection:'column',gap:'10px'}}>
            <div style={{fontSize:'13px',fontWeight:600,color:'#fff'}}>Portal Only</div>
            <div style={{fontSize:'11px',color:'var(--nv-text-dim)',lineHeight:'1.5',flex:1}}>
              Pulls code and restarts the portal server. Runners keep their current code.
            </div>
            <button className="btn" onClick={() => setConfirmAction("deploy-portal")}
              disabled={isBusy || !deployBranch.trim()}
              style={{width:'100%',justifyContent:'center',background:'rgba(118,185,0,0.1)',color:'var(--nv-green)',border:'1px solid rgba(118,185,0,0.25)',fontWeight:600}}>
              {activeAction === "deploy-portal" ? <><span className="spinner" style={{marginRight:'6px'}}></span>Deploying…</> : <><IconDownload /> Deploy Portal</>}
            </button>
          </div>

          <div style={{display:'flex',flexDirection:'column',gap:'10px'}}>
            <div style={{fontSize:'13px',fontWeight:600,color:'#fff'}}>Portal + Runners</div>
            <div style={{fontSize:'11px',color:'var(--nv-text-dim)',lineHeight:'1.5',flex:1}}>
              Pulls code, restarts portal, and signals all runners to update and restart.
            </div>
            <button className="btn btn-primary" onClick={() => setConfirmAction("deploy-all")}
              disabled={isBusy || !deployBranch.trim()}
              style={{width:'100%',justifyContent:'center'}}>
              {activeAction === "deploy-all" ? <><span className="spinner" style={{marginRight:'6px'}}></span>Deploying…</> : <><IconDownload /> Deploy All</>}
            </button>
          </div>

          <div style={{display:'flex',flexDirection:'column',gap:'10px'}}>
            <div style={{fontSize:'13px',fontWeight:600,color:'#fff'}}>Runners Only</div>
            <div style={{fontSize:'11px',color:'var(--nv-text-dim)',lineHeight:'1.5',flex:1}}>
              Signals runners to pull and restart. The portal is not restarted.
            </div>
            <button className="btn" onClick={() => setConfirmAction("update-runners")}
              disabled={isBusy || !deployBranch.trim()}
              style={{width:'100%',justifyContent:'center',background:'rgba(100,180,255,0.1)',color:'#64b4ff',border:'1px solid rgba(100,180,255,0.25)',fontWeight:600}}>
              {activeAction === "update-runners" ? <><span className="spinner" style={{marginRight:'6px'}}></span>Updating…</> : <><IconRefresh /> Update Runners</>}
            </button>
          </div>
        </div>

        <div style={{marginTop:'12px',fontSize:'11px',color:'var(--nv-text-dim)',lineHeight:'1.5'}}>
          Target: <span className="mono" style={{color:'var(--nv-text-muted)'}}>{refLabel}</span>
        </div>
      </div>

      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'20px',marginBottom:'20px'}}>
        {/* Recent Commits Card */}
        <div className="card" style={{padding:'24px'}}>
          <div className="section-title" style={{marginBottom:'16px'}}>Recent Commits</div>
          <div style={{display:'flex',flexDirection:'column',gap:'0'}}>
            {(gitInfo.recent_commits || []).map((c, i) => (
              <div key={c.sha} style={{
                display:'flex',gap:'10px',alignItems:'flex-start',padding:'8px 0',
                borderBottom: i < gitInfo.recent_commits.length - 1 ? '1px solid var(--nv-border)' : 'none',
              }}>
                <span className="mono" style={{
                  fontSize:'11px',fontWeight:600,whiteSpace:'nowrap',
                  color: i === 0 ? 'var(--nv-green)' : 'var(--nv-text-dim)',
                }}>{c.short_sha}</span>
                <div style={{flex:1,minWidth:0}}>
                  <div style={{fontSize:'12px',color:'#fff',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>{c.message}</div>
                  <div style={{fontSize:'10px',color:'var(--nv-text-dim)',marginTop:'2px'}}>{c.date}</div>
                </div>
              </div>
            ))}
            {(!gitInfo.recent_commits || gitInfo.recent_commits.length === 0) && (
              <div style={{padding:'20px',textAlign:'center',color:'var(--nv-text-dim)',fontSize:'13px'}}>No commit history available</div>
            )}
          </div>
        </div>

        {/* Remotes Card */}
        <div className="card" style={{padding:'24px'}}>
          <div className="section-title" style={{marginBottom:'12px'}}>Remotes ({(gitInfo?.remotes || []).length})</div>
          <div style={{display:'flex',flexDirection:'column',gap:'0'}}>
            {(gitInfo?.remotes || []).map(r => (
              <div key={r.name} style={{display:'flex',alignItems:'center',gap:'12px',padding:'8px 0',borderBottom:'1px solid var(--nv-border)'}}>
                <span className="mono" style={{fontSize:'13px',fontWeight:600,color:'var(--nv-green)',minWidth:'100px'}}>{r.name}</span>
                <span className="mono" style={{fontSize:'12px',color:'var(--nv-text-muted)',wordBreak:'break-all',flex:1}}>{r.url || '(no URL)'}</span>
              </div>
            ))}
            {(gitInfo?.remotes || []).length === 0 && (
              <div style={{padding:'16px',textAlign:'center',color:'var(--nv-text-dim)',fontSize:'13px'}}>No git remotes configured</div>
            )}
          </div>
        </div>
      </div>

      {/* Deploy/Update Result */}
      {deployResult && (
        <div className="card" style={{padding:'20px',marginBottom:'20px',background:'rgba(118,185,0,0.05)',border:'1px solid rgba(118,185,0,0.2)'}}>
          <div style={{display:'flex',alignItems:'center',gap:'10px',marginBottom:'12px'}}>
            <IconCheck />
            <span style={{fontSize:'14px',fontWeight:600,color:'var(--nv-green)'}}>{deployResult.message}</span>
          </div>
          {deployResult.log && deployResult.log.length > 0 && (
            <pre className="mono" style={{
              fontSize:'11px',padding:'12px',borderRadius:'8px',
              background:'var(--nv-bg)',border:'1px solid var(--nv-border)',
              color:'var(--nv-text-muted)',maxHeight:'200px',overflow:'auto',
              whiteSpace:'pre-wrap',wordBreak:'break-all',lineHeight:'1.6',
            }}>{deployResult.log.join("\n")}</pre>
          )}
        </div>
      )}

      {/* Deploy/Update Error */}
      {deployError && (
        <div style={{marginBottom:'16px',padding:'14px 18px',borderRadius:'8px',background:'rgba(255,50,50,0.08)',border:'1px solid rgba(255,50,50,0.2)'}}>
          <div style={{fontSize:'13px',fontWeight:600,color:'#ff5050',marginBottom:'6px'}}>Operation Failed</div>
          <pre className="mono" style={{fontSize:'12px',color:'#ff5050',whiteSpace:'pre-wrap',wordBreak:'break-all',margin:0}}>{deployError}</pre>
        </div>
      )}

      {/* ===== MCP Server Configuration ===== */}
      <div className="card" style={{padding:'24px',marginBottom:'20px'}}>
        <div className="section-title" style={{marginBottom:'6px'}}>MCP Server</div>
        <div style={{fontSize:'12px',color:'var(--nv-text-dim)',lineHeight:'1.6',marginBottom:'16px'}}>
          The MCP (Model Context Protocol) server allows AI agents like Cursor, Claude Desktop, and custom tools to interact with this portal programmatically.
        </div>

        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'16px',marginBottom:'16px'}}>
          <div>
            <label style={labelStyle}>Server Status</label>
            <div style={{display:'flex',alignItems:'center',gap:'8px'}}>
              <span style={{width:'8px',height:'8px',borderRadius:'50%',background: portalSettings.mcp_enabled === 'true' ? 'var(--nv-green)' : '#ff5050'}}></span>
              <span style={{fontSize:'13px',fontWeight:600,color: portalSettings.mcp_enabled === 'true' ? 'var(--nv-green)' : '#ff5050'}}>
                {portalSettings.mcp_enabled === 'true' ? 'Enabled' : 'Disabled'}
              </span>
              <button className="btn btn-sm" style={{marginLeft:'8px',fontSize:'10px',padding:'2px 10px'}}
                onClick={() => saveMcpSetting('mcp_enabled', portalSettings.mcp_enabled === 'true' ? 'false' : 'true')}>
                {portalSettings.mcp_enabled === 'true' ? 'Disable' : 'Enable'}
              </button>
            </div>
          </div>
          <div>
            <label style={labelStyle}>Rate Limit (req/min per agent)</label>
            <div style={{display:'flex',gap:'8px',alignItems:'center'}}>
              <input className="input" style={{width:'80px'}} type="number"
                value={portalSettings.mcp_rate_limit || '60'}
                onChange={e => saveMcpSetting('mcp_rate_limit', e.target.value)} />
              <span style={{fontSize:'11px',color:'var(--nv-text-dim)'}}>requests/minute</span>
            </div>
          </div>
        </div>

        <div style={{marginBottom:'16px'}}>
          <label style={labelStyle}>Allowed Origins (CORS)</label>
          <input className="input" style={{width:'100%'}}
            value={portalSettings.mcp_allowed_origins || '*'}
            onBlur={e => saveMcpSetting('mcp_allowed_origins', e.target.value)}
            onChange={e => setPortalSettings({...portalSettings, mcp_allowed_origins: e.target.value})}
            placeholder="* (allow all)" />
          <div style={{fontSize:'10px',color:'var(--nv-text-dim)',marginTop:'4px'}}>Comma-separated origins, or * for all</div>
        </div>

        {mcpCursorConfig && (
          <div style={{marginBottom:'0'}}>
            <label style={labelStyle}>Cursor IDE Configuration</label>
            <div style={{fontSize:'11px',color:'var(--nv-text-dim)',marginBottom:'8px'}}>
              Add this to your <code className="mono" style={{background:'rgba(255,255,255,0.06)',padding:'1px 5px',borderRadius:'4px'}}>.cursor/mcp.json</code> to connect Cursor to this portal:
            </div>
            <div style={{position:'relative'}}>
              <pre className="mono" style={{
                fontSize:'11px',padding:'12px',borderRadius:'8px',
                background:'var(--nv-bg)',border:'1px solid var(--nv-border)',
                color:'var(--nv-green)',margin:0,overflow:'auto',
              }}>{JSON.stringify(mcpCursorConfig, null, 2)}</pre>
              <button className="btn btn-sm" style={{position:'absolute',top:'6px',right:'6px',fontSize:'10px',padding:'2px 8px'}}
                onClick={() => { navigator.clipboard.writeText(JSON.stringify(mcpCursorConfig, null, 2)); }}>
                Copy
              </button>
            </div>
          </div>
        )}
      </div>

      {/* ===== Exposed Tools ===== */}
      <div className="card" style={{padding:'24px',marginBottom:'20px'}}>
        <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'16px'}}>
          <div>
            <div className="section-title" style={{marginBottom:'4px'}}>Exposed MCP Tools</div>
            <div style={{fontSize:'12px',color:'var(--nv-text-dim)'}}>
              {mcpTools.length} tool{mcpTools.length !== 1 ? 's' : ''} registered — {mcpTools.filter(t => t.enabled).length} enabled
            </div>
          </div>
          <div style={{display:'flex',gap:'8px',alignItems:'center'}}>
            <input className="input" style={{width:'200px',fontSize:'12px'}} placeholder="Filter tools…"
              value={mcpToolFilter} onChange={e => setMcpToolFilter(e.target.value)} />
            <button className="btn btn-sm" onClick={fetchMcpTools} style={{fontSize:'10px',padding:'4px 10px'}}>
              <IconRefresh />
            </button>
          </div>
        </div>

        {mcpToolsLoading ? (
          <div style={{padding:'40px',textAlign:'center'}}>
            <span className="spinner"></span>
          </div>
        ) : (
          <div style={{borderRadius:'8px',border:'1px solid var(--nv-border)',overflow:'hidden'}}>
            <table style={{width:'100%',borderCollapse:'collapse',fontSize:'12px'}}>
              <thead>
                <tr style={{background:'rgba(255,255,255,0.02)'}}>
                  <th style={{textAlign:'left',padding:'8px 12px',color:'var(--nv-text-dim)',fontWeight:600,fontSize:'10px',textTransform:'uppercase',letterSpacing:'0.04em'}}>Tool</th>
                  <th style={{textAlign:'left',padding:'8px 12px',color:'var(--nv-text-dim)',fontWeight:600,fontSize:'10px',textTransform:'uppercase',letterSpacing:'0.04em'}}>Category</th>
                  <th style={{textAlign:'left',padding:'8px 12px',color:'var(--nv-text-dim)',fontWeight:600,fontSize:'10px',textTransform:'uppercase',letterSpacing:'0.04em'}}>Description</th>
                  <th style={{textAlign:'center',padding:'8px 12px',color:'var(--nv-text-dim)',fontWeight:600,fontSize:'10px',textTransform:'uppercase',letterSpacing:'0.04em',width:'80px'}}>Status</th>
                </tr>
              </thead>
              <tbody>
                {mcpTools
                  .filter(t => !mcpToolFilter || t.name.toLowerCase().includes(mcpToolFilter.toLowerCase())
                    || t.category.toLowerCase().includes(mcpToolFilter.toLowerCase())
                    || t.description.toLowerCase().includes(mcpToolFilter.toLowerCase()))
                  .map(tool => (
                  <tr key={tool.key} style={{borderTop:'1px solid var(--nv-border)'}}>
                    <td style={{padding:'8px 12px'}}>
                      <span className="mono" style={{color:'#fff',fontWeight:600,fontSize:'12px'}}>{tool.name}</span>
                      {tool.tags && tool.tags.length > 0 && (
                        <div style={{marginTop:'2px',display:'flex',gap:'4px'}}>
                          {tool.tags.map(tag => (
                            <span key={tag} style={{fontSize:'9px',padding:'1px 5px',borderRadius:'3px',
                              background:'rgba(100,180,255,0.1)',color:'#64b4ff',fontWeight:500}}>{tag}</span>
                          ))}
                        </div>
                      )}
                    </td>
                    <td style={{padding:'8px 12px',color:'var(--nv-text-muted)',fontSize:'11px'}}>{tool.category}</td>
                    <td style={{padding:'8px 12px',color:'var(--nv-text-dim)',fontSize:'11px',maxWidth:'300px'}}>{tool.description}</td>
                    <td style={{padding:'8px 12px',textAlign:'center'}}>
                      <button className="btn btn-sm" style={{
                        fontSize:'10px',padding:'2px 10px',minWidth:'60px',
                        background: tool.enabled ? 'rgba(118,185,0,0.1)' : 'rgba(255,50,50,0.08)',
                        color: tool.enabled ? 'var(--nv-green)' : '#ff5050',
                        border: `1px solid ${tool.enabled ? 'rgba(118,185,0,0.3)' : 'rgba(255,50,50,0.2)'}`,
                      }} onClick={() => toggleMcpTool(tool.key, !tool.enabled)}>
                        {tool.enabled ? 'Enabled' : 'Disabled'}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* ===== Agent Activity Log ===== */}
      <div className="card" style={{padding:'24px',marginBottom:'20px'}}>
        <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'16px'}}>
          <div>
            <div className="section-title" style={{marginBottom:'4px'}}>Agent Activity Log</div>
            <div style={{fontSize:'12px',color:'var(--nv-text-dim)'}}>MCP tool invocations from AI agents</div>
          </div>
          <button className="btn btn-sm" onClick={fetchMcpAuditLog} style={{fontSize:'10px',padding:'4px 10px'}}>
            <IconRefresh />
          </button>
        </div>

        {/* Stats Cards */}
        {mcpAuditStats && (
          <div style={{display:'grid',gridTemplateColumns:'repeat(4, 1fr)',gap:'12px',marginBottom:'16px'}}>
            <div style={{padding:'12px 16px',borderRadius:'8px',background:'rgba(255,255,255,0.02)',border:'1px solid var(--nv-border)'}}>
              <div style={{fontSize:'20px',fontWeight:700,color:'#fff'}}>{mcpAuditStats.total_requests}</div>
              <div style={{fontSize:'10px',color:'var(--nv-text-dim)',textTransform:'uppercase',letterSpacing:'0.04em',marginTop:'2px'}}>Total Requests</div>
            </div>
            <div style={{padding:'12px 16px',borderRadius:'8px',background:'rgba(255,255,255,0.02)',border:'1px solid var(--nv-border)'}}>
              <div style={{fontSize:'20px',fontWeight:700,color:'var(--nv-green)'}}>{mcpAuditStats.unique_agents}</div>
              <div style={{fontSize:'10px',color:'var(--nv-text-dim)',textTransform:'uppercase',letterSpacing:'0.04em',marginTop:'2px'}}>Unique Agents</div>
            </div>
            <div style={{padding:'12px 16px',borderRadius:'8px',background:'rgba(255,255,255,0.02)',border:'1px solid var(--nv-border)'}}>
              <div style={{fontSize:'20px',fontWeight:700,color:'var(--nv-green)'}}>{mcpAuditStats.success_count}</div>
              <div style={{fontSize:'10px',color:'var(--nv-text-dim)',textTransform:'uppercase',letterSpacing:'0.04em',marginTop:'2px'}}>Successful</div>
            </div>
            <div style={{padding:'12px 16px',borderRadius:'8px',background:'rgba(255,255,255,0.02)',border:'1px solid var(--nv-border)'}}>
              <div style={{fontSize:'20px',fontWeight:700,color: mcpAuditStats.error_count > 0 ? '#ff5050' : 'var(--nv-text-muted)'}}>{mcpAuditStats.error_count}</div>
              <div style={{fontSize:'10px',color:'var(--nv-text-dim)',textTransform:'uppercase',letterSpacing:'0.04em',marginTop:'2px'}}>Errors</div>
            </div>
          </div>
        )}

        {/* Top tools and agents */}
        {mcpAuditStats && (mcpAuditStats.top_tools?.length > 0 || mcpAuditStats.top_agents?.length > 0) && (
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'16px',marginBottom:'16px'}}>
            {mcpAuditStats.top_tools?.length > 0 && (
              <div style={{padding:'12px 16px',borderRadius:'8px',background:'rgba(255,255,255,0.02)',border:'1px solid var(--nv-border)'}}>
                <div style={{fontSize:'10px',fontWeight:600,color:'var(--nv-text-dim)',textTransform:'uppercase',letterSpacing:'0.04em',marginBottom:'8px'}}>Most Used Tools</div>
                {mcpAuditStats.top_tools.map(t => (
                  <div key={t.tool_name} style={{display:'flex',justifyContent:'space-between',alignItems:'center',padding:'3px 0'}}>
                    <span className="mono" style={{fontSize:'11px',color:'var(--nv-text-muted)'}}>{t.tool_name}</span>
                    <span style={{fontSize:'11px',fontWeight:600,color:'#fff'}}>{t.count}</span>
                  </div>
                ))}
              </div>
            )}
            {mcpAuditStats.top_agents?.length > 0 && (
              <div style={{padding:'12px 16px',borderRadius:'8px',background:'rgba(255,255,255,0.02)',border:'1px solid var(--nv-border)'}}>
                <div style={{fontSize:'10px',fontWeight:600,color:'var(--nv-text-dim)',textTransform:'uppercase',letterSpacing:'0.04em',marginBottom:'8px'}}>Top Agents</div>
                {mcpAuditStats.top_agents.map(a => (
                  <div key={a.agent_name} style={{display:'flex',justifyContent:'space-between',alignItems:'center',padding:'3px 0'}}>
                    <span style={{fontSize:'11px',color:'var(--nv-text-muted)'}}>{a.agent_name}</span>
                    <span style={{fontSize:'11px',fontWeight:600,color:'#fff'}}>{a.count}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Filters */}
        <div style={{display:'flex',gap:'10px',marginBottom:'12px'}}>
          <input className="input" style={{flex:1,fontSize:'12px'}} placeholder="Filter by tool name…"
            value={mcpAuditFilter.tool_name}
            onChange={e => setMcpAuditFilter({...mcpAuditFilter, tool_name: e.target.value})} />
          <input className="input" style={{flex:1,fontSize:'12px'}} placeholder="Filter by agent name…"
            value={mcpAuditFilter.agent_name}
            onChange={e => setMcpAuditFilter({...mcpAuditFilter, agent_name: e.target.value})} />
          <button className="btn btn-sm" onClick={fetchMcpAuditLog} style={{fontSize:'10px',padding:'4px 10px'}}>Apply</button>
        </div>

        {/* Log Table */}
        {mcpAuditLoading ? (
          <div style={{padding:'40px',textAlign:'center'}}>
            <span className="spinner"></span>
          </div>
        ) : mcpAuditLog.length === 0 ? (
          <div style={{padding:'40px',textAlign:'center',color:'var(--nv-text-dim)',fontSize:'13px'}}>
            No agent activity recorded yet. Agents will appear here once they start making MCP requests.
          </div>
        ) : (
          <div style={{borderRadius:'8px',border:'1px solid var(--nv-border)',overflow:'hidden',maxHeight:'400px',overflowY:'auto'}}>
            <table style={{width:'100%',borderCollapse:'collapse',fontSize:'11px'}}>
              <thead style={{position:'sticky',top:0,background:'var(--nv-surface)'}}>
                <tr>
                  <th style={{textAlign:'left',padding:'6px 10px',color:'var(--nv-text-dim)',fontWeight:600,fontSize:'9px',textTransform:'uppercase',letterSpacing:'0.04em'}}>Time</th>
                  <th style={{textAlign:'left',padding:'6px 10px',color:'var(--nv-text-dim)',fontWeight:600,fontSize:'9px',textTransform:'uppercase',letterSpacing:'0.04em'}}>Agent</th>
                  <th style={{textAlign:'left',padding:'6px 10px',color:'var(--nv-text-dim)',fontWeight:600,fontSize:'9px',textTransform:'uppercase',letterSpacing:'0.04em'}}>Tool</th>
                  <th style={{textAlign:'left',padding:'6px 10px',color:'var(--nv-text-dim)',fontWeight:600,fontSize:'9px',textTransform:'uppercase',letterSpacing:'0.04em'}}>Arguments</th>
                  <th style={{textAlign:'right',padding:'6px 10px',color:'var(--nv-text-dim)',fontWeight:600,fontSize:'9px',textTransform:'uppercase',letterSpacing:'0.04em',width:'60px'}}>ms</th>
                  <th style={{textAlign:'center',padding:'6px 10px',color:'var(--nv-text-dim)',fontWeight:600,fontSize:'9px',textTransform:'uppercase',letterSpacing:'0.04em',width:'50px'}}>OK</th>
                </tr>
              </thead>
              <tbody>
                {mcpAuditLog.map(entry => (
                  <tr key={entry.id} style={{borderTop:'1px solid var(--nv-border)'}}>
                    <td style={{padding:'5px 10px',color:'var(--nv-text-dim)',whiteSpace:'nowrap',fontSize:'10px'}}>
                      {entry.timestamp ? new Date(entry.timestamp).toLocaleString() : '—'}
                    </td>
                    <td style={{padding:'5px 10px',color:'var(--nv-text-muted)',fontSize:'11px'}}>{entry.agent_name || entry.agent_id || '—'}</td>
                    <td style={{padding:'5px 10px'}}>
                      <span className="mono" style={{color:'#fff',fontWeight:500,fontSize:'11px'}}>{entry.tool_name}</span>
                    </td>
                    <td style={{padding:'5px 10px',color:'var(--nv-text-dim)',maxWidth:'200px',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap',fontSize:'10px'}}>
                      {entry.arguments || '—'}
                    </td>
                    <td style={{padding:'5px 10px',textAlign:'right',color:'var(--nv-text-muted)',fontSize:'10px',fontFamily:'var(--font-mono)'}}>
                      {entry.duration_ms != null ? Math.round(entry.duration_ms) : '—'}
                    </td>
                    <td style={{padding:'5px 10px',textAlign:'center'}}>
                      <span style={{
                        display:'inline-block',width:'6px',height:'6px',borderRadius:'50%',
                        background: entry.success ? 'var(--nv-green)' : '#ff5050',
                      }}></span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Confirmation Modal */}
      {confirm && (
        <div className="modal-overlay" onClick={() => setConfirmAction(null)}>
          <div className="modal-content" style={{maxWidth:'480px'}} onClick={e => e.stopPropagation()}>
            <div className="modal-head">
              <h2 style={{fontSize:'16px',fontWeight:700,color:'#fff'}}>{confirm.title}</h2>
              <button className="btn btn-ghost btn-icon" onClick={() => setConfirmAction(null)} style={{borderRadius:'50%'}}><IconX /></button>
            </div>
            <div style={{padding:'24px'}}>
              <div style={{fontSize:'14px',color:'var(--nv-text-muted)',lineHeight:'1.7',marginBottom:'16px'}}>
                This will:
              </div>
              <ol style={{fontSize:'13px',color:'#fff',lineHeight:'2',paddingLeft:'20px',marginBottom:'20px'}}>
                {confirm.steps.map((s, i) => <li key={i}>{s}</li>)}
              </ol>
              {confirm.note && (
                <div style={{padding:'10px 14px',borderRadius:'8px',background:'rgba(100,180,255,0.06)',border:'1px solid rgba(100,180,255,0.15)',color:'#64b4ff',fontSize:'12px',marginBottom:'16px',lineHeight:'1.6'}}>
                  {confirm.note}
                </div>
              )}
              {gitInfo.is_dirty && confirmAction !== "update-runners" && (
                <div style={{padding:'10px 14px',borderRadius:'8px',background:'rgba(255,184,77,0.08)',border:'1px solid rgba(255,184,77,0.25)',color:'#ffb84d',fontSize:'12px',marginBottom:'16px'}}>
                  <strong>Warning:</strong> You have uncommitted changes. They will be stashed before the deploy.
                </div>
              )}
              {confirmAction !== "update-runners" && (
                <div style={{fontSize:'12px',color:'var(--nv-text-dim)'}}>
                  The portal will be briefly unavailable during restart. This page will reconnect automatically.
                </div>
              )}
            </div>
            <div className="modal-foot">
              <button className="btn btn-secondary" onClick={() => setConfirmAction(null)}>Cancel</button>
              <button className="btn btn-primary" onClick={confirm.onConfirm} style={{flex:1,justifyContent:'center'}}>
                {confirm.btnLabel}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
