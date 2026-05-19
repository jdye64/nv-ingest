// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dashboard UI router — gateway-only.
//!
//! Wire-compatible with `service.routers.dashboard`. The same SPA assets
//! ship from `<repo>/dashboard/static/` for both Python and Rust runtimes
//! and call this exact endpoint set:
//!
//! * `GET  /v1/dashboard/`              — SPA shell (`index.html`)
//! * `GET  /v1/dashboard/static/...`    — JSX / SVG / etc (mounted in `app.rs`)
//! * `GET  /v1/dashboard/api/overview`  — cluster status snapshot
//! * `GET  /v1/dashboard/api/jobs`      — SSE event firehose
//! * `GET  /v1/dashboard/api/jobs/snapshot`              — REST fallback
//! * `GET  /v1/dashboard/api/jobs/list?status=&offset=&limit=&sort=`
//! * `GET  /v1/dashboard/api/jobs/{job_id}`              — single-job view
//! * `GET  /v1/dashboard/api/jobs/{job_id}/documents`    — paginated docs
//! * `GET  /v1/dashboard/api/vdb/tables`                 — vdb metadata
//! * `POST /v1/dashboard/api/vdb/query`                  — vdb search proxy

use std::convert::Infallible;
use std::path::PathBuf;
use std::time::Duration;

use axum::body::Body;
use axum::extract::{Path as AxumPath, Query, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::Stream;
use serde::Deserialize;
use serde_json::{json, Value as JsonValue};
use tokio::time::Instant;
use tokio_stream::wrappers::ReceiverStream;

use nrr_core::tracker::{DocumentRecord, DocumentStatus, JobAggregate, JobAggregateStatus};
use nrr_pipeline::PoolType;

use crate::state::SharedState;

pub fn router() -> Router<SharedState> {
    Router::new()
        .route("/", get(index))
        .route("/api/overview", get(overview))
        .route("/api/jobs", get(jobs_sse))
        .route("/api/jobs/snapshot", get(jobs_snapshot))
        .route("/api/jobs/list", get(jobs_list))
        .route("/api/jobs/:job_id", get(jobs_detail))
        .route("/api/jobs/:job_id/documents", get(jobs_documents))
        .route("/api/vdb/tables", get(vdb_tables))
        .route("/api/vdb/query", post(vdb_query))
}

// ────────────────────────── helpers ──────────────────────────

fn json_error(status: StatusCode, detail: impl Into<String>) -> Response {
    (status, Json(json!({ "detail": detail.into() }))).into_response()
}

/// Resolve the on-disk dashboard assets directory. Honours the
/// `NEMO_RETRIEVER_DASHBOARD_DIR` env var (set in the Docker image) and
/// falls back to a few sensible defaults so `cargo run` from the workspace
/// finds the dev tree without extra setup.
pub fn resolve_static_dir() -> Option<PathBuf> {
    if let Ok(env) = std::env::var("NEMO_RETRIEVER_DASHBOARD_DIR") {
        let p = PathBuf::from(env);
        if p.join("index.html").is_file() {
            return Some(p);
        }
        if p.join("static").join("index.html").is_file() {
            return Some(p.join("static"));
        }
    }
    // Walk a few likely repo-relative spots (dev convenience).
    for candidate in [
        // From workspace root.
        PathBuf::from("dashboard/static"),
        // From the binary's working dir if it lives one level deeper.
        PathBuf::from("../dashboard/static"),
        PathBuf::from("../../dashboard/static"),
    ] {
        if candidate.join("index.html").is_file() {
            return Some(candidate);
        }
    }
    None
}

// ────────────────────────── SPA shell ──────────────────────────

async fn index() -> Response {
    let Some(dir) = resolve_static_dir() else {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Dashboard UI not found (NEMO_RETRIEVER_DASHBOARD_DIR unset and no dev tree)",
        );
    };
    let path = dir.join("index.html");
    match tokio::fs::read(&path).await {
        Ok(bytes) => Response::builder()
            .status(StatusCode::OK)
            .header("content-type", HeaderValue::from_static("text/html; charset=utf-8"))
            .header("cache-control", HeaderValue::from_static("no-cache"))
            .body(Body::from(bytes))
            .unwrap(),
        Err(err) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("read {}: {err}", path.display()),
        ),
    }
}

// ────────────────────────── /api/overview ──────────────────────────

async fn fetch_pool_stats(
    client: &reqwest::Client,
    base_url: &str,
) -> JsonValue {
    let url = format!("{}/v1/admin/pool-stats", base_url.trim_end_matches('/'));
    let url_legacy = format!("{}/v1/admin/pool_stats", base_url.trim_end_matches('/'));
    // Prefer the canonical (hyphen) URL the Rust runtime exposes; fall
    // back to the underscore form for cross-runtime compatibility.
    for u in [url, url_legacy] {
        match client.get(&u).timeout(Duration::from_secs(2)).send().await {
            Ok(r) if r.status().is_success() => {
                if let Ok(v) = r.json::<JsonValue>().await {
                    return v;
                }
            }
            _ => continue,
        }
    }
    json!({})
}

async fn overview(State(state): State<SharedState>) -> Response {
    let cfg = &state.config;

    let mut backends = serde_json::Map::new();
    let mut pool_stats = serde_json::Map::new();
    if let Some(proxy) = &state.gateway_proxy {
        backends.insert("realtime".into(), proxy.check_backend(PoolType::Realtime).await);
        backends.insert("batch".into(), proxy.check_backend(PoolType::Batch).await);
        // Fan out to each backend for live pool depth so the UI can
        // surface scaling pressure without going through Prometheus.
        let rt = fetch_pool_stats(&state.http_client, &cfg.gateway.realtime_url);
        let bt = fetch_pool_stats(&state.http_client, &cfg.gateway.batch_url);
        let (rt, bt) = tokio::join!(rt, bt);
        // Each worker returns a flat map keyed by pool name. Merge them
        // into a single dict the UI can index by `realtime` / `batch`.
        for stats in [&rt, &bt] {
            // Worker pool stats look like:
            //   { "mode": "realtime", "realtime": { ...counts... } }
            // OR (legacy underscore) { "pools": { "realtime": {...} } }
            if let Some(obj) = stats.as_object() {
                if let Some(pools) = obj.get("pools").and_then(|v| v.as_object()) {
                    for (k, v) in pools {
                        pool_stats.insert(k.clone(), v.clone());
                    }
                }
                for key in ["realtime", "batch"] {
                    if let Some(v) = obj.get(key) {
                        if v.is_object() {
                            pool_stats.insert(key.to_string(), v.clone());
                        }
                    }
                }
            }
        }
    } else if let Some(pool) = &state.pipeline_pool {
        // Standalone / worker pod — read local stats inline.
        let stats = pool.stats();
        if let Some(obj) = stats.as_object() {
            for key in ["realtime", "batch"] {
                if let Some(v) = obj.get(key) {
                    pool_stats.insert(key.to_string(), v.clone());
                }
            }
        }
    }

    // VDB health is best-effort.
    let mut vdb_status: JsonValue = JsonValue::Null;
    if cfg.vectordb.enabled && !cfg.vectordb.vectordb_url.is_empty() {
        let url = format!("{}/v1/health", cfg.vectordb.vectordb_url.trim_end_matches('/'));
        if let Ok(r) = state
            .http_client
            .get(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
        {
            if r.status().is_success() {
                if let Ok(v) = r.json::<JsonValue>().await {
                    vdb_status = v;
                }
            }
        }
    }

    let job_summary = state.job_tracker.summary();

    let worker_config = json!({
        "realtime_workers": cfg.pipeline.realtime_workers,
        "realtime_queue_size": cfg.pipeline.realtime_queue_size,
        "batch_workers": cfg.pipeline.batch_workers,
        "batch_queue_size": cfg.pipeline.batch_queue_size,
    });

    let gateway_info = json!({
        "realtime_url": cfg.gateway.realtime_url,
        "batch_url": cfg.gateway.batch_url,
    });

    Json(json!({
        "mode": cfg.mode.as_str(),
        "backends": JsonValue::Object(backends),
        "pool_stats": JsonValue::Object(pool_stats),
        "vectordb": vdb_status,
        "job_summary": job_summary,
        "worker_config": worker_config,
        "gateway": gateway_info,
    }))
    .into_response()
}

// ────────────────────────── /api/jobs (SSE) ──────────────────────────

fn serialize_job(agg: &JobAggregate) -> JsonValue {
    json!({
        "job_id": agg.job_id,
        "status": agg.status.as_str(),
        "expected_documents": agg.expected_documents,
        "counts": agg.counts,
        "created_at": agg.created_at,
        "started_at": agg.started_at,
        "finalized_at": agg.finalized_at,
        "elapsed_s": agg.elapsed_s,
        "label": agg.label,
        "document_ids": agg.document_ids,
    })
}

fn serialize_document(rec: &DocumentRecord) -> JsonValue {
    json!({
        "id": rec.id,
        "job_id": rec.job_id,
        "status": rec.status.as_str(),
        "submitted_at": rec.submitted_at,
        "started_at": rec.started_at,
        "completed_at": rec.completed_at,
        "elapsed_s": rec.elapsed_s,
        "result_rows": rec.result_rows,
        "error": rec.error,
        "filename": rec.filename,
    })
}

async fn jobs_sse(
    State(state): State<SharedState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // We push events through a bounded mpsc; this matches the per-job
    // SSE handler in `routes::ingest` and avoids pulling in `async-stream`.
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);

    // Initial snapshot — bundle summary + jobs + documents so the SPA
    // can render immediately without an extra REST hop.
    let summary = state.job_tracker.summary();
    let jobs: Vec<JsonValue> = state
        .job_tracker
        .all_jobs()
        .iter()
        .map(serialize_job)
        .collect();
    let documents: Vec<JsonValue> = state
        .job_tracker
        .all_documents()
        .iter()
        .map(serialize_document)
        .collect();
    let snapshot = json!({
        "type": "snapshot",
        "summary": summary,
        "jobs": jobs,
        "documents": documents,
    });

    let bus = state.event_bus.clone();
    let tracker = state.job_tracker.clone();
    tokio::spawn(async move {
        // Snapshot first — fail-fast if the client already disconnected.
        if tx
            .send(Ok(Event::default()
                .event("snapshot")
                .data(snapshot.to_string())))
            .await
            .is_err()
        {
            return;
        }

        let mut sub = bus.subscribe(None);
        let mut last_heartbeat = Instant::now();
        loop {
            let next = tokio::time::timeout(Duration::from_secs(5), sub.recv()).await;
            match next {
                Ok(Ok(Some(event))) => {
                    let evt_type = event
                        .get("type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let sse_event_name = if evt_type.starts_with("job_") {
                        "job_lifecycle"
                    } else {
                        "job_update"
                    };
                    if tx
                        .send(Ok(Event::default()
                            .event(sse_event_name)
                            .data(event.to_string())))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
                Ok(Ok(None)) => break,
                Ok(Err(_lagged)) => continue,
                Err(_timeout) => {} // fall through to heartbeat
            }
            if last_heartbeat.elapsed() >= Duration::from_secs(5) {
                let hb = json!({
                    "type": "heartbeat",
                    "summary": tracker.summary(),
                });
                if tx
                    .send(Ok(Event::default()
                        .event("heartbeat")
                        .data(hb.to_string())))
                    .await
                    .is_err()
                {
                    break;
                }
                last_heartbeat = Instant::now();
            }
        }
    });

    Sse::new(ReceiverStream::new(rx))
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(30)))
}

// ────────────────────────── /api/jobs/snapshot ──────────────────────────

async fn jobs_snapshot(State(state): State<SharedState>) -> Response {
    let summary = state.job_tracker.summary();
    let jobs: Vec<JsonValue> = state
        .job_tracker
        .all_jobs()
        .iter()
        .map(serialize_job)
        .collect();
    let documents: Vec<JsonValue> = state
        .job_tracker
        .all_documents()
        .iter()
        .map(serialize_document)
        .collect();
    Json(json!({
        "summary": summary,
        "jobs": jobs,
        "documents": documents,
    }))
    .into_response()
}

// ────────────────────────── /api/jobs/list ──────────────────────────

#[derive(Debug, Deserialize)]
struct JobsListQuery {
    status: Option<String>,
    #[serde(default)]
    offset: usize,
    #[serde(default = "default_list_limit")]
    limit: usize,
    #[serde(default = "default_sort")]
    sort: String,
}

fn default_list_limit() -> usize {
    50
}

fn default_sort() -> String {
    "created_desc".to_string()
}

async fn jobs_list(
    State(state): State<SharedState>,
    Query(q): Query<JobsListQuery>,
) -> Response {
    if q.limit < 1 || q.limit > 500 {
        return json_error(StatusCode::BAD_REQUEST, "limit must be in [1, 500]");
    }
    const VALID_SORTS: &[&str] =
        &["created_desc", "created_asc", "finalized_desc", "finalized_asc"];
    if !VALID_SORTS.contains(&q.sort.as_str()) {
        return json_error(
            StatusCode::BAD_REQUEST,
            format!("sort must be one of {VALID_SORTS:?}, got {:?}", q.sort),
        );
    }

    let all = state.job_tracker.all_jobs();
    let total = all.len();

    let mut filtered: Vec<JobAggregate> = if let Some(s) = &q.status {
        let parsed = parse_job_status(s);
        if parsed.is_none() {
            return json_error(
                StatusCode::BAD_REQUEST,
                format!("status must be a valid JobAggregateStatus, got {s:?}"),
            );
        }
        let want = parsed.unwrap();
        all.into_iter().filter(|j| j.status == want).collect()
    } else {
        all
    };

    let key_created = |j: &JobAggregate| j.created_at.clone();
    let key_finalized = |j: &JobAggregate| j.finalized_at.clone().unwrap_or_default();
    match q.sort.as_str() {
        "created_desc" => {
            filtered.sort_by_key(key_created);
            filtered.reverse();
        }
        "created_asc" => filtered.sort_by_key(key_created),
        "finalized_desc" => {
            filtered.sort_by_key(key_finalized);
            filtered.reverse();
        }
        "finalized_asc" => filtered.sort_by_key(key_finalized),
        _ => {}
    }

    let total_filtered = filtered.len();
    let page: Vec<JsonValue> = filtered
        .into_iter()
        .skip(q.offset)
        .take(q.limit)
        .map(|j| serialize_job(&j))
        .collect();

    Json(json!({
        "jobs": page,
        "total": total,
        "total_filtered": total_filtered,
        "offset": q.offset,
        "limit": q.limit,
        "sort": q.sort,
    }))
    .into_response()
}

fn parse_job_status(s: &str) -> Option<JobAggregateStatus> {
    match s {
        "pending" => Some(JobAggregateStatus::Pending),
        "processing" => Some(JobAggregateStatus::Processing),
        "completed" => Some(JobAggregateStatus::Completed),
        "failed" => Some(JobAggregateStatus::Failed),
        "partial_success" => Some(JobAggregateStatus::PartialSuccess),
        _ => None,
    }
}

fn parse_doc_status(s: &str) -> Option<DocumentStatus> {
    match s {
        "pending" => Some(DocumentStatus::Pending),
        "processing" => Some(DocumentStatus::Processing),
        "completed" => Some(DocumentStatus::Completed),
        "failed" => Some(DocumentStatus::Failed),
        _ => None,
    }
}

// ────────────────────────── /api/jobs/{job_id} ──────────────────────────

async fn jobs_detail(
    State(state): State<SharedState>,
    AxumPath(job_id): AxumPath<String>,
) -> Response {
    let Some(agg) = state.job_tracker.get_job(&job_id) else {
        return json_error(StatusCode::NOT_FOUND, format!("Job {job_id:?} not found"));
    };
    let docs = state.job_tracker.job_documents(&job_id);
    const SAMPLE_CAP: usize = 500;
    let truncated = docs.len() > SAMPLE_CAP;
    let docs_page: Vec<JsonValue> = docs
        .iter()
        .take(SAMPLE_CAP)
        .map(serialize_document)
        .collect();

    let mut body = serialize_job(&agg);
    if let Some(obj) = body.as_object_mut() {
        obj.insert("documents".into(), JsonValue::Array(docs_page));
        obj.insert("documents_truncated".into(), JsonValue::Bool(truncated));
    }
    Json(body).into_response()
}

// ────────────────── /api/jobs/{job_id}/documents ───────────────────

#[derive(Debug, Deserialize)]
struct JobDocumentsQuery {
    status: Option<String>,
    #[serde(default)]
    offset: usize,
    #[serde(default = "default_doc_limit")]
    limit: usize,
}

fn default_doc_limit() -> usize {
    100
}

async fn jobs_documents(
    State(state): State<SharedState>,
    AxumPath(job_id): AxumPath<String>,
    Query(q): Query<JobDocumentsQuery>,
) -> Response {
    if q.limit < 1 || q.limit > 1000 {
        return json_error(StatusCode::BAD_REQUEST, "limit must be in [1, 1000]");
    }
    if state.job_tracker.get_job(&job_id).is_none() {
        return json_error(StatusCode::NOT_FOUND, format!("Job {job_id:?} not found"));
    }
    let docs = state.job_tracker.job_documents(&job_id);
    let total = docs.len();
    let filtered: Vec<DocumentRecord> = if let Some(s) = &q.status {
        let Some(want) = parse_doc_status(s) else {
            return json_error(
                StatusCode::BAD_REQUEST,
                format!("status must be a valid DocumentStatus, got {s:?}"),
            );
        };
        docs.into_iter().filter(|d| d.status == want).collect()
    } else {
        docs
    };
    let total_filtered = filtered.len();
    let page: Vec<JsonValue> = filtered
        .into_iter()
        .skip(q.offset)
        .take(q.limit)
        .map(|d| serialize_document(&d))
        .collect();
    Json(json!({
        "job_id": job_id,
        "total": total,
        "total_filtered": total_filtered,
        "offset": q.offset,
        "limit": q.limit,
        "items": page,
    }))
    .into_response()
}

// ────────────────────────── /api/vdb/tables ──────────────────────────

async fn vdb_tables(State(state): State<SharedState>) -> Response {
    let cfg = &state.config.vectordb;
    if !cfg.enabled || cfg.vectordb_url.is_empty() {
        return Json(json!({ "error": "VectorDB not enabled", "tables": [] })).into_response();
    }
    let url = format!("{}/v1/health", cfg.vectordb_url.trim_end_matches('/'));
    match state
        .http_client
        .get(&url)
        .timeout(Duration::from_secs(10))
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => match r.json::<JsonValue>().await {
            Ok(health) => {
                let table = json!({
                    "name": health.get("table").and_then(|v| v.as_str()).unwrap_or(""),
                    "total_rows": health.get("total_rows").cloned().unwrap_or(json!(0)),
                    "exists": health.get("table_exists").and_then(|v| v.as_bool()).unwrap_or(false),
                });
                Json(json!({ "tables": [table] })).into_response()
            }
            Err(err) => Json(json!({ "error": err.to_string(), "tables": [] })).into_response(),
        },
        Ok(r) => Json(json!({
            "error": format!("vdb health HTTP {}", r.status().as_u16()),
            "tables": [],
        }))
        .into_response(),
        Err(err) => Json(json!({ "error": err.to_string(), "tables": [] })).into_response(),
    }
}

// ────────────────────────── /api/vdb/query ──────────────────────────

#[derive(Debug, Deserialize)]
struct VdbQueryRequest {
    query: String,
    #[serde(default = "default_top_k")]
    top_k: u32,
}

fn default_top_k() -> u32 {
    10
}

async fn vdb_query(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(body): Json<VdbQueryRequest>,
) -> Response {
    let _ = headers; // reserved for future passthrough
    if body.top_k < 1 || body.top_k > 1000 {
        return json_error(StatusCode::BAD_REQUEST, "top_k must be in [1, 1000]");
    }
    let cfg = &state.config.vectordb;
    if !cfg.enabled || cfg.vectordb_url.is_empty() {
        return json_error(StatusCode::NOT_IMPLEMENTED, "VectorDB not enabled");
    }
    let url = format!("{}/v1/query", cfg.vectordb_url.trim_end_matches('/'));
    let payload = json!({ "query": body.query, "top_k": body.top_k });
    match state
        .http_client
        .post(&url)
        .timeout(Duration::from_secs(60))
        .json(&payload)
        .send()
        .await
    {
        Ok(r) => {
            let status = r.status();
            let bytes = r.bytes().await.unwrap_or_default();
            if status.is_success() {
                Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", HeaderValue::from_static("application/json"))
                    .body(Body::from(bytes))
                    .unwrap()
            } else {
                let preview: String = String::from_utf8_lossy(&bytes)
                    .chars()
                    .take(500)
                    .collect();
                json_error(
                    StatusCode::from_u16(status.as_u16())
                        .unwrap_or(StatusCode::BAD_GATEWAY),
                    preview,
                )
            }
        }
        Err(err) => json_error(StatusCode::BAD_GATEWAY, format!("VDB query failed: {err}")),
    }
}
