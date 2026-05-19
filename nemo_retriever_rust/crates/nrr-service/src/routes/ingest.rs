// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ingest endpoints (J1+ shape): jobs, document/page/whole uploads, status,
//! sidecars, internal gateway callback, and per-job SSE event stream.
//!
//! Wire-compatible with `service.routers.ingest`. Field names, status codes,
//! and JSON shapes match exactly so existing Python clients work unchanged.

use std::convert::Infallible;
use std::time::Duration;

use axum::body::Bytes as AxumBytes;
use axum::extract::{DefaultBodyLimit, Path, Query, Request, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use bytes::Bytes;
use futures::stream::{self, Stream};
use serde::Deserialize;
use serde_json::{json, Value as JsonValue};
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use nrr_core::models::{
    DocumentIngestAccepted, DocumentStatusResponse, IngestAccepted, IngestRequest,
    JobAggregateResponse, JobCreateRequest, JobCreatedResponse, JobDocumentsPage,
    JobStatusResponse, PageIngestAccepted, SidecarUploadResponse,
};
use nrr_core::policy::validate_pipeline_spec;
use nrr_core::tracker::DocumentRecord;
use nrr_core::util::{classify_file, sha256_hex, utc_now_iso};
use nrr_pipeline::{PoolType, WorkItem};

use crate::state::SharedState;

const GATEWAY_DOC_ID_HEADER: &str = "X-Gateway-Document-Id";
const GATEWAY_CALLBACK_HEADER: &str = "X-Gateway-Callback-Url";
#[allow(dead_code)]
const GATEWAY_PIPELINE_SPEC_HEADER: &str = "X-Gateway-Pipeline-Spec";
const GATEWAY_JOB_ID_HEADER: &str = "X-Gateway-Job-Id";
const PAGE_THRESHOLD_FOR_BATCH: u32 = 5;

pub fn router() -> Router<SharedState> {
    // Multipart uploads can be hundreds of MB. Disable axum's default 2 MiB
    // request-body cap so the YAML-driven `resources.max_upload_bytes` is the
    // only ceiling — that limit is enforced explicitly inside each upload
    // handler after we know the actual file size.
    Router::new()
        // Job CRUD
        .route("/ingest/job", post(create_job))
        .route("/ingest/job/:job_id", get(get_job))
        .route("/ingest/job/:job_id/documents", get(get_job_documents))
        .route(
            "/ingest/job/:job_id/document/:document_id",
            get(get_job_document),
        )
        // Document / page / whole uploads (body limits disabled — see above)
        .route(
            "/ingest/job/:job_id/document",
            post(submit_document_to_job).layer(DefaultBodyLimit::disable()),
        )
        .route(
            "/ingest/job/:job_id/page",
            post(submit_page_to_job).layer(DefaultBodyLimit::disable()),
        )
        .route(
            "/ingest/job/:job_id/whole",
            post(submit_whole_document_to_job).layer(DefaultBodyLimit::disable()),
        )
        // Sidecars
        .route(
            "/ingest/sidecar",
            post(upload_sidecar).layer(DefaultBodyLimit::disable()),
        )
        .route("/ingest/sidecar/:sidecar_id", delete(delete_sidecar))
        // Status polling
        .route("/ingest/status/:item_id", get(ingest_status))
        .route("/ingest/page/status/:page_id", get(ingest_status))
        .route("/ingest/document/status/:document_id", get(ingest_status))
        .route("/ingest/status/batch", post(ingest_status_batch))
        // SSE per-job stream
        .route("/ingest/job/:job_id/events", get(ingest_job_events))
        // Internal callback (worker → gateway)
        .route("/internal/job-callback", post(job_callback))
}

// ────────────────────────── helpers ──────────────────────────

fn json_error(status: StatusCode, detail: impl Into<String>) -> Response {
    let body = json!({ "detail": detail.into() });
    (status, Json(body)).into_response()
}

fn role_label(state: &SharedState) -> &'static str {
    state.config.mode.as_str()
}

fn record_prometheus(
    state: &SharedState,
    endpoint: &str,
    status_class: &str,
    file_size: u64,
    is_page: bool,
) {
    let role = role_label(state);
    state
        .metrics
        .ingest_requests_total
        .with_label_values(&[role, endpoint, status_class])
        .inc();
    if file_size > 0 {
        state
            .metrics
            .ingest_bytes_total
            .with_label_values(&[role, endpoint])
            .inc_by(file_size);
    }
    if is_page {
        state
            .metrics
            .ingest_pages_total
            .with_label_values(&[role])
            .inc();
    } else {
        state
            .metrics
            .ingest_documents_total
            .with_label_values(&[role])
            .inc();
    }
}

fn build_callback_url(state: &SharedState) -> String {
    let port = state.config.server.port;
    if let Ok(pod_ip) = std::env::var("POD_IP") {
        format!("http://{pod_ip}:{port}/v1/internal/job-callback")
    } else {
        format!("http://localhost:{port}/v1/internal/job-callback")
    }
}

fn check_upload_size(state: &SharedState, size: u64) -> Result<(), Response> {
    let limit = state.config.resources.max_upload_bytes;
    if size > limit {
        return Err(json_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("upload size {size} exceeds limit of {limit} bytes"),
        ));
    }
    Ok(())
}

/// Drain a non-2xx upstream response into a `(detail_string, rebuilt_response)`
/// tuple. The detail string is what we record in the JobTracker so the
/// client's status query surfaces the **actual** worker error message
/// instead of a generic "Worker returned HTTP X" placeholder.
async fn drain_for_diagnostics(resp: Response) -> (String, Response) {
    let (parts, body) = resp.into_parts();
    let bytes = match axum::body::to_bytes(body, 64 * 1024).await {
        Ok(b) => b,
        Err(_) => Bytes::new(),
    };
    let preview: String = String::from_utf8_lossy(&bytes)
        .chars()
        .take(1024)
        .collect();
    let detail = if preview.is_empty() {
        format!("Worker returned HTTP {}", parts.status)
    } else {
        format!("Worker returned HTTP {}: {}", parts.status, preview)
    };
    let rebuilt = Response::from_parts(parts, axum::body::Body::from(bytes));
    (detail, rebuilt)
}

/// Routing heuristic that **never parses the PDF**.
///
/// Parsing PDFs in the gateway is a hot-path foot-gun: `lopdf` is
/// CPU-bound, sometimes spends 100+ ms on malformed inputs, and the
/// gateway pod has no business doing per-document work. We instead
/// route on cheap signals only:
///
/// 1. Single-page uploads (`page_number` set) → Realtime.
/// 2. Explicit PDF split spec → Batch (caller asked for it).
/// 3. File size threshold (`PAGE_THRESHOLD_FOR_BATCH` × ~50 KiB/page
///    rough average) → Batch for big files, Realtime otherwise.
///
/// The worker can re-route internally if it ever needs to.
fn route_by_size(file_size: usize, meta: &IngestRequest) -> PoolType {
    if meta.page_number.is_some() {
        return PoolType::Realtime;
    }
    if meta
        .pipeline
        .as_ref()
        .and_then(|s| s.pdf_split.as_ref())
        .is_some()
    {
        return PoolType::Batch;
    }
    // ~50 KiB / page is a conservative ballpark for text-heavy PDFs.
    // Anything past that is best handled by the batch pool.
    const ROUGH_BYTES_PER_PAGE: usize = 50 * 1024;
    if file_size >= (PAGE_THRESHOLD_FOR_BATCH as usize) * ROUGH_BYTES_PER_PAGE {
        PoolType::Batch
    } else {
        PoolType::Realtime
    }
}

/// Read a multipart upload from raw `body` bytes + the request's
/// `content-type` header, with NO per-field or per-stream size cap.
///
/// Axum's built-in [`axum::extract::Multipart`] uses `multer`'s
/// `Constraints::default()` which enforces an 8 MiB whole-stream and
/// 2 MiB per-field cap — far too small for typical PDF ingest. We feed
/// `multer::Multipart::with_constraints` directly with no `size_limit`
/// set so the YAML-driven `resources.max_upload_bytes` (enforced after
/// parsing) is the only ceiling.
async fn parse_upload(
    headers: &HeaderMap,
    body: Bytes,
) -> Result<(Bytes, Option<String>, Option<String>, IngestRequest, MultipartExtras), Response> {
    use futures::StreamExt;

    let boundary = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .and_then(|ct| multer::parse_boundary(ct).ok())
        .ok_or_else(|| {
            json_error(
                StatusCode::BAD_REQUEST,
                "missing or invalid multipart boundary in content-type header",
            )
        })?;

    // multer wants a Stream<Item = Result<Bytes, ...>>. The body already
    // sits in memory as a single Bytes chunk, so wrap it as a single-item
    // stream — this avoids re-allocation and keeps zero-copy semantics.
    let chunks =
        stream::once(async move { Ok::<_, std::io::Error>(body) });
    let constraints = multer::Constraints::new().allowed_fields(vec![
        "file",
        "metadata",
        "document_id",
        "page_number",
        "filename",
        "ttl_s",
        "consume_on_read",
    ]);
    let mut multipart = multer::Multipart::with_constraints(chunks, boundary, constraints);

    let mut file_bytes: Option<Bytes> = None;
    let mut filename: Option<String> = None;
    let mut content_type: Option<String> = None;
    let mut metadata: IngestRequest = IngestRequest::default();
    let mut extras = MultipartExtras::default();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| json_error(StatusCode::BAD_REQUEST, format!("multipart error: {e}")))?
    {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                filename = field.file_name().map(|s| s.to_string());
                content_type = field.content_type().map(|s| s.to_string());
                // Read the field as a stream of chunks ourselves so we
                // bypass multer's built-in per-field cap.
                let mut buf = Vec::with_capacity(64 * 1024);
                let mut field = field;
                while let Some(chunk) = field.next().await {
                    let chunk = chunk.map_err(|e| {
                        json_error(StatusCode::BAD_REQUEST, format!("read file: {e}"))
                    })?;
                    buf.extend_from_slice(&chunk);
                }
                file_bytes = Some(Bytes::from(buf));
            }
            "metadata" => {
                let raw = field
                    .text()
                    .await
                    .map_err(|e| json_error(StatusCode::BAD_REQUEST, format!("read metadata: {e}")))?;
                metadata = serde_json::from_str(&raw).map_err(|e| {
                    json_error(StatusCode::BAD_REQUEST, format!("invalid metadata JSON: {e}"))
                })?;
            }
            "document_id" => extras.document_id = field.text().await.ok(),
            "page_number" => {
                if let Ok(s) = field.text().await {
                    extras.page_number = s.parse().ok();
                }
            }
            "filename" => extras.filename_field = field.text().await.ok(),
            "ttl_s" => {
                if let Ok(s) = field.text().await {
                    extras.ttl_s = s.parse().ok();
                }
            }
            "consume_on_read" => {
                if let Ok(s) = field.text().await {
                    extras.consume_on_read = Some(s.parse().unwrap_or(true));
                }
            }
            _ => {}
        }
    }
    let bytes = file_bytes
        .ok_or_else(|| json_error(StatusCode::BAD_REQUEST, "missing 'file' field"))?;
    Ok((bytes, filename, content_type, metadata, extras))
}

#[derive(Debug, Default)]
struct MultipartExtras {
    document_id: Option<String>,
    page_number: Option<u32>,
    filename_field: Option<String>,
    ttl_s: Option<f64>,
    consume_on_read: Option<bool>,
}

fn validated_spec(state: &SharedState, meta: &IngestRequest) -> Result<Option<JsonValue>, Response> {
    let Some(spec) = &meta.pipeline else {
        return Ok(None);
    };
    let policy = state
        .config
        .pipeline_overrides
        .to_policy(state.config.nim_endpoints.caption_invoke_url.is_some());
    match validate_pipeline_spec(spec, &policy) {
        Ok(s) => Ok(Some(serde_json::to_value(s).unwrap_or(JsonValue::Null))),
        Err(err) => Err(json_error(
            StatusCode::from_u16(err.status_code()).unwrap_or(StatusCode::FORBIDDEN),
            err.to_string(),
        )),
    }
}

// ────────────────────────── job CRUD ──────────────────────────

async fn create_job(
    State(state): State<SharedState>,
    Json(body): Json<JobCreateRequest>,
) -> Response {
    let job_id = Uuid::new_v4().simple().to_string();
    match state.job_tracker.register_job(
        &job_id,
        body.expected_documents,
        body.label.clone(),
        body.metadata,
    ) {
        Ok(agg) => (
            StatusCode::CREATED,
            Json(JobCreatedResponse {
                job_id: agg.job_id,
                expected_documents: agg.expected_documents,
                status: agg.status.as_str().into(),
                created_at: agg.created_at,
                label: agg.label,
            }),
        )
            .into_response(),
        Err(err) => json_error(
            StatusCode::from_u16(err.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            err.to_string(),
        ),
    }
}

#[derive(Debug, Deserialize, Default)]
struct GetJobQuery {
    #[serde(default)]
    include_documents: bool,
}

async fn get_job(
    State(state): State<SharedState>,
    Path(job_id): Path<String>,
    Query(q): Query<GetJobQuery>,
) -> Response {
    let Some(agg) = state.job_tracker.get_job(&job_id) else {
        return json_error(StatusCode::NOT_FOUND, format!("Job {job_id:?} not found"));
    };
    let documents: Option<Vec<JsonValue>> = if q.include_documents {
        let docs = state.job_tracker.job_documents(&job_id);
        let cap = 10_000;
        Some(
            docs.into_iter()
                .take(cap)
                .map(|d| serde_json::to_value(d).unwrap_or(JsonValue::Null))
                .collect(),
        )
    } else {
        None
    };
    Json(JobAggregateResponse {
        job_id: agg.job_id,
        expected_documents: agg.expected_documents,
        status: agg.status.as_str().into(),
        created_at: agg.created_at,
        started_at: agg.started_at,
        finalized_at: agg.finalized_at,
        elapsed_s: agg.elapsed_s,
        label: agg.label,
        counts: agg.counts,
        document_ids: agg.document_ids,
        documents,
    })
    .into_response()
}

#[derive(Debug, Deserialize, Default)]
struct DocsQuery {
    status: Option<String>,
    #[serde(default)]
    offset: u64,
    #[serde(default = "default_limit")]
    limit: u64,
}

fn default_limit() -> u64 {
    100
}

async fn get_job_documents(
    State(state): State<SharedState>,
    Path(job_id): Path<String>,
    Query(q): Query<DocsQuery>,
) -> Response {
    if !(1..=1000).contains(&q.limit) {
        return json_error(StatusCode::BAD_REQUEST, "limit must be in [1, 1000]");
    }
    let Some(agg) = state.job_tracker.get_job(&job_id) else {
        return json_error(StatusCode::NOT_FOUND, format!("Job {job_id:?} not found"));
    };
    let docs = state.job_tracker.job_documents(&job_id);
    let total = docs.len() as u64;
    let filtered: Vec<DocumentRecord> = if let Some(s) = q.status.as_deref() {
        let valid = ["pending", "processing", "completed", "failed"];
        if !valid.contains(&s) {
            return json_error(
                StatusCode::BAD_REQUEST,
                format!("status must be one of {valid:?}, got {s:?}"),
            );
        }
        docs.into_iter().filter(|d| d.status.as_str() == s).collect()
    } else {
        docs
    };
    let total_filtered = filtered.len() as u64;
    let page: Vec<DocumentRecord> = filtered
        .into_iter()
        .skip(q.offset as usize)
        .take(q.limit as usize)
        .collect();
    let items: Vec<DocumentStatusResponse> = page
        .into_iter()
        .map(|d| record_to_response(&d, None))
        .collect();
    Json(JobDocumentsPage {
        job_id: agg.job_id,
        total,
        total_filtered,
        offset: q.offset,
        limit: q.limit,
        items,
    })
    .into_response()
}

async fn get_job_document(
    State(state): State<SharedState>,
    Path((job_id, document_id)): Path<(String, String)>,
) -> Response {
    if state.job_tracker.get_job(&job_id).is_none() {
        return json_error(StatusCode::NOT_FOUND, format!("Job {job_id:?} not found"));
    }
    let Some(rec) = state.job_tracker.get_document(&document_id) else {
        return json_error(
            StatusCode::NOT_FOUND,
            format!("Document {document_id:?} not found in job {job_id:?}"),
        );
    };
    if rec.job_id != job_id {
        return json_error(
            StatusCode::NOT_FOUND,
            format!("Document {document_id:?} not found in job {job_id:?}"),
        );
    }
    let is_terminal = rec.status.is_terminal();
    let result_data = if is_terminal {
        state.job_tracker.consume_result_data(&document_id)
    } else {
        None
    };
    let body = record_to_response(&rec, result_data);
    let status = if is_terminal {
        StatusCode::OK
    } else {
        StatusCode::ACCEPTED
    };
    (status, Json(body)).into_response()
}

fn record_to_response(
    rec: &DocumentRecord,
    result_data: Option<Vec<JsonValue>>,
) -> DocumentStatusResponse {
    DocumentStatusResponse {
        document_id: rec.id.clone(),
        job_id: rec.job_id.clone(),
        status: rec.status.as_str().into(),
        submitted_at: rec.submitted_at.clone(),
        started_at: rec.started_at.clone(),
        completed_at: rec.completed_at.clone(),
        elapsed_s: rec.elapsed_s,
        filename: rec.filename.clone(),
        result_rows: rec.result_rows,
        result_data: result_data.or_else(|| rec.result_data.clone()),
        error: rec.error.clone(),
    }
}

// ────────────────────────── upload paths ──────────────────────────

async fn submit_document_to_job(
    State(state): State<SharedState>,
    Path(job_id): Path<String>,
    headers: HeaderMap,
    body: AxumBytes,
) -> Response {
    // Hot-path: in gateway mode we deliberately do NOT parse the multipart
    // body. Parsing PDFs / re-encoding uploads on the gateway turns it into
    // a CPU bottleneck and starves the async runtime, which surfaces as
    // `httpx.ReadError` on the client side. Forward the raw body verbatim.
    // `AxumBytes` is just `bytes::Bytes` re-exported, so this is a refcount
    // bump — no allocation.
    let raw_body: Bytes = body;

    if let Err(r) = check_upload_size(&state, raw_body.len() as u64) {
        return r;
    }
    if !state.config.mode.is_worker() && state.job_tracker.get_job(&job_id).is_none() {
        return json_error(StatusCode::NOT_FOUND, format!("Job {job_id:?} not found"));
    }

    let document_id = headers
        .get(GATEWAY_DOC_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| Uuid::new_v4().simple().to_string());
    let gw_callback = headers
        .get(GATEWAY_CALLBACK_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let gw_job_id = headers
        .get(GATEWAY_JOB_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| job_id.clone());
    let content_sha256 = sha256_hex(&raw_body);
    let now = utc_now_iso();

    if state.config.mode.is_gateway() {
        // Cheap routing decision based on raw body size only.
        let route = route_by_size(raw_body.len(), &IngestRequest::default());

        let _ = state
            .job_tracker
            .register_document(&document_id, &job_id, None);
        state.job_tracker.mark_processing(&document_id);

        let Some(proxy) = &state.gateway_proxy else {
            return json_error(StatusCode::SERVICE_UNAVAILABLE, "Gateway proxy not initialised");
        };
        let mut extra = HeaderMap::new();
        extra.insert(
            HeaderName::from_static("x-gateway-document-id"),
            HeaderValue::from_str(&document_id).unwrap(),
        );
        extra.insert(
            HeaderName::from_static("x-gateway-job-id"),
            HeaderValue::from_str(&job_id).unwrap(),
        );
        let cb = build_callback_url(&state);
        extra.insert(
            HeaderName::from_static("x-gateway-callback-url"),
            HeaderValue::from_str(&cb).unwrap(),
        );

        // Forward original Content-Type (with the client's boundary) and
        // the original raw body verbatim. No re-encoding, no allocation
        // beyond a single Bytes clone (refcounted).
        let path = format!("/v1/ingest/job/{job_id}/document");
        let mut fwd_headers = HeaderMap::new();
        if let Some(ct) = headers.get("content-type") {
            fwd_headers.insert(HeaderName::from_static("content-type"), ct.clone());
        }
        let resp = proxy
            .forward(Method::POST, &path, &fwd_headers, raw_body.clone(), route, Some(&extra))
            .await;
        if !resp.status().is_success() {
            let (detail, rebuilt) = drain_for_diagnostics(resp).await;
            state.job_tracker.mark_failed(&document_id, detail, None);
            return rebuilt;
        }
        record_prometheus(
            &state,
            "/v1/ingest/job/document",
            "2xx",
            raw_body.len() as u64,
            false,
        );
        return Json(IngestAccepted {
            document_id,
            job_id: Some(job_id),
            content_sha256,
            status: "accepted".into(),
            created_at: now,
        })
        .into_response();
    }

    // Worker / standalone: parse the multipart and submit to the pool.
    let (file_bytes, filename, content_type, meta, _extras) =
        match parse_upload(&headers, raw_body).await {
            Ok(v) => v,
            Err(r) => return r,
        };
    let validated = match validated_spec(&state, &meta) {
        Ok(v) => v,
        Err(r) => return r,
    };
    let route = route_by_size(file_bytes.len(), &meta);

    // ── worker / standalone ──
    let _ = classify_file(filename.as_deref().unwrap_or(""), content_type.as_deref());
    if gw_callback.is_none() {
        let _ = state
            .job_tracker
            .register_document(&document_id, &gw_job_id, filename.clone());
    }
    let item = WorkItem {
        id: document_id.clone(),
        payload: file_bytes.clone(),
        filename: filename.clone(),
        callback_url: gw_callback.clone(),
        job_id: Some(gw_job_id.clone()),
        pipeline_spec: validated,
    };
    let Some(pool) = &state.pipeline_pool else {
        return json_error(StatusCode::SERVICE_UNAVAILABLE, "Pipeline pool not initialised");
    };
    if !pool.submit(route, item).await {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            [(HeaderName::from_static("retry-after"), "5")],
            Json(json!({ "detail": "pool is at capacity" })),
        )
            .into_response();
    }
    record_prometheus(
        &state,
        "/v1/ingest/job/document",
        "2xx",
        file_bytes.len() as u64,
        false,
    );
    Json(IngestAccepted {
        document_id,
        job_id: Some(gw_job_id),
        content_sha256,
        status: "accepted".into(),
        created_at: now,
    })
    .into_response()
}

async fn submit_page_to_job(
    State(state): State<SharedState>,
    Path(job_id): Path<String>,
    headers: HeaderMap,
    body: AxumBytes,
) -> Response {
    let raw_body: Bytes = body;

    if let Err(r) = check_upload_size(&state, raw_body.len() as u64) {
        return r;
    }
    if !state.config.mode.is_worker() && state.job_tracker.get_job(&job_id).is_none() {
        return json_error(StatusCode::NOT_FOUND, format!("Job {job_id:?} not found"));
    }

    let page_id = headers
        .get(GATEWAY_DOC_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| Uuid::new_v4().simple().to_string());
    let content_sha256 = sha256_hex(&raw_body);
    let now = utc_now_iso();
    let gw_callback = headers
        .get(GATEWAY_CALLBACK_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let gw_job_id = headers
        .get(GATEWAY_JOB_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| job_id.clone());

    if state.config.mode.is_gateway() {
        // Pages always go to realtime. Forward verbatim — no parsing.
        let _ = state.job_tracker.register_document(&page_id, &job_id, None);
        state.job_tracker.mark_processing(&page_id);

        let Some(proxy) = &state.gateway_proxy else {
            return json_error(StatusCode::SERVICE_UNAVAILABLE, "Gateway proxy not initialised");
        };
        let mut extra = HeaderMap::new();
        extra.insert(
            HeaderName::from_static("x-gateway-document-id"),
            HeaderValue::from_str(&page_id).unwrap(),
        );
        extra.insert(
            HeaderName::from_static("x-gateway-job-id"),
            HeaderValue::from_str(&job_id).unwrap(),
        );
        let cb = build_callback_url(&state);
        extra.insert(
            HeaderName::from_static("x-gateway-callback-url"),
            HeaderValue::from_str(&cb).unwrap(),
        );
        let path = format!("/v1/ingest/job/{job_id}/page");
        let mut fwd_headers = HeaderMap::new();
        if let Some(ct) = headers.get("content-type") {
            fwd_headers.insert(HeaderName::from_static("content-type"), ct.clone());
        }
        let resp = proxy
            .forward(
                Method::POST,
                &path,
                &fwd_headers,
                raw_body.clone(),
                PoolType::Realtime,
                Some(&extra),
            )
            .await;
        if !resp.status().is_success() {
            let (detail, rebuilt) = drain_for_diagnostics(resp).await;
            state.job_tracker.mark_failed(&page_id, detail, None);
            return rebuilt;
        }
        // Pages need a structured response — the worker's response IS that
        // structured response, so just pass it through.
        record_prometheus(&state, "/v1/ingest/job/page", "2xx", raw_body.len() as u64, true);
        return resp;
    }

    // Worker / standalone: parse and submit to the realtime pool.
    let (file_bytes, filename, _content_type, _meta, extras) =
        match parse_upload(&headers, raw_body).await {
            Ok(v) => v,
            Err(r) => return r,
        };
    let document_id = match extras
        .document_id
        .ok_or_else(|| json_error(StatusCode::BAD_REQUEST, "missing 'document_id' field"))
    {
        Ok(v) => v,
        Err(r) => return r,
    };
    let page_number = match extras
        .page_number
        .ok_or_else(|| json_error(StatusCode::BAD_REQUEST, "missing 'page_number' field"))
    {
        Ok(v) => v,
        Err(r) => return r,
    };
    if gw_callback.is_none() {
        let _ = state.job_tracker.register_document(
            &page_id,
            &gw_job_id,
            extras.filename_field.clone().or(filename.clone()),
        );
    }
    let item = WorkItem {
        id: page_id.clone(),
        payload: file_bytes.clone(),
        filename: filename.clone(),
        callback_url: gw_callback.clone(),
        job_id: Some(gw_job_id.clone()),
        pipeline_spec: None,
    };
    let Some(pool) = &state.pipeline_pool else {
        return json_error(StatusCode::SERVICE_UNAVAILABLE, "Pipeline pool not initialised");
    };
    if !pool.submit(PoolType::Realtime, item).await {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({ "detail": "realtime pool at capacity" })),
        )
            .into_response();
    }

    record_prometheus(
        &state,
        "/v1/ingest/job/page",
        "2xx",
        file_bytes.len() as u64,
        true,
    );

    Json(PageIngestAccepted {
        page_id,
        document_id,
        page_number,
        content_sha256,
        status: "accepted".into(),
        created_at: now,
    })
    .into_response()
}

async fn submit_whole_document_to_job(
    State(state): State<SharedState>,
    Path(job_id): Path<String>,
    headers: HeaderMap,
    body: AxumBytes,
) -> Response {
    let raw_body: Bytes = body;

    if let Err(r) = check_upload_size(&state, raw_body.len() as u64) {
        return r;
    }
    if !state.config.mode.is_worker() && state.job_tracker.get_job(&job_id).is_none() {
        return json_error(StatusCode::NOT_FOUND, format!("Job {job_id:?} not found"));
    }

    let document_id = headers
        .get(GATEWAY_DOC_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| Uuid::new_v4().simple().to_string());
    let gw_callback = headers
        .get(GATEWAY_CALLBACK_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let gw_job_id = headers
        .get(GATEWAY_JOB_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| job_id.clone());
    let content_sha256 = sha256_hex(&raw_body);
    let now = utc_now_iso();

    if state.config.mode.is_gateway() {
        // Whole-document → batch pool. Forward verbatim.
        let _ = state
            .job_tracker
            .register_document(&document_id, &job_id, None);
        state.job_tracker.mark_processing(&document_id);

        let Some(proxy) = &state.gateway_proxy else {
            return json_error(StatusCode::SERVICE_UNAVAILABLE, "Gateway proxy not initialised");
        };
        let mut extra = HeaderMap::new();
        extra.insert(
            HeaderName::from_static("x-gateway-document-id"),
            HeaderValue::from_str(&document_id).unwrap(),
        );
        extra.insert(
            HeaderName::from_static("x-gateway-job-id"),
            HeaderValue::from_str(&job_id).unwrap(),
        );
        let cb = build_callback_url(&state);
        extra.insert(
            HeaderName::from_static("x-gateway-callback-url"),
            HeaderValue::from_str(&cb).unwrap(),
        );
        let path = format!("/v1/ingest/job/{job_id}/whole");
        let mut fwd_headers = HeaderMap::new();
        if let Some(ct) = headers.get("content-type") {
            fwd_headers.insert(HeaderName::from_static("content-type"), ct.clone());
        }
        let resp = proxy
            .forward(
                Method::POST,
                &path,
                &fwd_headers,
                raw_body.clone(),
                PoolType::Batch,
                Some(&extra),
            )
            .await;
        if !resp.status().is_success() {
            let (detail, rebuilt) = drain_for_diagnostics(resp).await;
            state.job_tracker.mark_failed(&document_id, detail, None);
            return rebuilt;
        }
        record_prometheus(&state, "/v1/ingest/job/whole", "2xx", raw_body.len() as u64, false);
        return resp;
    }

    // Worker / standalone: parse and submit.
    let (file_bytes, filename, content_type, meta, _extras) =
        match parse_upload(&headers, raw_body).await {
            Ok(v) => v,
            Err(r) => return r,
        };
    let validated = match validated_spec(&state, &meta) {
        Ok(v) => v,
        Err(r) => return r,
    };
    let classification = classify_file(
        filename.as_deref().unwrap_or(""),
        content_type.as_deref(),
    );

    if gw_callback.is_none() {
        let _ = state
            .job_tracker
            .register_document(&document_id, &gw_job_id, filename.clone());
    }
    let item = WorkItem {
        id: document_id.clone(),
        payload: file_bytes.clone(),
        filename: filename.clone(),
        callback_url: gw_callback.clone(),
        job_id: Some(gw_job_id.clone()),
        pipeline_spec: validated,
    };
    let Some(pool) = &state.pipeline_pool else {
        return json_error(StatusCode::SERVICE_UNAVAILABLE, "Pipeline pool not initialised");
    };
    if !pool.submit(PoolType::Batch, item).await {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({ "detail": "batch pool at capacity" })),
        )
            .into_response();
    }

    record_prometheus(
        &state,
        "/v1/ingest/job/whole",
        "2xx",
        file_bytes.len() as u64,
        false,
    );

    Json(DocumentIngestAccepted {
        document_id,
        filename: classification.filename,
        file_size_bytes: file_bytes.len() as u64,
        content_sha256,
        status: "accepted".into(),
        created_at: now,
    })
    .into_response()
}

// ────────────────────────── status ──────────────────────────

async fn ingest_status(
    State(state): State<SharedState>,
    Path(item_id): Path<String>,
) -> Response {
    let Some(rec) = state.job_tracker.get_document(&item_id) else {
        return json_error(
            StatusCode::NOT_FOUND,
            format!("No tracked document with id={item_id:?}"),
        );
    };
    let is_terminal = rec.status.is_terminal();
    let result_data = if is_terminal {
        state.job_tracker.consume_result_data(&item_id)
    } else {
        None
    };
    let body = JobStatusResponse {
        id: rec.id.clone(),
        status: rec.status.as_str().into(),
        submitted_at: rec.submitted_at.clone(),
        started_at: rec.started_at.clone(),
        completed_at: rec.completed_at.clone(),
        elapsed_s: rec.elapsed_s,
        result_rows: rec.result_rows,
        result_data,
        error: rec.error.clone(),
    };
    let status = if is_terminal {
        StatusCode::OK
    } else {
        StatusCode::ACCEPTED
    };
    (status, Json(body)).into_response()
}

#[derive(Debug, Deserialize)]
struct StatusBatchRequest {
    ids: Vec<String>,
}

async fn ingest_status_batch(
    State(state): State<SharedState>,
    Json(body): Json<StatusBatchRequest>,
) -> Response {
    const MAX: usize = 1000;
    if body.ids.len() > MAX {
        return json_error(
            StatusCode::BAD_REQUEST,
            format!("too many ids ({}); max {MAX}", body.ids.len()),
        );
    }
    let mut items = serde_json::Map::with_capacity(body.ids.len());
    let mut terminal = 0u64;
    for id in &body.ids {
        match state.job_tracker.get_document(id) {
            Some(rec) => {
                let s = rec.status.as_str();
                if rec.status.is_terminal() {
                    terminal += 1;
                }
                items.insert(
                    id.clone(),
                    json!({
                        "status": s,
                        "job_id": rec.job_id,
                        "result_rows": rec.result_rows,
                        "elapsed_s": rec.elapsed_s,
                        "error": rec.error,
                    }),
                );
            }
            None => {
                items.insert(id.clone(), json!({ "status": "unknown" }));
            }
        }
    }
    Json(json!({
        "total": body.ids.len(),
        "terminal": terminal,
        "pending": body.ids.len() as u64 - terminal,
        "items": items,
    }))
    .into_response()
}

// ────────────────────────── sidecars ──────────────────────────

async fn upload_sidecar(
    State(state): State<SharedState>,
    headers: HeaderMap,
    body: AxumBytes,
) -> Response {
    let (file_bytes, filename, content_type, _meta, extras) =
        match parse_upload(&headers, body).await {
            Ok(v) => v,
            Err(r) => return r,
        };
    if let Err(r) = check_upload_size(&state, file_bytes.len() as u64) {
        return r;
    }
    if file_bytes.is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "Sidecar upload is empty");
    }
    let ttl = extras.ttl_s.unwrap_or(3600.0);
    let consume = extras.consume_on_read.unwrap_or(true);
    let entry = match state.sidecar_store.put(
        filename.clone().unwrap_or_else(|| "sidecar".into()),
        content_type.unwrap_or_else(|| "application/octet-stream".into()),
        file_bytes,
        None,
        ttl,
        consume,
    ) {
        Ok(e) => e,
        Err(err) => return json_error(StatusCode::TOO_MANY_REQUESTS, err.to_string()),
    };
    let body = SidecarUploadResponse {
        sidecar_id: entry.sidecar_id,
        filename: entry.filename,
        content_type: entry.content_type,
        size_bytes: entry.payload.len() as u64,
        expires_at: nrr_core::sidecar::SidecarStore::expires_at_iso(entry.expires_at),
    };
    (StatusCode::CREATED, Json(body)).into_response()
}

async fn delete_sidecar(
    State(state): State<SharedState>,
    Path(sidecar_id): Path<String>,
) -> Response {
    state.sidecar_store.delete(&sidecar_id);
    StatusCode::NO_CONTENT.into_response()
}

// ────────────────────────── internal callback ──────────────────────────

#[derive(Debug, Deserialize)]
struct CallbackBody {
    id: String,
    #[serde(default = "default_completed")]
    status: String,
    #[serde(default)]
    result_rows: Option<u64>,
    #[serde(default)]
    result_data: Option<Vec<JsonValue>>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    elapsed_s: Option<f64>,
}

fn default_completed() -> String {
    "completed".into()
}

async fn job_callback(
    State(state): State<SharedState>,
    Json(body): Json<CallbackBody>,
) -> Response {
    if body.status == "failed" {
        state.job_tracker.mark_failed(
            &body.id,
            body.error.unwrap_or_else(|| "unknown error".into()),
            body.elapsed_s,
        );
    } else {
        state.job_tracker.mark_completed(
            &body.id,
            body.result_rows.unwrap_or(0),
            body.result_data,
            body.elapsed_s,
        );
    }
    Json(json!({ "ok": true })).into_response()
}

// ────────────────────────── SSE ──────────────────────────

async fn ingest_job_events(
    State(state): State<SharedState>,
    Path(job_id): Path<String>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, Response> {
    if state.job_tracker.get_job(&job_id).is_none() {
        return Err(json_error(
            StatusCode::NOT_FOUND,
            format!("Job {job_id:?} not found"),
        ));
    }
    let bus = state.event_bus.clone();
    let tracker = state.job_tracker.clone();
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);

    // Catch-up: re-emit terminal documents already in the tracker, then
    // pump live events for as long as the subscriber stays connected.
    let job_id_clone = job_id.clone();
    tokio::spawn(async move {
        for rec in tracker.job_documents(&job_id_clone) {
            if rec.status.is_terminal() {
                let payload = serde_json::to_string(&json!({
                    "type": rec.status.as_str(),
                    "id": rec.id,
                    "job_id": rec.job_id,
                    "status": rec.status.as_str(),
                    "result_rows": rec.result_rows,
                    "elapsed_s": rec.elapsed_s,
                    "error": rec.error,
                }))
                .unwrap_or_default();
                let _ = tx
                    .send(Ok(Event::default().event(rec.status.as_str()).data(payload)))
                    .await;
            }
        }
        let mut sub = bus.subscribe(Some(job_id_clone));
        loop {
            match sub.recv().await {
                Ok(Some(ev)) => {
                    let event_name = ev
                        .get("type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("status")
                        .to_string();
                    let _ = tx
                        .send(Ok(Event::default().event(event_name).data(ev.to_string())))
                        .await;
                }
                Ok(None) => break,
                Err(_) => break,
            }
        }
    });

    let stream = ReceiverStream::new(rx);
    Ok(Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(30))))
}

// (multipart re-encoder removed — gateway forwards raw bodies verbatim)

// Suppress unused-warning for the `Request` import (axum re-exports it via
// the prelude in some examples; we keep it for future passthrough).
#[allow(dead_code)]
fn _unused(_: Request) {}
