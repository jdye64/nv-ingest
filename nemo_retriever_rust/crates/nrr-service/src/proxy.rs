// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Gateway-mode HTTP proxy.
//!
//! Mirrors `service.services.proxy.GatewayProxy`. In gateway mode the
//! Rust pod has no local pipeline pool — every ingest request is
//! forwarded to the realtime or batch backend Service over HTTP.
//!
//! Concurrency model:
//!
//! * One long-lived `reqwest::Client` per backend so connection pools stay
//!   warm across requests.
//! * A per-pool [`tokio::sync::Semaphore`] caps concurrent in-flight
//!   forwards. This mirrors `httpx.Limits(max_connections=...)` in the
//!   Python proxy and prevents the gateway from opening unbounded TCP
//!   connections to a worker pod when a burst of client uploads arrives
//!   faster than workers can drain them.
//! * Bodies are passed to reqwest as zero-copy [`bytes::Bytes`] —
//!   reqwest implements `From<Bytes>` for its `Body` type, so we avoid
//!   the `Vec<u8>` allocation per request.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode};
use axum::response::Response;
use bytes::Bytes;
use reqwest::Client;
use serde::Serialize;
use tokio::sync::Semaphore;

use nrr_core::config::GatewayConfig;
use nrr_pipeline::PoolType;

#[derive(Clone)]
pub struct GatewayProxy {
    inner: Arc<ProxyInner>,
}

struct ProxyInner {
    realtime: Client,
    batch: Client,
    realtime_url: String,
    batch_url: String,
    realtime_sem: Arc<Semaphore>,
    batch_sem: Arc<Semaphore>,
    timeout_s: f64,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    detail: &'a str,
    gateway_error: bool,
    status_code: u16,
}

impl GatewayProxy {
    pub fn new(cfg: &GatewayConfig) -> Result<Self, reqwest::Error> {
        let make = |timeout_s: f64| -> Result<Client, reqwest::Error> {
            reqwest::Client::builder()
                .timeout(Duration::from_secs_f64(timeout_s))
                // Idle pool keeps warm sockets between bursts. Total in-flight
                // is bounded by the semaphore below, not by reqwest itself.
                .pool_max_idle_per_host(cfg.max_connections as usize)
                .pool_idle_timeout(Some(Duration::from_secs(90)))
                .tcp_keepalive(Duration::from_secs(60))
                // h2 keep-alive settings are inert for plaintext h1 backends.
                // Left in so they Just Work if backends are ever served over
                // TLS (where ALPN may negotiate h2).
                .http2_keep_alive_interval(Duration::from_secs(30))
                .http2_keep_alive_timeout(Duration::from_secs(10))
                .build()
        };
        let realtime = make(cfg.timeout_s)?;
        let batch = make(cfg.timeout_s)?;
        let max_conns = cfg.max_connections.max(1) as usize;
        tracing::info!(
            realtime = %cfg.realtime_url,
            batch = %cfg.batch_url,
            max_in_flight = max_conns,
            timeout_s = cfg.timeout_s,
            "gateway proxy initialised"
        );
        Ok(Self {
            inner: Arc::new(ProxyInner {
                realtime,
                batch,
                realtime_url: cfg.realtime_url.clone(),
                batch_url: cfg.batch_url.clone(),
                realtime_sem: Arc::new(Semaphore::new(max_conns)),
                batch_sem: Arc::new(Semaphore::new(max_conns)),
                timeout_s: cfg.timeout_s,
            }),
        })
    }

    fn select(&self, pool: PoolType) -> (&Client, &str, Arc<Semaphore>) {
        match pool {
            PoolType::Realtime => (
                &self.inner.realtime,
                &self.inner.realtime_url,
                Arc::clone(&self.inner.realtime_sem),
            ),
            PoolType::Batch => (
                &self.inner.batch,
                &self.inner.batch_url,
                Arc::clone(&self.inner.batch_sem),
            ),
        }
    }

    pub async fn forward(
        &self,
        method: Method,
        path: &str,
        headers: &HeaderMap,
        body: Bytes,
        pool: PoolType,
        extra_headers: Option<&HeaderMap>,
    ) -> Response {
        let (client, base_url, sem) = self.select(pool);
        let target = format!("{}{}", base_url.trim_end_matches('/'), path);

        // Hard cap on concurrent in-flight requests per backend (mirrors
        // httpx.Limits(max_connections=...)). When saturated this awaits
        // until a slot frees — preferable to opening unbounded sockets.
        let _permit = match sem.acquire_owned().await {
            Ok(p) => p,
            Err(_) => {
                return error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "gateway proxy semaphore closed",
                );
            }
        };

        // Zero-copy body handoff: reqwest's `Body` impls `From<Bytes>`.
        let mut req = client
            .request(reqwest_method(method.clone()), &target)
            .body(reqwest::Body::from(body));
        for (k, v) in headers.iter() {
            let kn = k.as_str().to_ascii_lowercase();
            if matches!(kn.as_str(), "host" | "transfer-encoding" | "content-length") {
                continue;
            }
            req = req.header(k.as_str(), v.as_bytes());
        }
        if let Some(h) = extra_headers {
            for (k, v) in h.iter() {
                req = req.header(k.as_str(), v.as_bytes());
            }
        }
        let resp = match req.send().await {
            Ok(r) => r,
            Err(err) => {
                let causes = source_chain(&err);
                let kind = classify_reqwest_error(&err);
                tracing::error!(
                    target = %target,
                    error = %err,
                    causes = ?causes,
                    kind,
                    "gateway forward failed"
                );
                let status = if err.is_timeout() {
                    StatusCode::GATEWAY_TIMEOUT
                } else {
                    StatusCode::BAD_GATEWAY
                };
                let detail = if causes.is_empty() {
                    format!("Gateway forward to {} failed: {} ({})", pool.as_str(), err, kind)
                } else {
                    format!(
                        "Gateway forward to {} failed: {} ({}) [caused by: {}]",
                        pool.as_str(),
                        err,
                        kind,
                        causes.join(" → ")
                    )
                };
                return error_response(status, &detail);
            }
        };
        let backend_status = resp.status();
        let backend_headers = resp.headers().clone();
        let bytes = match resp.bytes().await {
            Ok(b) => b,
            Err(err) => {
                let causes = source_chain(&err);
                tracing::error!(
                    target = %target,
                    error = %err,
                    causes = ?causes,
                    "failed to drain backend body"
                );
                return error_response(
                    StatusCode::BAD_GATEWAY,
                    &format!("backend body read failed: {err}"),
                );
            }
        };
        if !backend_status.is_success() {
            let preview = String::from_utf8_lossy(&bytes)
                .chars()
                .take(512)
                .collect::<String>();
            tracing::warn!(
                target = %target,
                status = backend_status.as_u16(),
                body = %preview,
                "backend returned non-2xx"
            );
        }
        let mut builder = Response::builder().status(backend_status.as_u16());
        for (k, v) in backend_headers.iter() {
            let kn = k.as_str().to_ascii_lowercase();
            if matches!(
                kn.as_str(),
                "transfer-encoding" | "content-encoding" | "content-length"
            ) {
                continue;
            }
            builder = builder.header(k.as_str(), v.as_bytes());
        }
        builder.body(Body::from(bytes)).unwrap()
    }

    pub async fn check_backend(&self, pool: PoolType) -> serde_json::Value {
        let (client, base_url, _sem) = self.select(pool);
        let url = format!("{}/v1/health", base_url.trim_end_matches('/'));
        match client
            .get(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
        {
            Ok(r) => serde_json::json!({ "status": "ok", "code": r.status().as_u16() }),
            Err(err) => serde_json::json!({
                "status": "unreachable",
                "error": err.to_string(),
                "causes": source_chain(&err),
            }),
        }
    }

    /// Per-pool semaphore stats, useful for debugging "why is the gateway
    /// stalling" issues.
    #[allow(dead_code)]
    pub fn pool_pressure(&self, pool: PoolType) -> serde_json::Value {
        let (_, _, sem) = self.select(pool);
        serde_json::json!({
            "available_permits": sem.available_permits(),
        })
    }

    /// Returns the configured per-request timeout (seconds). Exposed so
    /// other code paths can include it in error messages without baking
    /// it into the proxy's own log lines.
    #[allow(dead_code)]
    pub fn timeout_s(&self) -> f64 {
        self.inner.timeout_s
    }
}

fn reqwest_method(m: Method) -> reqwest::Method {
    reqwest::Method::from_bytes(m.as_str().as_bytes()).unwrap_or(reqwest::Method::POST)
}

/// Walk an error's `source()` chain into a Vec of strings. Useful because
/// reqwest's top-level Display is generic ("error sending request for url
/// (...)") and the actual root cause (DNS, ECONNRESET, h2 GOAWAY, etc.) is
/// only visible when you traverse the chain.
fn source_chain<E: std::error::Error + 'static>(err: &E) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut src: Option<&dyn std::error::Error> = err.source();
    while let Some(s) = src {
        out.push(s.to_string());
        src = s.source();
    }
    out
}

/// Coarse classification of a reqwest error so log lines and HTTP error
/// bodies are immediately greppable.
fn classify_reqwest_error(err: &reqwest::Error) -> &'static str {
    if err.is_timeout() {
        "timeout"
    } else if err.is_connect() {
        "connect"
    } else if err.is_request() {
        "request_build"
    } else if err.is_body() {
        "body"
    } else if err.is_decode() {
        "decode"
    } else if err.is_redirect() {
        "redirect"
    } else if err.is_status() {
        "status"
    } else {
        "other"
    }
}

fn error_response(status: StatusCode, detail: &str) -> Response {
    let body = serde_json::to_vec(&ErrorBody {
        detail,
        gateway_error: true,
        status_code: status.as_u16(),
    })
    .unwrap_or_else(|_| b"{}".to_vec());
    Response::builder()
        .status(status)
        .header("content-type", HeaderValue::from_static("application/json"))
        .body(Body::from(body))
        .unwrap()
}
