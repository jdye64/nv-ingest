// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Top-level Axum application factory.
//!
//! Mirrors `service.app.create_app`: builds shared state, wires up the
//! routers, applies middleware (request id, optional bearer auth), and
//! exposes a `/v1/health` probe.

use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::middleware;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::{Json, Router};
use serde_json::json;
use tower_http::request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer};
use tower_http::trace::TraceLayer;

use nrr_core::bus::EventBus;
use nrr_core::config::ServiceConfig;
use nrr_core::sidecar::SidecarStore;
use nrr_core::tracker::JobTracker;
use nrr_pipeline::{PipelinePool, PoolType, WorkFn};

use crate::auth::require_bearer;
use crate::metrics::Metrics;
use crate::proxy::GatewayProxy;
use crate::routes;
use crate::state::{AppState, SharedState};

/// Inputs the caller assembles before building the Axum app. Splitting
/// pipeline construction from `build_app` keeps the service crate free of
/// any executor-specific code (`InMemoryExecutor` lives in `nrr-pipeline`).
pub struct AppConfig {
    pub config: Arc<ServiceConfig>,
    pub realtime_work_fn: Option<Arc<dyn WorkFn>>,
    pub batch_work_fn: Option<Arc<dyn WorkFn>>,
}

pub async fn build_app(input: AppConfig) -> (Router, SharedState) {
    let cfg = input.config.clone();
    let mode = cfg.mode;
    let metrics = Metrics::shared();
    let event_bus = EventBus::new();
    let job_tracker = JobTracker::new();
    job_tracker.set_event_bus(event_bus.clone());
    let sidecar_store = SidecarStore::new();
    let http_client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .pool_max_idle_per_host(64)
        .pool_idle_timeout(Some(Duration::from_secs(90)))
        .tcp_keepalive(Duration::from_secs(60))
        .build()
        .expect("reqwest client");

    let pipeline_pool: Option<Arc<PipelinePool>> = if mode.is_gateway() {
        None
    } else {
        let pool = PipelinePool::new(
            &cfg.pipeline,
            mode,
            input.realtime_work_fn,
            input.batch_work_fn,
            Some(job_tracker.clone()),
            http_client.clone(),
        );
        Some(Arc::new(pool))
    };

    let gateway_proxy = if mode.is_gateway() {
        Some(
            GatewayProxy::new(&cfg.gateway)
                .expect("gateway proxy build (reqwest config)"),
        )
    } else {
        None
    };

    let state: SharedState = Arc::new(AppState {
        config: cfg.clone(),
        job_tracker,
        event_bus,
        sidecar_store,
        pipeline_pool,
        gateway_proxy,
        metrics,
        http_client,
    });

    // Update pool capacity gauges so the HPA has data immediately.
    if let Some(pool) = state.pipeline_pool.as_ref() {
        let stats = pool.stats();
        if let Some(rt) = stats.get("realtime") {
            if let Some(n) = rt.get("max_queue_size").and_then(|v| v.as_u64()) {
                state
                    .metrics
                    .pool_max_queue_size
                    .with_label_values(&[PoolType::Realtime.as_str()])
                    .set(n as i64);
            }
        }
        if let Some(bt) = stats.get("batch") {
            if let Some(n) = bt.get("max_queue_size").and_then(|v| v.as_u64()) {
                state
                    .metrics
                    .pool_max_queue_size
                    .with_label_values(&[PoolType::Batch.as_str()])
                    .set(n as i64);
            }
        }
        state
            .metrics
            .pool_workers
            .with_label_values(&[PoolType::Realtime.as_str()])
            .set(cfg.pipeline.realtime_workers as i64);
        state
            .metrics
            .pool_workers
            .with_label_values(&[PoolType::Batch.as_str()])
            .set(cfg.pipeline.batch_workers as i64);
    }

    let mut v1 = Router::new()
        .merge(routes::ingest::router())
        .merge(routes::admin::router())
        .merge(routes::metrics::router())
        .merge(routes::query::router())
        .route("/health", get(health));

    // Dashboard SPA + JSON API — gateway-only (matches Python service).
    // The static asset directory is resolved from
    // `NEMO_RETRIEVER_DASHBOARD_DIR` (set by the Docker image) with a
    // dev-mode fallback that walks the workspace.
    if mode.is_gateway() {
        let dashboard_router = routes::dashboard::router();
        let mut v1_dash = Router::new().nest("/dashboard", dashboard_router);
        if let Some(static_dir) = routes::dashboard::resolve_static_dir() {
            tracing::info!(
                static_dir = %static_dir.display(),
                "mounting dashboard static assets at /v1/dashboard/static"
            );
            v1_dash = v1_dash.nest_service(
                "/dashboard/static",
                tower_http::services::ServeDir::new(static_dir),
            );
        } else {
            tracing::warn!(
                "dashboard SPA assets not found; the UI will return 500. Set \
                 NEMO_RETRIEVER_DASHBOARD_DIR to the path of <repo>/dashboard/static."
            );
        }
        v1 = v1.merge(v1_dash);
    }

    // Optional bearer-token auth layer.
    if state.config.auth.api_token.is_some() {
        v1 = v1.layer(middleware::from_fn_with_state(
            state.clone(),
            require_bearer,
        ));
    }

    let app = Router::new()
        .nest("/v1", v1)
        .layer(SetRequestIdLayer::x_request_id(MakeRequestUuid))
        .layer(PropagateRequestIdLayer::x_request_id())
        .layer(TraceLayer::new_for_http())
        .with_state(state.clone());

    (app, state)
}

async fn health(State(state): State<SharedState>) -> impl IntoResponse {
    let mut body = json!({
        "status": "ok",
        "mode": state.config.mode.as_str(),
        "version": env!("CARGO_PKG_VERSION"),
        "runtime": "rust",
    });
    if let Some(proxy) = &state.gateway_proxy {
        body["backends"] = json!({
            "realtime": proxy.check_backend(PoolType::Realtime).await,
            "batch": proxy.check_backend(PoolType::Batch).await,
        });
    }
    Json(body)
}
