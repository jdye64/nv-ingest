// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared state injected into every route handler via `Arc<AppState>`.

use std::sync::Arc;

use nrr_core::bus::EventBus;
use nrr_core::config::ServiceConfig;
use nrr_core::sidecar::SidecarStore;
use nrr_core::tracker::JobTracker;
use nrr_pipeline::PipelinePool;

use crate::metrics::Metrics;
use crate::proxy::GatewayProxy;

/// Process-wide app state. All inner fields are Arc/Clone-friendly so this
/// struct itself is cheap to clone — Axum routes receive `State<Arc<...>>`.
pub struct AppState {
    pub config: Arc<ServiceConfig>,
    pub job_tracker: JobTracker,
    pub event_bus: EventBus,
    pub sidecar_store: SidecarStore,
    pub pipeline_pool: Option<Arc<PipelinePool>>,
    pub gateway_proxy: Option<GatewayProxy>,
    pub metrics: Metrics,
    /// Shared HTTP client used by job callbacks + downstream proxying.
    pub http_client: reqwest::Client,
}

impl AppState {
    pub fn role(&self) -> &'static str {
        self.config.mode.as_str()
    }
}

pub type SharedState = Arc<AppState>;
