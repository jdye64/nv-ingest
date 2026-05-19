// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core types for the Rust nemo-retriever service.
//!
//! Module overview:
//! * [`config`]   — `ServiceConfig` (YAML-backed), mirrors `service/config.py`.
//! * [`models`]   — wire-format request/response/spec structs.
//! * [`policy`]   — pipeline-overrides allow-list / reject / allow-all logic.
//! * [`tracker`]  — in-memory `JobTracker` (jobs + per-document records).
//! * [`bus`]      — broadcast SSE event bus.
//! * [`sidecar`]  — TTL-bounded in-memory sidecar payload store.
//! * [`util`]     — shared helpers (file-type classification, time, redaction).

pub mod bus;
pub mod config;
pub mod models;
pub mod policy;
pub mod sidecar;
pub mod tracker;
pub mod util;

pub use config::{
    AuthConfig, GatewayConfig, LoggingConfig, NimEndpointsConfig, PipelineOverridesConfig,
    PipelinePoolConfig, ResourceLimitsConfig, ServerConfig, ServiceConfig, ServiceMode,
    SinksConfig, VectorDbConfig,
};
