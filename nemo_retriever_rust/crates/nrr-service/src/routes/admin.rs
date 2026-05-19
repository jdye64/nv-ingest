// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `/v1` admin endpoints: pool stats + pipeline config introspection.

use axum::extract::State;
use axum::routing::get;
use axum::{Json, Router};
use serde_json::json;

use crate::state::SharedState;

pub fn router() -> Router<SharedState> {
    Router::new()
        .route("/ingest/pipeline-config", get(pipeline_config))
        // Both naming conventions are accepted: the Rust runtime prefers
        // hyphens (`pool-stats`) but the Python service used underscores
        // (`pool_stats`). The dashboard fan-out tries both forms so the
        // overview view works against either runtime mid-migration.
        .route("/admin/pool-stats", get(pool_stats))
        .route("/admin/pool_stats", get(pool_stats))
}

async fn pool_stats(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let stats = state
        .pipeline_pool
        .as_ref()
        .map(|p| p.stats())
        .unwrap_or_else(|| json!({}));
    Json(stats)
}

async fn pipeline_config(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let mode = state.config.mode.as_str();
    let policy = state
        .config
        .pipeline_overrides
        .to_policy(state.config.nim_endpoints.caption_invoke_url.is_some())
        .describe();
    let pool_stats = state
        .pipeline_pool
        .as_ref()
        .map(|p| p.stats())
        .unwrap_or_else(|| json!({}));
    Json(json!({
        "source": mode,
        "mode": mode,
        "pool_stats": pool_stats,
        "allowed_overrides": policy,
        "nim_endpoints": {
            "page_elements_invoke_url": state.config.nim_endpoints.page_elements_invoke_url,
            "embed_invoke_url": state.config.nim_endpoints.embed_invoke_url,
            "ocr_invoke_url": state.config.nim_endpoints.ocr_invoke_url,
            "table_structure_invoke_url": state.config.nim_endpoints.table_structure_invoke_url,
            "graphic_elements_invoke_url": state.config.nim_endpoints.graphic_elements_invoke_url,
            "caption_invoke_url": state.config.nim_endpoints.caption_invoke_url,
        }
    }))
}
