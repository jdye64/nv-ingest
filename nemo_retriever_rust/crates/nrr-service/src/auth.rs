// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bearer-token authentication middleware.
//!
//! Mirrors `nemo_retriever.service.auth.BearerAuthMiddleware`. When
//! `auth.api_token` is set in the YAML config, every request must carry
//! `Authorization: Bearer <token>` — except for paths in `bypass_paths`
//! (the health probe, OpenAPI docs, etc.).

use std::sync::Arc;

use axum::{
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

use crate::state::SharedState;

pub async fn require_bearer(
    State(state): State<SharedState>,
    req: Request,
    next: Next,
) -> Response {
    let auth = &state.config.auth;
    let Some(expected) = auth.api_token.as_deref() else {
        return next.run(req).await;
    };
    let path = req.uri().path();
    if auth.bypass_paths.iter().any(|p| path.starts_with(p.as_str())) {
        return next.run(req).await;
    }
    let header_name = auth.header_name.as_str();
    let provided = req
        .headers()
        .get(header_name)
        .and_then(|v| v.to_str().ok())
        .map(str::trim);
    let token = provided.and_then(|raw| {
        if let Some(rest) = raw.strip_prefix("Bearer ") {
            Some(rest.trim())
        } else {
            None
        }
    });
    if token.map(|t| t == expected).unwrap_or(false) {
        return next.run(req).await;
    }
    let body = Json(json!({
        "detail": "missing or invalid bearer token",
        "header": header_name,
    }));
    (StatusCode::UNAUTHORIZED, body).into_response()
}

/// Tiny helper used by the app builder so test code can assert that auth
/// was wired without poking at handler internals.
pub fn auth_enabled(state: &Arc<crate::state::AppState>) -> bool {
    state.config.auth.api_token.is_some()
}
