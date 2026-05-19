// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `/v1/query` — semantic search proxied to the dedicated vectordb pod.
//!
//! Wire-compatible with the Python service: gateway and standalone modes
//! forward to `<vectordb_url>/v1/query`; worker modes return 404.

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Router;
use bytes::Bytes;
use serde_json::json;

use crate::state::SharedState;

pub fn router() -> Router<SharedState> {
    Router::new().route("/query", post(query))
}

async fn query(State(state): State<SharedState>, body: Bytes) -> Response {
    if !state.config.vectordb.enabled {
        return error(StatusCode::NOT_FOUND, "VectorDB is not enabled");
    }
    if state.config.mode.is_worker() {
        return error(
            StatusCode::NOT_FOUND,
            "Query endpoint is not available on worker pods. Use the gateway.",
        );
    }
    let target = format!(
        "{}/v1/query",
        state.config.vectordb.vectordb_url.trim_end_matches('/')
    );
    let mut headers = HeaderMap::new();
    headers.insert("content-type", "application/json".parse().unwrap());
    let resp = state
        .http_client
        .post(&target)
        .body(body.to_vec())
        .send()
        .await;
    match resp {
        Ok(r) => {
            let status = r.status();
            let bytes = r.bytes().await.unwrap_or_default();
            Response::builder()
                .status(status.as_u16())
                .header("content-type", "application/json")
                .body(Body::from(bytes))
                .unwrap()
        }
        Err(err) => error(
            StatusCode::BAD_GATEWAY,
            &format!("Failed to reach VectorDB: {err}"),
        ),
    }
}

fn error(status: StatusCode, detail: &str) -> Response {
    let body = serde_json::to_vec(&json!({ "detail": detail })).unwrap_or_default();
    (status, [("content-type", "application/json")], body).into_response()
}
