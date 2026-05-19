// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP client that POSTs row batches to the dedicated vectordb pod.

use std::sync::Arc;
use std::time::Duration;

use reqwest::{Client, ClientBuilder};
use serde::Serialize;
use thiserror::Error;

use crate::schema::LanceRow;

#[derive(Debug, Error)]
pub enum VectorDbError {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("vectordb returned status {status}: {body}")]
    Status { status: u16, body: String },
}

#[derive(Clone)]
pub struct VectorDbClient {
    inner: Arc<VectorDbInner>,
}

struct VectorDbInner {
    http: Client,
    base_url: String,
}

#[derive(Serialize)]
struct WriteRequest<'a> {
    rows: &'a [LanceRow],
}

impl VectorDbClient {
    pub fn new(base_url: impl Into<String>) -> Result<Self, VectorDbError> {
        let http = ClientBuilder::new()
            .timeout(Duration::from_secs(30))
            .pool_max_idle_per_host(8)
            .pool_idle_timeout(Some(Duration::from_secs(90)))
            .tcp_keepalive(Duration::from_secs(60))
            .build()?;
        Ok(Self {
            inner: Arc::new(VectorDbInner {
                http,
                base_url: base_url.into(),
            }),
        })
    }

    /// Fire-and-forget POST of rows to the vectordb service.
    /// Mirrors `_post_rows_to_vectordb` from the Python service.
    pub async fn write_rows(&self, rows: &[LanceRow], filename: &str) -> Result<(), VectorDbError> {
        if rows.is_empty() {
            return Ok(());
        }
        let url = format!(
            "{}/internal/vectordb/write",
            self.inner.base_url.trim_end_matches('/')
        );
        let req = WriteRequest { rows };
        let resp = self
            .inner
            .http
            .post(&url)
            .json(&req)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            tracing::warn!(
                filename,
                status = status.as_u16(),
                body = %body,
                "vectordb write rejected"
            );
            return Err(VectorDbError::Status {
                status: status.as_u16(),
                body,
            });
        }
        tracing::info!(filename, rows = rows.len(), "vectordb write OK");
        Ok(())
    }
}
