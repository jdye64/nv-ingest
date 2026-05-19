// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Async Rust SDK for the nemo-retriever HTTP API.
//!
//! Mirrors `nemo_retriever.service.client.RetrieverClient`. All methods
//! return `Result<T, ClientError>` and use a shared `reqwest::Client` so
//! HTTP/2 connection re-use happens automatically.

use std::collections::BTreeMap;
use std::time::Duration;

use bytes::Bytes;
use reqwest::multipart::{Form, Part};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value as JsonValue;
use thiserror::Error;
use url::Url;

#[derive(Debug, Error)]
pub enum ClientError {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("server returned status {status}: {body}")]
    Status { status: u16, body: String },
    #[error("invalid base url: {0}")]
    Url(#[from] url::ParseError),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Clone)]
pub struct Client {
    http: reqwest::Client,
    base: String,
    bearer: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CreateJob {
    pub expected_documents: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, JsonValue>,
}

impl Client {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self::with_options(base_url, None, Duration::from_secs(120))
    }

    pub fn with_options(
        base_url: impl Into<String>,
        bearer: Option<String>,
        timeout: Duration,
    ) -> Self {
        let http = reqwest::Client::builder()
            .timeout(timeout)
            .pool_max_idle_per_host(8)
            .pool_idle_timeout(Some(Duration::from_secs(90)))
            .tcp_keepalive(Duration::from_secs(60))
            .build()
            .expect("reqwest client");
        Self {
            http,
            base: base_url.into(),
            bearer,
        }
    }

    fn url(&self, path: &str) -> Result<Url, ClientError> {
        let base = self.base.trim_end_matches('/');
        let full = if path.starts_with('/') {
            format!("{base}{path}")
        } else {
            format!("{base}/{path}")
        };
        Ok(Url::parse(&full)?)
    }

    fn auth(&self, mut req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(token) = &self.bearer {
            req = req.bearer_auth(token);
        }
        req
    }

    async fn get<R: DeserializeOwned>(&self, path: &str) -> Result<R, ClientError> {
        let url = self.url(path)?;
        let resp = self.auth(self.http.get(url)).send().await?;
        Self::parse(resp).await
    }

    async fn post_json<B: Serialize, R: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<R, ClientError> {
        let url = self.url(path)?;
        let resp = self.auth(self.http.post(url).json(body)).send().await?;
        Self::parse(resp).await
    }

    async fn parse<R: DeserializeOwned>(resp: reqwest::Response) -> Result<R, ClientError> {
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(ClientError::Status {
                status: status.as_u16(),
                body,
            });
        }
        let bytes = resp.bytes().await?;
        let parsed: R = serde_json::from_slice(&bytes)?;
        Ok(parsed)
    }

    /// `GET /v1/health`
    pub async fn health(&self) -> Result<JsonValue, ClientError> {
        self.get("/v1/health").await
    }

    /// `POST /v1/ingest/job`
    pub async fn create_job(&self, req: &CreateJob) -> Result<JsonValue, ClientError> {
        self.post_json("/v1/ingest/job", req).await
    }

    /// `GET /v1/ingest/job/{job_id}`
    pub async fn get_job(&self, job_id: &str) -> Result<JsonValue, ClientError> {
        self.get(&format!("/v1/ingest/job/{job_id}")).await
    }

    /// Multipart upload of a single document to a job.
    pub async fn upload_document(
        &self,
        job_id: &str,
        file_bytes: Bytes,
        filename: &str,
        metadata: Option<JsonValue>,
    ) -> Result<JsonValue, ClientError> {
        let url = self.url(&format!("/v1/ingest/job/{job_id}/document"))?;
        let mut form = Form::new().part(
            "file",
            Part::stream(file_bytes)
                .file_name(filename.to_string())
                .mime_str("application/octet-stream")
                .map_err(ClientError::Http)?,
        );
        if let Some(meta) = metadata {
            form = form.text("metadata", meta.to_string());
        }
        let resp = self
            .auth(self.http.post(url).multipart(form))
            .send()
            .await?;
        Self::parse(resp).await
    }
}
