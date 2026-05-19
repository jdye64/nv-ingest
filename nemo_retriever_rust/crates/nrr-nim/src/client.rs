// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared `reqwest`-based HTTP client used by every NIM endpoint.

use std::error::Error as _;
use std::sync::Arc;
use std::time::Duration;

use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest::{Client, ClientBuilder};
use serde::de::DeserializeOwned;
use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum NimError {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("nim returned status {status}: {body}")]
    Status { status: u16, body: String },
    #[error("invalid header value: {0}")]
    Header(#[from] reqwest::header::InvalidHeaderValue),
    #[error("missing endpoint: tried to call {endpoint:?} but it is not configured")]
    MissingEndpoint { endpoint: &'static str },
    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),
}

#[derive(Clone)]
pub struct NimClient {
    inner: Arc<NimClientInner>,
}

struct NimClientInner {
    http: Client,
    api_key: Option<String>,
}

pub struct NimClientBuilder {
    api_key: Option<String>,
    timeout: Duration,
    pool_max_per_host: usize,
    user_agent: String,
}

impl Default for NimClientBuilder {
    fn default() -> Self {
        Self {
            api_key: None,
            timeout: Duration::from_secs(120),
            pool_max_per_host: 32,
            user_agent: format!("nemo-retriever-rust/{}", env!("CARGO_PKG_VERSION")),
        }
    }
}

impl NimClientBuilder {
    pub fn api_key(mut self, k: Option<impl Into<String>>) -> Self {
        self.api_key = k.map(|s| s.into());
        self
    }
    pub fn timeout(mut self, d: Duration) -> Self {
        self.timeout = d;
        self
    }
    pub fn pool_max_per_host(mut self, n: usize) -> Self {
        self.pool_max_per_host = n;
        self
    }
    pub fn user_agent(mut self, ua: impl Into<String>) -> Self {
        self.user_agent = ua.into();
        self
    }
    pub fn build(self) -> Result<NimClient, NimError> {
        // NOTE: do NOT call `http2_prior_knowledge()` here. In-cluster NIMs
        // are served over plain HTTP and only speak HTTP/1.1 — forcing the
        // HTTP/2 connection preface against an h1 server makes the server
        // close the connection, which surfaces as a transport error.
        // For HTTPS endpoints reqwest negotiates HTTP/2 via ALPN
        // automatically, so we leave protocol selection to reqwest. The
        // h2 keep-alive settings below are inert when the negotiated
        // protocol is HTTP/1.1.
        let http = ClientBuilder::new()
            .timeout(self.timeout)
            .pool_max_idle_per_host(self.pool_max_per_host)
            .pool_idle_timeout(Some(Duration::from_secs(90)))
            .http2_keep_alive_interval(Duration::from_secs(30))
            .http2_keep_alive_timeout(Duration::from_secs(10))
            .tcp_keepalive(Duration::from_secs(60))
            .user_agent(self.user_agent)
            .build()?;
        Ok(NimClient {
            inner: Arc::new(NimClientInner {
                http,
                api_key: self.api_key,
            }),
        })
    }
}

impl NimClient {
    pub fn builder() -> NimClientBuilder {
        NimClientBuilder::default()
    }

    pub fn http(&self) -> &Client {
        &self.inner.http
    }

    pub fn api_key(&self) -> Option<&str> {
        self.inner.api_key.as_deref()
    }

    /// Build a header map with Authorization (when an API key is set) and
    /// Content-Type: application/json. Per-call extra headers are merged in.
    pub fn json_headers(&self, extra: Option<&HeaderMap>) -> Result<HeaderMap, NimError> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Some(k) = self.api_key() {
            let v = HeaderValue::from_str(&format!("Bearer {}", k))?;
            headers.insert(AUTHORIZATION, v);
        }
        if let Some(extra) = extra {
            headers.extend(extra.clone());
        }
        Ok(headers)
    }

    /// POST a JSON body and decode the JSON response. The `endpoint_label`
    /// is used purely for error messages — pass the YAML key name, e.g.
    /// `"embed_invoke_url"`.
    pub async fn post_json<B: Serialize, R: DeserializeOwned>(
        &self,
        url: &str,
        body: &B,
        endpoint_label: &'static str,
    ) -> Result<R, NimError> {
        let headers = self.json_headers(None)?;
        let resp = self
            .inner
            .http
            .post(url)
            .headers(headers)
            .json(body)
            .send()
            .await
            .map_err(|e| {
                // Walk the error source chain so we see the *real* cause
                // (DNS failure, connection reset, h2 negotiation, etc.).
                let mut cause: Vec<String> = Vec::new();
                let mut src: Option<&dyn std::error::Error> = e.source();
                while let Some(s) = src {
                    cause.push(s.to_string());
                    src = s.source();
                }
                tracing::warn!(
                    error = %e,
                    causes = ?cause,
                    endpoint = endpoint_label,
                    url,
                    "nim transport failure"
                );
                NimError::Http(e)
            })?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(NimError::Status {
                status: status.as_u16(),
                body,
            });
        }
        let parsed = resp.json::<R>().await?;
        Ok(parsed)
    }
}
