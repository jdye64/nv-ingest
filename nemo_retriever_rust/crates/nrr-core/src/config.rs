// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `ServiceConfig` — YAML-backed configuration for the Rust service mode.
//!
//! This mirrors `nemo_retriever.service.config.ServiceConfig` from the Python
//! implementation. Discovery precedence is identical:
//!   1. Explicit `--config /path/to/retriever-service.yaml`
//!   2. `./retriever-service.yaml` in the current working directory
//!   3. Bundled default (`assets/retriever-service.yaml`, embedded at build time)

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::policy::{PipelineOverridesPolicy, SinkUrlAllowlist};

/// Bundled default config (mirrors the Python package data file).
pub const BUNDLED_CONFIG: &str = include_str!("../assets/retriever-service.yaml");

/// Runtime role for the service.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ServiceMode {
    Standalone,
    Gateway,
    Realtime,
    Batch,
}

impl ServiceMode {
    pub fn as_str(self) -> &'static str {
        match self {
            ServiceMode::Standalone => "standalone",
            ServiceMode::Gateway => "gateway",
            ServiceMode::Realtime => "realtime",
            ServiceMode::Batch => "batch",
        }
    }

    pub fn is_worker(self) -> bool {
        matches!(self, ServiceMode::Realtime | ServiceMode::Batch)
    }

    pub fn is_gateway(self) -> bool {
        matches!(self, ServiceMode::Gateway)
    }
}

impl Default for ServiceMode {
    fn default() -> Self {
        ServiceMode::Standalone
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
}

fn default_host() -> String {
    "0.0.0.0".into()
}
fn default_port() -> u16 {
    7670
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoggingConfig {
    #[serde(default = "default_log_level")]
    pub level: String,
    #[serde(default = "default_log_file")]
    pub file: String,
    /// Free-form format string. Honoured by the Python service via `logging`'s
    /// percent-style formatter; in Rust we render via `tracing_subscriber`'s
    /// default formatter and treat this field as advisory.
    #[serde(default = "default_log_format")]
    pub format: String,
}

fn default_log_level() -> String {
    "INFO".into()
}
fn default_log_file() -> String {
    "retriever-service.log".into()
}
fn default_log_format() -> String {
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s".into()
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            file: default_log_file(),
            format: default_log_format(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NimEndpointsConfig {
    #[serde(default)]
    pub page_elements_invoke_url: Option<String>,
    #[serde(default)]
    pub graphic_elements_invoke_url: Option<String>,
    #[serde(default)]
    pub table_structure_invoke_url: Option<String>,
    #[serde(default)]
    pub ocr_invoke_url: Option<String>,
    #[serde(default)]
    pub embed_invoke_url: Option<String>,
    #[serde(default)]
    pub rerank_invoke_url: Option<String>,
    #[serde(default)]
    pub caption_invoke_url: Option<String>,
    #[serde(default)]
    pub caption_model_name: Option<String>,
    #[serde(default)]
    pub api_key: Option<String>,

    /// Maximum number of in-flight embed-NIM requests across all worker
    /// tasks in this pod. Acts as backpressure so the embed pod is never
    /// asked to satisfy more concurrent batches than its memory ceiling
    /// allows. Tuned conservatively because each request can be a
    /// `[N_pages, max_seq_len]` tensor (i.e. dozens of MB of pinned host
    /// memory in Triton). Default 8 matches a single-GPU embed pod with
    /// the stock NIM container; raise it after observing steady-state
    /// memory headroom.
    #[serde(default = "default_embed_max_concurrency")]
    pub embed_max_concurrency: u32,

    /// Maximum number of texts (pages) sent in a single embed request.
    /// Documents larger than this are split into multiple sequential
    /// calls behind the global semaphore. Keeps peak request size bounded
    /// even when one document is being embedded by one worker.
    #[serde(default = "default_embed_max_pages_per_request")]
    pub embed_max_pages_per_request: u32,
}

fn default_embed_max_concurrency() -> u32 {
    8
}

fn default_embed_max_pages_per_request() -> u32 {
    4
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceLimitsConfig {
    #[serde(default)]
    pub max_memory_mb: Option<u64>,
    #[serde(default)]
    pub max_cpu_cores: Option<u32>,
    #[serde(default)]
    pub gpu_devices: Vec<String>,
    /// Reject uploads larger than this many bytes before buffering.
    #[serde(default = "default_max_upload_bytes")]
    pub max_upload_bytes: u64,
}

fn default_max_upload_bytes() -> u64 {
    500_000_000
}

impl Default for ResourceLimitsConfig {
    fn default() -> Self {
        Self {
            max_memory_mb: None,
            max_cpu_cores: None,
            gpu_devices: Vec::new(),
            max_upload_bytes: default_max_upload_bytes(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthConfig {
    #[serde(default)]
    pub api_token: Option<String>,
    #[serde(default = "default_auth_header_name")]
    pub header_name: String,
    #[serde(default = "default_auth_bypass_paths")]
    pub bypass_paths: Vec<String>,
}

fn default_auth_header_name() -> String {
    "Authorization".into()
}

fn default_auth_bypass_paths() -> Vec<String> {
    vec![
        "/v1/health".into(),
        "/docs".into(),
        "/openapi.json".into(),
        "/redoc".into(),
    ]
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            api_token: None,
            header_name: default_auth_header_name(),
            bypass_paths: default_auth_bypass_paths(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GatewayConfig {
    #[serde(default = "default_realtime_url")]
    pub realtime_url: String,
    #[serde(default = "default_batch_url")]
    pub batch_url: String,
    #[serde(default = "default_gw_timeout_s")]
    pub timeout_s: f64,
    #[serde(default = "default_max_connections")]
    pub max_connections: u32,
}

fn default_realtime_url() -> String {
    "http://nemo-retriever-realtime:7670".into()
}
fn default_batch_url() -> String {
    "http://nemo-retriever-batch:7670".into()
}
fn default_gw_timeout_s() -> f64 {
    300.0
}
fn default_max_connections() -> u32 {
    100
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            realtime_url: default_realtime_url(),
            batch_url: default_batch_url(),
            timeout_s: default_gw_timeout_s(),
            max_connections: default_max_connections(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelinePoolConfig {
    #[serde(default = "default_realtime_workers")]
    pub realtime_workers: u32,
    #[serde(default = "default_realtime_queue_size")]
    pub realtime_queue_size: u32,
    #[serde(default = "default_batch_workers")]
    pub batch_workers: u32,
    #[serde(default = "default_batch_queue_size")]
    pub batch_queue_size: u32,
}

fn default_realtime_workers() -> u32 {
    8
}
fn default_realtime_queue_size() -> u32 {
    2048
}
fn default_batch_workers() -> u32 {
    16
}
fn default_batch_queue_size() -> u32 {
    4096
}

impl Default for PipelinePoolConfig {
    fn default() -> Self {
        Self {
            realtime_workers: default_realtime_workers(),
            realtime_queue_size: default_realtime_queue_size(),
            batch_workers: default_batch_workers(),
            batch_queue_size: default_batch_queue_size(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorDbConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_lancedb_uri")]
    pub lancedb_uri: String,
    #[serde(default = "default_table_name")]
    pub table_name: String,
    #[serde(default = "default_embed_model")]
    pub embed_model: String,
    #[serde(default = "default_vectordb_url")]
    pub vectordb_url: String,
}

fn default_lancedb_uri() -> String {
    "/data/vectordb".into()
}
fn default_table_name() -> String {
    "nemo_retriever".into()
}
fn default_embed_model() -> String {
    "nvidia/llama-nemotron-embed-1b-v2".into()
}
fn default_vectordb_url() -> String {
    "http://nemo-retriever-vectordb:7671".into()
}

impl Default for VectorDbConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            lancedb_uri: default_lancedb_uri(),
            table_name: default_table_name(),
            embed_model: default_embed_model(),
            vectordb_url: default_vectordb_url(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SinksConfig {
    #[serde(default)]
    pub storage_uri_schemes: Vec<String>,
    #[serde(default)]
    pub webhook_url_prefixes: Vec<String>,
    #[serde(default)]
    pub vdb_uri_schemes: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OverridesMode {
    Reject,
    AllowList,
    AllowAll,
}

impl Default for OverridesMode {
    fn default() -> Self {
        OverridesMode::AllowList
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineOverridesConfig {
    #[serde(default)]
    pub mode: OverridesMode,
    #[serde(default)]
    pub extra_extract_keys: Vec<String>,
    #[serde(default)]
    pub extra_embed_keys: Vec<String>,
    #[serde(default)]
    pub extra_dedup_keys: Vec<String>,
    #[serde(default)]
    pub extra_split_keys: Vec<String>,
    #[serde(default)]
    pub extra_store_keys: Vec<String>,
    #[serde(default)]
    pub extra_webhook_keys: Vec<String>,
    #[serde(default)]
    pub extra_vdb_upload_keys: Vec<String>,
    #[serde(default)]
    pub extra_caption_keys: Vec<String>,
    #[serde(default)]
    pub sinks: SinksConfig,
}

impl PipelineOverridesConfig {
    /// Materialize a runtime policy. `caption_enabled` is derived from
    /// `nim_endpoints.caption_invoke_url` so clients can only override
    /// caption settings when the operator has wired up a VLM endpoint.
    pub fn to_policy(&self, caption_enabled: bool) -> PipelineOverridesPolicy {
        PipelineOverridesPolicy {
            mode: self.mode,
            extra_extract_keys: self.extra_extract_keys.iter().cloned().collect::<HashSet<_>>(),
            extra_embed_keys: self.extra_embed_keys.iter().cloned().collect::<HashSet<_>>(),
            extra_dedup_keys: self.extra_dedup_keys.iter().cloned().collect::<HashSet<_>>(),
            extra_split_keys: self.extra_split_keys.iter().cloned().collect::<HashSet<_>>(),
            extra_store_keys: self.extra_store_keys.iter().cloned().collect::<HashSet<_>>(),
            extra_webhook_keys: self
                .extra_webhook_keys
                .iter()
                .cloned()
                .collect::<HashSet<_>>(),
            extra_vdb_upload_keys: self
                .extra_vdb_upload_keys
                .iter()
                .cloned()
                .collect::<HashSet<_>>(),
            extra_caption_keys: self
                .extra_caption_keys
                .iter()
                .cloned()
                .collect::<HashSet<_>>(),
            sinks: SinkUrlAllowlist {
                storage_uri_schemes: self.sinks.storage_uri_schemes.clone(),
                webhook_url_prefixes: self.sinks.webhook_url_prefixes.clone(),
                vdb_uri_schemes: self.sinks.vdb_uri_schemes.clone(),
            },
            caption_enabled,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServiceConfig {
    #[serde(default)]
    pub mode: ServiceMode,
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
    #[serde(default)]
    pub nim_endpoints: NimEndpointsConfig,
    #[serde(default)]
    pub resources: ResourceLimitsConfig,
    #[serde(default)]
    pub auth: AuthConfig,
    #[serde(default)]
    pub gateway: GatewayConfig,
    #[serde(default)]
    pub pipeline: PipelinePoolConfig,
    #[serde(default)]
    pub vectordb: VectorDbConfig,
    #[serde(default)]
    pub pipeline_overrides: PipelineOverridesConfig,
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("config file not found: {0}")]
    NotFound(PathBuf),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("yaml parse error: {0}")]
    Yaml(#[from] serde_yaml::Error),
}

/// Result of [`discover_config_path`]: the chosen file plus a human-readable
/// origin label for log/diag purposes.
pub struct DiscoveredConfig {
    pub path: Option<PathBuf>,
    pub origin: &'static str,
}

pub fn discover_config_path(explicit: Option<&Path>) -> Result<DiscoveredConfig, ConfigError> {
    if let Some(p) = explicit {
        if !p.is_file() {
            return Err(ConfigError::NotFound(p.to_path_buf()));
        }
        return Ok(DiscoveredConfig {
            path: Some(p.to_path_buf()),
            origin: "explicit",
        });
    }
    let cwd_candidate = std::env::current_dir()
        .ok()
        .map(|cwd| cwd.join("retriever-service.yaml"));
    if let Some(p) = cwd_candidate {
        if p.is_file() {
            return Ok(DiscoveredConfig {
                path: Some(p),
                origin: "cwd",
            });
        }
    }
    Ok(DiscoveredConfig {
        path: None,
        origin: "bundled",
    })
}

/// Load a [`ServiceConfig`] using the discovery precedence above. Optional
/// overrides are dotted keys (e.g. `"server.port"`) mapping to `serde_yaml`
/// values; they are merged on top of the YAML before deserialization.
pub fn load_config(
    explicit: Option<&Path>,
    overrides: &[(String, serde_yaml::Value)],
) -> Result<ServiceConfig, ConfigError> {
    let discovered = discover_config_path(explicit)?;
    let raw = match &discovered.path {
        Some(p) => std::fs::read_to_string(p)?,
        None => BUNDLED_CONFIG.to_string(),
    };
    let mut value: serde_yaml::Value = serde_yaml::from_str(&raw)?;
    if !value.is_mapping() {
        value = serde_yaml::Value::Mapping(Default::default());
    }
    for (dotted, v) in overrides {
        merge_dotted(&mut value, dotted, v.clone());
    }
    let cfg: ServiceConfig = serde_yaml::from_value(value)?;
    Ok(cfg)
}

fn merge_dotted(target: &mut serde_yaml::Value, dotted: &str, val: serde_yaml::Value) {
    let parts: Vec<&str> = dotted.split('.').collect();
    let mut cursor = target;
    for part in &parts[..parts.len() - 1] {
        let map = match cursor {
            serde_yaml::Value::Mapping(m) => m,
            _ => {
                *cursor = serde_yaml::Value::Mapping(Default::default());
                match cursor {
                    serde_yaml::Value::Mapping(m) => m,
                    _ => unreachable!(),
                }
            }
        };
        let key = serde_yaml::Value::String((*part).to_string());
        if !map.contains_key(&key) {
            map.insert(key.clone(), serde_yaml::Value::Mapping(Default::default()));
        }
        cursor = map.get_mut(&key).unwrap();
    }
    let last_key = serde_yaml::Value::String(parts[parts.len() - 1].to_string());
    if let serde_yaml::Value::Mapping(m) = cursor {
        m.insert(last_key, val);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_round_trip() {
        let cfg = ServiceConfig::default();
        let s = serde_yaml::to_string(&cfg).unwrap();
        let _round: ServiceConfig = serde_yaml::from_str(&s).unwrap();
    }

    #[test]
    fn bundled_yaml_parses() {
        let _cfg: ServiceConfig = serde_yaml::from_str(BUNDLED_CONFIG).unwrap();
    }

    #[test]
    fn dotted_override_applies() {
        let cfg = load_config(
            None,
            &[(
                "server.port".into(),
                serde_yaml::Value::Number(9000u64.into()),
            )],
        )
        .unwrap();
        assert_eq!(cfg.server.port, 9000);
    }
}
