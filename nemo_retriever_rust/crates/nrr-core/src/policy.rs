// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline-overrides policy.
//!
//! Mirrors `nemo_retriever.service.policy`:
//! * `reject` — every override is denied.
//! * `allow_list` — only the built-in audited keys plus operator-supplied
//!   `extra_*_keys` are accepted. Endpoint URLs and API keys are ALWAYS denied.
//! * `allow_all` — every key passes except the endpoint/api_key denylist.
//!
//! Sink endpoints (image store / webhook / vdb_upload) are gated by URL/scheme
//! allow-lists in [`SinkUrlAllowlist`].

use std::collections::HashSet;

use serde_json::Value as JsonValue;
use thiserror::Error;

use crate::config::OverridesMode;
use crate::models::PipelineSpec;

/// Keys that NEVER ride a per-request override regardless of policy mode.
/// They are entirely server-owned (NIM endpoint URLs, API keys, model IDs).
const TRUST_OWNED_EXTRACT_KEYS: &[&str] = &[
    "invoke_url",
    "api_key",
    "page_elements_invoke_url",
    "page_elements_api_key",
    "ocr_invoke_url",
    "ocr_api_key",
    "graphic_elements_invoke_url",
    "table_structure_invoke_url",
    "nemotron_parse_invoke_url",
];
const TRUST_OWNED_EMBED_KEYS: &[&str] =
    &["embed_invoke_url", "embedding_endpoint", "api_key"];
const TRUST_OWNED_CAPTION_KEYS: &[&str] = &["endpoint_url", "api_key", "model_name"];

/// Default audited "shape" keys per stage. These are the knobs operators
/// have already approved for tenant override (chunk sizes, batch sizes,
/// output column names …). Anything else is denied unless `allow_all`
/// is in effect.
const DEFAULT_EXTRACT_KEYS: &[&str] = &[
    "extract_text",
    "extract_charts",
    "extract_tables",
    "extract_infographics",
    "text_depth",
    "chunk_size",
    "chunk_overlap",
];
const DEFAULT_EMBED_KEYS: &[&str] = &[
    "embed_modality",
    "inference_batch_size",
    "input_type",
    "nim_http_max_concurrent",
    "text_column",
    "output_column",
];
const DEFAULT_DEDUP_KEYS: &[&str] = &["enabled", "method", "threshold"];
const DEFAULT_SPLIT_KEYS: &[&str] = &["pages_per_chunk", "max_pages"];
const DEFAULT_STORE_KEYS: &[&str] = &["uri", "compression"];
const DEFAULT_WEBHOOK_KEYS: &[&str] = &["url", "headers", "method"];
const DEFAULT_VDB_UPLOAD_KEYS: &[&str] = &[
    "lancedb_uri",
    "table_name",
    "overwrite",
    "vector_dim",
    "meta_dataframe_id",
];
const DEFAULT_CAPTION_KEYS: &[&str] = &[
    "prompt",
    "system_prompt",
    "batch_size",
    "max_tokens",
    "temperature",
];

#[derive(Debug, Clone)]
pub struct SinkUrlAllowlist {
    pub storage_uri_schemes: Vec<String>,
    pub webhook_url_prefixes: Vec<String>,
    pub vdb_uri_schemes: Vec<String>,
}

impl SinkUrlAllowlist {
    fn check_uri_scheme(uri: &str, schemes: &[String]) -> bool {
        if schemes.iter().any(|s| s == "*") {
            return true;
        }
        schemes.iter().any(|s| uri.starts_with(s.as_str()))
    }

    pub fn allow_storage(&self, uri: &str) -> bool {
        Self::check_uri_scheme(uri, &self.storage_uri_schemes)
    }
    pub fn allow_webhook(&self, url: &str) -> bool {
        if self.webhook_url_prefixes.iter().any(|p| p == "*") {
            return true;
        }
        self.webhook_url_prefixes
            .iter()
            .any(|p| url.starts_with(p.as_str()))
    }
    pub fn allow_vdb_upload(&self, uri: &str) -> bool {
        Self::check_uri_scheme(uri, &self.vdb_uri_schemes)
    }
}

#[derive(Debug, Clone)]
pub struct PipelineOverridesPolicy {
    pub mode: OverridesMode,
    pub extra_extract_keys: HashSet<String>,
    pub extra_embed_keys: HashSet<String>,
    pub extra_dedup_keys: HashSet<String>,
    pub extra_split_keys: HashSet<String>,
    pub extra_store_keys: HashSet<String>,
    pub extra_webhook_keys: HashSet<String>,
    pub extra_vdb_upload_keys: HashSet<String>,
    pub extra_caption_keys: HashSet<String>,
    pub sinks: SinkUrlAllowlist,
    pub caption_enabled: bool,
}

impl PipelineOverridesPolicy {
    pub fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "mode": match self.mode {
                OverridesMode::Reject => "reject",
                OverridesMode::AllowList => "allow_list",
                OverridesMode::AllowAll => "allow_all",
            },
            "extract_keys": merged_keys(DEFAULT_EXTRACT_KEYS, &self.extra_extract_keys),
            "embed_keys": merged_keys(DEFAULT_EMBED_KEYS, &self.extra_embed_keys),
            "dedup_keys": merged_keys(DEFAULT_DEDUP_KEYS, &self.extra_dedup_keys),
            "split_keys": merged_keys(DEFAULT_SPLIT_KEYS, &self.extra_split_keys),
            "store_keys": merged_keys(DEFAULT_STORE_KEYS, &self.extra_store_keys),
            "webhook_keys": merged_keys(DEFAULT_WEBHOOK_KEYS, &self.extra_webhook_keys),
            "vdb_upload_keys": merged_keys(DEFAULT_VDB_UPLOAD_KEYS, &self.extra_vdb_upload_keys),
            "caption_keys": merged_keys(DEFAULT_CAPTION_KEYS, &self.extra_caption_keys),
            "caption_enabled": self.caption_enabled,
            "sinks": {
                "storage_uri_schemes": &self.sinks.storage_uri_schemes,
                "webhook_url_prefixes": &self.sinks.webhook_url_prefixes,
                "vdb_uri_schemes": &self.sinks.vdb_uri_schemes,
            }
        })
    }
}

fn merged_keys(defaults: &[&str], extras: &HashSet<String>) -> Vec<String> {
    let mut out: Vec<String> = defaults.iter().map(|s| (*s).to_string()).collect();
    out.extend(extras.iter().cloned());
    out.sort();
    out.dedup();
    out
}

#[derive(Debug, Error)]
pub enum PolicyError {
    #[error("forbidden override key: {key}")]
    ForbiddenKey { key: String, status_code: u16 },
    #[error("override mode is 'reject'; client overrides are not allowed")]
    Rejected { status_code: u16 },
    #[error("forbidden sink uri/url: {uri}")]
    ForbiddenSink { uri: String, status_code: u16 },
    #[error("caption stage requested but caption_invoke_url is not configured")]
    CaptionDisabled { status_code: u16 },
    #[error("malformed pipeline spec: {0}")]
    Malformed(String),
}

impl PolicyError {
    pub fn status_code(&self) -> u16 {
        match self {
            PolicyError::ForbiddenKey { status_code, .. } => *status_code,
            PolicyError::Rejected { status_code } => *status_code,
            PolicyError::ForbiddenSink { status_code, .. } => *status_code,
            PolicyError::CaptionDisabled { status_code } => *status_code,
            PolicyError::Malformed(_) => 400,
        }
    }
}

/// Validate a client-supplied [`PipelineSpec`] against the policy.
///
/// Returns the spec unchanged on success. The server merges its own
/// trust-owned defaults (URLs / API keys) on top of the returned spec at
/// pipeline build time, so the validated spec only carries "shape" knobs.
pub fn validate_pipeline_spec(
    spec: &PipelineSpec,
    policy: &PipelineOverridesPolicy,
) -> Result<PipelineSpec, PolicyError> {
    if spec.is_empty() {
        return Ok(spec.clone());
    }

    if matches!(policy.mode, OverridesMode::Reject) {
        return Err(PolicyError::Rejected { status_code: 403 });
    }

    let mode = policy.mode;

    check_params_block(
        spec.extract_params.as_ref(),
        DEFAULT_EXTRACT_KEYS,
        &policy.extra_extract_keys,
        TRUST_OWNED_EXTRACT_KEYS,
        mode,
    )?;
    check_params_block(
        spec.embed_params.as_ref(),
        DEFAULT_EMBED_KEYS,
        &policy.extra_embed_keys,
        TRUST_OWNED_EMBED_KEYS,
        mode,
    )?;
    check_params_block(
        spec.dedup_params.as_ref(),
        DEFAULT_DEDUP_KEYS,
        &policy.extra_dedup_keys,
        &[],
        mode,
    )?;
    check_params_block(
        spec.split_config.as_ref(),
        DEFAULT_SPLIT_KEYS,
        &policy.extra_split_keys,
        &[],
        mode,
    )?;
    check_params_block(
        spec.vdb_upload_params.as_ref(),
        DEFAULT_VDB_UPLOAD_KEYS,
        &policy.extra_vdb_upload_keys,
        &[],
        mode,
    )?;

    if let Some(store) = &spec.store_params {
        check_params_block(
            Some(store),
            DEFAULT_STORE_KEYS,
            &policy.extra_store_keys,
            &[],
            mode,
        )?;
        if let Some(uri) = store.get("uri").and_then(|v| v.as_str()) {
            if !policy.sinks.allow_storage(uri) {
                return Err(PolicyError::ForbiddenSink {
                    uri: uri.into(),
                    status_code: 403,
                });
            }
        }
    }

    if let Some(webhook) = &spec.webhook_params {
        check_params_block(
            Some(webhook),
            DEFAULT_WEBHOOK_KEYS,
            &policy.extra_webhook_keys,
            &[],
            mode,
        )?;
        if let Some(url) = webhook.get("url").and_then(|v| v.as_str()) {
            if !policy.sinks.allow_webhook(url) {
                return Err(PolicyError::ForbiddenSink {
                    uri: url.into(),
                    status_code: 403,
                });
            }
        }
    }

    if let Some(vdb) = &spec.vdb_upload_params {
        if let Some(uri) = vdb.get("lancedb_uri").and_then(|v| v.as_str()) {
            if !policy.sinks.allow_vdb_upload(uri) {
                return Err(PolicyError::ForbiddenSink {
                    uri: uri.into(),
                    status_code: 403,
                });
            }
        }
    }

    if spec.caption_params.is_some() {
        if !policy.caption_enabled {
            return Err(PolicyError::CaptionDisabled { status_code: 403 });
        }
        check_params_block(
            spec.caption_params.as_ref(),
            DEFAULT_CAPTION_KEYS,
            &policy.extra_caption_keys,
            TRUST_OWNED_CAPTION_KEYS,
            mode,
        )?;
    }

    Ok(spec.clone())
}

fn check_params_block(
    params: Option<&JsonValue>,
    defaults: &[&str],
    extras: &HashSet<String>,
    denied: &[&str],
    mode: OverridesMode,
) -> Result<(), PolicyError> {
    let Some(JsonValue::Object(map)) = params else {
        return Ok(());
    };
    for (k, _) in map.iter() {
        if denied.iter().any(|d| *d == k.as_str()) {
            return Err(PolicyError::ForbiddenKey {
                key: k.clone(),
                status_code: 403,
            });
        }
        if matches!(mode, OverridesMode::AllowAll) {
            continue;
        }
        let allowed = defaults.iter().any(|d| *d == k.as_str()) || extras.contains(k);
        if !allowed {
            return Err(PolicyError::ForbiddenKey {
                key: k.clone(),
                status_code: 403,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn policy(mode: OverridesMode) -> PipelineOverridesPolicy {
        PipelineOverridesPolicy {
            mode,
            extra_extract_keys: HashSet::new(),
            extra_embed_keys: HashSet::new(),
            extra_dedup_keys: HashSet::new(),
            extra_split_keys: HashSet::new(),
            extra_store_keys: HashSet::new(),
            extra_webhook_keys: HashSet::new(),
            extra_vdb_upload_keys: HashSet::new(),
            extra_caption_keys: HashSet::new(),
            sinks: SinkUrlAllowlist {
                storage_uri_schemes: vec![],
                webhook_url_prefixes: vec![],
                vdb_uri_schemes: vec![],
            },
            caption_enabled: false,
        }
    }

    #[test]
    fn reject_mode_blocks_overrides() {
        let p = policy(OverridesMode::Reject);
        let mut spec = PipelineSpec::default();
        spec.embed_params = Some(json!({ "inference_batch_size": 32 }));
        let err = validate_pipeline_spec(&spec, &p).unwrap_err();
        assert_eq!(err.status_code(), 403);
    }

    #[test]
    fn allow_list_blocks_endpoint_keys() {
        let p = policy(OverridesMode::AllowList);
        let mut spec = PipelineSpec::default();
        spec.embed_params = Some(json!({ "embed_invoke_url": "http://attacker" }));
        let err = validate_pipeline_spec(&spec, &p).unwrap_err();
        assert_eq!(err.status_code(), 403);
    }

    #[test]
    fn allow_list_accepts_default_keys() {
        let p = policy(OverridesMode::AllowList);
        let mut spec = PipelineSpec::default();
        spec.embed_params = Some(json!({ "inference_batch_size": 64 }));
        validate_pipeline_spec(&spec, &p).unwrap();
    }

    #[test]
    fn allow_all_still_blocks_endpoint_keys() {
        let p = policy(OverridesMode::AllowAll);
        let mut spec = PipelineSpec::default();
        spec.embed_params = Some(json!({ "embed_invoke_url": "http://attacker" }));
        let err = validate_pipeline_spec(&spec, &p).unwrap_err();
        assert_eq!(err.status_code(), 403);
    }
}
