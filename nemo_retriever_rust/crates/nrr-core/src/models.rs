// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wire-format request and response structs.
//!
//! These mirror the pydantic models under
//! `nemo_retriever.service.models.{requests,responses,pipeline_spec}`.
//! Field names use the same JSON keys so the Rust service is drop-in
//! compatible with existing Python clients.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

// ─────────────────────────────────────────────────────────────────────
// PipelineSpec
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtractionMode {
    Pdf,
    Image,
    Auto,
    Text,
    Html,
    Audio,
}

impl Default for ExtractionMode {
    fn default() -> Self {
        ExtractionMode::Pdf
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageName {
    Extract,
    Dedup,
    Caption,
    Embed,
    Store,
    Filter,
    Webhook,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PdfSplitSpec {
    #[serde(default = "default_pages_per_chunk")]
    pub pages_per_chunk: u32,
}

fn default_pages_per_chunk() -> u32 {
    32
}

impl Default for PdfSplitSpec {
    fn default() -> Self {
        Self {
            pages_per_chunk: default_pages_per_chunk(),
        }
    }
}

/// Wire representation of a per-request pipeline override.
///
/// Each `*_params` field is intentionally `serde_json::Value` so the wire
/// format does not need to track every params-model field change.
/// `policy::validate_pipeline_spec` decides which keys are admissible.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineSpec {
    #[serde(default)]
    pub extraction_mode: ExtractionMode,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extract_params: Option<JsonValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embed_params: Option<JsonValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dedup_params: Option<JsonValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caption_params: Option<JsonValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub store_params: Option<JsonValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vdb_upload_params: Option<JsonValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub webhook_params: Option<JsonValue>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split_config: Option<JsonValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pdf_split: Option<PdfSplitSpec>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stage_order: Vec<StageName>,
}

impl PipelineSpec {
    pub fn is_empty(&self) -> bool {
        matches!(self.extraction_mode, ExtractionMode::Pdf)
            && self.extract_params.is_none()
            && self.embed_params.is_none()
            && self.dedup_params.is_none()
            && self.caption_params.is_none()
            && self.store_params.is_none()
            && self.vdb_upload_params.is_none()
            && self.webhook_params.is_none()
            && self.split_config.is_none()
            && self.pdf_split.is_none()
            && self.stage_order.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Request bodies
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IngestRequest {
    /// Legacy free-form client tag. In J3+ the canonical job id is in the
    /// URL path; this field is retained for backward compatibility but
    /// upload routes ignore it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page_number: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_pages: Option<u32>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, JsonValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pipeline: Option<PipelineSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobCreateRequest {
    pub expected_documents: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, JsonValue>,
}

// ─────────────────────────────────────────────────────────────────────
// Response bodies
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestAccepted {
    pub document_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
    pub content_sha256: String,
    pub status: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageIngestAccepted {
    pub page_id: String,
    pub document_id: String,
    pub page_number: u32,
    pub content_sha256: String,
    pub status: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentIngestAccepted {
    pub document_id: String,
    pub filename: String,
    pub file_size_bytes: u64,
    pub content_sha256: String,
    pub status: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobStatusResponse {
    pub id: String,
    pub status: String,
    pub submitted_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elapsed_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_rows: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_data: Option<Vec<JsonValue>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SidecarUploadResponse {
    pub sidecar_id: String,
    pub filename: String,
    pub content_type: String,
    pub size_bytes: u64,
    pub expires_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobCreatedResponse {
    pub job_id: String,
    pub expected_documents: u32,
    pub status: String,
    pub created_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobAggregateResponse {
    pub job_id: String,
    pub expected_documents: u32,
    pub status: String,
    pub created_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finalized_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elapsed_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default)]
    pub counts: BTreeMap<String, u64>,
    #[serde(default)]
    pub document_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub documents: Option<Vec<JsonValue>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentStatusResponse {
    pub document_id: String,
    pub job_id: String,
    pub status: String,
    pub submitted_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elapsed_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_rows: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_data: Option<Vec<JsonValue>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobDocumentsPage {
    pub job_id: String,
    pub total: u64,
    pub total_filtered: u64,
    pub offset: u64,
    pub limit: u64,
    #[serde(default)]
    pub items: Vec<DocumentStatusResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub mode: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backends: Option<JsonValue>,
}
