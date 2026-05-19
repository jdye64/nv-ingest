// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LanceDB row schema (must match `vdb/lancedb_schema.py`).

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

/// One LanceDB row as the vectordb pod expects to receive it. Field names
/// and JSON shapes are identical to `build_lancedb_row` in Python.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanceRow {
    pub vector: Vec<f32>,
    #[serde(default)]
    pub pdf_page: String,
    #[serde(default)]
    pub filename: String,
    #[serde(default)]
    pub pdf_basename: String,
    #[serde(default)]
    pub page_number: i32,
    /// JSON-encoded `{ "source_id": ... }` envelope.
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub source_id: String,
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub text: String,
    /// JSON-encoded metadata blob (page number + chunking hints + detection counts).
    #[serde(default)]
    pub metadata: String,
    #[serde(default)]
    pub stored_image_uri: String,
    #[serde(default)]
    pub content_type: String,
    /// JSON-encoded normalized bounding box, or empty.
    #[serde(default)]
    pub bbox_xyxy_norm: String,
}

/// Compose a single [`LanceRow`] from the building blocks the pipeline
/// has on hand. The Python implementation reaches into a pandas
/// `itertuples()` row object — in Rust we ask the caller to assemble the
/// inputs explicitly. This keeps the schema layer pipeline-agnostic.
pub fn build_lancedb_row(
    embedding: Vec<f32>,
    path: &str,
    page_number: i32,
    text: &str,
    metadata: &JsonValue,
    content_type: Option<&str>,
    stored_image_uri: Option<&str>,
    bbox_xyxy_norm: Option<&JsonValue>,
) -> LanceRow {
    let path_obj = std::path::Path::new(path);
    let filename = path_obj
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();
    let pdf_basename = path_obj
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();
    let pdf_page = if !pdf_basename.is_empty() && page_number >= 0 {
        format!("{}_{}", pdf_basename, page_number)
    } else {
        String::new()
    };
    let source_id = if !path.is_empty() {
        path.to_string()
    } else if !filename.is_empty() {
        filename.clone()
    } else {
        pdf_basename.clone()
    };
    LanceRow {
        vector: embedding,
        pdf_page,
        filename,
        pdf_basename,
        page_number,
        source: serde_json::json!({ "source_id": path }).to_string(),
        source_id,
        path: path.to_string(),
        text: text.to_string(),
        metadata: serde_json::to_string(metadata).unwrap_or_else(|_| "{}".to_string()),
        stored_image_uri: stored_image_uri.unwrap_or("").to_string(),
        content_type: content_type.unwrap_or("").to_string(),
        bbox_xyxy_norm: bbox_xyxy_norm
            .map(|v| serde_json::to_string(v).unwrap_or_default())
            .unwrap_or_default(),
    }
}

/// Build many rows from parallel slices.
///
/// Each input slice must have the same length as `embeddings`. When
/// `metadata`, `texts`, etc. are shorter, the missing entries default
/// to empty/null.
#[allow(clippy::too_many_arguments)]
pub fn build_lancedb_rows(
    embeddings: Vec<Vec<f32>>,
    paths: &[String],
    page_numbers: &[i32],
    texts: &[String],
    metadatas: &[JsonValue],
) -> Vec<LanceRow> {
    embeddings
        .into_iter()
        .enumerate()
        .map(|(i, emb)| {
            let path = paths.get(i).map(|s| s.as_str()).unwrap_or("");
            let page = page_numbers.get(i).copied().unwrap_or(-1);
            let text = texts.get(i).map(|s| s.as_str()).unwrap_or("");
            let meta = metadatas
                .get(i)
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));
            build_lancedb_row(emb, path, page, text, &meta, None, None, None)
        })
        .collect()
}
