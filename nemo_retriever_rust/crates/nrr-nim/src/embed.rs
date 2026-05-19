// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NVIDIA Embedding NIM client.
//!
//! Wire format mirrors OpenAI's `/v1/embeddings` schema, which is what the
//! NV-Ingest embedding NIMs expose. Both text and base64-encoded image
//! payloads are supported via the `input_type` field.

use serde::{Deserialize, Serialize};

use crate::client::{NimClient, NimError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputType {
    Passage,
    Query,
    Image,
}

impl Default for InputType {
    fn default() -> Self {
        InputType::Passage
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbedRequest<'a> {
    pub model: &'a str,
    /// Vector of texts (or base64 image strings when `input_type=image`).
    pub input: Vec<String>,
    pub input_type: InputType,
    /// Forwarded verbatim; valid values: "float", "base64".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<&'a str>,
    /// Forwarded verbatim; valid values: "NONE", "START", "END".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbedResponse {
    pub data: Vec<EmbeddingRecord>,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub usage: Option<EmbedUsage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingRecord {
    pub embedding: Vec<f32>,
    #[serde(default)]
    pub index: u32,
    #[serde(default)]
    pub object: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbedUsage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
}

/// Convenience: call the embedding NIM with a list of texts and return the
/// raw embedding vectors in input order.
pub async fn embed_texts(
    client: &NimClient,
    endpoint: &str,
    model: &str,
    inputs: Vec<String>,
    input_type: InputType,
) -> Result<Vec<Vec<f32>>, NimError> {
    let req = EmbedRequest {
        model,
        input: inputs,
        input_type,
        encoding_format: Some("float"),
        truncate: Some("END"),
        dimensions: None,
    };
    // Endpoint arrives as the YAML "embed_invoke_url" — most deployments
    // already include the trailing /v1/embeddings, but we'll fix the path
    // up if not so callers don't have to remember.
    let url = if endpoint.ends_with("/embeddings") || endpoint.ends_with("/embeddings/") {
        endpoint.to_string()
    } else if endpoint.ends_with("/v1") || endpoint.ends_with("/v1/") {
        format!("{}/embeddings", endpoint.trim_end_matches('/'))
    } else {
        format!("{}/v1/embeddings", endpoint.trim_end_matches('/'))
    };

    let resp: EmbedResponse = client.post_json(&url, &req, "embed_invoke_url").await?;
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(resp.data.len());
    let mut indexed: Vec<(u32, Vec<f32>)> =
        resp.data.into_iter().map(|d| (d.index, d.embedding)).collect();
    indexed.sort_by_key(|(i, _)| *i);
    for (_, e) in indexed {
        out.push(e);
    }
    Ok(out)
}
