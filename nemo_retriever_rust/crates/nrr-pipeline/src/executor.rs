// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline executor: extract → embed → vdb fan-out for a single document.
//!
//! Compared to the Python implementation, this version:
//!
//! * Skips the ProcessPoolExecutor isolation wrapper. PDFium is not in the
//!   Rust hot path (we use `lopdf`), so no fork-server is needed.
//! * Runs PDF parsing inside `tokio::task::spawn_blocking` to keep the
//!   async runtime responsive for HTTP I/O and NIM calls.
//! * Pipelines NIM calls through HTTP/2 keep-alive — `reqwest::Client` is
//!   shared across every worker so the embed / OCR sockets stay warm.

use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use serde_json::{json, Value as JsonValue};
use tokio::sync::Semaphore;
use tracing::info;

use nrr_core::config::{NimEndpointsConfig, ServiceConfig};
use nrr_nim::{client::NimClient, embed::embed_texts, embed::InputType};
use nrr_vdb::{client::VectorDbClient, schema::build_lancedb_rows};

use crate::pdf;
use crate::pool::{WorkFn, WorkItem, WorkResult};

/// Default model name used when the spec doesn't override it.
const DEFAULT_EMBED_MODEL: &str = "nvidia/llama-nemotron-embed-1b-v2";

/// Concrete `WorkFn` for the realtime pool.
///
/// In practice realtime and batch differ only in queue sizing and the
/// `extract_text` shortcut, so we expose the same struct from both.
pub struct InMemoryExecutor {
    cfg: Arc<ServiceConfig>,
    nim: NimClient,
    vdb: Option<VectorDbClient>,
    /// Shared backpressure for the embed NIM. Bounded so the worker pool
    /// can never have more than `nim_endpoints.embed_max_concurrency`
    /// simultaneous embed requests in flight, regardless of how many
    /// documents are being processed in parallel. Cloned cheaply across
    /// realtime + batch executors so they share the same budget.
    embed_sem: Arc<Semaphore>,
    /// Maximum pages per single embed request; enforced by the executor
    /// via sequential chunking under the semaphore above.
    embed_max_pages: usize,
}

impl InMemoryExecutor {
    pub fn new(
        cfg: Arc<ServiceConfig>,
        nim: NimClient,
        vdb: Option<VectorDbClient>,
    ) -> Self {
        let permits = cfg.nim_endpoints.embed_max_concurrency.max(1) as usize;
        let embed_sem = Arc::new(Semaphore::new(permits));
        let embed_max_pages =
            cfg.nim_endpoints.embed_max_pages_per_request.max(1) as usize;
        Self {
            cfg,
            nim,
            vdb,
            embed_sem,
            embed_max_pages,
        }
    }

    /// Reuse the same semaphore + chunk budget across multiple executor
    /// instances (e.g. one per pool) so realtime + batch workers compete
    /// for a single embed-NIM budget rather than each having their own.
    pub fn with_embed_budget(
        cfg: Arc<ServiceConfig>,
        nim: NimClient,
        vdb: Option<VectorDbClient>,
        embed_sem: Arc<Semaphore>,
        embed_max_pages: usize,
    ) -> Self {
        Self {
            cfg,
            nim,
            vdb,
            embed_sem,
            embed_max_pages,
        }
    }

    pub fn embed_semaphore(&self) -> Arc<Semaphore> {
        Arc::clone(&self.embed_sem)
    }

    pub fn embed_max_pages(&self) -> usize {
        self.embed_max_pages
    }
}

#[async_trait::async_trait]
impl WorkFn for InMemoryExecutor {
    async fn run(&self, item: WorkItem) -> Result<WorkResult> {
        run_pipeline(
            &self.cfg.nim_endpoints,
            &self.nim,
            self.vdb.as_ref(),
            self.cfg.vectordb.embed_model.as_str(),
            &self.embed_sem,
            self.embed_max_pages,
            &item,
        )
        .await
    }
}

/// Run the in-memory pipeline for a single [`WorkItem`].
///
/// * Reads the per-page text from the uploaded PDF (or treats arbitrary
///   uploads as a single text blob).
/// * Calls the embedding NIM in one batched HTTP/2 request.
/// * Builds LanceDB-shaped rows and posts them to the vectordb pod.
/// * Returns lightweight summary rows so the status endpoint can show
///   per-document text snippets without OOMing on big payloads.
pub async fn run_pipeline(
    nim_cfg: &NimEndpointsConfig,
    nim: &NimClient,
    vdb: Option<&VectorDbClient>,
    embed_model_default: &str,
    embed_sem: &Arc<Semaphore>,
    embed_max_pages: usize,
    item: &WorkItem,
) -> Result<WorkResult> {
    let filename = item.filename.clone().unwrap_or_else(|| item.id.clone());

    // ── Step 1: extract per-page text on a blocking thread.
    let payload: Bytes = item.payload.clone();
    let extracted: Vec<String> = tokio::task::spawn_blocking(move || extract_per_page(&payload))
        .await
        .map_err(|e| anyhow!("extract task panicked: {e}"))??;
    info!(
        item_id = item.id,
        filename, pages = extracted.len(),
        "extracted text"
    );

    // ── Step 2: embed via remote NIM (HTTP/2 keep-alive across workers).
    //
    // The embed pod has a hard memory ceiling (cgroup limit minus the
    // tmpfs /dev/shm reservation), and each request is a
    // `[N_pages, max_seq_len]` int64 tensor that materialises pinned host
    // memory inside Triton. With unbounded fan-out from N workers each
    // sending the full document in one shot, the embed pod gets OOMKilled
    // mid-run (we observed exit 137 after ~200 concurrent calls).
    //
    // Two-layer mitigation:
    //   1. Chunk per-document inputs into at most `embed_max_pages` pages
    //      per HTTP call. Bounds the worst-case request size from this
    //      worker even if a document has hundreds of pages.
    //   2. Acquire one permit from `embed_sem` per chunk. The semaphore
    //      is shared across every worker in the process, so the embed
    //      pod can never see more than `embed_max_concurrency` in-flight
    //      requests regardless of how many documents are being processed
    //      in parallel.
    let embed_url = nim_cfg
        .embed_invoke_url
        .as_deref()
        .ok_or_else(|| anyhow!("embed_invoke_url is not configured on this worker"))?;

    let embed_model = pick_embed_model(item, embed_model_default);
    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(extracted.len());
    for chunk in extracted.chunks(embed_max_pages) {
        // Permit acquisition is FIFO under tokio::sync::Semaphore, so
        // documents make forward progress even when the queue is deep.
        // The permit is released on drop at the end of this block.
        let _permit = embed_sem
            .clone()
            .acquire_owned()
            .await
            .map_err(|e| anyhow!("embed semaphore closed: {e}"))?;
        let part = embed_texts(
            nim,
            embed_url,
            &embed_model,
            chunk.to_vec(),
            InputType::Passage,
        )
        .await
        .context("embed call to NIM")?;
        if part.len() != chunk.len() {
            return Err(anyhow!(
                "embedding count {} does not match chunk size {} (item {})",
                part.len(),
                chunk.len(),
                item.id,
            ));
        }
        embeddings.extend(part);
    }

    if embeddings.len() != extracted.len() {
        return Err(anyhow!(
            "embedding count {} does not match input count {}",
            embeddings.len(),
            extracted.len()
        ));
    }

    // ── Step 3: build LanceDB rows.
    let path = filename.clone();
    let paths = vec![path.clone(); extracted.len()];
    let pages: Vec<i32> = (1..=extracted.len() as i32).collect();
    let metadatas: Vec<JsonValue> = pages
        .iter()
        .map(|p| json!({ "page_number": *p }))
        .collect();
    let rows = build_lancedb_rows(embeddings, &paths, &pages, &extracted, &metadatas);

    // ── Step 4: best-effort write to vectordb (mirrors Python fan-out).
    if let Some(client) = vdb {
        if let Err(err) = client.write_rows(&rows, &filename).await {
            tracing::warn!(filename, error = %err, "vectordb write failed; continuing");
        }
    }

    // ── Step 5: summary rows for the status endpoint (text-only, capped).
    //
    // CRITICAL: `String::truncate(n)` panics if `n` is NOT on a UTF-8 char
    // boundary. PDF text routinely contains multi-byte glyphs (em-dashes,
    // smart quotes, accents) and ToUnicode CMaps occasionally produce
    // garbled-but-valid UTF-8 sequences. We MUST truncate by char count,
    // not by byte index — otherwise the worker panics, the tokio worker
    // thread dies, and the whole pod crashes mid-pipeline.
    const MAX_SNIPPET_CHARS: usize = 500;
    let result_data: Vec<JsonValue> = rows
        .iter()
        .map(|r| {
            let text = truncate_utf8_safe(&r.text, MAX_SNIPPET_CHARS);
            json!({
                "page_number": r.page_number,
                "filename": r.filename,
                "text": text,
                "vector_dim": r.vector.len(),
            })
        })
        .collect();

    Ok(WorkResult {
        result_rows: rows.len() as u64,
        result_data: Some(result_data),
    })
}

fn pick_embed_model(item: &WorkItem, default_name: &str) -> String {
    // Per-request override gets first say; otherwise fall back to the
    // server-default. Trust-owned URL/api_key are never honoured.
    if let Some(spec) = &item.pipeline_spec {
        if let Some(name) = spec
            .get("embed_params")
            .and_then(|p| p.get("embedding_model"))
            .and_then(|v| v.as_str())
        {
            return name.to_string();
        }
    }
    if !default_name.is_empty() {
        default_name.to_string()
    } else {
        DEFAULT_EMBED_MODEL.to_string()
    }
}

/// Extract text per page. PDF uploads use `lopdf`; everything else is
/// treated as one UTF-8 page.
fn extract_per_page(bytes: &Bytes) -> Result<Vec<String>> {
    if looks_like_pdf(bytes) {
        match pdf::extract_text(bytes) {
            Ok(pages) => Ok(pages),
            Err(err) => {
                tracing::warn!(error = %err, "pdf extract fell back to single-blob text");
                Ok(vec![String::from_utf8_lossy(bytes).to_string()])
            }
        }
    } else {
        Ok(vec![String::from_utf8_lossy(bytes).to_string()])
    }
}

fn looks_like_pdf(bytes: &Bytes) -> bool {
    bytes.starts_with(b"%PDF-")
}

/// UTF-8-safe variant of `String::truncate`: caps the string at
/// `max_chars` Unicode scalar values and appends `…` if anything was
/// dropped. Never panics, regardless of input encoding edge cases.
fn truncate_utf8_safe(s: &str, max_chars: usize) -> String {
    let mut iter = s.chars();
    let mut out = String::with_capacity(s.len().min(max_chars * 4 + 4));
    let mut taken = 0usize;
    for c in iter.by_ref() {
        if taken >= max_chars {
            out.push('…');
            return out;
        }
        out.push(c);
        taken += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_handles_multibyte_boundary() {
        // 250 ASCII chars + an em-dash repeated 250 times. Byte length
        // straddles the multi-byte glyph at the ~500-char mark.
        let s: String = "a".repeat(250) + &"—".repeat(250);
        let out = truncate_utf8_safe(&s, 500);
        assert!(out.chars().count() <= 501);
    }

    #[test]
    fn truncate_short_string_passthrough() {
        assert_eq!(truncate_utf8_safe("hi", 500), "hi");
    }

    #[test]
    fn truncate_empty() {
        assert_eq!(truncate_utf8_safe("", 500), "");
    }

    #[test]
    fn truncate_pure_ascii_long() {
        let s = "x".repeat(1000);
        let out = truncate_utf8_safe(&s, 500);
        assert!(out.ends_with('…'));
        assert_eq!(out.chars().count(), 501);
    }
}
