// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal PDF helpers — page count + per-page text extraction.
//!
//! `lopdf` is a pure-Rust PDF parser; we use it for routing decisions
//! (small docs → realtime pool, larger docs → batch pool) and for the
//! cheap-text-extract path. Heavy-weight chart / table / image work is
//! delegated to remote NIMs (see `nrr-nim`).

use anyhow::{anyhow, Context, Result};
use bytes::Bytes;

/// Return the number of pages in a PDF, or 1 on parse failure (matches the
/// Python `_count_pdf_pages` fallback behaviour).
pub fn count_pages(bytes: &Bytes) -> u32 {
    match lopdf::Document::load_mem(bytes.as_ref()) {
        Ok(doc) => doc.get_pages().len() as u32,
        Err(err) => {
            tracing::warn!(error = %err, "could not parse PDF page count; defaulting to 1");
            1
        }
    }
}

/// Extract the raw text of every page, returned as one string per page in
/// document order. Uses `lopdf::Document::extract_text` per page index.
pub fn extract_text(bytes: &Bytes) -> Result<Vec<String>> {
    let doc = lopdf::Document::load_mem(bytes.as_ref()).context("parse pdf")?;
    let pages = doc.get_pages();
    let mut out = Vec::with_capacity(pages.len());
    for (page_num, _) in pages.iter() {
        let text = doc
            .extract_text(&[*page_num])
            .map_err(|e| anyhow!("extract page {page_num}: {e}"))?;
        out.push(text);
    }
    Ok(out)
}
