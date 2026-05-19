// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OCR / table-structure / graphic-elements NIM clients.
//!
//! These NIMs all share the same call shape: POST a base64-encoded image
//! plus a small JSON envelope, receive JSON detections back. The exact
//! response schema differs per model — callers handle the result as raw
//! `serde_json::Value` for now (the pipeline does the same in Python).

use bytes::Bytes;
use serde::Serialize;
use serde_json::Value as JsonValue;

use crate::client::{NimClient, NimError};

#[derive(Debug, Serialize)]
struct ImagePayload<'a> {
    /// Base64-encoded image bytes (without the `data:image/...;base64,` prefix).
    image: &'a str,
}

/// Minimal helper used by `ocr` / `table_structure` / `graphic_elements`.
/// Always POSTs `{ "image": "<b64>" }` and returns the raw JSON response.
pub async fn invoke_image_endpoint(
    client: &NimClient,
    endpoint: &str,
    image_bytes: &Bytes,
    endpoint_label: &'static str,
) -> Result<JsonValue, NimError> {
    use base64ct_lite::base64_encode;
    let b64 = base64_encode(image_bytes);
    let payload = ImagePayload { image: &b64 };
    let url = endpoint.trim_end_matches('/').to_string();
    client.post_json(&url, &payload, endpoint_label).await
}

/// Lightweight base64 encoder so we don't pull a heavyweight crate just for
/// one call site. Implemented in a private module so we can swap the
/// implementation later without touching callers.
mod base64ct_lite {
    use bytes::Bytes;
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    pub fn base64_encode(bytes: &Bytes) -> String {
        let mut out = String::with_capacity((bytes.len() + 2) / 3 * 4);
        let mut i = 0;
        while i + 3 <= bytes.len() {
            let n = (bytes[i] as u32) << 16 | (bytes[i + 1] as u32) << 8 | (bytes[i + 2] as u32);
            out.push(ALPHABET[((n >> 18) & 0x3f) as usize] as char);
            out.push(ALPHABET[((n >> 12) & 0x3f) as usize] as char);
            out.push(ALPHABET[((n >> 6) & 0x3f) as usize] as char);
            out.push(ALPHABET[(n & 0x3f) as usize] as char);
            i += 3;
        }
        let rem = bytes.len() - i;
        if rem == 1 {
            let n = (bytes[i] as u32) << 16;
            out.push(ALPHABET[((n >> 18) & 0x3f) as usize] as char);
            out.push(ALPHABET[((n >> 12) & 0x3f) as usize] as char);
            out.push('=');
            out.push('=');
        } else if rem == 2 {
            let n = (bytes[i] as u32) << 16 | (bytes[i + 1] as u32) << 8;
            out.push(ALPHABET[((n >> 18) & 0x3f) as usize] as char);
            out.push(ALPHABET[((n >> 12) & 0x3f) as usize] as char);
            out.push(ALPHABET[((n >> 6) & 0x3f) as usize] as char);
            out.push('=');
        }
        out
    }
}
