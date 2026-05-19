// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Async HTTP clients for the NVIDIA NIM microservices.
//!
//! In service mode the Python pipeline calls these endpoints over HTTP
//! instead of loading GPU models locally. The Rust port follows the same
//! pattern: every model invocation is an HTTP/2 request, multiplexed across
//! a shared `reqwest::Client` so we get connection-pool re-use without per-
//! request handshake cost.
//!
//! Supported endpoints:
//!
//! | Endpoint              | Purpose                                            |
//! |-----------------------|----------------------------------------------------|
//! | `embed_invoke_url`    | OpenAI-style `/v1/embeddings` for text + image     |
//! | `ocr_invoke_url`      | Per-page OCR for image regions                     |
//! | `page_elements_*`     | Detect bounding boxes on a page                    |
//! | `graphic_elements_*`  | Detect chart / figure regions                      |
//! | `table_structure_*`   | Convert table images to structured cells           |
//! | `caption_invoke_url`  | VLM caption (chat-completions style)               |
//!
//! All clients share the same retry / timeout policy and forward `api_key`
//! as `Authorization: Bearer <token>`.

pub mod client;
pub mod embed;
pub mod ocr;
pub mod page_elements;

pub use client::{NimClient, NimClientBuilder, NimError};
pub use embed::{EmbedRequest, EmbedResponse, EmbeddingRecord, InputType};
