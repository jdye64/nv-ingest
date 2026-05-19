// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Vector-DB sink for the Rust pipeline.
//!
//! Mirrors the Python pipeline's `_post_rows_to_vectordb` helper: rather than
//! linking LanceDB into every worker pod, we POST pre-built rows to a
//! dedicated `vectordb` Service. That keeps worker pods CPU-only and lets
//! the LanceDB pod own a single PVC.
//!
//! In a follow-up commit we can also add a direct LanceDB writer behind a
//! Cargo feature flag for users who want a single-binary deployment.

pub mod client;
pub mod schema;

pub use client::{VectorDbClient, VectorDbError};
pub use schema::{build_lancedb_row, build_lancedb_rows, LanceRow};
