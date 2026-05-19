// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-memory pipeline executor for the Rust service.
//!
//! Replaces the Python `pipeline_executor.py` + `pipeline_pool.py` pair with
//! an idiomatic Tokio implementation:
//!
//! * `pool` — bounded mpsc per pool; one Tokio task per worker; back-pressure
//!   reports `false` from `submit()` so the HTTP layer can return 429.
//! * `executor` — runs the extract → embed → vdb pipeline for a single
//!   `WorkItem`. CPU-bound steps (PDF parsing) live inside
//!   `tokio::task::spawn_blocking` so they never starve the runtime.
//! * `pdf` — minimal PDF page-count + text extraction via `lopdf`. The
//!   heavy chart / table / OCR work is delegated to remote NIMs the same
//!   way the Python pipeline does it.
//!
//! Removed by design (per the user's brief): `batch` run_mode, Ray Data,
//! ProcessPoolExecutor isolation. Async tasks plus HTTP/2 connection
//! re-use are the speed multipliers here.

pub mod executor;
pub mod pdf;
pub mod pool;

pub use executor::InMemoryExecutor;
pub use pool::{ClosureWorkFn, PipelinePool, PoolType, WorkFn, WorkItem, WorkResult};
