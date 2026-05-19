// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Axum route handlers — one module per logical group.
//!
//! Wire-compatible mirror of `service.routers.{ingest,admin,metrics}`. The
//! module is intentionally flat; sub-modules export `router()` factories
//! that are merged in `app::build_app`.

pub mod admin;
pub mod dashboard;
pub mod ingest;
pub mod metrics;
pub mod query;
