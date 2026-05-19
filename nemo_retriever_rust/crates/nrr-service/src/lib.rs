// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Axum HTTP server: gateway / standalone / realtime / batch roles.

pub mod app;
pub mod auth;
pub mod logging;
pub mod metrics;
pub mod proxy;
pub mod routes;
pub mod state;

pub use app::{build_app, AppConfig};
pub use state::AppState;
