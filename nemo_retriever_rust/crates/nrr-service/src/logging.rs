// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `tracing` setup.
//!
//! Always logs to stderr (the canonical destination for containerised pods).
//! Optionally attaches a daily-rotating file appender when `logging.file` is
//! set AND the parent directory exists (or can be created). When the file
//! sink can't be opened — for example because a non-root container has no
//! write access to the configured log directory — we fall back to stderr
//! only, log a warning, and keep running. This mirrors the Python service's
//! tolerant `logging.FileHandler` behaviour and avoids panicking pods on
//! permission-denied.

use std::path::Path;

use nrr_core::config::LoggingConfig;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

/// Returns a `WorkerGuard` when a file appender was successfully attached
/// (must be kept alive for the appender to flush). Returns `None` if no
/// file sink was configured, or if the file sink could not be opened.
pub fn init(config: &LoggingConfig) -> Option<WorkerGuard> {
    let level = match config.level.to_ascii_uppercase().as_str() {
        "TRACE" => LevelFilter::TRACE,
        "DEBUG" => LevelFilter::DEBUG,
        "INFO" => LevelFilter::INFO,
        "WARN" | "WARNING" => LevelFilter::WARN,
        "ERROR" => LevelFilter::ERROR,
        _ => LevelFilter::INFO,
    };
    let env_filter = EnvFilter::builder()
        .with_default_directive(level.into())
        .from_env_lossy();

    let console_layer = tracing_subscriber::fmt::layer()
        .with_target(true)
        .with_writer(std::io::stderr);

    // `Option<L>` implements `Layer<S>` when `L: Layer<S>`, so the
    // file_layer branch slots into the registry whether present or not.
    let (file_layer, guard, file_warning) = match try_open_file_appender(&config.file) {
        Ok(Some((writer, guard))) => {
            let layer = tracing_subscriber::fmt::layer()
                .with_target(true)
                .with_writer(writer);
            (Some(layer), Some(guard), None)
        }
        Ok(None) => (None, None, None),
        Err(err) => (
            None,
            None,
            Some(format!(
                "tracing: file appender disabled — {}: {err}",
                config.file
            )),
        ),
    };

    let _ = tracing_subscriber::registry()
        .with(env_filter)
        .with(console_layer)
        .with(file_layer)
        .try_init();

    if let Some(msg) = file_warning {
        // Subscriber is now initialised, so this lands on stderr like
        // every other log entry.
        tracing::warn!("{msg}");
    }

    guard
}

/// Construct a daily-rolling file appender's writer at `file_path` if the
/// parent directory either exists or can be created AND the directory is
/// writable. Empty path → `Ok(None)` (no file sink, but no error). Missing
/// parent we can't create, or non-writable directory → `Err`, so the caller
/// can warn but still boot with stderr only.
fn try_open_file_appender(
    file_path: &str,
) -> std::io::Result<Option<(tracing_appender::non_blocking::NonBlocking, WorkerGuard)>> {
    if file_path.trim().is_empty() {
        return Ok(None);
    }
    let path = Path::new(file_path);
    let dir: &Path = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("retriever-service.log")
        .to_string();
    if !dir.exists() {
        std::fs::create_dir_all(dir)?;
    }
    // Probe writability before handing the path to the appender, since
    // `RollingFileAppender::new` panics on first write if the directory
    // exists but isn't writable.
    let probe = dir.join(format!(".{}.probe", std::process::id()));
    std::fs::File::create(&probe)?;
    let _ = std::fs::remove_file(&probe);

    let appender = RollingFileAppender::new(Rotation::DAILY, dir, file_name);
    let (writer, guard) = tracing_appender::non_blocking(appender);
    Ok(Some((writer, guard)))
}
