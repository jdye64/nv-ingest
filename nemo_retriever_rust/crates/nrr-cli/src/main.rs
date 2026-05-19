// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `retriever-rs` — entry-point binary for the Rust nemo-retriever service.
//!
//! Subcommands:
//! * `service start` — boot the Axum HTTP server.
//! * `service config` — print the resolved service configuration as JSON.
//! * `client health` — simple liveness probe against a running service.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use tracing::info;

use nrr_core::config::{load_config, ServiceConfig};
use nrr_service::app::{build_app, AppConfig};

#[derive(Debug, Parser)]
#[command(
    name = "retriever-rs",
    version,
    about = "NeMo Retriever Rust runtime",
    arg_required_else_help = true
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Service-mode subcommands.
    #[command(subcommand)]
    Service(ServiceCommand),
    /// Lightweight client subcommands.
    #[command(subcommand)]
    Client(ClientCommand),
}

#[derive(Debug, Subcommand)]
enum ServiceCommand {
    /// Boot the HTTP server.
    Start {
        /// Path to retriever-service.yaml (otherwise auto-discovered).
        #[arg(long)]
        config: Option<PathBuf>,
        /// Override server.host (e.g. 127.0.0.1).
        #[arg(long)]
        host: Option<String>,
        /// Override server.port.
        #[arg(long)]
        port: Option<u16>,
        /// Override mode (standalone, gateway, realtime, batch).
        #[arg(long)]
        mode: Option<String>,
    },
    /// Print the resolved configuration as JSON and exit.
    Config {
        #[arg(long)]
        config: Option<PathBuf>,
    },
}

#[derive(Debug, Subcommand)]
enum ClientCommand {
    /// GET /v1/health on the supplied URL.
    Health {
        #[arg(long, default_value = "http://localhost:7670")]
        url: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Service(sc) => run_service(sc).await,
        Command::Client(cc) => run_client(cc).await,
    }
}

async fn run_service(cmd: ServiceCommand) -> Result<()> {
    match cmd {
        ServiceCommand::Start {
            config,
            host,
            port,
            mode,
        } => {
            let mut overrides: Vec<(String, serde_yaml::Value)> = Vec::new();
            if let Some(h) = host {
                overrides.push(("server.host".into(), serde_yaml::Value::String(h)));
            }
            if let Some(p) = port {
                overrides.push(("server.port".into(), serde_yaml::Value::Number(p.into())));
            }
            if let Some(m) = mode {
                overrides.push(("mode".into(), serde_yaml::Value::String(m)));
            }
            let cfg = load_config(config.as_deref(), &overrides)
                .context("load service config")?;
            let _guard = nrr_service::logging::init(&cfg.logging);
            info!(mode = cfg.mode.as_str(), port = cfg.server.port, "boot");

            let cfg = Arc::new(cfg);
            let (work_rt, work_batch) = build_executors(&cfg);
            let (app, state) = build_app(AppConfig {
                config: cfg.clone(),
                realtime_work_fn: work_rt,
                batch_work_fn: work_batch,
            })
            .await;

            let addr: SocketAddr = format!("{}:{}", cfg.server.host, cfg.server.port)
                .parse()
                .map_err(|e| anyhow!("invalid server.host/port: {e}"))?;
            info!(%addr, role = state.role(), "listening");
            let listener = tokio::net::TcpListener::bind(addr)
                .await
                .with_context(|| format!("bind {addr}"))?;
            axum::serve(listener, app)
                .with_graceful_shutdown(shutdown_signal())
                .await
                .context("axum::serve")?;
            Ok(())
        }
        ServiceCommand::Config { config } => {
            let cfg = load_config(config.as_deref(), &[])?;
            let json = serde_json::to_string_pretty(&cfg)?;
            println!("{json}");
            Ok(())
        }
    }
}

fn build_executors(
    cfg: &Arc<ServiceConfig>,
) -> (
    Option<Arc<dyn nrr_pipeline::WorkFn>>,
    Option<Arc<dyn nrr_pipeline::WorkFn>>,
) {
    if cfg.mode.is_gateway() {
        return (None, None);
    }
    let nim = nrr_nim::NimClient::builder()
        .api_key(cfg.nim_endpoints.api_key.clone())
        .build()
        .expect("nim client build");
    let vdb = if cfg.vectordb.enabled && !cfg.vectordb.vectordb_url.is_empty() {
        match nrr_vdb::VectorDbClient::new(&cfg.vectordb.vectordb_url) {
            Ok(v) => Some(v),
            Err(err) => {
                tracing::warn!(error = %err, "vectordb client init failed; continuing without VDB");
                None
            }
        }
    } else {
        None
    };
    let executor = nrr_pipeline::InMemoryExecutor::new(cfg.clone(), nim, vdb);
    let executor: Arc<dyn nrr_pipeline::WorkFn> = Arc::new(executor);
    let rt = matches!(
        cfg.mode,
        nrr_core::config::ServiceMode::Standalone | nrr_core::config::ServiceMode::Realtime
    )
    .then(|| executor.clone());
    let bt = matches!(
        cfg.mode,
        nrr_core::config::ServiceMode::Standalone | nrr_core::config::ServiceMode::Batch
    )
    .then(|| executor);
    (rt, bt)
}

async fn shutdown_signal() {
    use tokio::signal;
    let ctrl_c = async { let _ = signal::ctrl_c().await; };
    #[cfg(unix)]
    let term = async {
        if let Ok(mut sig) = signal::unix::signal(signal::unix::SignalKind::terminate()) {
            sig.recv().await;
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = term => {},
    }
    info!("shutdown signal received");
}

async fn run_client(cmd: ClientCommand) -> Result<()> {
    match cmd {
        ClientCommand::Health { url } => {
            let client = nrr_client::Client::new(&url);
            let resp = client.health().await?;
            println!("{}", serde_json::to_string_pretty(&resp)?);
            Ok(())
        }
    }
}
