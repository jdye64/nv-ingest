// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline pool — bounded mpsc queue + N tokio worker tasks per pool.
//!
//! Mirrors `nemo_retriever.service.services.pipeline_pool.PipelinePool`:
//!
//! * `realtime` pool — small worker count, optimised for latency.
//! * `batch` pool — larger worker count, optimised for throughput.
//!
//! The `PipelinePool` is mode-aware: in `gateway` mode neither pool is
//! created; in `realtime`/`batch` mode only the matching pool starts; in
//! `standalone` both run side by side.

use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use parking_lot::Mutex;
use serde_json::Value as JsonValue;
use tokio::sync::mpsc;
use tokio::sync::OwnedSemaphorePermit;
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

use nrr_core::config::{PipelinePoolConfig, ServiceMode};
use nrr_core::tracker::JobTracker;

/// Identifies which pool a [`WorkItem`] should land in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PoolType {
    Realtime,
    Batch,
}

impl PoolType {
    pub fn as_str(self) -> &'static str {
        match self {
            PoolType::Realtime => "realtime",
            PoolType::Batch => "batch",
        }
    }
}

/// Unit of work submitted to a pool. Equivalent to the Python `WorkItem`
/// pydantic model.
#[derive(Debug, Clone)]
pub struct WorkItem {
    pub id: String,
    pub payload: Bytes,
    pub filename: Option<String>,
    pub callback_url: Option<String>,
    pub job_id: Option<String>,
    /// Validated per-request pipeline overrides. None = run the legacy
    /// startup-baked pipeline.
    pub pipeline_spec: Option<JsonValue>,
}

/// Result returned from a pool's work function. The optional `result_data`
/// field is bounded JSON-safe rows that the status endpoint surfaces back
/// to the client.
#[derive(Debug, Default, Clone)]
pub struct WorkResult {
    pub result_rows: u64,
    pub result_data: Option<Vec<JsonValue>>,
}

// `#[async_trait]` is required so the trait can be used behind `dyn`.
// Without it the desugared `impl Future` return type is not dyn-compatible.
#[async_trait::async_trait]
pub trait WorkFn: Send + Sync + 'static {
    async fn run(&self, item: WorkItem) -> anyhow::Result<WorkResult>;
}

/// Function-pointer adapter so plain async closures can be used as a
/// `WorkFn` without writing a struct.
pub struct ClosureWorkFn<F>(pub F);

#[async_trait::async_trait]
impl<F, Fut> WorkFn for ClosureWorkFn<F>
where
    F: Fn(WorkItem) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = anyhow::Result<WorkResult>> + Send + 'static,
{
    async fn run(&self, item: WorkItem) -> anyhow::Result<WorkResult> {
        (self.0)(item).await
    }
}

struct Pool {
    name: &'static str,
    tx: mpsc::Sender<WorkItem>,
    workers: Vec<JoinHandle<()>>,
    semaphore: Arc<tokio::sync::Semaphore>,
    capacity: usize,
    queue_depth: Arc<Mutex<usize>>,
    processed: Arc<Mutex<u64>>,
}

impl Pool {
    fn new(
        name: &'static str,
        num_workers: usize,
        max_queue_size: usize,
        work_fn: Arc<dyn WorkFn>,
        tracker: Option<JobTracker>,
        http_client: reqwest::Client,
    ) -> Self {
        let (tx, mut rx) = mpsc::channel::<WorkItem>(max_queue_size.max(1));
        let semaphore = Arc::new(tokio::sync::Semaphore::new(max_queue_size.max(1)));
        let queue_depth = Arc::new(Mutex::new(0usize));
        let processed = Arc::new(Mutex::new(0u64));

        // Single dispatcher loop hands items to a small number of worker
        // tasks via a fan-out semaphore. We keep a Vec<JoinHandle> only
        // for the dispatcher so shutdown is a single .await.
        let work_fn_dispatcher = Arc::clone(&work_fn);
        let queue_depth_d = Arc::clone(&queue_depth);
        let processed_d = Arc::clone(&processed);
        let workers_sem = Arc::new(tokio::sync::Semaphore::new(num_workers.max(1)));
        let tracker_d = tracker.clone();
        let http_d = http_client.clone();
        let dispatcher: JoinHandle<()> = tokio::spawn(async move {
            while let Some(item) = rx.recv().await {
                {
                    let mut d = queue_depth_d.lock();
                    *d = d.saturating_sub(1);
                }
                let permit = match workers_sem.clone().acquire_owned().await {
                    Ok(p) => p,
                    Err(_) => break,
                };
                let work_fn = Arc::clone(&work_fn_dispatcher);
                let processed = Arc::clone(&processed_d);
                let tracker = tracker_d.clone();
                let http = http_d.clone();
                let pool_name = name;
                tokio::spawn(async move {
                    Self::run_one(pool_name, item, work_fn, processed, tracker, http, permit)
                        .await;
                });
            }
        });

        Self {
            name,
            tx,
            workers: vec![dispatcher],
            semaphore,
            capacity: max_queue_size.max(1),
            queue_depth,
            processed,
        }
    }

    async fn run_one(
        pool_name: &'static str,
        item: WorkItem,
        work_fn: Arc<dyn WorkFn>,
        processed: Arc<Mutex<u64>>,
        tracker: Option<JobTracker>,
        http: reqwest::Client,
        _permit: OwnedSemaphorePermit,
    ) {
        let item_id = item.id.clone();
        let callback_url = item.callback_url.clone();
        if let Some(t) = &tracker {
            t.mark_processing(&item_id);
        }
        let t0 = Instant::now();
        // Catch panics from the work function so a single bad document
        // can never bring down the pod. A panicking work future is treated
        // as a failed item with the panic message recorded as the error.
        let outcome = {
            use futures::FutureExt;
            let fut = std::panic::AssertUnwindSafe(work_fn.run(item));
            match fut.catch_unwind().await {
                Ok(r) => r,
                Err(panic_payload) => {
                    let msg = panic_to_string(&panic_payload);
                    error!(
                        pool = pool_name,
                        item_id,
                        panic = %msg,
                        "work function panicked — recording as failed item"
                    );
                    Err(anyhow::anyhow!("worker panicked: {msg}"))
                }
            }
        };
        let elapsed = t0.elapsed().as_secs_f64();

        match outcome {
            Ok(result) => {
                if let Some(url) = &callback_url {
                    fire_gateway_callback(
                        &http,
                        url,
                        &item_id,
                        "completed",
                        Some(result.result_rows),
                        result.result_data.clone(),
                        None,
                    )
                    .await;
                } else if let Some(t) = &tracker {
                    t.mark_completed(
                        &item_id,
                        result.result_rows,
                        result.result_data,
                        Some(elapsed),
                    );
                }
                {
                    let mut p = processed.lock();
                    *p += 1;
                }
                tracing::debug!(pool = pool_name, item_id, elapsed_s = elapsed, "ok");
            }
            Err(err) => {
                let detail = format!("{err:#}");
                if let Some(url) = &callback_url {
                    fire_gateway_callback(
                        &http,
                        url,
                        &item_id,
                        "failed",
                        None,
                        None,
                        Some(detail.clone()),
                    )
                    .await;
                } else if let Some(t) = &tracker {
                    t.mark_failed(&item_id, detail.clone(), Some(elapsed));
                }
                error!(pool = pool_name, item_id, error = %detail, "work failed");
            }
        }
    }

    async fn submit(&self, item: WorkItem) -> bool {
        // Try to reserve a queue slot; if the channel is at capacity,
        // immediately reject so the HTTP layer can return 429.
        let permit = match self.semaphore.clone().try_acquire_owned() {
            Ok(p) => p,
            Err(_) => return false,
        };
        // Forget the permit: it represents a logical queue slot. The
        // dispatcher restores capacity by decrementing `queue_depth`
        // and we just track the current depth for stats.
        std::mem::forget(permit);
        {
            let mut d = self.queue_depth.lock();
            *d += 1;
        }
        match self.tx.send(item).await {
            Ok(()) => true,
            Err(_) => {
                // Channel closed during shutdown; release our slot.
                let mut d = self.queue_depth.lock();
                *d = d.saturating_sub(1);
                self.semaphore.add_permits(1);
                false
            }
        }
    }

    async fn shutdown(self) {
        drop(self.tx);
        for h in self.workers {
            let _ = h.await;
        }
        info!(pool = self.name, "pool shut down");
    }

    fn stats(&self) -> serde_json::Value {
        serde_json::json!({
            "name": self.name,
            "max_queue_size": self.capacity,
            "queue_depth": *self.queue_depth.lock(),
            "processed": *self.processed.lock(),
        })
    }
}

/// Best-effort extraction of a panic payload's message. Panic payloads
/// are `Box<dyn Any + Send>`; we try the two most common types
/// (`&'static str` and `String`) and fall back to a generic placeholder.
fn panic_to_string(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        return (*s).to_string();
    }
    if let Some(s) = payload.downcast_ref::<String>() {
        return s.clone();
    }
    "unknown panic payload".to_string()
}

async fn fire_gateway_callback(
    http: &reqwest::Client,
    url: &str,
    item_id: &str,
    status: &str,
    result_rows: Option<u64>,
    result_data: Option<Vec<JsonValue>>,
    error: Option<String>,
) {
    let mut payload = serde_json::json!({
        "id": item_id,
        "status": status,
        "result_rows": result_rows.unwrap_or(0),
    });
    if let Some(d) = result_data {
        payload["result_data"] = JsonValue::Array(d);
    }
    if let Some(e) = error {
        payload["error"] = JsonValue::String(e);
    }
    let res = http
        .post(url)
        .json(&payload)
        .timeout(Duration::from_secs(10))
        .send()
        .await;
    match res {
        Ok(r) if r.status().is_success() => {}
        Ok(r) => warn!(item_id, status = r.status().as_u16(), "gateway callback non-2xx"),
        Err(e) => warn!(item_id, error = %e, "gateway callback failed"),
    }
}

/// Manages realtime + batch pools per the configured `ServiceMode`.
pub struct PipelinePool {
    realtime: Option<Pool>,
    batch: Option<Pool>,
    mode: ServiceMode,
}

impl PipelinePool {
    pub fn new(
        config: &PipelinePoolConfig,
        mode: ServiceMode,
        realtime_work_fn: Option<Arc<dyn WorkFn>>,
        batch_work_fn: Option<Arc<dyn WorkFn>>,
        tracker: Option<JobTracker>,
        http_client: reqwest::Client,
    ) -> Self {
        let realtime = if matches!(mode, ServiceMode::Standalone | ServiceMode::Realtime) {
            realtime_work_fn.map(|f| {
                Pool::new(
                    "realtime",
                    config.realtime_workers as usize,
                    config.realtime_queue_size as usize,
                    f,
                    tracker.clone(),
                    http_client.clone(),
                )
            })
        } else {
            None
        };
        let batch = if matches!(mode, ServiceMode::Standalone | ServiceMode::Batch) {
            batch_work_fn.map(|f| {
                Pool::new(
                    "batch",
                    config.batch_workers as usize,
                    config.batch_queue_size as usize,
                    f,
                    tracker,
                    http_client,
                )
            })
        } else {
            None
        };
        info!(
            mode = mode.as_str(),
            realtime_workers = config.realtime_workers,
            realtime_queue = config.realtime_queue_size,
            batch_workers = config.batch_workers,
            batch_queue = config.batch_queue_size,
            "pipeline pool initialised"
        );
        Self { realtime, batch, mode }
    }

    pub fn mode(&self) -> ServiceMode {
        self.mode
    }

    pub async fn submit(&self, pool: PoolType, item: WorkItem) -> bool {
        let p = match pool {
            PoolType::Realtime => self.realtime.as_ref(),
            PoolType::Batch => self.batch.as_ref(),
        };
        match p {
            Some(p) => p.submit(item).await,
            None => false,
        }
    }

    pub fn has_capacity(&self, pool: PoolType) -> bool {
        let p = match pool {
            PoolType::Realtime => self.realtime.as_ref(),
            PoolType::Batch => self.batch.as_ref(),
        };
        match p {
            Some(p) => *p.queue_depth.lock() < p.capacity,
            None => false,
        }
    }

    pub fn stats(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        m.insert("mode".into(), JsonValue::String(self.mode.as_str().into()));
        if let Some(p) = &self.realtime {
            m.insert("realtime".into(), p.stats());
        }
        if let Some(p) = &self.batch {
            m.insert("batch".into(), p.stats());
        }
        JsonValue::Object(m)
    }

    pub async fn shutdown(self) {
        if let Some(p) = self.realtime {
            p.shutdown().await;
        }
        if let Some(p) = self.batch {
            p.shutdown().await;
        }
    }
}
