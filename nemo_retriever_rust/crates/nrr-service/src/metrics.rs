// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus metrics — mirror of `service.services.prometheus`.
//!
//! Exposes the same metric names (`nemo_retriever_*`) so existing
//! prometheus-adapter / ServiceMonitor rules in the Helm chart continue
//! to scrape the Rust service unchanged.

use once_cell::sync::Lazy;
use prometheus::{
    Encoder, GaugeVec, HistogramOpts, HistogramVec, IntCounterVec, IntGaugeVec, Opts, Registry,
    TextEncoder,
};

#[derive(Clone)]
pub struct Metrics {
    pub registry: std::sync::Arc<Registry>,
    pub ingest_requests_total: IntCounterVec,
    pub ingest_bytes_total: IntCounterVec,
    pub ingest_documents_total: IntCounterVec,
    pub ingest_pages_total: IntCounterVec,
    pub gateway_forward_duration: HistogramVec,
    pub pool_max_queue_size: IntGaugeVec,
    pub pool_workers: IntGaugeVec,
    pub pool_queue_depth: IntGaugeVec,
    pub pool_queue_depth_ratio: GaugeVec,
    pub pool_processed_total: IntCounterVec,
    pub pool_processing_duration: HistogramVec,
}

pub static METRICS: Lazy<Metrics> = Lazy::new(Metrics::build);

impl Metrics {
    pub fn shared() -> Self {
        METRICS.clone()
    }

    fn build() -> Self {
        let registry = std::sync::Arc::new(Registry::new());

        let ingest_requests_total = IntCounterVec::new(
            Opts::new(
                "nemo_retriever_ingest_requests_total",
                "Total ingest requests handled, by role/endpoint/status class",
            ),
            &["role", "endpoint", "status"],
        )
        .unwrap();
        let ingest_bytes_total = IntCounterVec::new(
            Opts::new(
                "nemo_retriever_ingest_bytes_total",
                "Total bytes uploaded, by role/endpoint",
            ),
            &["role", "endpoint"],
        )
        .unwrap();
        let ingest_documents_total = IntCounterVec::new(
            Opts::new(
                "nemo_retriever_ingest_documents_total",
                "Total documents accepted (whole-doc uploads), by role",
            ),
            &["role"],
        )
        .unwrap();
        let ingest_pages_total = IntCounterVec::new(
            Opts::new(
                "nemo_retriever_ingest_pages_total",
                "Total pages accepted, by role",
            ),
            &["role"],
        )
        .unwrap();
        let gateway_forward_duration = HistogramVec::new(
            HistogramOpts::new(
                "nemo_retriever_gateway_forward_duration_seconds",
                "Gateway → backend forwarding duration",
            )
            .buckets(vec![0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]),
            &["backend"],
        )
        .unwrap();
        let pool_max_queue_size = IntGaugeVec::new(
            Opts::new(
                "nemo_retriever_pool_max_queue_size",
                "Configured max queue size per pool",
            ),
            &["pool"],
        )
        .unwrap();
        let pool_workers = IntGaugeVec::new(
            Opts::new(
                "nemo_retriever_pool_workers",
                "Configured worker count per pool",
            ),
            &["pool"],
        )
        .unwrap();
        let pool_queue_depth = IntGaugeVec::new(
            Opts::new(
                "nemo_retriever_pool_queue_depth",
                "Current queued items per pool",
            ),
            &["pool"],
        )
        .unwrap();
        let pool_queue_depth_ratio = GaugeVec::new(
            Opts::new(
                "nemo_retriever_pool_queue_depth_ratio",
                "Current queued items / max queue size per pool, in [0,1]",
            ),
            &["pool"],
        )
        .unwrap();
        let pool_processed_total = IntCounterVec::new(
            Opts::new(
                "nemo_retriever_pool_processed_total",
                "Total work items processed per pool, by terminal outcome",
            ),
            &["pool", "outcome"],
        )
        .unwrap();
        let pool_processing_duration = HistogramVec::new(
            HistogramOpts::new(
                "nemo_retriever_pool_processing_duration_seconds",
                "Per-item pipeline duration, observed inside the worker loop",
            )
            .buckets(vec![0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0]),
            &["pool"],
        )
        .unwrap();

        registry
            .register(Box::new(ingest_requests_total.clone()))
            .unwrap();
        registry
            .register(Box::new(ingest_bytes_total.clone()))
            .unwrap();
        registry
            .register(Box::new(ingest_documents_total.clone()))
            .unwrap();
        registry
            .register(Box::new(ingest_pages_total.clone()))
            .unwrap();
        registry
            .register(Box::new(gateway_forward_duration.clone()))
            .unwrap();
        registry
            .register(Box::new(pool_max_queue_size.clone()))
            .unwrap();
        registry.register(Box::new(pool_workers.clone())).unwrap();
        registry
            .register(Box::new(pool_queue_depth.clone()))
            .unwrap();
        registry
            .register(Box::new(pool_queue_depth_ratio.clone()))
            .unwrap();
        registry
            .register(Box::new(pool_processed_total.clone()))
            .unwrap();
        registry
            .register(Box::new(pool_processing_duration.clone()))
            .unwrap();

        Self {
            registry,
            ingest_requests_total,
            ingest_bytes_total,
            ingest_documents_total,
            ingest_pages_total,
            gateway_forward_duration,
            pool_max_queue_size,
            pool_workers,
            pool_queue_depth,
            pool_queue_depth_ratio,
            pool_processed_total,
            pool_processing_duration,
        }
    }

    /// Render the registry to the Prometheus text-exposition format.
    pub fn render(&self) -> String {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buf = Vec::with_capacity(4096);
        if let Err(err) = encoder.encode(&metric_families, &mut buf) {
            return format!("# encoding error: {err}\n");
        }
        String::from_utf8(buf).unwrap_or_else(|_| "# non-utf8 metrics".into())
    }
}
