// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-process event bus that broadcasts SSE events to subscribers.
//!
//! Mirrors `nemo_retriever.service.services.event_bus`. The Rust impl uses a
//! `tokio::sync::broadcast` channel so subscribers can receive events
//! concurrently without contending on a global lock. Filtering by `job_id`
//! happens on the subscriber side (the channel itself broadcasts every
//! event); slow subscribers are dropped with a warning rather than blocking
//! producers.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::Value as JsonValue;
use tokio::sync::broadcast;

const DEFAULT_CAPACITY: usize = 4096;

/// One published event. The `job_id` key (when present) is what subscribers
/// filter on.
#[derive(Debug, Clone)]
pub struct EventEnvelope {
    pub job_id: Option<String>,
    pub event: JsonValue,
}

#[derive(Clone)]
pub struct EventBus {
    inner: Arc<EventBusInner>,
}

struct EventBusInner {
    tx: broadcast::Sender<EventEnvelope>,
    next_sub_id: AtomicU64,
    subscriber_count: Arc<Mutex<u64>>,
}

impl EventBus {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let (tx, _rx) = broadcast::channel(capacity);
        Self {
            inner: Arc::new(EventBusInner {
                tx,
                next_sub_id: AtomicU64::new(1),
                subscriber_count: Arc::new(Mutex::new(0)),
            }),
        }
    }

    /// Publish an event to all subscribers. Best-effort; if the channel has
    /// no receivers the event is silently dropped (matches the Python bus's
    /// behaviour when there are no SSE clients connected).
    pub fn publish(&self, event: JsonValue, job_id: Option<String>) {
        let env = EventEnvelope { job_id, event };
        let _ = self.inner.tx.send(env);
    }

    /// Register a new subscriber. Returns the subscriber id and a
    /// [`Subscription`] handle that yields events for `job_id_filter`
    /// (or every event if `None`).
    pub fn subscribe(&self, job_id_filter: Option<String>) -> Subscription {
        let id = self.inner.next_sub_id.fetch_add(1, Ordering::Relaxed);
        {
            let mut c = self.inner.subscriber_count.lock();
            *c += 1;
        }
        Subscription {
            id,
            rx: self.inner.tx.subscribe(),
            filter: job_id_filter,
            counter: Arc::clone(&self.inner.subscriber_count),
        }
    }

    pub fn subscriber_count(&self) -> u64 {
        *self.inner.subscriber_count.lock()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Subscription {
    pub id: u64,
    rx: broadcast::Receiver<EventEnvelope>,
    filter: Option<String>,
    counter: Arc<Mutex<u64>>,
}

impl Subscription {
    /// Receive the next matching event. Returns `Ok(None)` when the channel
    /// is closed; returns `Ok(Some(event))` on the next event matching this
    /// subscription's filter; returns `Err(Lagged)` when the slow subscriber
    /// fell behind and the broadcast channel had to drop events.
    pub async fn recv(&mut self) -> Result<Option<JsonValue>, RecvError> {
        loop {
            match self.rx.recv().await {
                Ok(env) => match (&self.filter, &env.job_id) {
                    (Some(f), Some(j)) if f == j => return Ok(Some(env.event)),
                    (None, _) => return Ok(Some(env.event)),
                    _ => continue,
                },
                Err(broadcast::error::RecvError::Closed) => return Ok(None),
                Err(broadcast::error::RecvError::Lagged(n)) => return Err(RecvError::Lagged(n)),
            }
        }
    }
}

impl Drop for Subscription {
    fn drop(&mut self) {
        let mut c = self.counter.lock();
        *c = c.saturating_sub(1);
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RecvError {
    #[error("subscription lagged by {0} events")]
    Lagged(u64),
}
