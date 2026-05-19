// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-memory sidecar payload store with TTL eviction.
//!
//! Mirrors `nemo_retriever.service.services.sidecar_store`. Tenants upload
//! ancillary files (CSV/JSON/parquet metadata) via `POST /v1/ingest/sidecar`
//! and reference them by id from a subsequent ingest call.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use dashmap::DashMap;
use uuid::Uuid;

const DEFAULT_MAX_BYTES: usize = 256 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct SidecarEntry {
    pub sidecar_id: String,
    pub filename: String,
    pub content_type: String,
    pub payload: Bytes,
    pub owner_token: Option<String>,
    pub expires_at: f64,
    pub consume_on_read: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum SidecarError {
    #[error("sidecar store is at capacity ({current} bytes used, limit {limit} bytes)")]
    AtCapacity { current: usize, limit: usize },
}

#[derive(Clone)]
pub struct SidecarStore {
    inner: Arc<SidecarInner>,
}

struct SidecarInner {
    entries: DashMap<String, SidecarEntry>,
    max_bytes: usize,
    used_bytes: parking_lot::Mutex<usize>,
}

impl SidecarStore {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_MAX_BYTES)
    }

    pub fn with_capacity(max_bytes: usize) -> Self {
        Self {
            inner: Arc::new(SidecarInner {
                entries: DashMap::new(),
                max_bytes,
                used_bytes: parking_lot::Mutex::new(0),
            }),
        }
    }

    pub fn put(
        &self,
        filename: impl Into<String>,
        content_type: impl Into<String>,
        payload: Bytes,
        owner_token: Option<String>,
        ttl_s: f64,
        consume_on_read: bool,
    ) -> Result<SidecarEntry, SidecarError> {
        self.evict_expired();
        let size = payload.len();
        {
            let mut used = self.inner.used_bytes.lock();
            if *used + size > self.inner.max_bytes {
                return Err(SidecarError::AtCapacity {
                    current: *used,
                    limit: self.inner.max_bytes,
                });
            }
            *used += size;
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        let entry = SidecarEntry {
            sidecar_id: Uuid::new_v4().simple().to_string(),
            filename: filename.into(),
            content_type: content_type.into(),
            payload,
            owner_token,
            expires_at: now + ttl_s.max(0.0),
            consume_on_read,
        };
        self.inner.entries.insert(entry.sidecar_id.clone(), entry.clone());
        Ok(entry)
    }

    pub fn get(&self, id: &str) -> Option<SidecarEntry> {
        self.evict_expired();
        self.inner.entries.get(id).map(|e| e.clone())
    }

    /// Take ownership of a sidecar payload. When the entry was created with
    /// `consume_on_read=true`, this also removes it from the store.
    pub fn consume(&self, id: &str) -> Option<SidecarEntry> {
        self.evict_expired();
        let consume_on_read = self
            .inner
            .entries
            .get(id)
            .map(|e| e.consume_on_read)
            .unwrap_or(false);
        if consume_on_read {
            let entry = self.inner.entries.remove(id)?.1;
            self.account_release(entry.payload.len());
            Some(entry)
        } else {
            self.inner.entries.get(id).map(|e| e.clone())
        }
    }

    pub fn delete(&self, id: &str) -> bool {
        match self.inner.entries.remove(id) {
            Some((_, entry)) => {
                self.account_release(entry.payload.len());
                true
            }
            None => false,
        }
    }

    fn account_release(&self, n: usize) {
        let mut used = self.inner.used_bytes.lock();
        *used = used.saturating_sub(n);
    }

    fn evict_expired(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        let mut expired: Vec<String> = Vec::new();
        for entry in self.inner.entries.iter() {
            if entry.expires_at < now {
                expired.push(entry.key().clone());
            }
        }
        for id in expired {
            if let Some((_, entry)) = self.inner.entries.remove(&id) {
                self.account_release(entry.payload.len());
            }
        }
    }

    pub fn used_bytes(&self) -> usize {
        *self.inner.used_bytes.lock()
    }

    pub fn capacity_bytes(&self) -> usize {
        self.inner.max_bytes
    }

    pub fn len(&self) -> usize {
        self.inner.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.entries.is_empty()
    }

    /// Convert seconds-since-epoch into an ISO-8601 UTC string for the
    /// `expires_at` response field.
    pub fn expires_at_iso(secs: f64) -> String {
        let dt = chrono::DateTime::<chrono::Utc>::from_timestamp(
            secs.trunc() as i64,
            ((secs.fract().abs()) * 1e9) as u32,
        )
        .unwrap_or_else(chrono::Utc::now);
        dt.to_rfc3339_opts(chrono::SecondsFormat::Micros, true)
    }
}

impl Default for SidecarStore {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(dead_code)]
const _SAMPLE_TTL: Duration = Duration::from_secs(3600);
