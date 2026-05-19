// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-memory `JobTracker` that mirrors `service.services.job_tracker`.
//!
//! Two layers:
//! * [`DocumentRecord`] — one per uploaded file (was `JobRecord` historically).
//! * [`JobAggregate`]   — one per client-issued job; rolls up per-document
//!   counts and exposes a derived terminal status.
//!
//! All writes are guarded by a single `parking_lot::Mutex`. Reads return
//! defensive clones so callers can hold them without keeping the lock.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};
use thiserror::Error;

use crate::bus::EventBus;
use crate::util::utc_now_iso;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "lowercase")]
pub enum DocumentStatus {
    Pending,
    Processing,
    Completed,
    Failed,
}

impl DocumentStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            DocumentStatus::Pending => "pending",
            DocumentStatus::Processing => "processing",
            DocumentStatus::Completed => "completed",
            DocumentStatus::Failed => "failed",
        }
    }
    pub fn is_terminal(self) -> bool {
        matches!(self, DocumentStatus::Completed | DocumentStatus::Failed)
    }
    pub const ALL: [DocumentStatus; 4] = [
        DocumentStatus::Pending,
        DocumentStatus::Processing,
        DocumentStatus::Completed,
        DocumentStatus::Failed,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobAggregateStatus {
    Pending,
    Processing,
    Completed,
    Failed,
    PartialSuccess,
}

impl JobAggregateStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            JobAggregateStatus::Pending => "pending",
            JobAggregateStatus::Processing => "processing",
            JobAggregateStatus::Completed => "completed",
            JobAggregateStatus::Failed => "failed",
            JobAggregateStatus::PartialSuccess => "partial_success",
        }
    }
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            JobAggregateStatus::Completed
                | JobAggregateStatus::Failed
                | JobAggregateStatus::PartialSuccess
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentRecord {
    pub id: String,
    pub job_id: String,
    pub status: DocumentStatus,
    pub submitted_at: String,
    #[serde(default)]
    pub started_at: Option<String>,
    #[serde(default)]
    pub completed_at: Option<String>,
    #[serde(default)]
    pub elapsed_s: Option<f64>,
    #[serde(default)]
    pub result_rows: Option<u64>,
    #[serde(default)]
    pub result_data: Option<Vec<JsonValue>>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub filename: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobAggregate {
    pub job_id: String,
    pub expected_documents: u32,
    pub document_ids: Vec<String>,
    /// Counts keyed by `DocumentStatus::as_str()`.
    pub counts: BTreeMap<String, u64>,
    pub status: JobAggregateStatus,
    pub created_at: String,
    #[serde(default)]
    pub started_at: Option<String>,
    #[serde(default)]
    pub finalized_at: Option<String>,
    #[serde(default)]
    pub elapsed_s: Option<f64>,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, JsonValue>,
}

#[derive(Debug, Error)]
pub enum JobTrackerError {
    #[error("job {0:?} not found")]
    JobNotFound(String),
    #[error("job {0:?} is at capacity")]
    JobFull(String),
    #[error("job {0:?} has already finalized")]
    JobFinalized(String),
    #[error("job {0:?} already exists")]
    JobAlreadyExists(String),
    #[error("document {0:?} already registered")]
    DocumentAlreadyRegistered(String),
    #[error("invalid expected_documents: must be > 0")]
    InvalidExpectedDocuments,
}

impl JobTrackerError {
    pub fn http_status(&self) -> u16 {
        match self {
            JobTrackerError::JobNotFound(_) => 404,
            JobTrackerError::JobFull(_) | JobTrackerError::JobFinalized(_) => 409,
            JobTrackerError::JobAlreadyExists(_) => 409,
            JobTrackerError::DocumentAlreadyRegistered(_) => 409,
            JobTrackerError::InvalidExpectedDocuments => 400,
        }
    }
}

const DEFAULT_TTL_S: f64 = 4.0 * 3600.0;
const DEFAULT_MAX_JOBS: usize = 200_000;
const EVICTION_INTERVAL: u64 = 50;

#[derive(Clone)]
pub struct JobTracker {
    inner: Arc<Mutex<TrackerInner>>,
    bus: Arc<Mutex<Option<EventBus>>>,
}

struct TrackerInner {
    jobs: HashMap<String, JobAggregate>,
    documents: HashMap<String, DocumentRecord>,
    started_mono: HashMap<String, Instant>,
    job_started_mono: HashMap<String, Instant>,
    progress_published: HashMap<String, u32>,
    reg_count: u64,
    ttl_s: f64,
    max_jobs: usize,
    progress_step: u32,
}

impl JobTracker {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(TrackerInner {
                jobs: HashMap::new(),
                documents: HashMap::new(),
                started_mono: HashMap::new(),
                job_started_mono: HashMap::new(),
                progress_published: HashMap::new(),
                reg_count: 0,
                ttl_s: DEFAULT_TTL_S,
                max_jobs: DEFAULT_MAX_JOBS,
                progress_step: 10,
            })),
            bus: Arc::new(Mutex::new(None)),
        }
    }

    pub fn set_event_bus(&self, bus: EventBus) {
        *self.bus.lock() = Some(bus);
    }

    pub fn set_progress_step(&self, step: u32) {
        if step > 0 {
            self.inner.lock().progress_step = step;
        }
    }

    // ── job lifecycle ────────────────────────────────────────────────

    pub fn register_job(
        &self,
        job_id: &str,
        expected_documents: u32,
        label: Option<String>,
        metadata: BTreeMap<String, JsonValue>,
    ) -> Result<JobAggregate, JobTrackerError> {
        if expected_documents == 0 {
            return Err(JobTrackerError::InvalidExpectedDocuments);
        }
        let agg = {
            let mut g = self.inner.lock();
            if g.jobs.contains_key(job_id) {
                return Err(JobTrackerError::JobAlreadyExists(job_id.into()));
            }
            let mut counts = BTreeMap::new();
            for s in DocumentStatus::ALL.iter() {
                counts.insert(s.as_str().to_string(), 0u64);
            }
            let agg = JobAggregate {
                job_id: job_id.into(),
                expected_documents,
                document_ids: Vec::new(),
                counts,
                status: JobAggregateStatus::Pending,
                created_at: utc_now_iso(),
                started_at: None,
                finalized_at: None,
                elapsed_s: None,
                label,
                metadata,
            };
            g.jobs.insert(job_id.into(), agg.clone());
            g.reg_count += 1;
            if g.reg_count % EVICTION_INTERVAL == 0 {
                Self::evict_locked(&mut g);
            }
            agg
        };
        self.publish_job_event("job_created", &agg);
        Ok(agg)
    }

    pub fn get_job(&self, job_id: &str) -> Option<JobAggregate> {
        self.inner.lock().jobs.get(job_id).cloned()
    }

    pub fn all_jobs(&self) -> Vec<JobAggregate> {
        self.inner.lock().jobs.values().cloned().collect()
    }

    pub fn job_documents(&self, job_id: &str) -> Vec<DocumentRecord> {
        let g = self.inner.lock();
        let Some(agg) = g.jobs.get(job_id) else {
            return Vec::new();
        };
        agg.document_ids
            .iter()
            .filter_map(|id| g.documents.get(id).cloned())
            .collect()
    }

    pub fn all_documents(&self) -> Vec<DocumentRecord> {
        self.inner.lock().documents.values().cloned().collect()
    }

    pub fn get_document(&self, document_id: &str) -> Option<DocumentRecord> {
        self.inner.lock().documents.get(document_id).cloned()
    }

    /// Take ownership of the document's `result_data` (and clear it from
    /// memory). Mirrors `consume_result_data` in the Python tracker.
    pub fn consume_result_data(&self, document_id: &str) -> Option<Vec<JsonValue>> {
        let mut g = self.inner.lock();
        let rec = g.documents.get_mut(document_id)?;
        rec.result_data.take()
    }

    // ── document lifecycle ───────────────────────────────────────────

    pub fn register_document(
        &self,
        document_id: &str,
        job_id: &str,
        filename: Option<String>,
    ) -> Result<DocumentRecord, JobTrackerError> {
        let mut g = self.inner.lock();
        let agg = g
            .jobs
            .get_mut(job_id)
            .ok_or_else(|| JobTrackerError::JobNotFound(job_id.into()))?;
        if agg.status.is_terminal() {
            return Err(JobTrackerError::JobFinalized(job_id.into()));
        }
        if agg.document_ids.len() >= agg.expected_documents as usize {
            return Err(JobTrackerError::JobFull(job_id.into()));
        }
        if g.documents.contains_key(document_id) {
            return Err(JobTrackerError::DocumentAlreadyRegistered(
                document_id.into(),
            ));
        }
        let rec = DocumentRecord {
            id: document_id.into(),
            job_id: job_id.into(),
            status: DocumentStatus::Pending,
            submitted_at: utc_now_iso(),
            started_at: None,
            completed_at: None,
            elapsed_s: None,
            result_rows: None,
            result_data: None,
            error: None,
            filename,
        };
        g.documents.insert(document_id.into(), rec.clone());
        let agg = g.jobs.get_mut(job_id).unwrap();
        agg.document_ids.push(document_id.into());
        let p = agg.counts.entry("pending".into()).or_insert(0);
        *p += 1;
        Ok(rec)
    }

    pub fn mark_processing(&self, document_id: &str) {
        let bus_event = {
            let mut g = self.inner.lock();
            // Step 1 — find job_id and validate transition (short borrow).
            let job_id = match g.documents.get_mut(document_id) {
                Some(rec) if rec.status == DocumentStatus::Pending => {
                    rec.status = DocumentStatus::Processing;
                    rec.started_at = Some(utc_now_iso());
                    rec.job_id.clone()
                }
                _ => return,
            };
            // Step 2 — record start time on the side maps.
            g.started_mono.insert(document_id.into(), Instant::now());
            // Step 3 — adjust aggregate counts.
            Self::adjust_counts_locked(
                &mut g,
                &job_id,
                DocumentStatus::Pending,
                DocumentStatus::Processing,
            );
            // Step 4 — promote job to processing if first transition.
            let promoted = match g.jobs.get_mut(&job_id) {
                Some(agg) if agg.status == JobAggregateStatus::Pending => {
                    agg.status = JobAggregateStatus::Processing;
                    agg.started_at = Some(utc_now_iso());
                    true
                }
                _ => false,
            };
            if promoted {
                g.job_started_mono.insert(job_id.clone(), Instant::now());
                g.jobs.get(&job_id).cloned()
            } else {
                None
            }
        };
        if let Some(agg) = bus_event {
            self.publish_job_event("job_started", &agg);
        }
    }

    pub fn mark_completed(
        &self,
        document_id: &str,
        result_rows: u64,
        result_data: Option<Vec<JsonValue>>,
        elapsed_s: Option<f64>,
    ) {
        self.mark_terminal(
            document_id,
            DocumentStatus::Completed,
            result_rows,
            result_data,
            None,
            elapsed_s,
        );
    }

    pub fn mark_failed(&self, document_id: &str, error: String, elapsed_s: Option<f64>) {
        self.mark_terminal(
            document_id,
            DocumentStatus::Failed,
            0,
            None,
            Some(error),
            elapsed_s,
        );
    }

    fn mark_terminal(
        &self,
        document_id: &str,
        new_status: DocumentStatus,
        result_rows: u64,
        result_data: Option<Vec<JsonValue>>,
        error: Option<String>,
        elapsed_s: Option<f64>,
    ) {
        let (doc_snap, progress_snap, finalized_snap) = {
            let mut g = self.inner.lock();
            // Step 1 — validate transition and capture old status + job_id.
            let (old_status, job_id) = match g.documents.get(document_id) {
                Some(rec) if !rec.status.is_terminal() => (rec.status, rec.job_id.clone()),
                _ => return,
            };
            // Step 2 — compute elapsed_s from side map (must release docs borrow).
            let computed_elapsed = match elapsed_s {
                Some(e) => Some(e),
                None => g
                    .started_mono
                    .remove(document_id)
                    .map(|t| (t.elapsed().as_secs_f64() * 10000.0).round() / 10000.0),
            };
            // Step 3 — mutate the document record.
            let doc_snap = {
                let rec = g.documents.get_mut(document_id).expect("just read above");
                rec.status = new_status;
                rec.completed_at = Some(utc_now_iso());
                rec.result_rows = Some(result_rows);
                rec.result_data = result_data;
                rec.error = error;
                rec.elapsed_s = computed_elapsed;
                rec.clone()
            };
            // Step 4 — adjust aggregate counts.
            Self::adjust_counts_locked(&mut g, &job_id, old_status, new_status);

            // Step 5 — read counts and decide finalize/progress (short borrow).
            let (terminal_count, expected, agg_terminal) = match g.jobs.get(&job_id) {
                Some(agg) => (
                    agg.counts.get("completed").copied().unwrap_or(0)
                        + agg.counts.get("failed").copied().unwrap_or(0),
                    agg.expected_documents,
                    agg.status.is_terminal(),
                ),
                None => (0, 0, false),
            };
            let mut progress_snap: Option<JobAggregate> = None;
            let mut finalized_snap: Option<JobAggregate> = None;
            if terminal_count as u32 == expected && !agg_terminal {
                // Step 6a — finalize: remove start time first to avoid concurrent borrows.
                let t0 = g.job_started_mono.remove(&job_id);
                if let Some(agg) = g.jobs.get_mut(&job_id) {
                    agg.status = Self::derive_terminal_status_locked(agg);
                    agg.finalized_at = Some(utc_now_iso());
                    if let Some(t0) = t0 {
                        agg.elapsed_s =
                            Some((t0.elapsed().as_secs_f64() * 10000.0).round() / 10000.0);
                    }
                    finalized_snap = Some(agg.clone());
                }
            } else if terminal_count > 0 {
                let last = g.progress_published.get(&job_id).copied().unwrap_or(0);
                let step = g.progress_step as u64;
                if terminal_count - last as u64 >= step {
                    g.progress_published
                        .insert(job_id.clone(), terminal_count as u32);
                    progress_snap = g.jobs.get(&job_id).cloned();
                }
            }
            (doc_snap, progress_snap, finalized_snap)
        };

        self.publish_document_event(&doc_snap);
        if let Some(p) = progress_snap {
            self.publish_job_event("job_progress", &p);
        }
        if let Some(f) = finalized_snap {
            let event_name = match f.status {
                JobAggregateStatus::Failed => "job_failed",
                JobAggregateStatus::PartialSuccess => "job_partial",
                _ => "job_finalized",
            };
            self.publish_job_event(event_name, &f);
        }
    }

    // ── internal helpers ─────────────────────────────────────────────

    fn adjust_counts_locked(
        g: &mut TrackerInner,
        job_id: &str,
        old: DocumentStatus,
        new: DocumentStatus,
    ) {
        let Some(agg) = g.jobs.get_mut(job_id) else {
            return;
        };
        let entry = agg.counts.entry(old.as_str().into()).or_insert(0);
        *entry = entry.saturating_sub(1);
        let entry = agg.counts.entry(new.as_str().into()).or_insert(0);
        *entry += 1;
    }

    fn derive_terminal_status_locked(agg: &JobAggregate) -> JobAggregateStatus {
        let completed = agg.counts.get("completed").copied().unwrap_or(0);
        let failed = agg.counts.get("failed").copied().unwrap_or(0);
        if failed == 0 && completed > 0 {
            return JobAggregateStatus::Completed;
        }
        if completed == 0 && failed > 0 {
            return JobAggregateStatus::Failed;
        }
        if completed > 0 && failed > 0 {
            return JobAggregateStatus::PartialSuccess;
        }
        JobAggregateStatus::Failed
    }

    fn evict_locked(g: &mut TrackerInner) {
        use chrono::DateTime;
        let now = chrono::Utc::now();
        let mut expired: Vec<String> = Vec::new();
        for (jid, agg) in g.jobs.iter() {
            if !agg.status.is_terminal() {
                continue;
            }
            if let Some(fin) = agg.finalized_at.as_deref() {
                if let Ok(dt) = DateTime::parse_from_rfc3339(fin) {
                    let age = now.signed_duration_since(dt.with_timezone(&chrono::Utc));
                    if age.num_seconds() as f64 > g.ttl_s {
                        expired.push(jid.clone());
                    }
                }
            }
        }
        for jid in &expired {
            Self::drop_job_locked(g, jid);
        }
        if g.jobs.len() > g.max_jobs {
            let mut terminal: Vec<(String, String)> = g
                .jobs
                .iter()
                .filter(|(_, a)| a.status.is_terminal())
                .map(|(k, v)| (k.clone(), v.finalized_at.clone().unwrap_or_default()))
                .collect();
            terminal.sort_by(|a, b| a.1.cmp(&b.1));
            let excess = g.jobs.len() - g.max_jobs;
            for (jid, _) in terminal.into_iter().take(excess) {
                Self::drop_job_locked(g, &jid);
            }
        }
    }

    fn drop_job_locked(g: &mut TrackerInner, job_id: &str) {
        let Some(agg) = g.jobs.remove(job_id) else {
            return;
        };
        for did in &agg.document_ids {
            g.documents.remove(did);
            g.started_mono.remove(did);
        }
        g.job_started_mono.remove(job_id);
        g.progress_published.remove(job_id);
    }

    fn publish_document_event(&self, rec: &DocumentRecord) {
        let bus = self.bus.lock().clone();
        let Some(bus) = bus else { return };
        let event = json!({
            "type": rec.status.as_str(),
            "id": rec.id,
            "document_id": rec.id,
            "job_id": rec.job_id,
            "status": rec.status.as_str(),
            "result_rows": rec.result_rows,
            "elapsed_s": rec.elapsed_s,
            "error": rec.error,
            "filename": rec.filename,
        });
        bus.publish(event, Some(rec.job_id.clone()));
    }

    fn publish_job_event(&self, event_type: &str, agg: &JobAggregate) {
        let bus = self.bus.lock().clone();
        let Some(bus) = bus else { return };
        let completed = agg.counts.get("completed").copied().unwrap_or(0);
        let failed = agg.counts.get("failed").copied().unwrap_or(0);
        let terminal = completed + failed;
        let remaining = (agg.expected_documents as i64 - terminal as i64).max(0);
        let progress_pct = if agg.expected_documents > 0 {
            ((terminal as f64 * 100.0 / agg.expected_documents as f64) * 100.0).round() / 100.0
        } else {
            0.0
        };
        let event = json!({
            "type": event_type,
            "id": agg.job_id,
            "job_id": agg.job_id,
            "status": agg.status.as_str(),
            "expected_documents": agg.expected_documents,
            "counts": agg.counts,
            "completed": completed,
            "failed": failed,
            "remaining": remaining,
            "progress_pct": progress_pct,
            "elapsed_s": agg.elapsed_s,
            "started_at": agg.started_at,
            "finalized_at": agg.finalized_at,
            "label": agg.label,
        });
        bus.publish(event, Some(agg.job_id.clone()));
    }

    pub fn summary(&self) -> JsonValue {
        let g = self.inner.lock();
        let mut by_status: BTreeMap<&str, u64> = BTreeMap::new();
        for s in &[
            "pending",
            "processing",
            "completed",
            "failed",
            "partial_success",
        ] {
            by_status.insert(*s, 0);
        }
        for agg in g.jobs.values() {
            *by_status.entry(agg.status.as_str()).or_insert(0) += 1;
        }
        json!({
            "total_jobs": g.jobs.len(),
            "total_documents": g.documents.len(),
            "pending": by_status.get("pending").copied().unwrap_or(0),
            "processing": by_status.get("processing").copied().unwrap_or(0),
            "completed": by_status.get("completed").copied().unwrap_or(0),
            "failed": by_status.get("failed").copied().unwrap_or(0),
            "partial_success": by_status.get("partial_success").copied().unwrap_or(0),
        })
    }
}

impl Default for JobTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn job_lifecycle_completes() {
        let t = JobTracker::new();
        let job = t
            .register_job("j1", 2, Some("smoke".into()), Default::default())
            .unwrap();
        assert_eq!(job.expected_documents, 2);

        t.register_document("d1", "j1", Some("a.pdf".into())).unwrap();
        t.register_document("d2", "j1", Some("b.pdf".into())).unwrap();
        t.mark_processing("d1");
        t.mark_completed("d1", 5, None, None);
        let j = t.get_job("j1").unwrap();
        assert!(!j.status.is_terminal());

        t.mark_processing("d2");
        t.mark_completed("d2", 3, None, None);
        let j = t.get_job("j1").unwrap();
        assert!(j.status.is_terminal());
        assert_eq!(j.status, JobAggregateStatus::Completed);
    }

    #[test]
    fn partial_success_when_mixed() {
        let t = JobTracker::new();
        t.register_job("j", 2, None, Default::default()).unwrap();
        t.register_document("a", "j", None).unwrap();
        t.register_document("b", "j", None).unwrap();
        t.mark_completed("a", 1, None, None);
        t.mark_failed("b", "boom".into(), None);
        let j = t.get_job("j").unwrap();
        assert_eq!(j.status, JobAggregateStatus::PartialSuccess);
    }

    #[test]
    fn rejects_extra_uploads() {
        let t = JobTracker::new();
        t.register_job("j", 1, None, Default::default()).unwrap();
        t.register_document("a", "j", None).unwrap();
        let err = t.register_document("b", "j", None).unwrap_err();
        assert_eq!(err.http_status(), 409);
    }
}
