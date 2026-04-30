# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CRUD operations for all service-mode SQLite tables.

Every write method is wrapped with :func:`execute_with_retry` so that
transient ``OperationalError: database is locked`` errors (common with
16+ concurrent writer processes) are retried with exponential backoff
instead of crashing the worker.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone

from nemo_retriever.service.db.engine import DatabaseEngine, execute_with_retry
from nemo_retriever.service.models.document import Document, ProcessingStatus
from nemo_retriever.service.models.job import Job
from nemo_retriever.service.models.metrics import ProcessingMetric
from nemo_retriever.service.models.page_processing_log import PageProcessingLog
from nemo_retriever.service.models.page_result import PageResult

logger = logging.getLogger(__name__)


class Repository:
    """Data-access layer wrapping all service tables.

    All public write methods delegate to ``execute_with_retry`` so that
    SQLite lock contention is handled transparently.
    """

    def __init__(self, engine: DatabaseEngine) -> None:
        self._engine = engine

    @property
    def _conn(self):
        return self._engine.connection

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def insert_job(self, job: Job) -> None:
        def _do():
            row = job.to_row()
            cols = ", ".join(row.keys())
            placeholders = ", ".join(f":{k}" for k in row.keys())
            self._conn.execute(f"INSERT INTO jobs ({cols}) VALUES ({placeholders})", row)
            self._conn.commit()

        execute_with_retry(_do)

    def get_job(self, job_id: str) -> Job | None:
        cur = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cur.fetchone()
        return Job(**dict(row)) if row else None

    def get_job_by_sha(self, sha256: str) -> Job | None:
        cur = self._conn.execute("SELECT * FROM jobs WHERE content_sha256 = ?", (sha256,))
        row = cur.fetchone()
        return Job(**dict(row)) if row else None

    def increment_job_pages_submitted(self, job_id: str) -> int:
        def _do() -> int:
            now = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                "UPDATE jobs SET pages_submitted = pages_submitted + 1, updated_at = ? WHERE id = ?",
                (now, job_id),
            )
            self._conn.commit()
            cur = self._conn.execute("SELECT pages_submitted FROM jobs WHERE id = ?", (job_id,))
            row = cur.fetchone()
            return int(row["pages_submitted"]) if row else 0

        return execute_with_retry(_do)

    def increment_job_pages_completed(self, job_id: str) -> int:
        def _do() -> int:
            now = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                "UPDATE jobs SET pages_completed = pages_completed + 1, updated_at = ? WHERE id = ?",
                (now, job_id),
            )
            self._conn.commit()
            cur = self._conn.execute("SELECT pages_completed, total_pages FROM jobs WHERE id = ?", (job_id,))
            row = cur.fetchone()
            return int(row["pages_completed"]) if row else 0

        return execute_with_retry(_do)

    def update_job_status(self, job_id: str, status: ProcessingStatus) -> None:
        def _do():
            now = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                "UPDATE jobs SET processing_status = ?, updated_at = ? WHERE id = ?",
                (status.value, now, job_id),
            )
            self._conn.commit()

        execute_with_retry(_do)

    def cancel_job(self, job_id: str) -> dict[str, int]:
        """Mark a job as cancelled along with any of its still-pending documents.

        Returns a dict with counts of what was changed:
            ``{"documents_cancelled": N, "already_terminal": <bool as int>}``

        Already-terminal jobs (complete/failed/cancelled) are left alone and
        the caller is informed via ``already_terminal=1``.
        """

        def _do() -> dict[str, int]:
            now = datetime.now(timezone.utc).isoformat()
            cur = self._conn.execute(
                "SELECT processing_status FROM jobs WHERE id = ?",
                (job_id,),
            )
            row = cur.fetchone()
            if row is None:
                self._conn.commit()
                return {"documents_cancelled": 0, "already_terminal": 0, "found": 0}
            current = row["processing_status"]
            if current in ("complete", "failed", "cancelled"):
                self._conn.commit()
                return {"documents_cancelled": 0, "already_terminal": 1, "found": 1}

            cur = self._conn.execute(
                "UPDATE documents SET processing_status = ?, updated_at = ? "
                "WHERE job_id = ? AND processing_status IN ('queued','processing')",
                (ProcessingStatus.CANCELLED.value, now, job_id),
            )
            n_docs = cur.rowcount or 0

            self._conn.execute(
                "UPDATE jobs SET processing_status = ?, updated_at = ? WHERE id = ?",
                (ProcessingStatus.CANCELLED.value, now, job_id),
            )
            self._conn.commit()
            return {"documents_cancelled": n_docs, "already_terminal": 0, "found": 1}

        return execute_with_retry(_do)

    def get_documents_for_job(self, job_id: str) -> list[Document]:
        cur = self._conn.execute(
            "SELECT * FROM documents WHERE job_id = ? ORDER BY page_number",
            (job_id,),
        )
        return [Document(**dict(r)) for r in cur.fetchall()]

    def get_document_for_job_page(self, job_id: str, page_number: int) -> Document | None:
        """Look up the single :class:`Document` row corresponding to one input
        page of a job, where ``page_number`` is the 1-based input page number
        as uploaded (i.e. metadata.page_number on POST /v1/ingest)."""
        cur = self._conn.execute(
            "SELECT * FROM documents WHERE job_id = ? AND page_number = ? LIMIT 1",
            (job_id, page_number),
        )
        row = cur.fetchone()
        return Document(**dict(row)) if row else None

    def get_metrics_for_job(self, job_id: str) -> list[ProcessingMetric]:
        cur = self._conn.execute(
            "SELECT m.* FROM processing_metrics m "
            "JOIN documents d ON m.document_id = d.id "
            "WHERE d.job_id = ? ORDER BY m.model_name",
            (job_id,),
        )
        return [ProcessingMetric(**dict(r)) for r in cur.fetchall()]

    def list_jobs(
        self,
        *,
        status: str | None = None,
        since: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Job]:
        """List jobs, optionally filtered by status and/or created/updated-since timestamp.

        ``since`` is matched against ``updated_at`` so the call returns jobs that
        have changed (or were created) after that ISO-8601 instant.
        """
        sql_parts = ["SELECT * FROM jobs"]
        params: list[object] = []
        clauses: list[str] = []
        if status is not None:
            clauses.append("processing_status = ?")
            params.append(status)
        if since is not None:
            clauses.append("updated_at >= ?")
            params.append(since)
        if clauses:
            sql_parts.append("WHERE " + " AND ".join(clauses))
        sql_parts.append("ORDER BY updated_at DESC LIMIT ? OFFSET ?")
        params.extend([int(limit), int(offset)])

        cur = self._conn.execute(" ".join(sql_parts), params)
        return [Job(**dict(r)) for r in cur.fetchall()]

    def count_jobs_by_status(self) -> dict[str, int]:
        """Return ``{status: count}`` for every status that currently has jobs.

        Statuses that have zero rows are omitted; callers should default to 0.
        """
        cur = self._conn.execute("SELECT processing_status, COUNT(*) AS n FROM jobs GROUP BY processing_status")
        return {row["processing_status"]: int(row["n"]) for row in cur.fetchall()}

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    def insert_document(self, doc: Document) -> None:
        def _do():
            row = doc.to_row()
            cols = ", ".join(row.keys())
            placeholders = ", ".join(f":{k}" for k in row.keys())
            self._conn.execute(f"INSERT INTO documents ({cols}) VALUES ({placeholders})", row)
            self._conn.commit()

        execute_with_retry(_do)

    def insert_document_or_get(self, doc: Document) -> tuple[Document, bool]:
        """Atomically insert ``doc`` OR return the existing winner of a SHA race.

        SQLite's ``UNIQUE INDEX idx_documents_sha256`` enforces dedup across
        all worker processes / coroutines.  When two requests with identical
        bytes race past :meth:`get_document_by_sha`, the first ``INSERT``
        wins and the second raises :class:`sqlite3.IntegrityError`.  This
        method catches that, re-fetches the winning row, and returns it
        with ``existed=True`` so the caller can avoid double-counting the
        page in job tallies.

        Returns ``(document, existed)`` where:
          - ``existed=False`` — *we* inserted ``doc``; ``document is doc``
          - ``existed=True``  — someone else inserted first; ``document``
            is that winning row.
        """

        def _do() -> tuple[Document, bool]:
            row = doc.to_row()
            cols = ", ".join(row.keys())
            placeholders = ", ".join(f":{k}" for k in row.keys())
            try:
                self._conn.execute(
                    f"INSERT INTO documents ({cols}) VALUES ({placeholders})",
                    row,
                )
                self._conn.commit()
                return doc, False
            except sqlite3.IntegrityError:
                # Lost the race — fetch the winner.
                self._conn.commit()
                winner = self.get_document_by_sha(doc.content_sha256)
                if winner is None:
                    # Extremely unlikely (UNIQUE violation but no row?). Re-raise.
                    raise
                return winner, True

        return execute_with_retry(_do)

    def get_document(self, document_id: str) -> Document | None:
        cur = self._conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,))
        row = cur.fetchone()
        return Document(**dict(row)) if row else None

    def get_document_by_sha(self, sha256: str) -> Document | None:
        cur = self._conn.execute("SELECT * FROM documents WHERE content_sha256 = ?", (sha256,))
        row = cur.fetchone()
        return Document(**dict(row)) if row else None

    # ------------------------------------------------------------------
    # Spool support
    # ------------------------------------------------------------------

    def update_document_spool_path(self, document_id: str, spool_path: str | None) -> None:
        """Set or clear the on-disk spool location for a document."""

        def _do() -> None:
            now = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                "UPDATE documents SET spool_path = ?, updated_at = ? WHERE id = ?",
                (spool_path, now, document_id),
            )
            self._conn.commit()

        execute_with_retry(_do)

    def list_recoverable_spooled_documents(self) -> list[Document]:
        """Return documents whose bytes are still spooled and not yet processed.

        Used at startup to re-enqueue work that was accepted before a
        crash / restart but never reached a worker subprocess.  Only
        non-terminal statuses (``queued`` and ``processing``) are
        returned — anything terminal will be cleaned up by the spool
        sweeper instead.
        """
        cur = self._conn.execute(
            "SELECT * FROM documents " "WHERE spool_path IS NOT NULL " "  AND processing_status IN (?, ?)",
            (ProcessingStatus.QUEUED.value, ProcessingStatus.PROCESSING.value),
        )
        return [Document(**dict(row)) for row in cur.fetchall()]

    def list_evictable_spooled_documents(self, limit: int = 1000) -> list[tuple[str, str]]:
        """Return ``(document_id, spool_path)`` for terminal docs to evict.

        Bounded to *limit* rows per call so the sweeper can pace itself
        and not block the loop on a huge backlog.
        """
        cur = self._conn.execute(
            "SELECT id, spool_path FROM documents "
            "WHERE spool_path IS NOT NULL "
            "  AND processing_status IN (?, ?, ?) "
            "LIMIT ?",
            (
                ProcessingStatus.COMPLETE.value,
                ProcessingStatus.FAILED.value,
                ProcessingStatus.CANCELLED.value,
                limit,
            ),
        )
        return [(str(row["id"]), str(row["spool_path"])) for row in cur.fetchall()]

    def update_document_status(
        self,
        document_id: str,
        status: ProcessingStatus,
        *,
        total_pages: int | None = None,
    ) -> None:
        def _do():
            now = datetime.now(timezone.utc).isoformat()
            if total_pages is not None:
                self._conn.execute(
                    "UPDATE documents SET processing_status = ?, total_pages = ?, updated_at = ? WHERE id = ?",
                    (status.value, total_pages, now, document_id),
                )
            else:
                self._conn.execute(
                    "UPDATE documents SET processing_status = ?, updated_at = ? WHERE id = ?",
                    (status.value, now, document_id),
                )
            self._conn.commit()

        execute_with_retry(_do)

    def increment_pages_received(self, document_id: str) -> int:
        """Atomically increment ``pages_received`` and return the new count."""

        def _do() -> int:
            now = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                "UPDATE documents SET pages_received = pages_received + 1, updated_at = ? WHERE id = ?",
                (now, document_id),
            )
            self._conn.commit()
            cur = self._conn.execute("SELECT pages_received FROM documents WHERE id = ?", (document_id,))
            row = cur.fetchone()
            return int(row["pages_received"]) if row else 0

        return execute_with_retry(_do)

    # ------------------------------------------------------------------
    # Page results
    # ------------------------------------------------------------------

    def insert_page_result(self, page: PageResult) -> None:
        def _do():
            row = page.to_row()
            cols = ", ".join(row.keys())
            placeholders = ", ".join(f":{k}" for k in row.keys())
            self._conn.execute(f"INSERT INTO page_results ({cols}) VALUES ({placeholders})", row)
            self._conn.commit()

        execute_with_retry(_do)

    def get_page_results(self, document_id: str) -> list[PageResult]:
        cur = self._conn.execute(
            "SELECT * FROM page_results WHERE document_id = ? ORDER BY page_number",
            (document_id,),
        )
        return [PageResult(**dict(r)) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Processing metrics
    # ------------------------------------------------------------------

    def insert_metric(self, metric: ProcessingMetric) -> None:
        def _do():
            row = metric.to_row()
            cols = ", ".join(row.keys())
            placeholders = ", ".join(f":{k}" for k in row.keys())
            self._conn.execute(f"INSERT INTO processing_metrics ({cols}) VALUES ({placeholders})", row)
            self._conn.commit()

        execute_with_retry(_do)

    def insert_metrics(self, metrics: list[ProcessingMetric]) -> None:
        if not metrics:
            return

        def _do():
            for m in metrics:
                row = m.to_row()
                cols = ", ".join(row.keys())
                placeholders = ", ".join(f":{k}" for k in row.keys())
                self._conn.execute(f"INSERT INTO processing_metrics ({cols}) VALUES ({placeholders})", row)
            self._conn.commit()

        execute_with_retry(_do)

    def get_metrics(self, document_id: str) -> list[ProcessingMetric]:
        cur = self._conn.execute(
            "SELECT * FROM processing_metrics WHERE document_id = ?",
            (document_id,),
        )
        return [ProcessingMetric(**dict(r)) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Page processing log
    # ------------------------------------------------------------------

    def insert_page_processing_log(self, entry: PageProcessingLog) -> None:
        def _do():
            row = entry.to_row()
            cols = ", ".join(row.keys())
            placeholders = ", ".join(f":{k}" for k in row.keys())
            self._conn.execute(
                f"INSERT OR REPLACE INTO page_processing_log ({cols}) VALUES ({placeholders})",
                row,
            )
            self._conn.commit()

        execute_with_retry(_do)

    def get_all_page_processing_logs(self) -> list[PageProcessingLog]:
        cur = self._conn.execute("SELECT * FROM page_processing_log ORDER BY source_file, page_number")
        return [PageProcessingLog(**dict(r)) for r in cur.fetchall()]

    def get_page_processing_logs_for_job(self, job_id: str) -> list[PageProcessingLog]:
        cur = self._conn.execute(
            "SELECT * FROM page_processing_log WHERE job_id = ? ORDER BY source_file, page_number",
            (job_id,),
        )
        return [PageProcessingLog(**dict(r)) for r in cur.fetchall()]

    def get_processing_log_for_document(self, document_id: str) -> PageProcessingLog | None:
        """Return the most recently written processing-log row for a document.

        Used to surface ``failure_type`` and ``error_message`` on REST status
        responses so REST callers don't have to scrape the SSE stream.
        """
        cur = self._conn.execute(
            "SELECT * FROM page_processing_log WHERE document_id = ? " "ORDER BY completed_at DESC LIMIT 1",
            (document_id,),
        )
        row = cur.fetchone()
        return PageProcessingLog(**dict(row)) if row else None
