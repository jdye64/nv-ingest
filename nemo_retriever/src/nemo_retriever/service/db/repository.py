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

    def get_document(self, document_id: str) -> Document | None:
        cur = self._conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,))
        row = cur.fetchone()
        return Document(**dict(row)) if row else None

    def get_document_by_sha(self, sha256: str) -> Document | None:
        cur = self._conn.execute("SELECT * FROM documents WHERE content_sha256 = ?", (sha256,))
        row = cur.fetchone()
        return Document(**dict(row)) if row else None

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
