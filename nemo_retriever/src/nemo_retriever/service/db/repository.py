# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CRUD operations for all service-mode SQLite tables."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from nemo_retriever.service.db.engine import DatabaseEngine
from nemo_retriever.service.models.document import Document, ProcessingStatus
from nemo_retriever.service.models.metrics import ProcessingMetric
from nemo_retriever.service.models.page_result import PageResult

logger = logging.getLogger(__name__)


class Repository:
    """Data-access layer wrapping all three service tables."""

    def __init__(self, engine: DatabaseEngine) -> None:
        self._engine = engine

    @property
    def _conn(self):
        return self._engine.connection

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    def insert_document(self, doc: Document) -> None:
        row = doc.to_row()
        cols = ", ".join(row.keys())
        placeholders = ", ".join(f":{k}" for k in row.keys())
        self._conn.execute(f"INSERT INTO documents ({cols}) VALUES ({placeholders})", row)
        self._conn.commit()

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

    def increment_pages_received(self, document_id: str) -> int:
        """Atomically increment ``pages_received`` and return the new count."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE documents SET pages_received = pages_received + 1, updated_at = ? WHERE id = ?",
            (now, document_id),
        )
        self._conn.commit()
        cur = self._conn.execute("SELECT pages_received FROM documents WHERE id = ?", (document_id,))
        row = cur.fetchone()
        return int(row["pages_received"]) if row else 0

    # ------------------------------------------------------------------
    # Page results
    # ------------------------------------------------------------------

    def insert_page_result(self, page: PageResult) -> None:
        row = page.to_row()
        cols = ", ".join(row.keys())
        placeholders = ", ".join(f":{k}" for k in row.keys())
        self._conn.execute(f"INSERT INTO page_results ({cols}) VALUES ({placeholders})", row)
        self._conn.commit()

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
        row = metric.to_row()
        cols = ", ".join(row.keys())
        placeholders = ", ".join(f":{k}" for k in row.keys())
        self._conn.execute(f"INSERT INTO processing_metrics ({cols}) VALUES ({placeholders})", row)
        self._conn.commit()

    def insert_metrics(self, metrics: list[ProcessingMetric]) -> None:
        if not metrics:
            return
        for m in metrics:
            row = m.to_row()
            cols = ", ".join(row.keys())
            placeholders = ", ".join(f":{k}" for k in row.keys())
            self._conn.execute(f"INSERT INTO processing_metrics ({cols}) VALUES ({placeholders})", row)
        self._conn.commit()

    def get_metrics(self, document_id: str) -> list[ProcessingMetric]:
        cur = self._conn.execute(
            "SELECT * FROM processing_metrics WHERE document_id = ?",
            (document_id,),
        )
        return [ProcessingMetric(**dict(r)) for r in cur.fetchall()]
