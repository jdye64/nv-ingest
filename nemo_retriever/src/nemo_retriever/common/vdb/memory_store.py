# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LanceDB storage for the agent-memory table.

The document-oriented :class:`~nemo_retriever.common.vdb.lancedb.LanceDB`
backend assumes bulk appends of extracted chunks. Memory is the opposite
workload: many small writes, and reads that are almost always scoped by
session or time before ranking. This store keeps that access pattern in one
place so the ingestor and the service share identical semantics.

Recall always prefilters. Applying a session predicate after a top-k vector
scan would return whatever survives from a globally ranked list, which for a
scoped recall is usually nothing.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

from nemo_retriever.common.vdb.memory_schema import (
    DEFAULT_MEMORY_TABLE,
    MEMORY_RESULT_COLUMNS,
    infer_memory_vector_dim,
    memory_schema,
    parse_memory_row,
)

logger = logging.getLogger(__name__)

#: Appends before the store asks LanceDB to compact. Memory writes arrive in
#: small batches, and every batch is a new fragment until compaction runs.
DEFAULT_OPTIMIZE_WRITE_THRESHOLD = 64


class MemoryStore:
    """LanceDB-backed reader and writer for the agent-memory table."""

    def __init__(
        self,
        *,
        uri: str = "lancedb",
        table_name: str = DEFAULT_MEMORY_TABLE,
        vector_dim: int | None = None,
        create_fts_index: bool = True,
        optimize_write_threshold: int = DEFAULT_OPTIMIZE_WRITE_THRESHOLD,
    ) -> None:
        self.uri = str(uri)
        self.table_name = str(table_name)
        self.vector_dim = int(vector_dim) if vector_dim else None
        self.create_fts_index = bool(create_fts_index)
        self.optimize_write_threshold = int(optimize_write_threshold)

        self._lock = threading.RLock()
        self._connection: Any = None
        self._table: Any = None
        self._writes_since_optimize = 0
        self._fts_ready = False

    # ------------------------------------------------------------------
    # Connection and table lifecycle
    # ------------------------------------------------------------------

    def _connect(self) -> Any:
        with self._lock:
            if self._connection is None:
                import lancedb

                self._connection = lancedb.connect(uri=self.uri)
            return self._connection

    def table_exists(self) -> bool:
        try:
            return self.table_name in self._connect().list_tables().tables
        except Exception:
            logger.debug("Could not list LanceDB tables at %s", self.uri, exc_info=True)
            return False

    def open(self) -> Any | None:
        """Return the memory table handle, or ``None`` when it does not exist yet."""
        with self._lock:
            if self._table is None:
                if not self.table_exists():
                    return None
                self._table = self._connect().open_table(self.table_name)
            _checkout_latest(self._table)
            return self._table

    def _require_table(self) -> Any:
        table = self.open()
        if table is None:
            raise MemoryTableMissingError(
                f"No memory table {self.table_name!r} at {self.uri!r}. Write a memory before reading."
            )
        return table

    def _ensure_table(self, rows: List[Dict[str, Any]]) -> Any:
        with self._lock:
            table = self.open()
            if table is not None:
                return table

            dim = self.vector_dim or infer_memory_vector_dim(rows)
            if dim <= 0:
                raise ValueError("Cannot create the memory table without at least one embedding vector")
            self.vector_dim = dim
            self._table = self._connect().create_table(
                self.table_name,
                data=list(rows),
                schema=memory_schema(dim),
                mode="create",
            )
            self._maybe_create_fts_index(self._table)
            return self._table

    def _maybe_create_fts_index(self, table: Any) -> None:
        if not self.create_fts_index or self._fts_ready:
            return
        try:
            table.create_fts_index("text", replace=True)
            self._fts_ready = True
        except Exception:
            # Hybrid recall degrades to dense-only; that is a ranking quality
            # question, not a correctness one, so do not fail the write.
            logger.debug("Could not create the memory full-text index", exc_info=True)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def write(self, rows: Sequence[Dict[str, Any]]) -> int:
        """Append memory rows, creating the table on first use."""
        rows = [dict(row) for row in rows]
        if not rows:
            return 0

        with self._lock:
            table = self.open()
            if table is None:
                self._ensure_table(rows)
                self._writes_since_optimize += 1
                return len(rows)

            table.add(rows)
            self._writes_since_optimize += 1
            self._fts_ready = False
            self.maybe_optimize()
            return len(rows)

    def supersede(self, where: str, *, valid_to: datetime | None = None) -> int:
        """Stamp ``valid_to`` on matching rows so they drop out of default recall."""
        table = self._require_table()
        moment = valid_to or datetime.now(timezone.utc)
        matched = self.count(where)
        if matched == 0:
            return 0
        table.update(where=where, values={"valid_to": moment})
        return matched

    def delete(self, where: str) -> int:
        """Remove matching rows outright."""
        table = self._require_table()
        matched = self.count(where)
        if matched == 0:
            return 0
        table.delete(where)
        self._writes_since_optimize += 1
        self.maybe_optimize()
        return matched

    def maybe_optimize(self) -> bool:
        """Compact when enough small appends have accumulated."""
        if self._writes_since_optimize < self.optimize_write_threshold:
            return False
        return self.optimize()

    def optimize(self) -> bool:
        """Compact fragments and refresh indexes. Returns whether it ran cleanly."""
        table = self.open()
        if table is None:
            return False
        try:
            table.optimize()
        except Exception:
            logger.warning("LanceDB optimization failed for memory table %r", self.table_name, exc_info=True)
            return False
        self._writes_since_optimize = 0
        self._maybe_create_fts_index(table)
        return True

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int = 10,
        where: str | None = None,
        query_text: str | None = None,
        hybrid: bool = False,
        nprobes: int = 64,
        refine_factor: int = 50,
    ) -> List[Dict[str, Any]]:
        """Rank memories by similarity to ``vector`` under an optional predicate."""
        table = self._require_table()
        columns = [column for column in MEMORY_RESULT_COLUMNS]

        if hybrid and query_text:
            try:
                return self._hybrid_search(
                    table,
                    vector,
                    query_text=query_text,
                    top_k=top_k,
                    where=where,
                    columns=columns,
                    nprobes=nprobes,
                    refine_factor=refine_factor,
                )
            except Exception:
                # A missing or stale full-text index is the common cause. Dense
                # recall still answers the question, so fall back rather than
                # failing the agent's turn.
                logger.debug("Hybrid memory recall failed; falling back to dense", exc_info=True)

        query = table.search([list(vector)], vector_column_name="vector")
        query = _apply_where(query, where)
        query = query.limit(top_k).refine_factor(refine_factor).nprobes(nprobes).select(columns)
        return [parse_memory_row(row) for row in query.to_list()]

    def _hybrid_search(
        self,
        table: Any,
        vector: Sequence[float],
        *,
        query_text: str,
        top_k: int,
        where: str | None,
        columns: List[str],
        nprobes: int,
        refine_factor: int,
    ) -> List[Dict[str, Any]]:
        query = (
            table.search(query_type="hybrid", vector_column_name="vector", fts_columns="text")
            .vector(list(vector))
            .text(query_text)
        )
        query = _apply_where(query, where)
        query = query.limit(top_k).refine_factor(refine_factor).nprobes(nprobes).select(columns)
        return [parse_memory_row(row) for row in query.to_list()]

    def scan(
        self,
        *,
        where: str | None = None,
        limit: int = 100,
        order_by: str = "occurred_at",
        ascending: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return rows in chronological order without touching the embedder."""
        table = self._require_table()
        query = table.search().select([column for column in MEMORY_RESULT_COLUMNS])
        query = _apply_where(query, where, prefilter=False)

        rows = [parse_memory_row(row) for row in query.limit(_scan_limit(limit)).to_list()]
        rows.sort(key=lambda row: (row.get(order_by) or ""), reverse=not ascending)
        return rows[:limit]

    def count(self, where: str | None = None) -> int:
        """Count rows matching a predicate."""
        table = self.open()
        if table is None:
            return 0
        try:
            if where:
                return int(table.count_rows(filter=where))
            return int(table.count_rows())
        except TypeError:
            # Older LanceDB builds accept the predicate positionally.
            return int(table.count_rows(where)) if where else int(table.count_rows())

    def distinct_sessions(self, where: str | None = None, *, limit: int = 10000) -> int:
        """Count distinct sessions, used for coarse namespace statistics."""
        table = self.open()
        if table is None:
            return 0
        query = table.search().select(["session_id"])
        query = _apply_where(query, where, prefilter=False)
        sessions = {row.get("session_id") for row in query.limit(limit).to_list()}
        sessions.discard("")
        sessions.discard(None)
        return len(sessions)

    def time_bounds(self, where: str | None = None, *, limit: int = 10000) -> tuple[Optional[str], Optional[str]]:
        """Return the earliest and latest ``occurred_at`` under a predicate."""
        table = self.open()
        if table is None:
            return None, None
        query = table.search().select(["occurred_at"])
        query = _apply_where(query, where, prefilter=False)
        stamps = sorted(
            value
            for value in (parse_memory_row(row).get("occurred_at") for row in query.limit(limit).to_list())
            if value
        )
        if not stamps:
            return None, None
        return stamps[0], stamps[-1]


class MemoryTableMissingError(RuntimeError):
    """Raised when a read targets a memory table that has never been written."""


def _apply_where(query: Any, where: str | None, *, prefilter: bool = True) -> Any:
    if not where:
        return query
    try:
        return query.where(where, prefilter=prefilter)
    except TypeError:
        # Plain scans and older builds do not accept the prefilter keyword.
        return query.where(where)


def _scan_limit(limit: int) -> int:
    # Sorting happens client-side, so pull a margin above the requested page to
    # avoid returning an arbitrary slice of the matching rows.
    return max(int(limit) * 4, int(limit))


def _checkout_latest(table: Any) -> None:
    """Advance a cached table handle to the latest committed version."""
    checkout_latest = getattr(table, "checkout_latest", None)
    if not callable(checkout_latest):
        return
    try:
        checkout_latest()
    except Exception as exc:  # noqa: BLE001 - version refresh is advisory.
        logger.debug("Could not advance the memory table handle: %s", exc)


def rows_from_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Materialize an iterable of prepared rows into a list."""
    return [dict(record) for record in records]
