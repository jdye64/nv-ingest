# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server-side agent memory, hosted alongside the VectorDB service.

The memory table lives in the same LanceDB directory as the document table but
under its own name and schema, so memory writes never disturb document
retrieval and the two can be compacted on different schedules.

Scope isolation is not optional here. The authorized scope is folded into the
stored namespace, so one tenant's recall can never reach another's memories
even if the client asks for a namespace it does not own.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, List, Optional, Sequence

from nemo_retriever.common.schemas.memory import (
    MemoryConsolidateRequest,
    MemoryConsolidateResult,
    MemoryFilter,
    MemoryForgetRequest,
    MemoryForgetResult,
    MemoryHit,
    MemoryRecallRequest,
    MemoryRecord,
    MemoryStats,
    MemoryTimelineRequest,
    MemoryWriteResult,
    compile_memory_filter,
)
from nemo_retriever.common.vdb.memory_schema import (
    DEFAULT_MEMORY_TABLE,
    build_memory_row,
    new_memory_id,
)
from nemo_retriever.memory.backends import rank_memory_rows, blank_to_none

logger = logging.getLogger(__name__)

#: Separator between the authorized scope and the caller-chosen namespace.
_SCOPE_SEPARATOR = "::"


def scoped_namespace(scope: str, namespace: Optional[str]) -> str:
    """Fold the authorized scope into the stored namespace."""
    clean_scope = (scope or "default").strip() or "default"
    clean_namespace = (namespace or "default").strip() or "default"
    return f"{clean_scope}{_SCOPE_SEPARATOR}{clean_namespace}"


def public_namespace(stored: str) -> str:
    """Strip the scope prefix so responses show the caller's own namespace."""
    _, separator, tail = str(stored).partition(_SCOPE_SEPARATOR)
    return tail if separator else str(stored)


class MemoryService:
    """Read and write agent memory for every scope served by this process."""

    def __init__(
        self,
        *,
        lancedb_uri: str,
        table_name: str = DEFAULT_MEMORY_TABLE,
        embed_fn: Callable[[Sequence[str]], List[List[float]]],
        max_write_batch: int = 512,
    ) -> None:
        from nemo_retriever.common.vdb.memory_store import MemoryStore

        self._store = MemoryStore(uri=lancedb_uri, table_name=table_name)
        self._embed_fn = embed_fn
        self._max_write_batch = int(max_write_batch)
        self._write_lock = threading.Lock()

    @property
    def store(self) -> Any:
        """Expose the store for maintenance endpoints."""
        return self._store

    def remember(self, records: Sequence[MemoryRecord], *, scope: str) -> MemoryWriteResult:
        """Embed and store memories under the caller's scope."""
        records = list(records)
        if not records:
            return MemoryWriteResult(memory_ids=[], written=0)
        if len(records) > self._max_write_batch:
            raise ValueError(
                f"Cannot write {len(records)} memories in one request; the limit is {self._max_write_batch}."
            )

        vectors = self._embed_fn([record.text for record in records])
        rows = []
        memory_ids = []
        for record, vector in zip(records, vectors):
            memory_id = record.memory_id or new_memory_id()
            memory_ids.append(memory_id)
            rows.append(
                build_memory_row(
                    text=record.text,
                    embedding=vector,
                    memory_id=memory_id,
                    memory_type=record.memory_type,
                    namespace=scoped_namespace(scope, record.namespace),
                    session_id=record.session_id,
                    agent_id=record.agent_id,
                    user_id=record.user_id,
                    role=record.role,
                    event_type=record.event_type,
                    occurred_at=record.occurred_at,
                    importance=record.importance,
                    confidence=record.confidence,
                    turn_index=record.turn_index,
                    source_memory_ids=record.source_memory_ids,
                    tags=record.tags,
                    metadata=record.metadata,
                )
            )

        with self._write_lock:
            written = self._store.write(rows)
        return MemoryWriteResult(memory_ids=memory_ids, written=written)

    def recall(self, request: MemoryRecallRequest, *, scope: str) -> List[MemoryHit]:
        """Rank memories in the caller's scope against a query."""
        namespace = scoped_namespace(scope, request.filter.namespace if request.filter else None)
        where = compile_memory_filter(_rescope(request.filter, namespace), namespace=namespace)
        vector = self._embed_fn([request.query])[0]

        fetch_k = request.top_k * 4 if request.recency_halflife_seconds else request.top_k
        rows = self._store.search(
            vector,
            top_k=fetch_k,
            where=where,
            query_text=request.query,
            hybrid=request.strategy == "hybrid",
        )
        hits = rank_memory_rows(
            rows,
            top_k=request.top_k,
            recency_halflife_seconds=request.recency_halflife_seconds,
        )
        return [_unscope(hit) for hit in hits]

    def timeline(self, request: MemoryTimelineRequest, *, scope: str) -> List[MemoryHit]:
        """Replay one session in the caller's scope."""
        namespace = scoped_namespace(scope, request.namespace)
        where = compile_memory_filter(
            MemoryFilter(
                namespace=namespace,
                session_id=request.session_id,
                occurred_after=request.since,
                occurred_before=request.until,
            ),
            namespace=namespace,
        )
        rows = self._store.scan(where=where, limit=request.limit, ascending=request.ascending)
        return [_unscope(MemoryHit(**blank_to_none(row))) for row in rows]

    def forget(self, request: MemoryForgetRequest, *, scope: str) -> MemoryForgetResult:
        """Retire memories in the caller's scope."""
        if request.memory_id:
            memory_filter = MemoryFilter(memory_ids=[request.memory_id], include_superseded=True)
        else:
            memory_filter = request.filter or MemoryFilter()
        if not memory_filter.is_targeted:
            raise ValueError("forget needs a narrower filter than a namespace; refusing to match every memory")
        namespace = scoped_namespace(scope, memory_filter.namespace)
        where = compile_memory_filter(_rescope(memory_filter, namespace), namespace=namespace)

        with self._write_lock:
            if request.hard:
                removed = self._store.delete(where)
                return MemoryForgetResult(matched=removed, forgotten=removed, hard=True)
            removed = self._store.supersede(where)
        return MemoryForgetResult(matched=removed, forgotten=removed, hard=False)

    def consolidate(
        self,
        request: MemoryConsolidateRequest,
        *,
        scope: str,
        llm: Any = None,
    ) -> MemoryConsolidateResult:
        """Distill this scope's episodic memories into semantic facts."""
        from nemo_retriever.memory.consolidation import consolidate_session

        return consolidate_session(_ScopedFacade(self, scope), request, llm=llm)

    def stats(self, *, scope: str, namespace: Optional[str] = None) -> MemoryStats:
        """Return coarse counters for one scoped namespace."""
        stored = scoped_namespace(scope, namespace)
        base = compile_memory_filter(MemoryFilter(include_superseded=True), namespace=stored)
        by_type = {}
        for memory_type in ("episodic", "semantic", "procedural"):
            where = compile_memory_filter(
                MemoryFilter(memory_type=memory_type, include_superseded=True),
                namespace=stored,
            )
            count = self._store.count(where)
            if count:
                by_type[memory_type] = count
        oldest, newest = self._store.time_bounds(base)
        return MemoryStats(
            namespace=public_namespace(stored),
            total=self._store.count(base),
            by_type=by_type,
            sessions=self._store.distinct_sessions(base),
            oldest_occurred_at=oldest,
            newest_occurred_at=newest,
        )

    def optimize(self) -> bool:
        """Compact the memory table. Maintenance for every scope at once."""
        with self._write_lock:
            return self._store.optimize()


class _ScopedFacade:
    """Bind one scope to a :class:`MemoryService` for scope-agnostic helpers.

    Consolidation is written against the backend interface used by
    :class:`~nemo_retriever.memory.backends.LocalMemoryBackend`, which has no
    scope argument. This adapter supplies it.
    """

    def __init__(self, service: MemoryService, scope: str) -> None:
        self._service = service
        self._scope = scope

    def write(self, records: Sequence[MemoryRecord]) -> MemoryWriteResult:
        return self._service.remember(records, scope=self._scope)

    def recall(self, request: MemoryRecallRequest) -> List[MemoryHit]:
        return self._service.recall(request, scope=self._scope)

    def timeline(self, request: MemoryTimelineRequest) -> List[MemoryHit]:
        return self._service.timeline(request, scope=self._scope)

    def forget(self, request: MemoryForgetRequest) -> MemoryForgetResult:
        return self._service.forget(request, scope=self._scope)


def _rescope(memory_filter: Optional[MemoryFilter], namespace: str) -> MemoryFilter:
    resolved = memory_filter.model_copy() if memory_filter is not None else MemoryFilter()
    resolved.namespace = namespace
    return resolved


def _unscope(hit: MemoryHit) -> MemoryHit:
    return hit.model_copy(update={"namespace": public_namespace(hit.namespace)})
