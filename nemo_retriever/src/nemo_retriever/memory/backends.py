# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage backends behind :class:`AgenticIngestor`.

Both backends implement :class:`MemoryBackend`, so the memory verbs are
written once on the ingestor and the deployment choice stays a constructor
argument. ``LocalMemoryBackend`` embeds and writes in-process against a
LanceDB directory; ``ServiceMemoryBackend`` forwards to a running retriever
service so several agents can share one memory.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

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
    utcnow,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class MemoryBackend(Protocol):
    """Operations every memory deployment must provide."""

    def write(self, records: Sequence[MemoryRecord]) -> MemoryWriteResult: ...

    def recall(self, request: MemoryRecallRequest) -> List[MemoryHit]: ...

    def timeline(self, request: MemoryTimelineRequest) -> List[MemoryHit]: ...

    def forget(self, request: MemoryForgetRequest) -> MemoryForgetResult: ...

    def consolidate(self, request: MemoryConsolidateRequest) -> MemoryConsolidateResult: ...

    def stats(self, namespace: str) -> MemoryStats: ...

    def optimize(self) -> bool: ...


# ----------------------------------------------------------------------
# Ranking helpers
# ----------------------------------------------------------------------


def _base_score(row: Dict[str, Any]) -> float:
    """Normalize backend-specific scores so higher always means better."""
    relevance = row.get("_relevance_score")
    if relevance is not None:
        return float(relevance)
    score = row.get("_score")
    if score is not None:
        return float(score)
    distance = row.get("_distance")
    if distance is not None:
        # Monotonic decreasing in distance and bounded, which keeps the recency
        # blend below from being dominated by one far-away outlier.
        return 1.0 / (1.0 + max(float(distance), 0.0))
    return 0.0


def _recency_weight(occurred_at: Optional[str], *, halflife_seconds: float, now: datetime) -> float:
    if not occurred_at:
        return 1.0
    try:
        moment = datetime.fromisoformat(occurred_at)
    except ValueError:
        return 1.0
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    age = max((now - moment).total_seconds(), 0.0)
    return math.exp(-math.log(2.0) * age / float(halflife_seconds))


def rank_memory_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    top_k: int,
    recency_halflife_seconds: Optional[float] = None,
) -> List[MemoryHit]:
    """Convert raw rows into ranked hits, optionally decaying by age."""
    now = utcnow()
    scored: List[tuple[float, Dict[str, Any]]] = []
    for row in rows:
        score = _base_score(row)
        if recency_halflife_seconds:
            score *= _recency_weight(
                row.get("occurred_at"),
                halflife_seconds=recency_halflife_seconds,
                now=now,
            )
        scored.append((score, row))

    if recency_halflife_seconds:
        scored.sort(key=lambda pair: pair[0], reverse=True)

    hits: List[MemoryHit] = []
    for score, row in scored[:top_k]:
        payload = {key: value for key, value in row.items() if not key.startswith("_")}
        payload.setdefault("memory_type", "episodic")
        hits.append(
            MemoryHit(
                **blank_to_none(payload),
                score=score,
                distance=row.get("_distance"),
            )
        )
    return hits


_OPTIONAL_TEXT_FIELDS = ("session_id", "agent_id", "user_id", "role", "event_type")


def blank_to_none(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Map the empty-string placeholders used at rest back to ``None``."""
    out = dict(payload)
    for field in _OPTIONAL_TEXT_FIELDS:
        if out.get(field) == "":
            out[field] = None
    if out.get("turn_index") == -1:
        out["turn_index"] = None
    return out


# ----------------------------------------------------------------------
# Local backend
# ----------------------------------------------------------------------


class LocalMemoryBackend:
    """Embed and store memories in-process against a local LanceDB directory."""

    def __init__(
        self,
        *,
        memory_uri: str = "lancedb",
        table_name: str = DEFAULT_MEMORY_TABLE,
        embed_kwargs: Optional[Dict[str, Any]] = None,
        embed_run_mode: str = "local",
        namespace: str = "default",
        optimize_write_threshold: int | None = None,
    ) -> None:
        from nemo_retriever.common.vdb.memory_store import (
            DEFAULT_OPTIMIZE_WRITE_THRESHOLD,
            MemoryStore,
        )
        from nemo_retriever.memory.embedding import MemoryEmbedder

        self.namespace = namespace
        self._embedder = MemoryEmbedder(run_mode=embed_run_mode, embed_kwargs=embed_kwargs)
        self._store = MemoryStore(
            uri=memory_uri,
            table_name=table_name,
            optimize_write_threshold=(
                optimize_write_threshold if optimize_write_threshold is not None else DEFAULT_OPTIMIZE_WRITE_THRESHOLD
            ),
        )

    @property
    def store(self) -> Any:
        """Expose the underlying store for consolidation and maintenance."""
        return self._store

    @property
    def embedder(self) -> Any:
        """Expose the embedder so document ingest can reuse the same model."""
        return self._embedder

    def write(self, records: Sequence[MemoryRecord]) -> MemoryWriteResult:
        records = list(records)
        if not records:
            return MemoryWriteResult(memory_ids=[], written=0)

        vectors = self._embedder.embed([record.text for record in records], input_type="passage")
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
                    namespace=record.namespace or self.namespace,
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

        written = self._store.write(rows)
        return MemoryWriteResult(memory_ids=memory_ids, written=written)

    def write_rows(self, rows: Sequence[Dict[str, Any]]) -> int:
        """Write prebuilt rows that already carry embeddings."""
        return self._store.write(rows)

    def recall(self, request: MemoryRecallRequest) -> List[MemoryHit]:
        where = compile_memory_filter(request.filter, namespace=self.namespace)
        vector = self._embedder.embed_one(request.query, input_type="query")

        # Recency blending reorders the candidate pool, so widen it first;
        # otherwise a slightly less similar but much fresher memory can never
        # enter the ranking.
        fetch_k = request.top_k * 4 if request.recency_halflife_seconds else request.top_k
        rows = self._store.search(
            vector,
            top_k=fetch_k,
            where=where,
            query_text=request.query,
            hybrid=request.strategy == "hybrid",
        )
        return rank_memory_rows(
            rows,
            top_k=request.top_k,
            recency_halflife_seconds=request.recency_halflife_seconds,
        )

    def timeline(self, request: MemoryTimelineRequest) -> List[MemoryHit]:
        where = compile_memory_filter(
            MemoryFilter(
                session_id=request.session_id,
                namespace=request.namespace,
                occurred_after=request.since,
                occurred_before=request.until,
            ),
            namespace=self.namespace,
        )
        rows = self._store.scan(where=where, limit=request.limit, ascending=request.ascending)
        return [MemoryHit(**blank_to_none(row)) for row in rows]

    def forget(self, request: MemoryForgetRequest) -> MemoryForgetResult:
        if request.memory_id:
            memory_filter = MemoryFilter(memory_ids=[request.memory_id], include_superseded=True)
        else:
            memory_filter = request.filter or MemoryFilter()
        if not memory_filter.is_targeted:
            raise ValueError("forget needs a narrower filter than a namespace; refusing to match every memory")
        where = compile_memory_filter(memory_filter, namespace=self.namespace)

        if request.hard:
            removed = self._store.delete(where)
            return MemoryForgetResult(matched=removed, forgotten=removed, hard=True)
        removed = self._store.supersede(where)
        return MemoryForgetResult(matched=removed, forgotten=removed, hard=False)

    def consolidate(self, request: MemoryConsolidateRequest) -> MemoryConsolidateResult:
        from nemo_retriever.memory.consolidation import consolidate_session

        return consolidate_session(self, request)

    def stats(self, namespace: str) -> MemoryStats:
        base = compile_memory_filter(MemoryFilter(include_superseded=True), namespace=namespace)
        total = self._store.count(base)
        by_type: Dict[str, int] = {}
        for memory_type in ("episodic", "semantic", "procedural"):
            where = compile_memory_filter(
                MemoryFilter(memory_type=memory_type, include_superseded=True),
                namespace=namespace,
            )
            count = self._store.count(where)
            if count:
                by_type[memory_type] = count
        oldest, newest = self._store.time_bounds(base)
        return MemoryStats(
            namespace=namespace,
            total=total,
            by_type=by_type,
            sessions=self._store.distinct_sessions(base),
            oldest_occurred_at=oldest,
            newest_occurred_at=newest,
        )

    def optimize(self) -> bool:
        return self._store.optimize()


# ----------------------------------------------------------------------
# Service backend
# ----------------------------------------------------------------------


class ServiceMemoryBackend:
    """Forward memory operations to a running retriever service."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:7670",
        api_token: str | None = None,
        scope: str | None = None,
        namespace: str = "default",
        client: Any = None,
    ) -> None:
        from nemo_retriever.service.client import RetrieverServiceClient

        self.namespace = namespace
        self._client = client or RetrieverServiceClient(
            base_url=base_url,
            api_token=api_token,
            scope=scope,
        )

    @property
    def client(self) -> Any:
        """Expose the underlying service client."""
        return self._client

    def write(self, records: Sequence[MemoryRecord]) -> MemoryWriteResult:
        records = list(records)
        if not records:
            return MemoryWriteResult(memory_ids=[], written=0)
        return self._client.remember(records)

    def recall(self, request: MemoryRecallRequest) -> List[MemoryHit]:
        return self._client.recall(request)

    def timeline(self, request: MemoryTimelineRequest) -> List[MemoryHit]:
        return self._client.memory_timeline(request)

    def forget(self, request: MemoryForgetRequest) -> MemoryForgetResult:
        return self._client.forget(request)

    def consolidate(self, request: MemoryConsolidateRequest) -> MemoryConsolidateResult:
        return self._client.consolidate_memory(request)

    def stats(self, namespace: str) -> MemoryStats:
        return self._client.memory_stats(namespace=namespace)

    def optimize(self) -> bool:
        # Compaction is a server-side maintenance concern in service mode.
        return False
