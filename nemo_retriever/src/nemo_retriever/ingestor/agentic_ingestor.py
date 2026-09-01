# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgenticIngestor: agent memory over NeMo Retriever.

Where :class:`~nemo_retriever.ingestor.graph_ingestor.GraphIngestor` and
:class:`~nemo_retriever.service.service_ingestor.ServiceIngestor` load
documents, this ingestor stores and recalls what an agent experienced. It
writes two kinds of memory:

* **episodic** — time-ordered events such as messages, tool calls, and
  observations, usually recalled within one session.
* **semantic** — durable facts distilled from episodes, recalled across
  sessions and superseded rather than deleted when they stop being true.

The backend is pluggable. ``backend="service"`` forwards to a running
retriever service so several agents share one memory; ``backend="local"``
embeds and stores in-process against a LanceDB directory with no server.

Writes never block the caller's turn on embedding. :meth:`remember` enqueues
and a batched flush does the embedding and storage work.

Usage::

    from nemo_retriever.ingestor.agentic_ingestor import AgenticIngestor

    memory = AgenticIngestor(backend="local", memory_uri="./agent-memory")

    with memory.session("thread-42", user_id="ada") as session:
        session.remember("The user prefers pytest over unittest.", memory_type="semantic")
        session.remember("Ran the test suite; 3 failures in test_ingest.py", role="assistant")

    hits = memory.recall("what testing framework does the user like?")
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from io import BytesIO
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Self,
    Sequence,
    Tuple,
    Union,
)

from nemo_retriever.common.params import (
    CaptionParams,
    DedupParams,
    EmbedParams,
    ExtractParams,
    StoreParams,
    VdbUploadParams,
    WebhookParams,
)
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
    RecallStrategy,
)
from nemo_retriever.common.vdb.memory_schema import DEFAULT_MEMORY_TABLE
from nemo_retriever.ingestor import ingestor

logger = logging.getLogger(__name__)

MemoryBackendName = Literal["service", "local"]

#: Flush automatically once this many memories are queued. Small enough that a
#: crash loses little, large enough that a chatty agent still batches.
DEFAULT_FLUSH_THRESHOLD = 16

#: Flush a partial batch this long after its first record was queued. Without
#: an age bound, a slow conversation could hold memories unwritten for as long
#: as it runs.
DEFAULT_FLUSH_INTERVAL_S = 30.0


class AgenticIngestor(ingestor):
    """Episodic and semantic memory for agents, with a pluggable backend.

    Parameters
    ----------
    backend
        ``"service"`` (default) talks to a running retriever service.
        ``"local"`` embeds and stores in-process.
    base_url
        Service endpoint, used when ``backend="service"``.
    api_token
        Bearer token for the service.
    scope
        Logical workspace scope sent as ``X-NRL-Scope``.
    memory_uri
        LanceDB directory, used when ``backend="local"``.
    table_name
        Physical memory table name for the local backend.
    namespace
        Logical partition applied to every write and, unless overridden, every
        read. Recall never crosses namespaces implicitly.
    agent_id
        Default agent identity stamped on writes.
    user_id
        Default user identity stamped on writes.
    session_id
        Default session, usually set through :meth:`session` instead.
    embed_kwargs
        Embedding overrides for the local backend.
    autoflush
        When ``True`` (default), queued memories flush on a size threshold and
        whenever a read needs to see them. Set ``False`` to control writes
        explicitly with :meth:`flush`.
    flush_threshold
        Queue depth that triggers an automatic flush.
    flush_interval_seconds
        Maximum age of a queued memory before the next write flushes the batch.
        Set to ``0`` to bound flushing by queue depth alone.
    """

    RUN_MODE = "agentic"
    SUPPORTS_SOURCE_METADATA = True
    UNSUPPORTED_VERBS = frozenset({"dedup", "caption", "store", "vdb_upload", "webhook"})

    def __init__(
        self,
        *,
        backend: MemoryBackendName = "service",
        base_url: str = "http://localhost:7670",
        api_token: Optional[str] = None,
        scope: Optional[str] = None,
        memory_uri: str = "lancedb",
        table_name: str = DEFAULT_MEMORY_TABLE,
        namespace: str = "default",
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        documents: Optional[List[str]] = None,
        embed_kwargs: Optional[Dict[str, Any]] = None,
        embed_run_mode: str = "local",
        autoflush: bool = True,
        flush_threshold: int = DEFAULT_FLUSH_THRESHOLD,
        flush_interval_seconds: float = DEFAULT_FLUSH_INTERVAL_S,
        memory_backend: Any = None,
    ) -> None:
        super().__init__(documents=documents)
        if backend not in ("service", "local"):
            raise ValueError(f"backend must be 'service' or 'local', got {backend!r}")
        if flush_threshold < 1:
            raise ValueError(f"flush_threshold must be at least 1, got {flush_threshold}")
        if flush_interval_seconds < 0:
            raise ValueError(f"flush_interval_seconds must be non-negative, got {flush_interval_seconds}")

        self._backend_name = backend
        self._namespace = namespace
        self._agent_id = agent_id
        self._user_id = user_id
        self._session_id = session_id
        self._autoflush = bool(autoflush)
        self._flush_threshold = int(flush_threshold)
        self._flush_interval_seconds = float(flush_interval_seconds)

        self._pending: List[MemoryRecord] = []
        self._pending_since: Optional[float] = None
        self._queue_lock = threading.Lock()
        self._written_ids: List[str] = []

        self._backend_options: Dict[str, Any] = {
            "base_url": base_url,
            "api_token": api_token,
            "scope": scope,
            "memory_uri": memory_uri,
            "table_name": table_name,
            "embed_kwargs": embed_kwargs,
            "embed_run_mode": embed_run_mode,
        }
        self._backend: Any = memory_backend

        # Document-ingest state, used by the inherited builder verbs.
        self._extract_params: Any = None
        self._embed_params: Any = None
        self._inline_texts: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Backend wiring
    # ------------------------------------------------------------------

    @property
    def backend(self) -> Any:
        """Return the memory backend, constructing it on first use."""
        if self._backend is None:
            self._backend = self._build_backend()
        return self._backend

    def _build_backend(self) -> Any:
        options = self._backend_options
        if self._backend_name == "service":
            from nemo_retriever.memory.backends import ServiceMemoryBackend

            return ServiceMemoryBackend(
                base_url=options["base_url"],
                api_token=options["api_token"],
                scope=options["scope"],
                namespace=self._namespace,
            )

        from nemo_retriever.memory.backends import LocalMemoryBackend

        return LocalMemoryBackend(
            memory_uri=options["memory_uri"],
            table_name=options["table_name"],
            embed_kwargs=options["embed_kwargs"],
            embed_run_mode=options["embed_run_mode"],
            namespace=self._namespace,
        )

    @property
    def namespace(self) -> str:
        """Logical partition applied to reads and writes."""
        return self._namespace

    # ------------------------------------------------------------------
    # Writing memory
    # ------------------------------------------------------------------

    def remember(
        self,
        text: str,
        *,
        memory_type: str = "episodic",
        session_id: Optional[str] = None,
        role: Optional[str] = None,
        event_type: Optional[str] = None,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Sequence[str] = (),
        importance: float = 0.5,
        confidence: float = 1.0,
        turn_index: Optional[int] = None,
        occurred_at: Optional[datetime] = None,
        source_memory_ids: Sequence[str] = (),
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryRecord:
        """Queue one memory for storage.

        Returns the queued record. Embedding and storage happen on flush, so
        the returned record has no identifier until then unless one was
        supplied through ``metadata``.
        """
        record = MemoryRecord(
            text=text,
            memory_type=memory_type,  # type: ignore[arg-type]
            namespace=self._namespace,
            session_id=session_id or self._session_id,
            agent_id=agent_id or self._agent_id,
            user_id=user_id or self._user_id,
            role=role,  # type: ignore[arg-type]
            event_type=event_type,  # type: ignore[arg-type]
            occurred_at=occurred_at,
            importance=importance,
            confidence=confidence,
            turn_index=turn_index,
            source_memory_ids=list(source_memory_ids),
            tags=self._merged_tags(tags),
            metadata=self._merged_metadata(metadata),
        )
        self._enqueue([record])
        return record

    def remember_many(self, records: Sequence[MemoryRecord]) -> List[MemoryRecord]:
        """Queue several prepared records at once.

        This is the path capture layers use. Defaults configured on the
        ingestor fill any field the caller left unset.
        """
        prepared = [self._apply_defaults(record) for record in records]
        self._enqueue(prepared)
        return prepared

    def _apply_defaults(self, record: MemoryRecord) -> MemoryRecord:
        updates: Dict[str, Any] = {"namespace": self._namespace}
        if record.session_id is None and self._session_id:
            updates["session_id"] = self._session_id
        if record.agent_id is None and self._agent_id:
            updates["agent_id"] = self._agent_id
        if record.user_id is None and self._user_id:
            updates["user_id"] = self._user_id
        if self._source_tags:
            updates["tags"] = self._merged_tags(record.tags)
        if self._source_metadata:
            updates["metadata"] = self._merged_metadata(record.metadata)
        return record.model_copy(update=updates)

    def _merged_tags(self, tags: Sequence[str]) -> List[str]:
        merged = list(self._source_tags)
        for tag in tags:
            text = str(tag).strip()
            if text and text not in merged:
                merged.append(text)
        return merged

    def _merged_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return {**self._source_metadata, **(metadata or {})}

    def _enqueue(self, records: Sequence[MemoryRecord]) -> None:
        if not records:
            return
        now = time.monotonic()
        with self._queue_lock:
            if not self._pending:
                self._pending_since = now
            self._pending.extend(records)
            should_flush = self._autoflush and (
                len(self._pending) >= self._flush_threshold or self._batch_is_stale(now)
            )
        if should_flush:
            self.flush()

    def _batch_is_stale(self, now: float) -> bool:
        if not self._flush_interval_seconds or self._pending_since is None:
            return False
        return (now - self._pending_since) >= self._flush_interval_seconds

    def flush(self) -> MemoryWriteResult:
        """Embed and store everything queued so far."""
        with self._queue_lock:
            batch = self._pending
            self._pending = []
            self._pending_since = None
        if not batch:
            return MemoryWriteResult(memory_ids=[], written=0)

        try:
            result = self.backend.write(batch)
        except Exception:
            # Put the batch back so a caller that handles the error can retry
            # instead of silently losing the agent's history.
            with self._queue_lock:
                self._pending = list(batch) + self._pending
                self._pending_since = self._pending_since or time.monotonic()
            raise

        self._written_ids.extend(result.memory_ids)
        return result

    @property
    def pending(self) -> int:
        """Number of memories queued but not yet stored."""
        with self._queue_lock:
            return len(self._pending)

    def _flush_before_read(self) -> None:
        if self._autoflush and self.pending:
            self.flush()

    # ------------------------------------------------------------------
    # Reading memory
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        *,
        top_k: int = 10,
        filter: Optional[MemoryFilter] = None,
        strategy: RecallStrategy = "hybrid",
        recency_halflife_seconds: Optional[float] = None,
        session_id: Optional[str] = None,
        memory_type: Optional[str] = None,
    ) -> List[MemoryHit]:
        """Rank stored memories against a natural-language query.

        ``session_id`` and ``memory_type`` are conveniences that build a
        filter for the common cases. Pass ``filter`` for anything richer; the
        shorthands then fill only the fields the filter left unset.
        """
        self._flush_before_read()
        request = MemoryRecallRequest(
            query=query,
            top_k=top_k,
            filter=self._resolve_filter(filter, session_id=session_id, memory_type=memory_type),
            strategy=strategy,
            recency_halflife_seconds=recency_halflife_seconds,
        )
        return list(self.backend.recall(request))

    def timeline(
        self,
        session_id: Optional[str] = None,
        *,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
        ascending: bool = True,
    ) -> List[MemoryHit]:
        """Replay a session in chronological order without embedding the query."""
        self._flush_before_read()
        resolved = session_id or self._session_id
        if not resolved:
            raise ValueError("timeline requires a session_id, either as an argument or on the ingestor")
        request = MemoryTimelineRequest(
            session_id=resolved,
            since=since,
            until=until,
            limit=limit,
            ascending=ascending,
            namespace=self._namespace,
        )
        return list(self.backend.timeline(request))

    def forget(
        self,
        *,
        memory_id: Optional[str] = None,
        filter: Optional[MemoryFilter] = None,
        hard: bool = False,
    ) -> MemoryForgetResult:
        """Retire memories.

        The default is a soft forget that stamps ``valid_to``, so the record
        drops out of recall but stays auditable. ``hard=True`` deletes rows.
        """
        self._flush_before_read()
        request = MemoryForgetRequest(memory_id=memory_id, filter=filter, hard=hard)
        return self.backend.forget(request)

    def consolidate(
        self,
        session_id: Optional[str] = None,
        *,
        max_episodes: int = 200,
        since: Optional[datetime] = None,
    ) -> MemoryConsolidateResult:
        """Distill episodic memories into durable semantic facts."""
        self._flush_before_read()
        request = MemoryConsolidateRequest(
            session_id=session_id or self._session_id,
            namespace=self._namespace,
            max_episodes=max_episodes,
            since=since,
        )
        return self.backend.consolidate(request)

    def stats(self) -> MemoryStats:
        """Return coarse counters for the configured namespace."""
        self._flush_before_read()
        return self.backend.stats(self._namespace)

    def optimize(self) -> bool:
        """Compact storage after a burst of small writes."""
        self._flush_before_read()
        return self.backend.optimize()

    def _resolve_filter(
        self,
        base: Optional[MemoryFilter],
        *,
        session_id: Optional[str] = None,
        memory_type: Optional[str] = None,
    ) -> MemoryFilter:
        resolved = base.model_copy() if base is not None else MemoryFilter()
        if resolved.namespace is None:
            resolved.namespace = self._namespace
        if session_id and resolved.session_id is None:
            resolved.session_id = session_id
        if memory_type and resolved.memory_type is None:
            resolved.memory_type = memory_type  # type: ignore[assignment]
        return resolved

    # ------------------------------------------------------------------
    # Session scoping
    # ------------------------------------------------------------------

    @contextmanager
    def session(
        self,
        session_id: Optional[str] = None,
        *,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Iterator["AgenticIngestor"]:
        """Scope writes to one session and flush on exit.

        Yields this same ingestor with session defaults applied, so a nested
        block does not need to thread identifiers through every call. Defaults
        are restored afterwards even when the body raises, and the queue is
        flushed so the session is durable before control returns.
        """
        import uuid

        previous = (self._session_id, self._agent_id, self._user_id)
        self._session_id = session_id or f"sess_{uuid.uuid4().hex[:16]}"
        if agent_id is not None:
            self._agent_id = agent_id
        if user_id is not None:
            self._user_id = user_id
        try:
            yield self
        finally:
            try:
                self.flush()
            finally:
                self._session_id, self._agent_id, self._user_id = previous

    def __enter__(self) -> "AgenticIngestor":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.flush()

    # ------------------------------------------------------------------
    # Document ingestion into the same memory namespace
    # ------------------------------------------------------------------

    def files(self, documents: Union[str, List[str]]) -> "AgenticIngestor":
        """Set reference documents to load into this memory namespace."""
        self._documents = [documents] if isinstance(documents, str) else list(documents)
        return self

    def texts(self, texts: Union[str, Sequence[str]]) -> Self:
        """Set raw inline text documents to load into this memory namespace."""
        from nemo_retriever.common.inline_text import normalize_inline_texts

        self._inline_texts = normalize_inline_texts(texts)
        return self

    def buffers(
        self,
        buffers: Union[Tuple[str, BytesIO], List[Tuple[str, BytesIO]]],
    ) -> "AgenticIngestor":
        """Set in-memory ``(name, BytesIO)`` buffers to load into memory."""
        if isinstance(buffers, tuple) and len(buffers) == 2 and isinstance(buffers[0], str):
            self._buffers = [buffers]
        else:
            self._buffers = list(buffers)
        return self

    def extract(self, params: Optional[ExtractParams] = None, **kwargs: Any) -> "AgenticIngestor":
        """Configure extraction for document loading."""
        self._extract_params = params or (ExtractParams(**kwargs) if kwargs else ExtractParams())
        return self

    def embed(self, params: Optional[EmbedParams] = None, **kwargs: Any) -> "AgenticIngestor":
        """Configure embedding for document loading."""
        self._embed_params = params or (EmbedParams(**kwargs) if kwargs else EmbedParams())
        return self

    def ingest(self, params: Any = None, **kwargs: Any) -> MemoryWriteResult:
        """Load the configured documents into memory as semantic records.

        Each extracted chunk becomes one ``memory_type="semantic"`` record with
        ``event_type="document"``, so reference material and remembered
        experience are recalled through the same query.

        Only ``backend="local"`` runs extraction in-process. In service mode,
        upload documents through the service ingest API instead.
        """
        _ = (params, kwargs)
        self._validate_input_sources(self._inline_texts)
        self.flush()

        if self._backend_name != "local":
            raise NotImplementedError(
                "AgenticIngestor.ingest() runs document extraction in-process and requires "
                "backend='local'. In service mode, upload documents with ServiceIngestor or "
                "the service ingest API, then recall them from the same namespace."
            )

        from nemo_retriever.memory.documents import ingest_documents_into_memory

        return ingest_documents_into_memory(
            backend=self.backend,
            documents=self._documents,
            buffers=self._buffers,
            inline_texts=self._inline_texts,
            namespace=self._namespace,
            agent_id=self._agent_id,
            user_id=self._user_id,
            tags=list(self._source_tags),
            metadata=dict(self._source_metadata),
            extract_params=self._extract_params,
        )

    # ------------------------------------------------------------------
    # Interface verbs this run mode does not provide
    # ------------------------------------------------------------------

    def dedup(self, params: Optional[DedupParams] = None, **kwargs: Any) -> "AgenticIngestor":
        """Not available: memory deduplication happens during consolidation."""
        self._not_implemented("dedup")

    def caption(self, params: Optional[CaptionParams] = None, **kwargs: Any) -> "AgenticIngestor":
        """Not available in agentic run mode."""
        self._not_implemented("caption")

    def store(self, params: Optional[StoreParams] = None, **kwargs: Any) -> "AgenticIngestor":
        """Not available in agentic run mode."""
        self._not_implemented("store")

    def vdb_upload(self, params: Optional[VdbUploadParams] = None, **kwargs: Any) -> "AgenticIngestor":
        """Not available: the memory table is the vector store for this mode."""
        self._not_implemented("vdb_upload")

    def webhook(self, params: Optional[WebhookParams] = None, **kwargs: Any) -> "AgenticIngestor":
        """Not available in agentic run mode."""
        self._not_implemented("webhook")

    def get_status(self) -> Dict[str, str]:
        """Return a coarse status mapping for the configured namespace."""
        stats = self.stats()
        return {
            "namespace": stats.namespace,
            "total": str(stats.total),
            "sessions": str(stats.sessions),
            "pending": str(self.pending),
        }
