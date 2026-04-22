# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ThreadPoolExecutor-based document processing for service mode.

Each worker thread owns a pre-built chain of instantiated operators (the same
graph that ``InprocessExecutor`` would run).  Documents are submitted as
futures; after pipeline completion the pool stores page-level results and
metrics directly in the database and publishes SSE events.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import pandas as pd

from nemo_retriever.service.config import ServiceConfig
from nemo_retriever.service.db.engine import DatabaseEngine
from nemo_retriever.service.db.repository import Repository
from nemo_retriever.service.event_bus import EventBus
from nemo_retriever.service.models.document import ProcessingStatus
from nemo_retriever.service.models.metrics import ProcessingMetric
from nemo_retriever.service.models.page_result import PageResult

logger = logging.getLogger(__name__)

_SERIALIZE_SKIP = frozenset({"bytes"})


def _safe_value(v: Any) -> Any:
    """Best-effort conversion to a JSON-serialisable type."""
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, bytes):
        return f"<bytes len={len(v)}>"
    if isinstance(v, (list, tuple)):
        return [_safe_value(i) for i in v]
    if isinstance(v, dict):
        return {str(k): _safe_value(val) for k, val in v.items()}
    return str(v)


def _row_to_page_content(row: dict[str, Any]) -> dict[str, Any]:
    """Extract a JSON-safe content dict from a pipeline output row."""
    return {k: _safe_value(v) for k, v in row.items() if k not in _SERIALIZE_SKIP}


def _extract_metrics_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull detection/invocation counts from well-known columns."""
    metrics: list[dict[str, Any]] = []
    for suffix in ("_num_detections", "_counts_by_label"):
        for col, val in row.items():
            if col.endswith(suffix) and val is not None:
                model_name = col[: -len(suffix)]
                metrics.append(
                    {
                        "model_name": model_name,
                        "detections_count": int(val) if isinstance(val, (int, float)) else 0,
                        "invocation_count": 1,
                        "pages_processed": 1,
                    }
                )
    return metrics


def _build_operator_chain() -> list[tuple[str, Any]]:
    """Build a linearised list of ``(name, operator_instance)`` pairs.

    This mirrors what ``InprocessExecutor`` does internally but we keep the
    instantiated operators around for repeated use within a single thread.

    No webhook is attached — the pool stores results directly.
    """
    from nemo_retriever.graph.ingestor_runtime import build_graph
    from nemo_retriever.graph.executor import InprocessExecutor
    from nemo_retriever.graph.operator_resolution import resolve_graph
    from nemo_retriever.utils.ray_resource_hueristics import gather_local_resources
    from nemo_retriever.params import ExtractParams

    graph = build_graph(
        extraction_mode="pdf",
        extract_params=ExtractParams(),
        stage_order=(),
    )
    resolved = resolve_graph(graph, gather_local_resources())
    nodes = InprocessExecutor._linearize(resolved)
    operators: list[tuple[str, Any]] = []
    for node in nodes:
        op = node.operator_class(**node.operator_kwargs)
        operators.append((node.name, op))
    return operators


class _ThreadState(threading.local):
    """Per-thread cached operator chain."""

    operators: list[tuple[str, Any]] | None = None


class ProcessingPool:
    """Manages a fixed-size thread pool where each thread holds a
    pre-initialised GraphPipeline operator chain.

    Call :meth:`submit` to queue a document for processing.
    """

    def __init__(
        self,
        config: ServiceConfig,
        db_engine: DatabaseEngine,
        event_bus: EventBus,
        event_loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._config = config
        self._db_engine = db_engine
        self._event_bus = event_bus
        self._event_loop = event_loop
        self._pool_size = config.processing.thread_pool_size
        self._executor: ThreadPoolExecutor | None = None
        self._thread_state = _ThreadState()
        self._active_jobs = threading.Semaphore(self._pool_size)

    def start(self) -> None:
        logger.info("Starting processing pool with %d workers", self._pool_size)
        self._executor = ThreadPoolExecutor(
            max_workers=self._pool_size,
            thread_name_prefix="retriever-worker",
        )

    def shutdown(self) -> None:
        if self._executor is not None:
            logger.info("Shutting down processing pool …")
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def _get_operators(self) -> list[tuple[str, Any]]:
        if self._thread_state.operators is None:
            logger.info(
                "Thread %s: building operator chain (first use)",
                threading.current_thread().name,
            )
            self._thread_state.operators = _build_operator_chain()
        return self._thread_state.operators

    @property
    def capacity(self) -> int:
        """Number of free worker slots available right now."""
        # Semaphore._value is CPython internal but reliable; fall back to 0.
        return getattr(self._active_jobs, "_value", 0)

    def has_capacity(self) -> bool:
        """Return ``True`` if at least one worker slot is free."""
        return self.capacity > 0

    def _publish_event(self, document_id: str, event: dict[str, Any]) -> None:
        """Thread-safe publish to the async event bus."""
        asyncio.run_coroutine_threadsafe(
            self._event_bus.publish(document_id, event),
            self._event_loop,
        )

    def _store_results(
        self,
        repo: Repository,
        document_id: str,
        content_sha256: str,
        df: pd.DataFrame,
    ) -> None:
        """Persist pipeline output rows as page results and metrics."""
        total_pages = len(df)
        repo.update_document_status(
            document_id,
            ProcessingStatus.PROCESSING,
            total_pages=total_pages,
        )

        for page_num, (_, row) in enumerate(df.iterrows()):
            row_dict = row.to_dict()
            content = _row_to_page_content(row_dict)

            page = PageResult(
                document_id=document_id,
                page_number=page_num,
                content_json=json.dumps(content),
            )
            repo.insert_page_result(page)

            row_metrics = _extract_metrics_from_row(row_dict)
            metric_objs = [
                ProcessingMetric(
                    document_id=document_id,
                    model_name=m["model_name"],
                    invocation_count=m.get("invocation_count", 1),
                    pages_processed=m.get("pages_processed", 1),
                    detections_count=m.get("detections_count", 0),
                )
                for m in row_metrics
            ]
            repo.insert_metrics(metric_objs)

            new_count = repo.increment_pages_received(document_id)

            self._publish_event(
                document_id,
                {
                    "event": "page_complete",
                    "document_id": document_id,
                    "page_number": page_num,
                    "pages_received": new_count,
                    "total_pages": total_pages,
                },
            )

        repo.update_document_status(document_id, ProcessingStatus.COMPLETE)
        self._publish_event(
            document_id,
            {
                "event": "document_complete",
                "document_id": document_id,
                "total_pages": total_pages,
            },
        )

    def _process_document(
        self,
        document_id: str,
        content_sha256: str,
        file_bytes: bytes,
        filename: str,
    ) -> None:
        """Run a single document through the in-process operator chain."""
        self._active_jobs.acquire()
        repo = Repository(self._db_engine)
        try:
            repo.update_document_status(document_id, ProcessingStatus.PROCESSING)
            df = pd.DataFrame([{"bytes": file_bytes, "path": filename}])

            operators = self._get_operators()
            for _name, op in operators:
                df = op.run(df)

            self._store_results(repo, document_id, content_sha256, df)
            logger.info("Document %s processing complete (%d rows)", document_id, len(df))

        except Exception:
            logger.exception("Document %s processing failed", document_id)
            try:
                repo.update_document_status(document_id, ProcessingStatus.FAILED)
                self._publish_event(
                    document_id,
                    {
                        "event": "status_change",
                        "document_id": document_id,
                        "status": "failed",
                    },
                )
            except Exception:
                logger.exception("Failed to mark document %s as FAILED", document_id)
        finally:
            self._active_jobs.release()

    def try_submit(
        self,
        document_id: str,
        content_sha256: str,
        file_bytes: bytes,
        filename: str,
    ) -> Future[None] | None:
        """Submit a document if capacity is available, otherwise return ``None``.

        Callers should check the return value and respond with a 503 when
        ``None`` is returned.
        """
        if self._executor is None:
            raise RuntimeError("ProcessingPool has not been started")
        if not self.has_capacity():
            return None
        return self._executor.submit(
            self._process_document,
            document_id,
            content_sha256,
            file_bytes,
            filename,
        )
