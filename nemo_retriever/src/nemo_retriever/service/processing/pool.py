# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ThreadPoolExecutor-based document processing for service mode.

Each worker thread owns a pre-built chain of instantiated operators (the same
graph that ``InprocessExecutor`` would run).  Documents are submitted as
futures; the webhook operator at the tail of the chain POSTs page-level
results back to ``/v1/ingest/internal_results``.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import pandas as pd

from nemo_retriever.service.config import ServiceConfig
from nemo_retriever.service.db.engine import DatabaseEngine
from nemo_retriever.service.db.repository import Repository
from nemo_retriever.service.models.document import ProcessingStatus

logger = logging.getLogger(__name__)


def _build_operator_chain(webhook_url: str) -> list[tuple[str, Any]]:
    """Build a linearised list of ``(name, operator_instance)`` pairs.

    This mirrors what ``InprocessExecutor`` does internally but we keep the
    instantiated operators around for repeated use within a single thread.
    """
    from nemo_retriever.graph.ingestor_runtime import build_graph
    from nemo_retriever.graph.executor import InprocessExecutor
    from nemo_retriever.graph.operator_resolution import resolve_graph
    from nemo_retriever.utils.ray_resource_hueristics import gather_local_resources
    from nemo_retriever.params import ExtractParams, WebhookParams

    graph = build_graph(
        extraction_mode="pdf",
        extract_params=ExtractParams(),
        webhook_params=WebhookParams(endpoint_url=webhook_url),
        stage_order=("webhook",),
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

    def __init__(self, config: ServiceConfig, db_engine: DatabaseEngine) -> None:
        self._config = config
        self._db_engine = db_engine
        self._pool_size = config.processing.thread_pool_size
        self._webhook_url = f"http://127.0.0.1:{config.server.port}/v1/ingest/internal_results"
        self._executor: ThreadPoolExecutor | None = None
        self._thread_state = _ThreadState()

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
            self._thread_state.operators = _build_operator_chain(self._webhook_url)
        return self._thread_state.operators

    def _process_document(self, document_id: str, file_bytes: bytes, filename: str) -> None:
        """Run a single document through the in-process operator chain."""
        repo = Repository(self._db_engine)
        try:
            repo.update_document_status(document_id, ProcessingStatus.PROCESSING)
            df = pd.DataFrame([{"bytes": file_bytes, "path": filename}])

            operators = self._get_operators()
            for _name, op in operators:
                df = op.run(df)

            doc = repo.get_document(document_id)
            if doc and doc.processing_status != ProcessingStatus.COMPLETE:
                repo.update_document_status(document_id, ProcessingStatus.COMPLETE)
                logger.info("Document %s processing complete", document_id)
        except Exception:
            logger.exception("Document %s processing failed", document_id)
            try:
                repo.update_document_status(document_id, ProcessingStatus.FAILED)
            except Exception:
                logger.exception("Failed to mark document %s as FAILED", document_id)

    def submit(self, document_id: str, file_bytes: bytes, filename: str) -> Future[None]:
        if self._executor is None:
            raise RuntimeError("ProcessingPool has not been started")
        return self._executor.submit(self._process_document, document_id, file_bytes, filename)
