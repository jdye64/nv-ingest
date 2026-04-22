# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ThreadPoolExecutor-based document processing for service mode.

Operator chains are pre-built at startup and shared across worker threads
via a ``queue.Queue``.  When the ``nim_endpoints`` config section provides
remote NIM endpoint URLs the pipeline operators make HTTP calls instead of
loading local GPU models — removing all VRAM requirements and allowing a
high ``pipeline_replicas`` count.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from nemo_retriever.service.config import NimEndpointsConfig, ServiceConfig
from nemo_retriever.service.db.engine import DatabaseEngine
from nemo_retriever.service.db.repository import Repository
from nemo_retriever.service.event_bus import EventBus
from nemo_retriever.service.models.document import ProcessingStatus
from nemo_retriever.service.models.metrics import ProcessingMetric
from nemo_retriever.service.models.page_processing_log import PageProcessingLog
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
    """Pull detection/invocation counts and per-label breakdowns from pipeline columns.

    The pipeline produces two companion columns per model:
    - ``<model>_num_detections`` → ``int``
    - ``<model>_counts_by_label`` → ``dict[str, int]``

    We merge them into a single metric dict keyed by model name.
    """
    by_model: dict[str, dict[str, Any]] = {}

    for col, val in row.items():
        if val is None:
            continue

        if col.endswith("_num_detections"):
            model = col[: -len("_num_detections")]
            entry = by_model.setdefault(
                model,
                {
                    "model_name": model,
                    "detections_count": 0,
                    "counts_by_label": {},
                    "invocation_count": 1,
                    "pages_processed": 1,
                },
            )
            entry["detections_count"] = int(val) if isinstance(val, (int, float)) else 0

        elif col.endswith("_counts_by_label"):
            model = col[: -len("_counts_by_label")]
            entry = by_model.setdefault(
                model,
                {
                    "model_name": model,
                    "detections_count": 0,
                    "counts_by_label": {},
                    "invocation_count": 1,
                    "pages_processed": 1,
                },
            )
            if isinstance(val, dict):
                entry["counts_by_label"] = {str(k): int(v) for k, v in val.items()}
                if entry["detections_count"] == 0:
                    entry["detections_count"] = sum(int(v) for v in val.values())

    return list(by_model.values())


# CUDA / PyTorch model loading is not thread-safe.  When using local GPU
# models the lock serialises chain construction.  With remote NIM endpoints
# the lock is still acquired for safety but init is near-instant.
_INIT_LOCK = threading.Lock()


def _build_params(nim: NimEndpointsConfig) -> tuple[Any, Any]:
    """Construct ``ExtractParams`` and (optionally) ``EmbedParams`` from NIM config.

    Returns ``(extract_params, embed_params)`` where *embed_params* is
    ``None`` when no ``embed_invoke_url`` is configured.
    """
    import os
    from nemo_retriever.params import ExtractParams, EmbedParams

    api_key = nim.api_key or os.environ.get("NVIDIA_API_KEY")

    extract_kwargs: dict[str, Any] = {}
    if nim.page_elements_invoke_url:
        extract_kwargs["page_elements_invoke_url"] = nim.page_elements_invoke_url
    if nim.ocr_invoke_url:
        extract_kwargs["ocr_invoke_url"] = nim.ocr_invoke_url
    if nim.table_structure_invoke_url:
        extract_kwargs["table_structure_invoke_url"] = nim.table_structure_invoke_url
    if nim.graphic_elements_invoke_url:
        extract_kwargs["graphic_elements_invoke_url"] = nim.graphic_elements_invoke_url
    if api_key:
        extract_kwargs["api_key"] = api_key

    embed_params = None
    if nim.embed_invoke_url:
        embed_kwargs: dict[str, Any] = {"embed_invoke_url": nim.embed_invoke_url}
        if api_key:
            embed_kwargs["api_key"] = api_key
        embed_params = EmbedParams(**embed_kwargs)

    return ExtractParams(**extract_kwargs), embed_params


def _build_operator_chain(
    replica_id: int,
    nim_endpoints: NimEndpointsConfig,
) -> list[tuple[str, Any]]:
    """Build a linearised list of ``(name, operator_instance)`` pairs.

    When ``nim_endpoints`` provides remote URLs the archetype operators
    resolve to lightweight CPU variants that call the NIM HTTP APIs instead
    of loading local GPU models.
    """
    from nemo_retriever.graph.ingestor_runtime import build_graph
    from nemo_retriever.graph.executor import InprocessExecutor
    from nemo_retriever.graph.operator_resolution import resolve_graph
    from nemo_retriever.utils.ray_resource_hueristics import gather_local_resources

    extract_params, embed_params = _build_params(nim_endpoints)

    has_remote = any(
        [
            nim_endpoints.page_elements_invoke_url,
            nim_endpoints.ocr_invoke_url,
            nim_endpoints.table_structure_invoke_url,
            nim_endpoints.graphic_elements_invoke_url,
            nim_endpoints.embed_invoke_url,
        ]
    )
    mode_label = "remote NIM" if has_remote else "local GPU"
    print(f">>> Building operator chain replica {replica_id} ({mode_label}) …")

    graph = build_graph(
        extraction_mode="pdf",
        extract_params=extract_params,
        embed_params=embed_params,
        stage_order=(),
    )
    resolved = resolve_graph(graph, gather_local_resources())
    nodes = InprocessExecutor._linearize(resolved)
    operators: list[tuple[str, Any]] = []
    for node in nodes:
        op = node.operator_class(**node.operator_kwargs)
        operators.append((node.name, op))
    print(f">>> Operator chain replica {replica_id} ready ({len(operators)} stages)")
    return operators


class ProcessingPool:
    """Manages a fixed-size thread pool that shares a small number of
    operator chain replicas.

    ``thread_pool_size`` controls I/O concurrency (how many documents can
    be in-flight at once — DB writes, event publishing, etc.).
    ``pipeline_replicas`` controls how many pre-built operator chains exist.

    The CPU actors in the pipeline (PDFium rendering, image cropping) use
    C/C++ libraries that are **not thread-safe**.  To prevent heap
    corruption, a single ``_chain_exec_lock`` serialises all operator chain
    execution so only one chain runs at a time.  Threads still overlap on
    everything *outside* the chain: uploads, DB writes, result storage, and
    SSE event publishing.
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
        self._num_replicas = config.processing.pipeline_replicas
        self._executor: ThreadPoolExecutor | None = None
        self._active_jobs = threading.Semaphore(self._pool_size)
        self._operator_pool: queue.Queue[list[tuple[str, Any]]] = queue.Queue(
            maxsize=self._num_replicas,
        )
        # Serialise operator chain execution to prevent C-library heap
        # corruption from concurrent pypdfium2 / image-processing calls.
        self._chain_exec_lock = threading.Lock()

    def start(self) -> None:
        nim = self._config.nim_endpoints
        has_remote = any(
            [
                nim.page_elements_invoke_url,
                nim.ocr_invoke_url,
                nim.table_structure_invoke_url,
                nim.graphic_elements_invoke_url,
                nim.embed_invoke_url,
            ]
        )
        mode_label = "remote NIM" if has_remote else "local GPU"
        logger.info(
            "Starting processing pool: %d threads, %d pipeline replicas (%s)",
            self._pool_size,
            self._num_replicas,
            mode_label,
        )
        print(
            f">>> Initialising {self._num_replicas} pipeline replica(s) "
            f"({mode_label}) across {self._pool_size} worker threads …"
        )
        with _INIT_LOCK:
            for i in range(self._num_replicas):
                chain = _build_operator_chain(i, nim)
                self._operator_pool.put(chain)
        print(f">>> All {self._num_replicas} pipeline replicas ready")
        self._executor = ThreadPoolExecutor(
            max_workers=self._pool_size,
            thread_name_prefix="retriever-worker",
        )

    def shutdown(self) -> None:
        if self._executor is not None:
            logger.info("Shutting down processing pool …")
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def _checkout_operators(self) -> list[tuple[str, Any]]:
        """Block until a shared operator chain is available."""
        return self._operator_pool.get()

    def _return_operators(self, operators: list[tuple[str, Any]]) -> None:
        self._operator_pool.put(operators)

    @property
    def capacity(self) -> int:
        """Number of free worker slots available right now."""
        return getattr(self._active_jobs, "_value", 0)

    def has_capacity(self) -> bool:
        """Return ``True`` if at least one worker slot is free."""
        return self.capacity > 0

    def _publish_event(self, document_id: str, event: dict[str, Any]) -> None:
        """Thread-safe publish to the async event bus.

        Events are published under the document_id key.  If a ``job_id`` is
        present it is also published under that key so job-level SSE
        subscriptions receive it.
        """
        asyncio.run_coroutine_threadsafe(
            self._event_bus.publish(document_id, event),
            self._event_loop,
        )
        job_id = event.get("job_id")
        if job_id:
            asyncio.run_coroutine_threadsafe(
                self._event_bus.publish(job_id, event),
                self._event_loop,
            )

    def _store_results(
        self,
        repo: Repository,
        document_id: str,
        content_sha256: str,
        df: pd.DataFrame,
        *,
        job_id: str | None = None,
        page_number: int = 1,
        processing_duration_ms: float = 0.0,
        started_at: str = "",
        completed_at: str = "",
    ) -> None:
        """Persist pipeline output rows as page results and metrics."""
        total_pages = len(df)
        thread_name = threading.current_thread().name

        doc = repo.get_document(document_id)
        source_file = doc.filename if doc else "unknown"
        if doc and doc.job_id:
            parent_job = repo.get_job(doc.job_id)
            if parent_job:
                source_file = parent_job.filename

        repo.update_document_status(
            document_id,
            ProcessingStatus.PROCESSING,
            total_pages=total_pages,
        )

        all_metrics: list[dict[str, Any]] = []
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
                    counts_by_label_json=json.dumps(m.get("counts_by_label", {})),
                )
                for m in row_metrics
            ]
            repo.insert_metrics(metric_objs)
            all_metrics.extend(row_metrics)

            repo.increment_pages_received(document_id)

        det_total = sum(m.get("detections_count", 0) for m in all_metrics)
        print(
            f">>> [{thread_name}] Doc {document_id[:8]} complete"
            f" — {len(df)} rows, {det_total} detections, {processing_duration_ms:.0f}ms"
        )

        log_entry = PageProcessingLog(
            id=PageProcessingLog.make_id(source_file, page_number),
            document_id=document_id,
            job_id=job_id,
            source_file=source_file,
            page_number=page_number,
            detection_count=det_total,
            processing_duration_ms=processing_duration_ms,
            started_at=started_at,
            completed_at=completed_at,
        )
        repo.insert_page_processing_log(log_entry)

        for m in all_metrics:
            self._publish_event(
                document_id,
                {
                    "event": "metrics_update",
                    "document_id": document_id,
                    "job_id": job_id,
                    "model_name": m["model_name"],
                    "invocation_count": m.get("invocation_count", 1),
                    "detections_count": m.get("detections_count", 0),
                },
            )

        page_complete_payload: dict[str, Any] = {
            "event": "page_complete",
            "document_id": document_id,
            "pages_received": len(df),
            "total_pages": total_pages,
        }
        if job_id:
            page_complete_payload["job_id"] = job_id
        self._publish_event(document_id, page_complete_payload)

        repo.update_document_status(document_id, ProcessingStatus.COMPLETE)
        self._publish_event(
            document_id,
            {
                "event": "document_complete",
                "document_id": document_id,
                "job_id": job_id,
                "total_pages": total_pages,
            },
        )

        if job_id:
            self._handle_job_completion(repo, job_id)

    def _handle_job_completion(self, repo: Repository, job_id: str) -> None:
        """Increment job progress; if all pages are done, mark the job complete."""
        pages_completed = repo.increment_job_pages_completed(job_id)
        job = repo.get_job(job_id)
        if job is None:
            return

        if pages_completed >= job.total_pages and job.total_pages > 0:
            repo.update_job_status(job_id, ProcessingStatus.COMPLETE)
            print(f">>> [job {job_id[:8]}] All {job.total_pages} pages of" f" {job.filename} complete")
            asyncio.run_coroutine_threadsafe(
                self._event_bus.publish(
                    job_id,
                    {
                        "event": "job_complete",
                        "job_id": job_id,
                        "filename": job.filename,
                        "total_pages": job.total_pages,
                    },
                ),
                self._event_loop,
            )

    def _process_document(
        self,
        document_id: str,
        content_sha256: str,
        file_bytes: bytes,
        filename: str,
        *,
        job_id: str | None = None,
        page_number: int = 1,
    ) -> None:
        """Run a single document through a shared operator chain."""
        self._active_jobs.acquire()
        thread_name = threading.current_thread().name
        repo = Repository(self._db_engine)
        operators = None
        try:
            print(f">>> [{thread_name}] Processing {filename} p{page_number} (doc {document_id[:8]})")
            repo.update_document_status(document_id, ProcessingStatus.PROCESSING)
            df = pd.DataFrame([{"bytes": file_bytes, "path": filename}])

            started_at = datetime.now(timezone.utc).isoformat()
            t0 = time.monotonic()
            operators = self._checkout_operators()
            with self._chain_exec_lock:
                for _name, op in operators:
                    df = op.run(df)
            self._return_operators(operators)
            operators = None
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            completed_at = datetime.now(timezone.utc).isoformat()

            self._store_results(
                repo,
                document_id,
                content_sha256,
                df,
                job_id=job_id,
                page_number=page_number,
                processing_duration_ms=elapsed_ms,
                started_at=started_at,
                completed_at=completed_at,
            )
            logger.info("Document %s processing complete (%d rows, %.0fms)", document_id, len(df), elapsed_ms)

        except BaseException as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.exception("Document %s processing failed", document_id)
            print(
                f">>> [{thread_name}] FAILED processing {filename} p{page_number} (doc {document_id[:8]}): {error_msg}"
            )
            try:
                repo.update_document_status(document_id, ProcessingStatus.FAILED)

                doc = repo.get_document(document_id)
                source_file = filename
                if doc and doc.job_id:
                    parent_job = repo.get_job(doc.job_id)
                    if parent_job:
                        source_file = parent_job.filename

                fail_log = PageProcessingLog(
                    id=PageProcessingLog.make_id(source_file, page_number),
                    document_id=document_id,
                    job_id=job_id,
                    source_file=source_file,
                    page_number=page_number,
                    status="failed",
                    error_message=error_msg,
                    detection_count=0,
                    processing_duration_ms=0.0,
                    started_at=datetime.now(timezone.utc).isoformat(),
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )
                repo.insert_page_processing_log(fail_log)

                self._publish_event(
                    document_id,
                    {
                        "event": "status_change",
                        "document_id": document_id,
                        "job_id": job_id,
                        "status": "failed",
                        "source_file": source_file,
                        "page_number": page_number,
                        "error": error_msg,
                    },
                )
                if job_id:
                    self._handle_job_completion(repo, job_id)
            except Exception:
                logger.exception("Failed to mark document %s as FAILED", document_id)
        finally:
            if operators is not None:
                self._return_operators(operators)
            self._active_jobs.release()

    def try_submit(
        self,
        document_id: str,
        content_sha256: str,
        file_bytes: bytes,
        filename: str,
        *,
        job_id: str | None = None,
        page_number: int = 1,
    ) -> Future[None] | None:
        """Submit a document if capacity is available, otherwise return ``None``."""
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
            job_id=job_id,
            page_number=page_number,
        )
