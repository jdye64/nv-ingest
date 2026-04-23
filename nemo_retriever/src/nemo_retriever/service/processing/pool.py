# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ProcessPoolExecutor-based document processing for service mode.

Each worker process builds its own operator chain at startup via
``_worker_initializer`` and keeps it alive for the lifetime of the
process.  This eliminates all C-library thread-safety issues (pypdfium2,
image processing) because every chain runs in its own address space.

When multiple NIM endpoint URLs are configured (comma-separated), the
pool distributes them round-robin across workers so that each worker
targets exactly one URL per endpoint type, balancing traffic evenly.

Pages are accumulated in a ``_BatchBuffer`` and dispatched to worker
processes in batches (default 32) so that NIM endpoints and GPUs see
larger inference batches.  A configurable timeout ensures small jobs
are dispatched promptly without waiting for a full batch.

Workers write results directly to SQLite (WAL mode supports concurrent
multi-process writers) and return a lightweight ``BatchWorkerResult`` to
the main process.  The main process uses the result to publish SSE
events and track job completion — the only operations that require the
in-process ``EventBus`` / ``asyncio`` event loop.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import multiprocessing
import os
import threading
import time
import typing
from concurrent.futures import Future, ProcessPoolExecutor
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

_SERIALIZE_SKIP = frozenset(
    {
        "bytes",
        "_page_document_id",
        "_page_job_id",
        "_page_number",
        "_page_filename",
    }
)


# ======================================================================
# Pure helper functions (used in both main and worker processes)
# ======================================================================


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
    return {k: _safe_value(v) for k, v in row.items() if k not in _SERIALIZE_SKIP}


def _extract_metrics_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull detection/invocation counts and per-label breakdowns from pipeline columns."""
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


_NIM_URL_FIELDS = (
    "page_elements_invoke_url",
    "ocr_invoke_url",
    "table_structure_invoke_url",
    "graphic_elements_invoke_url",
    "embed_invoke_url",
)


def _resolve_worker_nim_configs(
    nim: NimEndpointsConfig,
    num_workers: int,
) -> list[dict[str, Any]]:
    """Split comma-separated NIM URLs and round-robin them across workers.

    Returns *num_workers* plain dicts, each suitable for
    ``NimEndpointsConfig(**d)``.  Fields that contain a single URL (or
    are ``None``) are replicated identically to every worker.  Fields
    with multiple comma-separated URLs are distributed so that worker *i*
    gets ``urls[i % len(urls)]``.
    """
    base = nim.model_dump()

    parsed: dict[str, list[str]] = {}
    for field in _NIM_URL_FIELDS:
        raw = base.get(field)
        if raw and "," in raw:
            parsed[field] = [u.strip() for u in raw.split(",") if u.strip()]
        else:
            parsed[field] = [raw] if raw else []

    configs: list[dict[str, Any]] = []
    for i in range(num_workers):
        worker_dict = dict(base)
        for field in _NIM_URL_FIELDS:
            urls = parsed[field]
            if len(urls) > 1:
                worker_dict[field] = urls[i % len(urls)]
        configs.append(worker_dict)

    return configs


def _log_endpoint_distribution(nim: NimEndpointsConfig, num_workers: int) -> None:
    """Log a summary of how NIM endpoints are distributed across workers."""
    lines: list[str] = []
    for field in _NIM_URL_FIELDS:
        raw = getattr(nim, field, None)
        if not raw:
            continue
        urls = [u.strip() for u in raw.split(",") if u.strip()]
        n = len(urls)
        if n == 1:
            lines.append(f"  {field}: 1 endpoint -> all {num_workers} workers")
        else:
            per = num_workers // n
            remainder = num_workers % n
            if remainder == 0:
                lines.append(f"  {field}: {n} endpoints -> {per} workers each")
            else:
                lines.append(f"  {field}: {n} endpoints -> ~{per}-{per + 1} workers each")
            if n > num_workers:
                lines.append(
                    f"    WARNING: {n - num_workers} endpoint(s) will not be assigned (more URLs than workers)"
                )
    if lines:
        logger.info("NIM endpoint distribution across %d workers:\n%s", num_workers, "\n".join(lines))


def _build_params(nim: NimEndpointsConfig) -> tuple[Any, Any]:
    """Construct ``ExtractParams`` and (optionally) ``EmbedParams`` from NIM config."""
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
    replica_id: int | str,
    nim_endpoints: NimEndpointsConfig,
) -> list[tuple[str, Any]]:
    """Build a linearised list of ``(name, operator_instance)`` pairs."""
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
    logger.info("[pid %d] Building operator chain %s (%s)", os.getpid(), replica_id, mode_label)

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
    logger.info("[pid %d] Operator chain %s ready (%d stages)", os.getpid(), replica_id, len(operators))
    return operators


# ======================================================================
# Worker-process state and entry point
# ======================================================================

_worker_chain: list[tuple[str, Any]] | None = None
_worker_db_path: str | None = None


def _worker_initializer(
    db_path: str,
    nim_config_queue: multiprocessing.Queue,  # type: ignore[type-arg]
    fallback_nim_config: dict[str, Any],
) -> None:
    """Called exactly once per worker process by ProcessPoolExecutor.

    Each worker pops a unique NIM config from the shared queue so that
    multi-endpoint URLs are distributed round-robin.  If the queue is
    empty (more processes than configs — shouldn't happen) the fallback
    config is used so the worker still functions.
    """
    global _worker_chain, _worker_db_path
    try:
        import setproctitle

        setproctitle.setproctitle("nemo-retriever-worker")
    except ImportError:
        pass

    _worker_db_path = db_path
    try:
        nim_config_dict = nim_config_queue.get_nowait()
    except Exception:
        logger.warning("[pid %d] NIM config queue empty — using fallback config", os.getpid())
        nim_config_dict = fallback_nim_config

    nim = NimEndpointsConfig(**nim_config_dict)
    _worker_chain = _build_operator_chain(os.getpid(), nim)


@dataclasses.dataclass
class PageDescriptor:
    """Lightweight struct describing a single page to be batched."""

    document_id: str
    content_sha256: str
    file_bytes: bytes
    filename: str
    job_id: str | None = None
    page_number: int = 1


@dataclasses.dataclass
class WorkerResult:
    """Per-page result.  A batch returns a list of these."""

    document_id: str
    job_id: str | None
    source_file: str
    page_number: int
    success: bool
    error_message: str | None = None
    metrics: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    detection_count: int = 0
    processing_duration_ms: float = 0.0
    started_at: str = ""
    completed_at: str = ""
    total_pages: int = 0


@dataclasses.dataclass
class BatchWorkerResult:
    """Picklable result returned from a worker process to the main process.

    Contains one :class:`WorkerResult` per page in the batch.
    """

    results: list[WorkerResult] = dataclasses.field(default_factory=list)


# ======================================================================
# Provenance column used to track which output rows belong to which page
# after the content-explosion step expands 1 input row into N output rows.
# ======================================================================

_PROVENANCE_COL = "_page_document_id"
_PROVENANCE_JOB_COL = "_page_job_id"
_PROVENANCE_NUM_COL = "_page_number"
_PROVENANCE_FILE_COL = "_page_filename"


def _run_pipeline_batch(
    page_descriptors: list[dict[str, Any]],
    *,
    db_path: str = "",
) -> BatchWorkerResult:
    """Execute the operator chain on a batch of pages.

    Builds a multi-row DataFrame (one row per page), runs the full
    pipeline once, then splits the output back into per-page results
    using the ``_page_document_id`` provenance column.
    """
    pid = os.getpid()
    engine = DatabaseEngine(db_path)
    repo = Repository(engine)

    pages = [PageDescriptor(**d) for d in page_descriptors]
    n_pages = len(pages)

    logger.info(
        "[worker %d] Processing batch of %d pages (%s)",
        pid,
        n_pages,
        ", ".join(f"{p.filename} p{p.page_number}" for p in pages[:5]) + (" ..." if n_pages > 5 else ""),
    )

    for p in pages:
        try:
            repo.update_document_status(p.document_id, ProcessingStatus.PROCESSING)
        except Exception:
            pass

    rows = []
    for p in pages:
        rows.append(
            {
                "bytes": p.file_bytes,
                "path": p.filename,
                _PROVENANCE_COL: p.document_id,
                _PROVENANCE_JOB_COL: p.job_id,
                _PROVENANCE_NUM_COL: p.page_number,
                _PROVENANCE_FILE_COL: p.filename,
            }
        )
    df = pd.DataFrame(rows)

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()

    try:
        if _worker_chain is None:
            raise RuntimeError("Worker operator chain not initialised — _worker_initializer was never called")
        for stage_name, op in _worker_chain:
            df = op.run(df)
            if df is None:
                raise RuntimeError(
                    f"Operator '{stage_name}' ({type(op).__name__}) returned None instead of a DataFrame"
                )
    except BaseException as exc:
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error(
            "[worker %d] BATCH FAILED (%d pages): %s",
            pid,
            n_pages,
            error_msg,
            exc_info=True,
        )
        completed_at = datetime.now(timezone.utc).isoformat()
        fail_results: list[WorkerResult] = []
        for p in pages:
            _mark_page_failed(repo, p, error_msg, started_at, completed_at)
            fail_results.append(
                WorkerResult(
                    document_id=p.document_id,
                    job_id=p.job_id,
                    source_file=p.filename,
                    page_number=p.page_number,
                    success=False,
                    error_message=error_msg,
                )
            )
        return BatchWorkerResult(results=fail_results)

    elapsed_ms = (time.monotonic() - t0) * 1000.0
    completed_at = datetime.now(timezone.utc).isoformat()

    logger.info(
        "[worker %d] Batch of %d pages complete — %d output rows, %.0fms",
        pid,
        n_pages,
        len(df),
        elapsed_ms,
    )

    return _split_batch_results(
        repo,
        df,
        pages,
        elapsed_ms,
        started_at,
        completed_at,
    )


def _mark_page_failed(
    repo: Repository,
    p: PageDescriptor,
    error_msg: str,
    started_at: str,
    completed_at: str,
) -> None:
    """Mark a single page as failed in the DB."""
    try:
        repo.update_document_status(p.document_id, ProcessingStatus.FAILED)
        source_file = _resolve_source_file(repo, p)
        fail_log = PageProcessingLog(
            id=PageProcessingLog.make_id(source_file, p.page_number),
            document_id=p.document_id,
            job_id=p.job_id,
            source_file=source_file,
            page_number=p.page_number,
            status="failed",
            error_message=error_msg,
            detection_count=0,
            processing_duration_ms=0.0,
            started_at=started_at,
            completed_at=completed_at,
        )
        repo.insert_page_processing_log(fail_log)
    except Exception:
        pass


def _resolve_source_file(repo: Repository, p: PageDescriptor) -> str:
    """Look up the human-readable source filename for a page."""
    doc = repo.get_document(p.document_id)
    source_file = doc.filename if doc else p.filename
    if doc and doc.job_id:
        parent_job = repo.get_job(doc.job_id)
        if parent_job:
            source_file = parent_job.filename
    return source_file


def _split_batch_results(
    repo: Repository,
    df: pd.DataFrame,
    pages: list[PageDescriptor],
    elapsed_ms: float,
    started_at: str,
    completed_at: str,
) -> BatchWorkerResult:
    """Split the batched pipeline output back into per-page results.

    Groups output rows by ``_page_document_id``, extracts metrics from
    the first row of each group (before explosion duplicates), persists
    page results and metrics to SQLite, and assembles per-page
    :class:`WorkerResult` objects.
    """
    pid = os.getpid()
    page_lookup = {p.document_id: p for p in pages}

    if _PROVENANCE_COL not in df.columns:
        logger.warning("[worker %d] Provenance column missing — cannot split batch results", pid)
        results: list[WorkerResult] = []
        for p in pages:
            results.append(
                WorkerResult(
                    document_id=p.document_id,
                    job_id=p.job_id,
                    source_file=p.filename,
                    page_number=p.page_number,
                    success=False,
                    error_message="Provenance column lost during pipeline execution",
                )
            )
        return BatchWorkerResult(results=results)

    grouped = df.groupby(_PROVENANCE_COL, sort=False)
    results = []

    for doc_id, group_df in grouped:
        doc_id = str(doc_id)
        p = page_lookup.get(doc_id)
        if p is None:
            continue

        source_file = _resolve_source_file(repo, p)
        total_rows = len(group_df)

        repo.update_document_status(doc_id, ProcessingStatus.PROCESSING, total_pages=total_rows)

        page_metrics = _extract_metrics_from_row(group_df.iloc[0].to_dict())
        if page_metrics:
            metric_objs = [
                ProcessingMetric(
                    document_id=doc_id,
                    model_name=m["model_name"],
                    invocation_count=m.get("invocation_count", 1),
                    pages_processed=m.get("pages_processed", 1),
                    detections_count=m.get("detections_count", 0),
                    counts_by_label_json=json.dumps(m.get("counts_by_label", {})),
                )
                for m in page_metrics
            ]
            repo.insert_metrics(metric_objs)

        for page_num, (_, row) in enumerate(group_df.iterrows()):
            row_dict = row.to_dict()
            content = _row_to_page_content(row_dict)
            page_result = PageResult(
                document_id=doc_id,
                page_number=page_num,
                content_json=json.dumps(content),
            )
            repo.insert_page_result(page_result)
            repo.increment_pages_received(doc_id)

        det_total = sum(m.get("detections_count", 0) for m in page_metrics)

        log_entry = PageProcessingLog(
            id=PageProcessingLog.make_id(source_file, p.page_number),
            document_id=doc_id,
            job_id=p.job_id,
            source_file=source_file,
            page_number=p.page_number,
            detection_count=det_total,
            processing_duration_ms=elapsed_ms,
            started_at=started_at,
            completed_at=completed_at,
        )
        repo.insert_page_processing_log(log_entry)
        repo.update_document_status(doc_id, ProcessingStatus.COMPLETE)

        results.append(
            WorkerResult(
                document_id=doc_id,
                job_id=p.job_id,
                source_file=source_file,
                page_number=p.page_number,
                success=True,
                metrics=page_metrics,
                detection_count=det_total,
                processing_duration_ms=elapsed_ms,
                started_at=started_at,
                completed_at=completed_at,
                total_pages=total_rows,
            )
        )

    seen = {r.document_id for r in results}
    for p in pages:
        if p.document_id not in seen:
            _mark_page_failed(
                repo,
                p,
                "Page produced no output rows after pipeline execution",
                started_at,
                completed_at,
            )
            results.append(
                WorkerResult(
                    document_id=p.document_id,
                    job_id=p.job_id,
                    source_file=p.filename,
                    page_number=p.page_number,
                    success=False,
                    error_message="Page produced no output rows after pipeline execution",
                )
            )

    return BatchWorkerResult(results=results)


# ======================================================================
# Main-process pool manager
# ======================================================================


class _BatchBuffer:
    """Thread-safe buffer that accumulates pages and auto-flushes.

    A batch is dispatched to the executor when either:
    - The buffer reaches ``batch_size`` pages, **or**
    - ``timeout_s`` seconds have passed since the first page was added
      to the current (non-empty) buffer.

    The buffer enforces a hard cap of ``max_buffered`` pages to bound
    memory usage.  When the cap is reached, ``enqueue`` returns ``False``
    so the caller can 503 the client.

    IMPORTANT: the dispatch function is called **outside** the lock so
    that pickling large batches and submitting to the executor never
    blocks the event-loop thread from enqueuing new pages or receiving
    SSE events.
    """

    def __init__(
        self,
        batch_size: int,
        timeout_s: float,
        max_buffered: int,
        dispatch_fn: "typing.Callable[[list[dict[str, Any]]], None]",
    ) -> None:
        self._batch_size = batch_size
        self._timeout_s = timeout_s
        self._max_buffered = max_buffered
        self._dispatch_fn = dispatch_fn
        self._lock = threading.Lock()
        self._buffer: list[dict[str, Any]] = []
        self._timer: threading.Timer | None = None
        self._closed = False

    @property
    def buffered_count(self) -> int:
        with self._lock:
            return len(self._buffer)

    def enqueue(self, page: dict[str, Any]) -> bool:
        """Add a page to the buffer.  Returns ``False`` if full."""
        batch: list[dict[str, Any]] | None = None
        with self._lock:
            if self._closed:
                return False
            if len(self._buffer) >= self._max_buffered:
                return False
            self._buffer.append(page)
            if len(self._buffer) >= self._batch_size:
                batch = self._drain_locked()
            elif len(self._buffer) == 1:
                self._start_timer_locked()
        if batch is not None:
            self._safe_dispatch(batch)
        return True

    def flush(self) -> None:
        """Force-flush whatever is in the buffer."""
        batch: list[dict[str, Any]] | None = None
        with self._lock:
            batch = self._drain_locked()
        if batch is not None:
            self._safe_dispatch(batch)

    def close(self) -> None:
        """Flush remaining pages and prevent further enqueues."""
        batch: list[dict[str, Any]] | None = None
        with self._lock:
            self._closed = True
            batch = self._drain_locked()
        if batch is not None:
            self._safe_dispatch(batch)

    def _drain_locked(self) -> list[dict[str, Any]] | None:
        """Extract the current buffer contents.  Must hold ``_lock``."""
        self._cancel_timer_locked()
        if not self._buffer:
            return None
        batch = self._buffer[:]
        self._buffer.clear()
        return batch

    def _safe_dispatch(self, batch: list[dict[str, Any]]) -> None:
        """Dispatch a batch outside the lock, handling errors."""
        try:
            self._dispatch_fn(batch)
        except Exception:
            logger.exception("Failed to dispatch batch of %d pages", len(batch))

    def _start_timer_locked(self) -> None:
        self._cancel_timer_locked()
        self._timer = threading.Timer(self._timeout_s, self._on_timeout)
        self._timer.daemon = True
        self._timer.start()

    def _cancel_timer_locked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _on_timeout(self) -> None:
        batch: list[dict[str, Any]] | None = None
        with self._lock:
            batch = self._drain_locked()
        if batch is not None:
            self._safe_dispatch(batch)


class ProcessingPool:
    """Manages a ``ProcessPoolExecutor`` of isolated worker processes.

    Pages are accumulated in a ``_BatchBuffer`` and dispatched in
    batches (default 32) to worker processes.  Each worker process
    builds its own operator chain at startup, completely eliminating
    C-library thread-safety issues.  Workers write results to SQLite
    directly and return a ``BatchWorkerResult`` to the main process
    for SSE event publishing and job-completion tracking.
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
        self._num_workers = config.processing.num_workers
        self._batch_size = config.processing.batch_size
        self._batch_timeout_s = config.processing.batch_timeout_s
        self._executor: ProcessPoolExecutor | None = None
        self._buffer: _BatchBuffer | None = None
        self._in_flight = 0
        self._in_flight_lock = threading.Lock()

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
            "Starting processing pool: %d worker processes, batch_size=%d, timeout=%.1fs (%s)",
            self._num_workers,
            self._batch_size,
            self._batch_timeout_s,
            mode_label,
        )

        worker_nim_dicts = _resolve_worker_nim_configs(nim, self._num_workers)
        _log_endpoint_distribution(nim, self._num_workers)

        db_path = str(self._db_engine._db_path)

        ctx = multiprocessing.get_context("spawn")
        nim_config_queue: multiprocessing.Queue[dict[str, Any]] = ctx.Queue()
        for cfg in worker_nim_dicts:
            nim_config_queue.put(cfg)
        fallback_nim_config = nim.model_dump()

        logger.info("Initialising %d worker(s) — building operator chains", self._num_workers)
        self._executor = ProcessPoolExecutor(
            max_workers=self._num_workers,
            mp_context=ctx,
            initializer=_worker_initializer,
            initargs=(db_path, nim_config_queue, fallback_nim_config),
        )

        warmup_futures = [self._executor.submit(os.getpid) for _ in range(self._num_workers)]
        pids: set[int] = set()
        for fut in warmup_futures:
            pids.add(fut.result())
        logger.info("All %d worker process(es) initialised and ready", len(pids))

        max_buffered = self._num_workers * self._batch_size
        self._buffer = _BatchBuffer(
            batch_size=self._batch_size,
            timeout_s=self._batch_timeout_s,
            max_buffered=max_buffered,
            dispatch_fn=self._dispatch_batch,
        )
        logger.info(
            "Batch buffer ready — batch_size=%d, timeout=%.1fs, max_buffered=%d",
            self._batch_size,
            self._batch_timeout_s,
            max_buffered,
        )

    def shutdown(self) -> None:
        if self._buffer is not None:
            self._buffer.close()
            self._buffer = None
        if self._executor is not None:
            logger.info("Shutting down processing pool …")
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    @property
    def _pool_size(self) -> int:
        """Backwards-compatible accessor used by the ingest router."""
        return self._num_workers

    @property
    def capacity(self) -> int:
        buf_count = self._buffer.buffered_count if self._buffer else 0
        max_buf = self._num_workers * self._batch_size
        return max(0, max_buf - buf_count)

    def has_capacity(self) -> bool:
        return self.capacity > 0

    # ------------------------------------------------------------------
    # SSE event publishing (main process only)
    # ------------------------------------------------------------------

    def _publish_event(self, document_id: str, event: dict[str, Any]) -> None:
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

    # ------------------------------------------------------------------
    # Batch dispatch (called by _BatchBuffer from its timer/flush)
    # ------------------------------------------------------------------

    def _dispatch_batch(self, batch: list[dict[str, Any]]) -> None:
        """Submit a batch of pages to the executor as a single task."""
        if self._executor is None:
            raise RuntimeError("ProcessingPool has not been started")

        with self._in_flight_lock:
            self._in_flight += 1

        logger.debug("Dispatching batch of %d pages to executor", len(batch))
        fut: Future[BatchWorkerResult] = self._executor.submit(
            _run_pipeline_batch,
            batch,
            db_path=str(self._db_engine._db_path),
        )
        fut.add_done_callback(self._on_batch_result)

    # ------------------------------------------------------------------
    # Result callback (runs in main process on a callback thread)
    # ------------------------------------------------------------------

    def _on_batch_result(self, future: Future[BatchWorkerResult]) -> None:
        """Called in the main process when a batch finishes."""
        with self._in_flight_lock:
            self._in_flight -= 1

        try:
            batch_result = future.result()
        except Exception as exc:
            logger.exception("Worker process raised an unhandled exception: %s", exc)
            return

        for result in batch_result.results:
            self._handle_single_result(result)

    def _handle_single_result(self, result: WorkerResult) -> None:
        """Process one per-page result: publish SSE events and check job completion."""
        doc_id = result.document_id
        job_id = result.job_id

        if result.success:
            for m in result.metrics:
                self._publish_event(
                    doc_id,
                    {
                        "event": "metrics_update",
                        "document_id": doc_id,
                        "job_id": job_id,
                        "model_name": m["model_name"],
                        "invocation_count": m.get("invocation_count", 1),
                        "detections_count": m.get("detections_count", 0),
                    },
                )

            page_complete_payload: dict[str, Any] = {
                "event": "page_complete",
                "document_id": doc_id,
                "pages_received": result.total_pages,
                "total_pages": result.total_pages,
            }
            if job_id:
                page_complete_payload["job_id"] = job_id
            self._publish_event(doc_id, page_complete_payload)

            self._publish_event(
                doc_id,
                {
                    "event": "document_complete",
                    "document_id": doc_id,
                    "job_id": job_id,
                    "total_pages": result.total_pages,
                },
            )
        else:
            self._publish_event(
                doc_id,
                {
                    "event": "status_change",
                    "document_id": doc_id,
                    "job_id": job_id,
                    "status": "failed",
                    "source_file": result.source_file,
                    "page_number": result.page_number,
                    "error": result.error_message or "unknown error",
                },
            )

        if job_id:
            self._handle_job_completion(job_id)

    def _handle_job_completion(self, job_id: str) -> None:
        """Increment job progress; if all pages are done, mark the job complete."""
        repo = Repository(self._db_engine)
        pages_completed = repo.increment_job_pages_completed(job_id)
        job = repo.get_job(job_id)
        if job is None:
            return

        if pages_completed >= job.total_pages and job.total_pages > 0:
            repo.update_job_status(job_id, ProcessingStatus.COMPLETE)
            logger.info("[job %s] All %d pages of %s complete", job_id[:8], job.total_pages, job.filename)
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

    # ------------------------------------------------------------------
    # Submit (public API — enqueues into the batch buffer)
    # ------------------------------------------------------------------

    def try_submit(
        self,
        document_id: str,
        content_sha256: str,
        file_bytes: bytes,
        filename: str,
        *,
        job_id: str | None = None,
        page_number: int = 1,
    ) -> bool:
        """Enqueue a page for batched processing.

        Returns ``True`` if the page was accepted, ``False`` if the
        buffer is full (caller should 503).
        """
        if self._executor is None:
            raise RuntimeError("ProcessingPool has not been started")
        if self._buffer is None:
            raise RuntimeError("Batch buffer has not been initialised")

        page_dict = dataclasses.asdict(
            PageDescriptor(
                document_id=document_id,
                content_sha256=content_sha256,
                file_bytes=file_bytes,
                filename=filename,
                job_id=job_id,
                page_number=page_number,
            )
        )
        return self._buffer.enqueue(page_dict)
