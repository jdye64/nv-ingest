# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ServiceIngestor — fluent ingestor backed by a running retriever service.

Usage::

    from nemo_retriever import create_ingestor

    ing = create_ingestor(run_mode="service", base_url="http://localhost:7670")

    # Streaming — process each page as it arrives:
    for page in ing.files(["doc1.pdf", "doc2.pdf"]).ingest():
        print(page["_source_file"], page.get("text", "")[:80])

    # Collect into a DataFrame:
    import pandas as pd
    df = pd.DataFrame(ing.files(["doc1.pdf"]).ingest())

    # Async streaming:
    async for page in ing.ingest_async():
        ...
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from pathlib import Path
from typing import Any, AsyncGenerator, Generator, List, Optional, Union

from nemo_retriever.ingestor import ingestor

logger = logging.getLogger(__name__)

_SENTINEL = object()


class ServiceIngestor(ingestor):
    """Ingestor that delegates to a running retriever service over HTTP.

    The server-side pipeline is pre-configured via ``retriever-service.yaml``,
    so fluent pipeline methods (``.extract()``, ``.embed()``, etc.) are
    accepted but have no effect — a one-time info message is logged.

    Parameters
    ----------
    base_url
        Base URL of the running retriever service.
    max_concurrency
        Maximum concurrent page uploads.
    documents
        Initial list of file paths / glob patterns.
    """

    RUN_MODE = "service"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:7670",
        max_concurrency: int = 8,
        documents: Optional[List[str]] = None,
    ) -> None:
        super().__init__(documents=documents)
        self._base_url = base_url.rstrip("/")
        self._max_concurrency = max_concurrency
        self._pipeline_warning_emitted = False
        self._progress: dict[str, dict[str, int]] = {}
        self._progress_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Input configuration
    # ------------------------------------------------------------------

    def files(self, documents: Union[str, List[str]]) -> "ServiceIngestor":
        """Set the input file paths."""
        self._documents = [documents] if isinstance(documents, str) else list(documents)
        return self

    # ------------------------------------------------------------------
    # Execution — synchronous streaming generator
    # ------------------------------------------------------------------

    def ingest(self, params: Any = None, **kwargs: Any) -> Generator[dict[str, Any], None, None]:
        """Upload files to the service and yield page-result dicts as they arrive.

        Each yielded dict contains the page content (``text``, ``metadata``,
        etc.) plus ``_source_file``, ``_page_number``, and ``_document_id``.

        The generator blocks the caller only until the next page finishes on
        the server; there is no need to wait for all pages to complete.

        Progress is tracked internally — call :meth:`progress` at any point
        during iteration to check how many pages have completed per file.
        """
        if not self._documents:
            return

        with self._progress_lock:
            self._progress.clear()

        result_queue: queue.Queue[Any] = queue.Queue()
        error_holder: list[BaseException] = []

        def _on_file_submitted(filename: str, total_pages: int) -> None:
            with self._progress_lock:
                self._progress[filename] = {"total": total_pages, "completed": 0}

        async def _on_page(content: dict[str, Any]) -> None:
            result_queue.put(content)

        async def _run() -> None:
            from nemo_retriever.service.client import RetrieverServiceClient

            client = RetrieverServiceClient(
                base_url=self._base_url,
                max_concurrency=self._max_concurrency,
            )
            await client.ingest_documents(
                files=[Path(p) for p in self._documents],
                on_page_result=_on_page,
                on_file_submitted=_on_file_submitted,
                show_progress=False,
            )

        def _thread_target() -> None:
            try:
                asyncio.run(_run())
            except BaseException as exc:
                error_holder.append(exc)
            finally:
                result_queue.put(_SENTINEL)

        thread = threading.Thread(target=_thread_target, name="service-ingestor", daemon=True)
        thread.start()

        try:
            while True:
                item = result_queue.get()
                if item is _SENTINEL:
                    break
                src = item.get("_source_file", "")
                if src:
                    with self._progress_lock:
                        entry = self._progress.get(src)
                        if entry is not None:
                            entry["completed"] += 1
                yield item
        finally:
            thread.join(timeout=30.0)

        if error_holder:
            raise error_holder[0]

    # ------------------------------------------------------------------
    # Execution — async streaming generator
    # ------------------------------------------------------------------

    async def ingest_async(self, **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        """Async version of :meth:`ingest` for callers already in an event loop."""
        if not self._documents:
            return

        from nemo_retriever.service.client import RetrieverServiceClient

        result_queue: asyncio.Queue[Any] = asyncio.Queue()

        async def _on_page(content: dict[str, Any]) -> None:
            await result_queue.put(content)

        async def _driver() -> None:
            client = RetrieverServiceClient(
                base_url=self._base_url,
                max_concurrency=self._max_concurrency,
            )
            await client.ingest_documents(
                files=[Path(p) for p in self._documents],
                on_page_result=_on_page,
                show_progress=False,
            )
            await result_queue.put(_SENTINEL)

        driver_task = asyncio.create_task(_driver())

        try:
            while True:
                item = await result_queue.get()
                if item is _SENTINEL:
                    break
                yield item
        finally:
            if not driver_task.done():
                driver_task.cancel()
                try:
                    await driver_task
                except asyncio.CancelledError:
                    pass
            elif driver_task.exception():
                raise driver_task.exception()

    # ------------------------------------------------------------------
    # Progress tracking
    # ------------------------------------------------------------------

    def progress(self, filename: str | None = None) -> dict[str, Any]:
        """Return page-level progress for submitted files.

        Parameters
        ----------
        filename
            If given, return progress for that file only::

                {"total": 24, "completed": 18}

            If the file is not (yet) known, returns
            ``{"total": 0, "completed": 0}``.

        Returns
        -------
        dict
            When *filename* is ``None``, returns a dict keyed by filename::

                {
                    "report.pdf": {"total": 24, "completed": 24},
                    "invoice.pdf": {"total": 3, "completed": 1},
                }
        """
        with self._progress_lock:
            if filename is not None:
                entry = self._progress.get(filename)
                if entry is None:
                    return {"total": 0, "completed": 0}
                return dict(entry)
            return {k: dict(v) for k, v in self._progress.items()}

    # ------------------------------------------------------------------
    # Pipeline config methods — accepted but ignored in service mode
    # ------------------------------------------------------------------

    def _warn_pipeline_ignored(self, method: str) -> None:
        if not self._pipeline_warning_emitted:
            logger.info(
                "ServiceIngestor: pipeline method .%s() ignored — "
                "the server-side pipeline configuration is used instead.",
                method,
            )
            self._pipeline_warning_emitted = True

    def extract(self, params: Any = None, **kwargs: Any) -> "ServiceIngestor":
        self._warn_pipeline_ignored("extract")
        return self

    def extract_image_files(self, params: Any = None, **kwargs: Any) -> "ServiceIngestor":
        self._warn_pipeline_ignored("extract_image_files")
        return self

    def embed(self, params: Any = None, **kwargs: Any) -> "ServiceIngestor":
        self._warn_pipeline_ignored("embed")
        return self

    def split(self, params: Any = None, **kwargs: Any) -> "ServiceIngestor":
        self._warn_pipeline_ignored("split")
        return self

    def dedup(self, params: Any = None, **kwargs: Any) -> "ServiceIngestor":
        self._warn_pipeline_ignored("dedup")
        return self

    def caption(self, params: Any = None, **kwargs: Any) -> "ServiceIngestor":
        self._warn_pipeline_ignored("caption")
        return self

    def store(self, params: Any = None, **kwargs: Any) -> "ServiceIngestor":
        self._warn_pipeline_ignored("store")
        return self

    def store_embed(self) -> "ServiceIngestor":
        self._warn_pipeline_ignored("store_embed")
        return self

    def webhook(self, params: Any = None, **kwargs: Any) -> "ServiceIngestor":
        self._warn_pipeline_ignored("webhook")
        return self

    def filter(self) -> "ServiceIngestor":
        self._warn_pipeline_ignored("filter")
        return self

    def all_tasks(self) -> "ServiceIngestor":
        self._warn_pipeline_ignored("all_tasks")
        return self
