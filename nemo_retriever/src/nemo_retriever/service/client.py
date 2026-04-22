# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async CLI client for submitting documents to the retriever service."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_RETRY_AFTER = 5.0
_MAX_BACKOFF = 60.0


class RetrieverServiceClient:
    """Submits documents to a running retriever service and tracks results.

    Parameters
    ----------
    base_url
        The server base URL, e.g. ``http://localhost:8000``.
    max_concurrency
        Maximum number of simultaneous upload requests.
    """

    def __init__(self, base_url: str = "http://localhost:8000", max_concurrency: int = 8) -> None:
        self._base_url = base_url.rstrip("/")
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._capacity_event = asyncio.Event()
        self._capacity_event.set()

    def notify_capacity_available(self) -> None:
        """Signal that the server freed a slot (called from SSE listeners)."""
        self._capacity_event.set()

    async def _upload_file(
        self,
        client: httpx.AsyncClient,
        file_path: Path,
    ) -> dict[str, Any]:
        """Upload a single file, retrying on 503 with exponential backoff.

        When the server returns 503 (busy), the client waits for either:
        - An SSE ``document_complete`` event (signalled via :pyattr:`_capacity_event`)
        - The ``Retry-After`` header duration (fallback)

        before retrying.
        """
        async with self._semaphore:
            file_bytes = file_path.read_bytes()
            backoff = _DEFAULT_RETRY_AFTER

            while True:
                files = {"file": (file_path.name, file_bytes)}
                data = {"metadata": json.dumps({"filename": file_path.name})}
                resp = await client.post(f"{self._base_url}/v1/ingest", files=files, data=data)

                if resp.status_code != 503:
                    resp.raise_for_status()
                    return resp.json()

                retry_after = float(resp.headers.get("Retry-After", backoff))
                body = resp.json()
                pool_size = body.get("pool_size", "?")
                sys.stdout.write(
                    f"  [{file_path.name}] server busy ({pool_size} slots full)" f" — waiting {retry_after:.0f}s …\n"
                )

                self._capacity_event.clear()
                try:
                    await asyncio.wait_for(
                        self._capacity_event.wait(),
                        timeout=retry_after,
                    )
                    sys.stdout.write(f"  [{file_path.name}] capacity freed, retrying …\n")
                except asyncio.TimeoutError:
                    pass

                backoff = min(backoff * 1.5, _MAX_BACKOFF)

    async def _poll_until_complete(
        self,
        client: httpx.AsyncClient,
        document_id: str,
        poll_interval: float,
    ) -> dict[str, Any]:
        """Poll ``/v1/ingest/status/{id}`` until the document is complete or failed."""
        while True:
            resp = await client.get(f"{self._base_url}/v1/ingest/status/{document_id}")
            resp.raise_for_status()
            body = resp.json()
            status = body.get("status", "")
            _print_progress(document_id, body)
            if status in ("complete", "failed"):
                self.notify_capacity_available()
                return body
            await asyncio.sleep(poll_interval)

    async def _stream_sse(
        self,
        client: httpx.AsyncClient,
        document_id: str,
    ) -> dict[str, Any]:
        """Connect to the SSE stream and print events until ``document_complete``."""
        url = f"{self._base_url}/v1/ingest/stream/{document_id}"
        last_body: dict[str, Any] = {}
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            buffer = ""
            async for chunk in resp.aiter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    raw_event, buffer = buffer.split("\n\n", 1)
                    event = _parse_sse_event(raw_event)
                    if event is None:
                        continue
                    last_body = event
                    event_type = event.get("event", "message")
                    _print_sse_event(document_id, event)
                    if event_type == "document_complete":
                        self.notify_capacity_available()
                        final = await client.get(f"{self._base_url}/v1/ingest/status/{document_id}")
                        final.raise_for_status()
                        return final.json()

        self.notify_capacity_available()
        if last_body:
            final = await client.get(f"{self._base_url}/v1/ingest/status/{document_id}")
            final.raise_for_status()
            return final.json()
        return last_body

    async def ingest_documents(
        self,
        files: list[Path],
        use_sse: bool = True,
        poll_interval: float = 2.0,
    ) -> list[dict[str, Any]]:
        """Upload *files* and wait for all to complete.

        Uploads are submitted concurrently.  When the server responds with
        **503** the upload coroutine backs off and retries — either when an
        SSE ``document_complete`` event signals freed capacity, or after the
        ``Retry-After`` period.

        Returns the list of final result payloads.
        """
        timeout = httpx.Timeout(300.0, connect=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            upload_tasks = [self._upload_file(client, f) for f in files]
            accepted_list = await asyncio.gather(*upload_tasks, return_exceptions=True)

            doc_ids: list[str] = []
            for idx, result in enumerate(accepted_list):
                if isinstance(result, Exception):
                    _print_error(files[idx], result)
                    continue
                doc_id = result["document_id"]
                doc_ids.append(doc_id)
                sys.stdout.write(
                    f"  [{files[idx].name}] accepted — document_id={doc_id} " f"sha={result['content_sha256'][:12]}…\n"
                )

            if not doc_ids:
                sys.stdout.write("No documents were accepted.\n")
                return []

            sys.stdout.write(f"\nWaiting for {len(doc_ids)} document(s) …\n\n")

            if use_sse:
                wait_tasks = [self._stream_sse(client, did) for did in doc_ids]
            else:
                wait_tasks = [self._poll_until_complete(client, did, poll_interval) for did in doc_ids]

            results = await asyncio.gather(*wait_tasks, return_exceptions=True)
            final_results: list[dict[str, Any]] = []
            for idx, res in enumerate(results):
                if isinstance(res, Exception):
                    sys.stderr.write(f"  [doc {doc_ids[idx][:8]}…] error: {res}\n")
                else:
                    final_results.append(res)
                    _print_final(res)

            return final_results


def _parse_sse_event(raw: str) -> dict[str, Any] | None:
    """Parse a single SSE frame into a dict."""
    data_lines: list[str] = []
    for line in raw.strip().splitlines():
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
        elif line.startswith(":"):
            continue
    if not data_lines:
        return None
    try:
        return json.loads("".join(data_lines))
    except json.JSONDecodeError:
        return None


def _print_progress(doc_id: str, body: dict[str, Any]) -> None:
    pages = body.get("pages_received", 0)
    total = body.get("total_pages") or "?"
    status = body.get("status", "unknown")
    sys.stdout.write(f"  [{doc_id[:8]}…] {status} — {pages}/{total} pages\n")


def _print_sse_event(doc_id: str, event: dict[str, Any]) -> None:
    etype = event.get("event", "message")
    if etype == "page_complete":
        pg = event.get("page_number", "?")
        recv = event.get("pages_received", "?")
        total = event.get("total_pages", "?")
        sys.stdout.write(f"  [{doc_id[:8]}…] page {pg} complete ({recv}/{total})\n")
    elif etype == "metrics_update":
        model = event.get("model_name", "?")
        dets = event.get("detections_count", 0)
        sys.stdout.write(f"  [{doc_id[:8]}…] metrics: {model} — {dets} detections\n")
    elif etype == "document_complete":
        sys.stdout.write(f"  [{doc_id[:8]}…] COMPLETE\n")


def _print_final(body: dict[str, Any]) -> None:
    doc_id = body.get("document_id", "?")
    fname = body.get("filename", "?")
    status = body.get("status", "?")
    total = body.get("total_pages", 0)
    n_metrics = len(body.get("metrics", []))
    sys.stdout.write(
        f"\n  Result: {fname} (id={doc_id[:8]}…) — {status}, " f"{total} pages, {n_metrics} metric entries\n"
    )


def _print_error(file_path: Path, exc: Exception) -> None:
    sys.stderr.write(f"  [{file_path.name}] upload failed: {exc}\n")
