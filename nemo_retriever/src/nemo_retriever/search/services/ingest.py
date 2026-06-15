# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

import httpx

from nemo_retriever.search.config import SearchConfig
from nemo_retriever.search.models.responses import IngestJobStatus, IngestResponse
from nemo_retriever.search.services.documents import DocumentStore
from nemo_retriever.service.client import RetrieverServiceClient

_TERMINAL_JOB_STATUSES = frozenset({"completed", "failed", "partial_success"})


async def ingest_file_paths(
    paths: list[Path],
    config: SearchConfig,
    *,
    label: str | None = None,
    document_store: DocumentStore | None = None,
    wait: bool = True,
) -> IngestResponse:
    if not paths:
        return IngestResponse(
            job_id=None,
            documents_submitted=0,
            documents_succeeded=0,
            documents_failed=0,
            elapsed_s=0.0,
        )

    if document_store is not None:
        for path in paths:
            document_store.register_file(path.name, path.read_bytes(), display_name=path.name)

    client = RetrieverServiceClient(
        base_url=config.service_url,
        max_concurrency=config.ingest_max_concurrency,
        api_token=config.api_token,
    )
    t0 = time.monotonic()
    if wait:
        results = await client.ingest_documents(paths, show_progress=False)
        elapsed = time.monotonic() - t0
        return _ingest_response_from_results(paths, results, elapsed)

    result = await _submit_documents(client, paths, label=label)
    result.elapsed_s = round(time.monotonic() - t0, 3)
    return result


async def _submit_documents(
    client: RetrieverServiceClient,
    paths: list[Path],
    *,
    label: str | None = None,
) -> IngestResponse:
    pool_limits = httpx.Limits(max_connections=200, max_keepalive_connections=100)
    timeout = httpx.Timeout(timeout=None, connect=30.0)
    errors: list[str] = []

    async with httpx.AsyncClient(
        timeout=timeout,
        limits=pool_limits,
        headers=client._auth_headers,
    ) as http:
        job_id = await client._create_job(http, expected_documents=len(paths), label=label)
        upload_sem = asyncio.Semaphore(client._max_concurrency)

        async def _upload(path: Path) -> None:
            async with upload_sem:
                try:
                    await client._upload_one(http, path, job_id=job_id)
                except Exception as exc:
                    errors.append(f"{path.name}: {exc}")

        await asyncio.gather(*(_upload(path) for path in paths))

    failed = len(errors)
    submitted = len(paths)
    if failed >= submitted:
        status = "failed"
    else:
        status = "processing"

    return IngestResponse(
        job_id=job_id,
        documents_submitted=submitted,
        documents_succeeded=0,
        documents_failed=failed,
        elapsed_s=0.0,
        status=status,
        errors=errors[:20],
    )


def _ingest_response_from_results(
    paths: list[Path],
    results: list[dict],
    elapsed: float,
) -> IngestResponse:
    succeeded = sum(1 for r in results if r.get("status") == "completed")
    failed = len(paths) - succeeded
    errors: list[str] = []
    for r in results:
        if r.get("status") != "completed" and r.get("error"):
            errors.append(str(r.get("error")))

    job_id = results[0].get("job_id") if results else None
    status = "completed"
    if failed >= len(paths):
        status = "failed"
    elif failed:
        status = "partial_success"

    return IngestResponse(
        job_id=job_id,
        documents_submitted=len(paths),
        documents_succeeded=succeeded,
        documents_failed=failed,
        elapsed_s=round(elapsed, 3),
        status=status,
        errors=errors[:20],
    )


async def ingest_uploaded_bytes(
    files: list[tuple[str, bytes]],
    config: SearchConfig,
    *,
    label: str | None = None,
    document_store: DocumentStore | None = None,
    wait: bool = True,
) -> IngestResponse:
    """Write uploads to a temp dir and ingest via the retriever service client."""
    if not files:
        return IngestResponse(
            job_id=None,
            documents_submitted=0,
            documents_succeeded=0,
            documents_failed=0,
            elapsed_s=0.0,
        )

    if document_store is not None:
        for name, data in files:
            document_store.register_file(name, data)

    with tempfile.TemporaryDirectory(prefix="nemo-search-ingest-") as tmp:
        paths: list[Path] = []
        for name, data in files:
            dest = Path(tmp) / Path(name).name
            if dest.exists():
                stem = dest.stem
                suffix = dest.suffix
                dest = Path(tmp) / f"{stem}_{len(paths)}{suffix}"
            dest.write_bytes(data)
            paths.append(dest)
        return await ingest_file_paths(
            paths,
            config,
            label=label,
            document_store=None,
            wait=wait,
        )


def ingest_uploaded_bytes_sync(
    files: list[tuple[str, bytes]],
    config: SearchConfig,
    *,
    label: str | None = None,
    document_store: DocumentStore | None = None,
    wait: bool = True,
) -> IngestResponse:
    return asyncio.run(
        ingest_uploaded_bytes(
            files,
            config,
            label=label,
            document_store=document_store,
            wait=wait,
        )
    )


async def get_ingest_job_status(config: SearchConfig, job_id: str) -> IngestJobStatus:
    url = f"{config.service_url}/v1/ingest/job/{job_id}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers=config.auth_headers)
            resp.raise_for_status()
            body = resp.json()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Failed to fetch ingest job {job_id!r}: {exc}") from exc

    counts = body.get("counts") or {}
    errors: list[str] = []
    documents = body.get("documents")
    if isinstance(documents, list):
        for doc in documents:
            if isinstance(doc, dict) and doc.get("status") == "failed" and doc.get("error"):
                name = doc.get("filename") or doc.get("document_id") or "document"
                errors.append(f"{name}: {doc['error']}")

    status = str(body.get("status") or "processing")
    succeeded = int(counts.get("completed") or 0)
    failed = int(counts.get("failed") or 0)

    return IngestJobStatus(
        job_id=str(body.get("job_id") or job_id),
        status=status,
        expected_documents=int(body.get("expected_documents") or 0),
        counts={str(k): int(v) for k, v in counts.items()},
        elapsed_s=body.get("elapsed_s"),
        errors=errors[:20],
    )


def is_terminal_job_status(status: str) -> bool:
    return status in _TERMINAL_JOB_STATUSES
