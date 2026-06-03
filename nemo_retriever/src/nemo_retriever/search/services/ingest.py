# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

from nemo_retriever.search.config import SearchConfig
from nemo_retriever.search.models.responses import IngestResponse
from nemo_retriever.service.client import RetrieverServiceClient


async def ingest_file_paths(
    paths: list[Path],
    config: SearchConfig,
    *,
    label: str | None = None,
) -> IngestResponse:
    if not paths:
        return IngestResponse(
            job_id=None,
            documents_submitted=0,
            documents_succeeded=0,
            documents_failed=0,
            elapsed_s=0.0,
        )

    client = RetrieverServiceClient(
        base_url=config.service_url,
        max_concurrency=config.ingest_max_concurrency,
        api_token=config.api_token,
    )
    t0 = time.monotonic()
    results = await client.ingest_documents(paths, show_progress=False)
    elapsed = time.monotonic() - t0

    succeeded = sum(1 for r in results if r.get("status") == "completed")
    failed = len(paths) - succeeded
    errors: list[str] = []
    for r in results:
        if r.get("status") != "completed" and r.get("error"):
            errors.append(str(r.get("error")))

    job_id = None
    if results:
        job_id = results[0].get("job_id")

    return IngestResponse(
        job_id=job_id,
        documents_submitted=len(paths),
        documents_succeeded=succeeded,
        documents_failed=failed,
        elapsed_s=round(elapsed, 3),
        errors=errors[:20],
    )


async def ingest_uploaded_bytes(
    files: list[tuple[str, bytes]],
    config: SearchConfig,
    *,
    label: str | None = None,
) -> IngestResponse:
    """Write uploads to a temp dir and ingest via the service client."""
    if not files:
        return IngestResponse(
            job_id=None,
            documents_submitted=0,
            documents_succeeded=0,
            documents_failed=0,
            elapsed_s=0.0,
        )

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
        return await ingest_file_paths(paths, config, label=label)


def ingest_uploaded_bytes_sync(
    files: list[tuple[str, bytes]],
    config: SearchConfig,
    *,
    label: str | None = None,
) -> IngestResponse:
    return asyncio.run(ingest_uploaded_bytes(files, config, label=label))
