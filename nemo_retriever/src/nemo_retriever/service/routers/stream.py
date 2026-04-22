# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SSE streaming endpoints for real-time document processing events.

Three endpoints are provided:

- ``GET  /v1/ingest/stream/{document_id}`` — single-document stream (legacy)
- ``POST /v1/ingest/stream`` — session-level stream for multiple documents
  on a single SSE connection (preferred by the CLI client)
- ``POST /v1/ingest/stream/jobs`` — job-level stream for tracking pages
  across one or more jobs
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from nemo_retriever.service.db.repository import Repository
from nemo_retriever.service.event_bus import EventBus

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stream"])


# ------------------------------------------------------------------
# Shared SSE formatting
# ------------------------------------------------------------------


async def _single_doc_generator(
    event_bus: EventBus,
    document_id: str,
) -> AsyncIterator[str]:
    """Yield SSE frames for a single document until ``document_complete``."""
    queue = event_bus.subscribe(document_id)
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue

            event_type = event.get("event", "message")
            data = json.dumps(event)
            yield f"event: {event_type}\ndata: {data}\n\n"

            if event_type == "document_complete":
                break
    finally:
        event_bus.unsubscribe(document_id, queue)


async def _multi_doc_generator(
    event_bus: EventBus,
    document_ids: list[str],
) -> AsyncIterator[str]:
    """Yield SSE frames for *all* listed documents on a single connection.

    The stream stays open until every document has emitted a
    ``document_complete`` (or ``status_change`` with ``failed``) event.
    """
    queue = event_bus.subscribe_many(document_ids)
    pending = set(document_ids)
    try:
        while pending:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue

            event_type = event.get("event", "message")
            data = json.dumps(event)
            yield f"event: {event_type}\ndata: {data}\n\n"

            doc_id = event.get("document_id")
            if event_type == "document_complete" and doc_id in pending:
                pending.discard(doc_id)
            elif event_type == "status_change" and event.get("status") == "failed":
                pending.discard(doc_id)

        yield f"event: session_complete\ndata: {json.dumps({'completed': len(document_ids)})}\n\n"
    finally:
        event_bus.unsubscribe_many(document_ids, queue)


async def _job_stream_generator(
    event_bus: EventBus,
    job_ids: list[str],
) -> AsyncIterator[str]:
    """Yield SSE frames for one or more jobs until every job completes.

    Subscribes by *job_id* so it picks up all page-level and job-level events
    published under those keys. ``job_complete`` (or a ``status_change`` with
    ``failed``) is the terminal condition per job.
    """
    queue = event_bus.subscribe_many(job_ids)
    pending = set(job_ids)
    try:
        while pending:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue

            event_type = event.get("event", "message")
            data = json.dumps(event)
            yield f"event: {event_type}\ndata: {data}\n\n"

            jid = event.get("job_id")
            if event_type == "job_complete" and jid in pending:
                pending.discard(jid)
            elif event_type == "status_change" and event.get("status") == "failed":
                if jid in pending:
                    pending.discard(jid)

        yield f"event: session_complete\ndata: {json.dumps({'completed': len(job_ids)})}\n\n"
    finally:
        event_bus.unsubscribe_many(job_ids, queue)


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@router.get(
    "/ingest/stream/{document_id}",
    summary="Stream processing events for a single document via SSE",
)
async def stream_document_events(
    request: Request,
    document_id: str,
) -> StreamingResponse:
    repo: Repository = request.app.state.repository
    doc = repo.get_document(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    event_bus: EventBus = request.app.state.event_bus
    return StreamingResponse(
        _single_doc_generator(event_bus, document_id),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


class StreamSessionRequest(BaseModel):
    """Body for the multi-document SSE endpoint."""

    document_ids: list[str] = Field(..., min_length=1)


@router.post(
    "/ingest/stream",
    summary="Stream processing events for multiple documents on a single SSE connection",
)
async def stream_session_events(
    request: Request,
    body: StreamSessionRequest,
) -> StreamingResponse:
    repo: Repository = request.app.state.repository
    for doc_id in body.document_ids:
        doc = repo.get_document(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    event_bus: EventBus = request.app.state.event_bus
    return StreamingResponse(
        _multi_doc_generator(event_bus, body.document_ids),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


class JobStreamRequest(BaseModel):
    """Body for the job-level SSE endpoint."""

    job_ids: list[str] = Field(..., min_length=1)


@router.post(
    "/ingest/stream/jobs",
    summary="Stream processing events for one or more jobs on a single SSE connection",
)
async def stream_job_events(
    request: Request,
    body: JobStreamRequest,
) -> StreamingResponse:
    # Subscribe immediately without validating job existence.
    # Jobs are created on-the-fly as the first page for each file arrives,
    # so the SSE stream must be open BEFORE uploads begin to avoid missing
    # early events.  The EventBus is purely in-memory and doesn't need
    # the job row to exist.
    event_bus: EventBus = request.app.state.event_bus
    return StreamingResponse(
        _job_stream_generator(event_bus, body.job_ids),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
