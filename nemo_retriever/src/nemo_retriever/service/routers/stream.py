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

All three honour the standard SSE ``Last-Event-ID`` header.  When set, the
generator first replays buffered events with ``seq > last_event_id``
before subscribing to new ones, so a client that briefly disconnects can
resume without losing events.  Each frame includes an ``id: <seq>`` line
suitable for the browser ``EventSource`` API to track and re-send on
reconnect.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from nemo_retriever.service.db.repository import Repository
from nemo_retriever.service.event_bus import EventBus

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stream"])


# ------------------------------------------------------------------
# Per-endpoint event-type interest sets
# ------------------------------------------------------------------
#
# Each SSE endpoint declares the set of event types its subscribers
# actually consume.  The EventBus uses these as a hard filter so chatty
# per-page events that the endpoint doesn't forward (``metrics_update``,
# ``page_complete``, ``document_complete``) never even reach the
# subscriber's queue.  This is the principal mechanism that keeps
# ``stream_overflow`` from being triggered under normal load.

# Doc-level (single-document) endpoint — caller wants the whole story
# for one document, so accept everything.
_DOC_STREAM_EVENT_TYPES: frozenset[str] | None = None

# Multi-doc / session endpoint — caller needs status transitions and
# per-page completion + (optionally) per-page content; aggregate metrics
# don't help a session-level consumer.
_SESSION_STREAM_BASE_EVENTS: frozenset[str] = frozenset(
    {
        "page_complete",
        "document_complete",
        "status_change",
        "stream_overflow",
    }
)

# Job-level endpoint — caller is interested in job life-cycle events
# only (plus optionally the full per-page content payloads when content
# was requested).  Per-model metrics and per-page progress events are
# server-side-only details for this endpoint.
_JOB_STREAM_BASE_EVENTS: frozenset[str] = frozenset(
    {
        "job_started",
        "job_complete",
        "status_change",
        "stream_overflow",
    }
)


# ------------------------------------------------------------------
# Shared SSE formatting
# ------------------------------------------------------------------


def _format_event(event: dict, include_content: bool) -> str | None:
    """Return the wire-format SSE frame for *event*, or ``None`` to skip."""
    event_type = event.get("event", "message")
    if event_type == "page_result" and not include_content:
        return None
    seq = event.get("seq")
    id_line = f"id: {seq}\n" if isinstance(seq, int) else ""
    data = json.dumps(event)
    return f"{id_line}event: {event_type}\ndata: {data}\n\n"


def _parse_last_event_id(value: str | None) -> int:
    """Parse the ``Last-Event-ID`` header into an int (default 0)."""
    if not value:
        return 0
    try:
        return int(value.strip())
    except (TypeError, ValueError):
        return 0


async def _replay_buffered(
    event_bus: EventBus,
    keys: list[str],
    after_seq: int,
    include_content: bool,
) -> AsyncIterator[str]:
    if after_seq <= 0:
        return
    for evt in event_bus.replay(keys, after_seq=after_seq):
        frame = _format_event(evt, include_content)
        if frame is not None:
            yield frame


async def _single_doc_generator(
    event_bus: EventBus,
    document_id: str,
    *,
    after_seq: int = 0,
    include_content: bool = True,
) -> AsyncIterator[str]:
    """Yield SSE frames for a single document until ``document_complete``."""
    async for frame in _replay_buffered(event_bus, [document_id], after_seq, include_content):
        yield frame

    queue = event_bus.subscribe(document_id)
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue

            event_type = event.get("event", "message")
            frame = _format_event(event, include_content)
            if frame is not None:
                yield frame

            if event_type in ("document_complete", "stream_overflow"):
                break
    finally:
        event_bus.unsubscribe(document_id, queue)


async def _multi_doc_generator(
    event_bus: EventBus,
    document_ids: list[str],
    *,
    after_seq: int = 0,
    include_content: bool = False,
) -> AsyncIterator[str]:
    """Yield SSE frames for *all* listed documents on a single connection."""
    async for frame in _replay_buffered(event_bus, document_ids, after_seq, include_content):
        yield frame

    wanted = set(_SESSION_STREAM_BASE_EVENTS)
    if include_content:
        wanted.add("page_result")
    queue = event_bus.subscribe_many(document_ids, event_types=wanted)
    pending = set(document_ids)
    try:
        while pending:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue

            event_type = event.get("event", "message")

            if event_type == "stream_overflow":
                frame = _format_event(event, include_content)
                if frame is not None:
                    yield frame
                break

            frame = _format_event(event, include_content)
            if frame is not None:
                yield frame

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
    *,
    after_seq: int = 0,
    include_content: bool = False,
) -> AsyncIterator[str]:
    """Yield SSE frames for one or more jobs until every job completes."""
    async for frame in _replay_buffered(event_bus, job_ids, after_seq, include_content):
        yield frame

    wanted = set(_JOB_STREAM_BASE_EVENTS)
    if include_content:
        wanted.add("page_result")
    queue = event_bus.subscribe_many(job_ids, event_types=wanted)
    pending = set(job_ids)
    try:
        while pending:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue

            event_type = event.get("event", "message")

            if event_type == "stream_overflow":
                frame = _format_event(event, include_content)
                if frame is not None:
                    yield frame
                break

            frame = _format_event(event, include_content)
            if frame is not None:
                yield frame

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
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    repo: Repository = request.app.state.repository
    doc = repo.get_document(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    event_bus: EventBus = request.app.state.event_bus
    return StreamingResponse(
        _single_doc_generator(
            event_bus,
            document_id,
            after_seq=_parse_last_event_id(last_event_id),
        ),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


class StreamSessionRequest(BaseModel):
    """Body for the multi-document SSE endpoint."""

    document_ids: list[str] = Field(..., min_length=1)
    include_content: bool = Field(
        default=False,
        description="If true, include `page_result` events carrying the full per-page extracted content.",
    )


@router.post(
    "/ingest/stream",
    summary="Stream processing events for multiple documents on a single SSE connection",
)
async def stream_session_events(
    request: Request,
    body: StreamSessionRequest,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    repo: Repository = request.app.state.repository
    for doc_id in body.document_ids:
        doc = repo.get_document(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    event_bus: EventBus = request.app.state.event_bus
    return StreamingResponse(
        _multi_doc_generator(
            event_bus,
            body.document_ids,
            after_seq=_parse_last_event_id(last_event_id),
            include_content=body.include_content,
        ),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


class JobStreamRequest(BaseModel):
    """Body for the job-level SSE endpoint."""

    job_ids: list[str] = Field(..., min_length=1)
    include_content: bool = Field(
        default=False,
        description="If true, include `page_result` events carrying the full per-page extracted content.",
    )


@router.post(
    "/ingest/stream/jobs",
    summary="Stream processing events for one or more jobs on a single SSE connection",
)
async def stream_job_events(
    request: Request,
    body: JobStreamRequest,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    # Subscribe immediately without validating job existence.
    # Jobs are created on-the-fly as the first page for each file arrives,
    # so the SSE stream must be open BEFORE uploads begin to avoid missing
    # early events.  The EventBus is purely in-memory and doesn't need
    # the job row to exist.
    event_bus: EventBus = request.app.state.event_bus
    return StreamingResponse(
        _job_stream_generator(
            event_bus,
            body.job_ids,
            after_seq=_parse_last_event_id(last_event_id),
            include_content=body.include_content,
        ),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
