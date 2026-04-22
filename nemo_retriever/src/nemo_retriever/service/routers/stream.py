# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GET /v1/ingest/stream/{document_id} — Server-Sent Events endpoint."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from nemo_retriever.service.db.repository import Repository
from nemo_retriever.service.event_bus import EventBus

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stream"])


async def _event_generator(
    event_bus: EventBus,
    document_id: str,
) -> AsyncIterator[str]:
    """Yield SSE-formatted strings until the ``document_complete`` event."""
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


@router.get(
    "/ingest/stream/{document_id}",
    summary="Stream processing events for a document via SSE",
)
async def stream_document_events(
    request: Request,
    document_id: str,
) -> StreamingResponse:
    """Opens a Server-Sent Events stream that delivers real-time processing
    updates for the given document.

    Event types:
    - ``page_complete`` — fired each time a page finishes processing
    - ``metrics_update`` — per-model invocation statistics
    - ``status_change`` — processing status transitions
    - ``document_complete`` — terminal event; stream closes after this
    """
    repo: Repository = request.app.state.repository
    doc = repo.get_document(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    event_bus: EventBus = request.app.state.event_bus
    return StreamingResponse(
        _event_generator(event_bus, document_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
