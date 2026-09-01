# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public agent-memory routes.

The gateway authenticates the caller and forwards the authorized scope. The
VectorDB service owns the memory table and folds that scope into the stored
namespace, so a caller can never recall another tenant's memories by asking
for their namespace.
"""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, HTTPException, Request, Response

router = APIRouter(tags=["memory"])
logger = logging.getLogger(__name__)

#: Longer than the collection timeout: a remember call embeds a whole batch,
#: and consolidation additionally waits on a language model.
_MEMORY_TIMEOUT_S = 300.0


async def _forward(request: Request, suffix: str) -> Response:
    """Forward an authorized memory request to the internal VectorDB service."""

    from nemo_retriever.service.auth import authorized_scope, internal_auth_headers

    config = request.app.state.config
    if not config.vectordb.enabled:
        raise HTTPException(404, "VectorDB is not enabled in the service configuration.")
    if config.mode in ("realtime", "batch"):
        raise HTTPException(404, "Agent memory is available through the gateway.")

    target = f"{config.vectordb.vectordb_url.rstrip('/')}/v1/{suffix}"
    headers = {"Content-Type": request.headers.get("content-type", "application/json")}
    headers["X-NRL-Scope"] = authorized_scope(request)
    headers.update(internal_auth_headers(config.vectordb.internal_api_token))
    try:
        async with httpx.AsyncClient(timeout=_MEMORY_TIMEOUT_S) as client:
            response = await client.request(
                request.method,
                target,
                content=await request.body(),
                params=request.query_params,
                headers=headers,
            )
    except httpx.HTTPError as exc:
        logger.exception("Failed to proxy memory request to VectorDB at %s", target)
        raise HTTPException(502, "VectorDB service is unavailable.") from exc
    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "application/json"),
    )


@router.post("/memory/remember")
async def remember(request: Request) -> Response:
    """Store a batch of agent memories."""
    return await _forward(request, "memory/remember")


@router.post("/memory/recall")
async def recall(request: Request) -> Response:
    """Rank stored memories against a natural-language query."""
    return await _forward(request, "memory/recall")


@router.post("/memory/timeline")
async def timeline(request: Request) -> Response:
    """Replay one session's episodic memories in chronological order."""
    return await _forward(request, "memory/timeline")


@router.post("/memory/forget")
async def forget(request: Request) -> Response:
    """Retire memories by identifier or filter."""
    return await _forward(request, "memory/forget")


@router.post("/memory/consolidate")
async def consolidate(request: Request) -> Response:
    """Distill episodic memories into durable semantic facts."""
    return await _forward(request, "memory/consolidate")


@router.get("/memory/stats")
async def stats(request: Request) -> Response:
    """Return coarse counters for one memory namespace."""
    return await _forward(request, "memory/stats")


@router.post("/memory/optimize")
async def optimize(request: Request) -> Response:
    """Compact the memory table after a burst of small writes."""
    return await _forward(request, "memory/optimize")
