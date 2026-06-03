# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from nemo_retriever.search.models.requests import SearchRequest
from nemo_retriever.search.models.responses import CorpusStatus
from nemo_retriever.search.services.service_client import ServiceClient, ServiceUnavailableError

router = APIRouter(tags=["search"])


@router.get("/status", response_model=CorpusStatus)
async def corpus_status(request: Request) -> CorpusStatus:
    config = request.app.state.config
    client = ServiceClient(config)

    try:
        health = await client.get_service_health()
    except ServiceUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    vdb_health = await client.get_vectordb_health()
    vectordb_enabled = vdb_health is not None
    total_rows = 0
    table_exists = False
    table_name = None
    if vdb_health:
        total_rows = int(vdb_health.get("total_rows") or 0)
        table_exists = bool(vdb_health.get("table_exists"))
        table_name = vdb_health.get("table")

    message = None
    if not table_exists or total_rows == 0:
        message = "No ingested documents yet. Upload files with the + button or run retriever service ingest."

    return CorpusStatus(
        service_reachable=True,
        service_mode=health.get("mode"),
        vectordb_enabled=vectordb_enabled,
        total_rows=total_rows,
        table_exists=table_exists,
        table_name=table_name,
        service_url=config.service_url,
        message=message,
    )


@router.post("/search")
async def search(req: SearchRequest, request: Request) -> dict:
    config = request.app.state.config
    cache = request.app.state.hit_cache
    client = ServiceClient(config)

    try:
        raw = await client.query(req.query.strip(), req.top_k)
    except ServiceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        detail = str(exc)
        status = 422 if "No data has been ingested" in detail else 502
        raise HTTPException(status_code=status, detail=detail) from exc

    from nemo_retriever.search.services.hits import build_search_response

    response, hits = build_search_response(req.query.strip(), req.top_k, raw)
    cache.store(response.search_id, hits)
    return response.model_dump()
