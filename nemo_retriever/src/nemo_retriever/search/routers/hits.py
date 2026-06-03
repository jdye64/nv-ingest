# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response

from nemo_retriever.search.models.responses import HitExportFormat
from nemo_retriever.search.services.export import export_hit

router = APIRouter(tags=["hits"])


@router.get("/hits/{hit_id}/export")
async def export_hit_content(
    hit_id: str,
    request: Request,
    format: HitExportFormat = Query(default="text", alias="format"),
    download: bool = Query(default=False),
) -> Response:
    cache = request.app.state.hit_cache
    hit = cache.get(hit_id)
    if hit is None:
        raise HTTPException(
            status_code=404,
            detail="Hit not found or export expired. Run a new search to refresh hit exports.",
        )

    body, media_type = export_hit(hit, format)
    headers: dict[str, str] = {}
    if download:
        ext = "txt" if format in ("text", "summary") else "json"
        headers["Content-Disposition"] = f'attachment; filename="{hit_id}.{ext}"'

    return Response(content=body, media_type=media_type, headers=headers)
