# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from nemo_retriever.search.services.ingest import ingest_uploaded_bytes
from nemo_retriever.search.services.service_client import ServiceUnavailableError, ServiceClient

router = APIRouter(tags=["ingest"])


@router.post("/ingest")
async def ingest_documents(
    request: Request,
    files: list[UploadFile] = File(..., description="One or more documents to ingest."),
    label: str | None = Form(default=None),
) -> dict:
    config = request.app.state.config

    try:
        await ServiceClient(config).get_service_health()
    except ServiceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    uploads: list[tuple[str, bytes]] = []
    for upload in files:
        if not upload.filename:
            continue
        data = await upload.read()
        if data:
            uploads.append((upload.filename, data))

    if not uploads:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    try:
        result = await ingest_uploaded_bytes(uploads, config, label=label)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ingest failed: {type(exc).__name__}: {exc}") from exc

    return result.model_dump()
