# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""POST /v1/ingest/internal_results — webhook callback from the pipeline."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request

from nemo_retriever.service.db.repository import Repository
from nemo_retriever.service.event_bus import EventBus
from nemo_retriever.service.models.document import ProcessingStatus
from nemo_retriever.service.models.metrics import ProcessingMetric
from nemo_retriever.service.models.page_result import PageResult
from nemo_retriever.service.models.requests import InternalResultPayload

logger = logging.getLogger(__name__)

router = APIRouter(tags=["internal"])


@router.post(
    "/ingest/internal_results",
    status_code=200,
    summary="Receive page-level results from the processing pipeline",
)
async def receive_internal_results(
    request: Request,
    payload: InternalResultPayload,
) -> dict[str, str]:
    """Called by the ``WebhookNotifyOperator`` once per page.

    Stores the page result and metrics, increments the received-page counter,
    and — when all pages are in — marks the document complete.
    """
    repo: Repository = request.app.state.repository
    event_bus: EventBus = request.app.state.event_bus

    doc = repo.get_document_by_sha(payload.content_sha256)
    if doc is None:
        raise HTTPException(
            status_code=404,
            detail=f"No document with sha256={payload.content_sha256[:12]}…",
        )

    page = PageResult(
        document_id=doc.id,
        page_number=payload.page_number,
        content_json=json.dumps(payload.content),
    )
    repo.insert_page_result(page)

    metrics = [
        ProcessingMetric(
            document_id=doc.id,
            model_name=m.model_name,
            invocation_count=m.invocation_count,
            pages_processed=m.pages_processed,
            detections_count=m.detections_count,
            duration_ms=m.duration_ms,
        )
        for m in payload.metrics
    ]
    repo.insert_metrics(metrics)

    if payload.total_pages and doc.total_pages is None:
        repo.update_document_status(doc.id, doc.processing_status, total_pages=payload.total_pages)

    new_count = repo.increment_pages_received(doc.id)

    await event_bus.publish(
        doc.id,
        {
            "event": "page_complete",
            "document_id": doc.id,
            "page_number": payload.page_number,
            "pages_received": new_count,
            "total_pages": payload.total_pages,
            "content": payload.content,
        },
    )

    for m in payload.metrics:
        await event_bus.publish(
            doc.id,
            {
                "event": "metrics_update",
                "document_id": doc.id,
                "model_name": m.model_name,
                "invocation_count": m.invocation_count,
                "detections_count": m.detections_count,
            },
        )

    if payload.total_pages and new_count >= payload.total_pages:
        repo.update_document_status(doc.id, ProcessingStatus.COMPLETE)
        logger.info("Document %s complete (%d/%d pages)", doc.id, new_count, payload.total_pages)

        await event_bus.publish(
            doc.id,
            {
                "event": "document_complete",
                "document_id": doc.id,
                "total_pages": payload.total_pages,
            },
        )

    return {"status": "ok"}
