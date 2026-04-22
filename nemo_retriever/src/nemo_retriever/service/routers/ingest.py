# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""POST /v1/ingest and GET /v1/ingest/status/{document_id}."""

from __future__ import annotations

import hashlib
import json
import logging

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from nemo_retriever.service.db.repository import Repository
from nemo_retriever.service.models.document import Document, ProcessingStatus
from nemo_retriever.service.models.requests import IngestRequest
from nemo_retriever.service.models.responses import (
    IngestAccepted,
    IngestComplete,
    IngestStatus,
    MetricSummary,
    PageSummary,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ingest"])


@router.post(
    "/ingest",
    response_model=IngestAccepted,
    status_code=202,
    summary="Upload a single document for ingestion",
)
async def ingest_document(
    request: Request,
    file: UploadFile = File(..., description="The document to ingest"),
    metadata: str = Form(default="{}", description="JSON-encoded IngestRequest metadata"),
) -> IngestAccepted:
    """Accept a single document upload and queue it for processing.

    The file is read into memory, its SHA-256 is computed for dedup, and it is
    submitted to the processing thread pool.  The response is returned
    immediately with status ``queued``.
    """
    try:
        meta = IngestRequest(**json.loads(metadata))
    except (json.JSONDecodeError, Exception) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {exc}")

    file_bytes = await file.read()
    content_sha256 = hashlib.sha256(file_bytes).hexdigest()

    repo: Repository = request.app.state.repository
    existing = repo.get_document_by_sha(content_sha256)
    if existing is not None:
        logger.info("Duplicate document detected (sha=%s), returning existing record", content_sha256[:12])
        return IngestAccepted(
            document_id=existing.id,
            content_sha256=existing.content_sha256,
            status=existing.processing_status,
            created_at=existing.created_at,
        )

    filename = meta.filename or file.filename or "unknown"
    content_type = meta.content_type or file.content_type or "application/octet-stream"

    doc = Document(
        filename=filename,
        content_type=content_type,
        content_sha256=content_sha256,
        file_size_bytes=len(file_bytes),
        metadata_json=json.dumps(meta.metadata),
    )
    repo.insert_document(doc)
    logger.info("Document accepted: id=%s filename=%s sha=%s", doc.id, filename, content_sha256[:12])

    pool = request.app.state.processing_pool
    pool.submit(doc.id, file_bytes, filename)

    return IngestAccepted(
        document_id=doc.id,
        content_sha256=doc.content_sha256,
        status=doc.processing_status,
        created_at=doc.created_at,
    )


def _build_status_response(repo: Repository, doc: Document) -> IngestStatus | IngestComplete:
    """Build the appropriate status or complete response for *doc*."""
    pages = repo.get_page_results(doc.id)
    metrics_rows = repo.get_metrics(doc.id)

    page_summaries = [PageSummary(page_number=p.page_number, content=json.loads(p.content_json)) for p in pages]
    metric_summaries = [
        MetricSummary(
            model_name=m.model_name,
            invocation_count=m.invocation_count,
            pages_processed=m.pages_processed,
            detections_count=m.detections_count,
            duration_ms=m.duration_ms,
        )
        for m in metrics_rows
    ]

    if doc.processing_status == ProcessingStatus.COMPLETE and doc.total_pages is not None:
        return IngestComplete(
            document_id=doc.id,
            filename=doc.filename,
            content_sha256=doc.content_sha256,
            status=doc.processing_status,
            total_pages=doc.total_pages,
            pages_received=doc.pages_received,
            metrics=metric_summaries,
            pages=page_summaries,
            created_at=doc.created_at,
            updated_at=doc.updated_at,
        )

    return IngestStatus(
        document_id=doc.id,
        filename=doc.filename,
        content_sha256=doc.content_sha256,
        status=doc.processing_status,
        total_pages=doc.total_pages,
        pages_received=doc.pages_received,
        metrics=metric_summaries,
        pages=page_summaries,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


@router.get(
    "/ingest/status/{document_id}",
    response_model=IngestStatus | IngestComplete,
    summary="Get the processing status of a document",
)
async def get_ingest_status(
    request: Request,
    document_id: str,
) -> IngestStatus | IngestComplete:
    repo: Repository = request.app.state.repository
    doc = repo.get_document(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    return _build_status_response(repo, doc)
