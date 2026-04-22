# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""POST /v1/ingest, GET /v1/ingest/status/{document_id}, GET /v1/ingest/job/{job_id}."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from nemo_retriever.service.db.repository import Repository
from nemo_retriever.service.models.document import Document, ProcessingStatus
from nemo_retriever.service.models.job import Job
from nemo_retriever.service.models.requests import IngestRequest
from nemo_retriever.service.models.responses import (
    IngestAccepted,
    IngestComplete,
    IngestStatus,
    JobStatus,
    MetricSummary,
    PageSummary,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ingest"])


@router.post(
    "/ingest",
    response_model=IngestAccepted,
    status_code=202,
    summary="Upload a single document (or page) for ingestion",
    responses={
        503: {
            "description": "Server is at capacity. Retry after a worker slot frees up.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Server busy — all worker slots are in use.",
                        "retry_after": 5,
                        "capacity": 0,
                        "pool_size": 32,
                    }
                }
            },
        }
    },
)
async def ingest_document(
    request: Request,
    file: UploadFile = File(..., description="The document to ingest"),
    metadata: str = Form(default="{}", description="JSON-encoded IngestRequest metadata"),
) -> IngestAccepted | JSONResponse:
    pool = request.app.state.processing_pool

    if not pool.has_capacity():
        logger.debug(
            "Ingest rejected — pool at capacity (%d/%d)",
            pool._pool_size,
            pool._pool_size,
        )
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Server busy — all worker slots are in use.",
                "retry_after": 10,
                "capacity": 0,
                "pool_size": pool._pool_size,
            },
            headers={"Retry-After": "10"},
        )

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

        job_id = meta.job_id
        if job_id is not None:
            existing_job = repo.get_job(job_id)
            if existing_job is None:
                filename = meta.filename or file.filename or "unknown"
                job = Job(
                    id=job_id,
                    filename=filename,
                    content_sha256="",
                    total_pages=meta.total_pages or 0,
                )
                repo.insert_job(job)
                repo.update_job_status(job_id, ProcessingStatus.PROCESSING)

            repo.increment_job_pages_submitted(job_id)
            pages_completed = repo.increment_job_pages_completed(job_id)
            logger.info(
                "[job %s] Page %s/%s deduped (sha=%s), counting as complete",
                job_id[:8],
                meta.page_number,
                meta.total_pages,
                content_sha256[:8],
            )

            event_bus = request.app.state.event_bus
            loop = request.app.state.processing_pool._event_loop

            job = repo.get_job(job_id)
            if job and pages_completed >= job.total_pages and job.total_pages > 0:
                repo.update_job_status(job_id, ProcessingStatus.COMPLETE)
                logger.info("[job %s] All %d pages of %s complete", job_id[:8], job.total_pages, job.filename)
                asyncio.run_coroutine_threadsafe(
                    event_bus.publish(
                        job_id,
                        {
                            "event": "job_complete",
                            "job_id": job_id,
                            "filename": job.filename,
                            "total_pages": job.total_pages,
                        },
                    ),
                    loop,
                )
            else:
                asyncio.run_coroutine_threadsafe(
                    event_bus.publish(
                        job_id,
                        {
                            "event": "page_complete",
                            "document_id": existing.id,
                            "job_id": job_id,
                            "pages_received": 0,
                            "total_pages": meta.total_pages or 0,
                        },
                    ),
                    loop,
                )

        return IngestAccepted(
            document_id=existing.id,
            job_id=job_id or existing.job_id,
            content_sha256=existing.content_sha256,
            status=existing.processing_status,
            created_at=existing.created_at,
        )

    filename = meta.filename or file.filename or "unknown"
    content_type = meta.content_type or file.content_type or "application/octet-stream"

    # Handle job creation / update when the client pre-splits pages
    job_id = meta.job_id
    if job_id is not None:
        existing_job = repo.get_job(job_id)
        if existing_job is None:
            job = Job(
                id=job_id,
                filename=filename,
                content_sha256="",
                total_pages=meta.total_pages or 0,
            )
            repo.insert_job(job)
            repo.update_job_status(job_id, ProcessingStatus.PROCESSING)

        repo.increment_job_pages_submitted(job_id)
        logger.info("[job %s] Received page %s/%s of %s", job_id[:8], meta.page_number, meta.total_pages, filename)

    doc = Document(
        job_id=job_id,
        filename=filename,
        content_type=content_type,
        content_sha256=content_sha256,
        file_size_bytes=len(file_bytes),
        page_number=meta.page_number,
        metadata_json=json.dumps(meta.metadata),
    )
    repo.insert_document(doc)
    logger.info("Document accepted: id=%s filename=%s sha=%s", doc.id, filename, content_sha256[:12])

    future = pool.try_submit(
        doc.id,
        content_sha256,
        file_bytes,
        filename,
        job_id=job_id,
        page_number=meta.page_number or 1,
    )
    if future is None:
        repo.update_document_status(doc.id, ProcessingStatus.QUEUED)
        logger.warning("Document %s queued but pool became full between check and submit", doc.id)
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Server busy — all worker slots are in use.",
                "document_id": doc.id,
                "retry_after": 10,
                "capacity": 0,
                "pool_size": pool._pool_size,
            },
            headers={"Retry-After": "10"},
        )

    return IngestAccepted(
        document_id=doc.id,
        job_id=job_id,
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


@router.get(
    "/ingest/job/{job_id}",
    response_model=JobStatus,
    summary="Get the aggregated processing status of a job",
)
async def get_job_status(
    request: Request,
    job_id: str,
) -> JobStatus:
    repo: Repository = request.app.state.repository
    job = repo.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    metrics_rows = repo.get_metrics_for_job(job_id)

    agg: dict[str, MetricSummary] = {}
    for m in metrics_rows:
        key = m.model_name
        if key not in agg:
            agg[key] = MetricSummary(
                model_name=m.model_name,
                invocation_count=0,
                pages_processed=0,
                detections_count=0,
                duration_ms=0.0,
            )
        agg[key].invocation_count += m.invocation_count
        agg[key].pages_processed += m.pages_processed
        agg[key].detections_count += m.detections_count
        agg[key].duration_ms += m.duration_ms

    return JobStatus(
        job_id=job.id,
        filename=job.filename,
        status=job.processing_status,
        total_pages=job.total_pages,
        pages_submitted=job.pages_submitted,
        pages_completed=job.pages_completed,
        metrics=list(agg.values()),
        created_at=job.created_at,
        updated_at=job.updated_at,
    )
