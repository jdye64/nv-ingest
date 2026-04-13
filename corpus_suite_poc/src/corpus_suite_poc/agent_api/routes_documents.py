from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from corpus_suite_poc.agent_api.deps import CorpusDep
from corpus_suite_poc.agent_api.schemas import DocumentCreateResponse, DocumentSummary, PageSummary, ProcessAccepted
from corpus_suite_poc.store.db import Catalog

router = APIRouter(prefix="/v1/documents", tags=["documents"])


@router.post("", response_model=DocumentCreateResponse)
async def upload_document(
    corpus: CorpusDep,
    file: Annotated[UploadFile, File()],
    tenant_id: Annotated[str, Form()] = "default",
) -> DocumentCreateResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    doc_id = await corpus.ingest.register_upload(
        tenant_id=tenant_id,
        filename=file.filename or "upload.bin",
        data=data,
    )
    corpus.metrics.inc("documents_created")
    async with corpus.db.connection() as conn:
        cat = Catalog(conn)
        row = await cat.get_document(doc_id)
        await conn.commit()
    assert row is not None
    return DocumentCreateResponse(
        id=doc_id,
        status=row["status"],
        filename=row["filename"],
        mime=row.get("mime"),
        blob_sha256=row["blob_sha256"],
        byte_size=int(row["byte_size"]),
    )


@router.get("", response_model=list[DocumentSummary])
async def list_documents(corpus: CorpusDep, limit: int = 50) -> list[DocumentSummary]:
    async with corpus.db.connection() as conn:
        cat = Catalog(conn)
        rows = await cat.list_documents(limit=limit)
        await conn.commit()
    return [
        DocumentSummary(
            id=r["id"],
            status=r["status"],
            filename=r["filename"],
            mime=r.get("mime"),
            page_count=r.get("page_count"),
            created_at=r["created_at"],
        )
        for r in rows
    ]


@router.get("/{doc_id}", response_model=DocumentSummary)
async def get_document(corpus: CorpusDep, doc_id: str) -> DocumentSummary:
    async with corpus.db.connection() as conn:
        cat = Catalog(conn)
        row = await cat.get_document(doc_id)
        await conn.commit()
    if not row:
        raise HTTPException(status_code=404, detail="document not found")
    return DocumentSummary(
        id=row["id"],
        status=row["status"],
        filename=row["filename"],
        mime=row.get("mime"),
        page_count=row.get("page_count"),
        created_at=row["created_at"],
    )


@router.get("/{doc_id}/pages", response_model=list[PageSummary])
async def list_pages(corpus: CorpusDep, doc_id: str) -> list[PageSummary]:
    async with corpus.db.connection() as conn:
        cat = Catalog(conn)
        if not await cat.get_document(doc_id):
            raise HTTPException(status_code=404, detail="document not found")
        rows = await cat.list_pages(doc_id)
        await conn.commit()
    return [PageSummary(id=r["id"], page_index=int(r["page_index"]), status=r["status"]) for r in rows]


@router.post("/{doc_id}/process", response_model=ProcessAccepted)
async def process_document(
    corpus: CorpusDep,
    doc_id: str,
    background: BackgroundTasks,
) -> ProcessAccepted:
    async with corpus.db.connection() as conn:
        cat = Catalog(conn)
        if not await cat.get_document(doc_id):
            raise HTTPException(status_code=404, detail="document not found")
        await conn.commit()

    async def job() -> None:
        try:
            await corpus.processor.process_document(doc_id)
        except Exception:
            corpus.metrics.inc("errors")
            raise

    background.add_task(job)
    return ProcessAccepted(document_id=doc_id)
