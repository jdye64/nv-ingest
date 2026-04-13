from __future__ import annotations

from fastapi import APIRouter, HTTPException

from corpus_suite_poc.agent_api.deps import CorpusDep
from corpus_suite_poc.agent_api.schemas import ChunkDetail, Citation, QueryResponse
from corpus_suite_poc.store.db import Catalog

router = APIRouter(tags=["query"])


@router.get("/v1/query", response_model=QueryResponse)
async def query_corpus(
    corpus: CorpusDep,
    q: str,
    limit: int = 8,
    document_id: str | None = None,
) -> QueryResponse:
    corpus.metrics.inc("queries")
    try:
        hits = await corpus.query.search(q, limit=limit, document_id=document_id)
    except Exception as exc:  # noqa: BLE001
        corpus.metrics.inc("errors")
        raise HTTPException(status_code=400, detail=f"query failed: {exc}") from exc
    return QueryResponse(
        query=q,
        hits=[
            Citation(
                chunk_id=h.chunk_id,
                document_id=h.document_id,
                page_index=h.page_index,
                filename=h.filename,
                score=h.score,
                text=h.text,
            )
            for h in hits
        ],
    )


@router.get("/v1/chunks/{chunk_id}", response_model=ChunkDetail)
async def get_chunk(corpus: CorpusDep, chunk_id: str) -> ChunkDetail:
    async with corpus.db.connection() as conn:
        cat = Catalog(conn)
        row = await cat.get_chunk(chunk_id)
        await conn.commit()
    if not row:
        raise HTTPException(status_code=404, detail="chunk not found")
    return ChunkDetail(
        chunk_id=row["id"],
        document_id=row["document_id"],
        page_index=int(row["page_index"]),
        chunk_index=int(row["chunk_index"]),
        char_start=row.get("char_start"),
        char_end=row.get("char_end"),
        text=row["text"],
        filename=row.get("filename"),
    )
