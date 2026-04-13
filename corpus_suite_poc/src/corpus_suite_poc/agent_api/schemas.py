from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentCreateResponse(BaseModel):
    id: str
    status: str
    filename: str
    mime: str | None
    blob_sha256: str
    byte_size: int


class DocumentSummary(BaseModel):
    id: str
    status: str
    filename: str
    mime: str | None
    page_count: int | None
    created_at: str


class PageSummary(BaseModel):
    id: str
    page_index: int
    status: str


class ProcessAccepted(BaseModel):
    accepted: bool = True
    document_id: str


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    page_index: int
    filename: str | None = None
    score: float
    text: str = Field(description="Snippet text for the agent.")


class QueryResponse(BaseModel):
    query: str
    hits: list[Citation]


class ChunkDetail(BaseModel):
    chunk_id: str
    document_id: str
    page_index: int
    chunk_index: int
    char_start: int | None
    char_end: int | None
    text: str
    filename: str | None
