from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Sequence

import aiosqlite

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    filename TEXT NOT NULL,
    mime TEXT,
    blob_sha256 TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    page_count INTEGER,
    status TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_index INTEGER NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    UNIQUE(document_id, page_index)
);

CREATE TABLE IF NOT EXISTS page_steps (
    page_id TEXT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    step_name TEXT NOT NULL,
    status TEXT NOT NULL,
    output_json TEXT,
    started_at TEXT,
    finished_at TEXT,
    error TEXT,
    PRIMARY KEY (page_id, step_name)
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_id TEXT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    page_index INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    char_start INTEGER,
    char_end INTEGER,
    created_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    document_id UNINDEXED,
    page_index UNINDEXED,
    text,
    tokenize = 'porter unicode61'
);
"""


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()


@dataclass
class Database:
    path: str

    @asynccontextmanager
    async def connection(self):
        async with aiosqlite.connect(self.path) as conn:
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA foreign_keys=ON")
            await conn.executescript(SCHEMA)
            yield conn

    async def init(self) -> None:
        async with self.connection():
            pass


class Catalog:
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self.conn = conn

    async def insert_document(
        self,
        *,
        tenant_id: str,
        filename: str,
        mime: str | None,
        blob_sha256: str,
        byte_size: int,
        page_count: int | None,
        status: str,
    ) -> str:
        doc_id = str(uuid.uuid4())
        now = _utc_now()
        await self.conn.execute(
            """
            INSERT INTO documents
            (id, tenant_id, filename, mime, blob_sha256, byte_size, page_count, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (doc_id, tenant_id, filename, mime, blob_sha256, byte_size, page_count, status, now, now),
        )
        return doc_id

    async def update_document(
        self,
        doc_id: str,
        *,
        status: str | None = None,
        page_count: int | None = None,
        error: str | None = None,
    ) -> None:
        fields: list[str] = ["updated_at = ?"]
        args: list[Any] = [_utc_now()]
        if status is not None:
            fields.append("status = ?")
            args.append(status)
        if page_count is not None:
            fields.append("page_count = ?")
            args.append(page_count)
        if error is not None:
            fields.append("error = ?")
            args.append(error)
        args.append(doc_id)
        await self.conn.execute(
            f"UPDATE documents SET {', '.join(fields)} WHERE id = ?",
            args,
        )

    async def get_document(self, doc_id: str) -> dict[str, Any] | None:
        cur = await self.conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = await cur.fetchone()
        return dict(row) if row else None

    async def list_documents(self, limit: int = 50) -> list[dict[str, Any]]:
        cur = await self.conn.execute(
            "SELECT * FROM documents ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def insert_pages(self, document_id: str, page_indices: Sequence[int]) -> list[str]:
        page_ids: list[str] = []
        for idx in page_indices:
            pid = str(uuid.uuid4())
            page_ids.append(pid)
            await self.conn.execute(
                """
                INSERT INTO pages (id, document_id, page_index, status)
                VALUES (?, ?, ?, ?)
                """,
                (pid, document_id, idx, "pending"),
            )
        return page_ids

    async def list_pages(self, document_id: str) -> list[dict[str, Any]]:
        cur = await self.conn.execute(
            "SELECT * FROM pages WHERE document_id = ? ORDER BY page_index ASC",
            (document_id,),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def get_page(self, page_id: str) -> dict[str, Any] | None:
        cur = await self.conn.execute("SELECT * FROM pages WHERE id = ?", (page_id,))
        row = await cur.fetchone()
        return dict(row) if row else None

    async def update_page(self, page_id: str, *, status: str, error: str | None = None) -> None:
        await self.conn.execute(
            "UPDATE pages SET status = ?, error = ? WHERE id = ?",
            (status, error, page_id),
        )

    async def upsert_step(
        self,
        page_id: str,
        step_name: str,
        *,
        status: str,
        output: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        now = _utc_now()
        out = json.dumps(output) if output is not None else None
        await self.conn.execute(
            """
            INSERT INTO page_steps (page_id, step_name, status, output_json, started_at, finished_at, error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(page_id, step_name) DO UPDATE SET
              status=excluded.status,
              output_json=excluded.output_json,
              finished_at=excluded.finished_at,
              error=excluded.error
            """,
            (page_id, step_name, status, out, now, now, error),
        )

    async def insert_chunk(
        self,
        *,
        chunk_id: str,
        document_id: str,
        page_id: str,
        page_index: int,
        chunk_index: int,
        text: str,
        char_start: int | None,
        char_end: int | None,
    ) -> None:
        now = _utc_now()
        await self.conn.execute(
            """
            INSERT INTO chunks (
              id, document_id, page_id, page_index, chunk_index,
              text, char_start, char_end, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk_id,
                document_id,
                page_id,
                page_index,
                chunk_index,
                text,
                char_start,
                char_end,
                now,
            ),
        )
        await self.conn.execute(
            "INSERT INTO chunks_fts (chunk_id, document_id, page_index, text) VALUES (?, ?, ?, ?)",
            (chunk_id, document_id, page_index, text),
        )

    async def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        cur = await self.conn.execute(
            """
            SELECT c.*, d.filename
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id = ?
            """,
            (chunk_id,),
        )
        row = await cur.fetchone()
        return dict(row) if row else None

    async def delete_chunks_for_page(self, page_id: str) -> None:
        cur = await self.conn.execute("SELECT id FROM chunks WHERE page_id = ?", (page_id,))
        ids = [r[0] for r in await cur.fetchall()]
        for cid in ids:
            await self.conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (cid,))
        await self.conn.execute("DELETE FROM chunks WHERE page_id = ?", (page_id,))

    async def purge_document_pages(self, document_id: str) -> None:
        """Remove pages, step rows, chunks, and FTS rows for a document."""
        cur = await self.conn.execute("SELECT id FROM chunks WHERE document_id = ?", (document_id,))
        ids = [r[0] for r in await cur.fetchall()]
        for cid in ids:
            await self.conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (cid,))
        await self.conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        await self.conn.execute("DELETE FROM pages WHERE document_id = ?", (document_id,))
