from __future__ import annotations

import re
from dataclasses import dataclass
from corpus_suite_poc.store.db import Database


def _fts_query(q: str) -> str:
    """Build a conservative FTS5 prefix query from free text."""
    parts = [p for p in re.split(r"\s+", q.strip()) if p]
    if not parts:
        return ""
    escaped: list[str] = []
    for p in parts:
        p = re.sub(r'(["*])', r"\\\1", p)
        escaped.append(f"{p}*")
    return " AND ".join(escaped)


@dataclass
class Hit:
    chunk_id: str
    document_id: str
    page_index: int
    text: str
    score: float
    filename: str | None


@dataclass
class QueryEngine:
    db: Database

    async def search(self, q: str, *, limit: int = 8, document_id: str | None = None) -> list[Hit]:
        fts = _fts_query(q)
        if not fts:
            return []

        async with self.db.connection() as conn:
            if document_id:
                sql = """
                SELECT
                  c.id AS chunk_id,
                  c.document_id,
                  c.page_index,
                  c.text,
                  bm25(chunks_fts) AS score,
                  d.filename AS filename
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.chunk_id
                JOIN documents d ON d.id = c.document_id
                WHERE chunks_fts MATCH ? AND c.document_id = ?
                ORDER BY score
                LIMIT ?
                """
                cur = await conn.execute(sql, (fts, document_id, limit))
            else:
                sql = """
                SELECT
                  c.id AS chunk_id,
                  c.document_id,
                  c.page_index,
                  c.text,
                  bm25(chunks_fts) AS score,
                  d.filename AS filename
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.chunk_id
                JOIN documents d ON d.id = c.document_id
                WHERE chunks_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """
                cur = await conn.execute(sql, (fts, limit))
            rows = await cur.fetchall()

        out: list[Hit] = []
        for r in rows:
            out.append(
                Hit(
                    chunk_id=str(r["chunk_id"]),
                    document_id=str(r["document_id"]),
                    page_index=int(r["page_index"]),
                    text=str(r["text"]),
                    score=float(r["score"]),
                    filename=str(r["filename"]) if r["filename"] is not None else None,
                )
            )
        return out
