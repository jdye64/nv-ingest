from __future__ import annotations

from dataclasses import dataclass

from corpus_suite_poc.ingest.pdf_pages import sniff_mime
from corpus_suite_poc.store.blobs import BlobStore
from corpus_suite_poc.store.db import Catalog, Database


@dataclass
class IngestService:
    db: Database
    blobs: BlobStore

    async def register_upload(
        self,
        *,
        tenant_id: str,
        filename: str,
        data: bytes,
    ) -> str:
        mime = sniff_mime(filename, data)
        sha = self.blobs.put_bytes(data)
        async with self.db.connection() as conn:
            cat = Catalog(conn)
            doc_id = await cat.insert_document(
                tenant_id=tenant_id,
                filename=filename,
                mime=mime,
                blob_sha256=sha,
                byte_size=len(data),
                page_count=None,
                status="uploaded",
            )
            await conn.commit()
        return doc_id

    async def get_blob_for_document(self, doc_id: str) -> tuple[dict, bytes]:
        async with self.db.connection() as conn:
            cat = Catalog(conn)
            row = await cat.get_document(doc_id)
            if not row:
                raise KeyError("document not found")
            sha = row["blob_sha256"]
        data = self.blobs.read_bytes(sha)
        return dict(row), data
