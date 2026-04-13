from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from corpus_suite_poc.config import Settings
from corpus_suite_poc.ingest.pdf_pages import pages_for_blob
from corpus_suite_poc.ingest.service import IngestService
from corpus_suite_poc.ops.metrics import Metrics
from corpus_suite_poc.pipeline.runner import PagePipelineRunner
from corpus_suite_poc.store.db import Catalog, Database


@dataclass
class DocumentProcessor:
    db: Database
    ingest: IngestService
    runner: PagePipelineRunner
    settings: Settings
    metrics: Metrics

    async def process_document(self, doc_id: str) -> dict[str, Any]:
        row, data = await self.ingest.get_blob_for_document(doc_id)
        page_objs = pages_for_blob(mime=row.get("mime"), filename=row["filename"], data=data)
        texts = {p.page_index: p.text for p in page_objs}

        async with self.db.connection() as conn:
            cat = Catalog(conn)
            await cat.purge_document_pages(doc_id)
            await cat.update_document(
                doc_id,
                status="processing",
                page_count=len(page_objs),
                error=None,
            )
            await cat.insert_pages(doc_id, [p.page_index for p in page_objs])
            page_rows = await cat.list_pages(doc_id)
            await conn.commit()

        sem = asyncio.Semaphore(self.settings.max_concurrent_pages)

        async def run_one(pr: dict[str, Any]) -> None:
            async with sem:
                idx = int(pr["page_index"])
                await self.runner.run_page(
                    document_row=row,
                    page_row=pr,
                    file_bytes=data,
                    page_text=texts.get(idx, ""),
                )

        await asyncio.gather(*(run_one(pr) for pr in page_rows))

        async with self.db.connection() as conn:
            cat = Catalog(conn)
            pages = await cat.list_pages(doc_id)
            failed = [p for p in pages if p["status"] != "done"]
            if failed:
                await cat.update_document(doc_id, status="failed", error="one or more pages failed")
            else:
                await cat.update_document(doc_id, status="ready", error=None)
            await conn.commit()

        return {"document_id": doc_id, "page_count": len(page_objs), "failed_pages": len(failed)}
