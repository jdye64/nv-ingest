from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from corpus_suite_poc.config import Settings
from corpus_suite_poc.ops.metrics import Metrics
from corpus_suite_poc.pipeline.chunking import chunk_text
from corpus_suite_poc.pipeline.dag import STEP_ORDER
from corpus_suite_poc.store.db import Catalog, Database


@dataclass
class PagePipelineRunner:
    db: Database
    settings: Settings
    metrics: Metrics

    async def run_page(
        self,
        *,
        document_row: dict[str, Any],
        page_row: dict[str, Any],
        file_bytes: bytes,
        page_text: str,
    ) -> None:
        doc_id = document_row["id"]
        page_id = page_row["id"]
        page_index = int(page_row["page_index"])
        filename = document_row["filename"]
        mime = document_row.get("mime")

        state: dict[str, Any] = {}

        async with self.db.connection() as conn:
            cat = Catalog(conn)

            for step in STEP_ORDER:
                await cat.upsert_step(page_id, step, status="running", output=None, error=None)
                await conn.commit()
                try:
                    if step == "normalize":
                        state["mime"] = mime
                        state["filename"] = filename
                        out = {"mime": mime, "filename": filename, "bytes": len(file_bytes)}
                    elif step == "extract_text":
                        raw = page_text
                        state["raw_text"] = raw
                        out = {"chars": len(raw)}
                    elif step == "chunk":
                        raw = state.get("raw_text") or ""
                        chunks = chunk_text(
                            raw,
                            max_chars=self.settings.chunk_max_chars,
                            overlap=self.settings.chunk_overlap,
                        )
                        state["chunks"] = chunks
                        out = {"chunk_count": len(chunks)}
                    elif step == "index":
                        await cat.delete_chunks_for_page(page_id)
                        chunks = state.get("chunks") or []
                        for i, (cs, ce, txt) in enumerate(chunks):
                            cid = str(uuid.uuid4())
                            await cat.insert_chunk(
                                chunk_id=cid,
                                document_id=doc_id,
                                page_id=page_id,
                                page_index=page_index,
                                chunk_index=i,
                                text=txt,
                                char_start=cs,
                                char_end=ce,
                            )
                        out = {"indexed": len(chunks)}
                        self.metrics.inc("chunks_indexed", len(chunks))
                    else:
                        raise RuntimeError(f"unknown step {step}")

                    await cat.upsert_step(page_id, step, status="done", output=out, error=None)
                except Exception as exc:  # noqa: BLE001 - POC surfaces last error
                    await cat.upsert_step(
                        page_id,
                        step,
                        status="failed",
                        output=None,
                        error=str(exc),
                    )
                    await cat.update_page(page_id, status="failed", error=str(exc))
                    await cat.update_document(doc_id, status="failed", error=str(exc))
                    await conn.commit()
                    self.metrics.inc("errors")
                    raise
                await conn.commit()

            await cat.update_page(page_id, status="done", error=None)
            await conn.commit()

        self.metrics.inc("pages_processed")
