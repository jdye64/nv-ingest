# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load reference documents into a memory namespace.

Agents need their own experience and their reference material in one recall.
This runs the ordinary extraction graph through
:class:`~nemo_retriever.ingestor.graph_ingestor.GraphIngestor`, then writes the
resulting chunks into the memory table as semantic records instead of the
document table.

Embeddings produced by the graph are reused directly. Re-embedding the same
text through the memory embedder would double the cost for no benefit.
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence

from nemo_retriever.common.schemas.memory import MemoryWriteResult
from nemo_retriever.common.vdb.memory_schema import build_memory_row, new_memory_id

logger = logging.getLogger(__name__)

#: Graph column carrying the embedding payload produced by the embed stage.
_DEFAULT_EMBEDDING_COLUMN = "text_embeddings_1b_v2"


def ingest_documents_into_memory(
    *,
    backend: Any,
    documents: Sequence[str],
    buffers: Sequence[tuple[str, BytesIO]] = (),
    inline_texts: Optional[Sequence[str]] = None,
    namespace: str = "default",
    agent_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    extract_params: Any = None,
    embedding_column: str = _DEFAULT_EMBEDDING_COLUMN,
) -> MemoryWriteResult:
    """Extract, embed, and store documents as semantic memory records."""
    from nemo_retriever.ingestor.graph_ingestor import GraphIngestor

    graph = GraphIngestor(run_mode="inprocess", show_progress=False)
    if documents:
        graph.files(list(documents))
    if buffers:
        graph.buffers(list(buffers))
    if inline_texts:
        graph.texts(list(inline_texts))

    if extract_params is not None:
        graph.extract(extract_params)
    else:
        graph.extract()
    graph.embed()

    frame = graph.ingest()
    rows = _memory_rows_from_frame(
        frame,
        namespace=namespace,
        agent_id=agent_id,
        user_id=user_id,
        tags=list(tags or []),
        metadata=dict(metadata or {}),
        embedding_column=embedding_column,
    )
    if not rows:
        logger.warning("Document ingest produced no embedded chunks; nothing written to memory")
        return MemoryWriteResult(memory_ids=[], written=0)

    written = backend.write_rows(rows)
    return MemoryWriteResult(memory_ids=[row["memory_id"] for row in rows], written=written)


def _memory_rows_from_frame(
    frame: Any,
    *,
    namespace: str,
    agent_id: Optional[str],
    user_id: Optional[str],
    tags: List[str],
    metadata: Dict[str, Any],
    embedding_column: str,
) -> List[Dict[str, Any]]:
    from nemo_retriever.common.vdb.lancedb_schema import extract_embedding_from_row

    if frame is None or getattr(frame, "empty", True):
        return []

    rows: List[Dict[str, Any]] = []
    for record in frame.itertuples(index=False):
        embedding = extract_embedding_from_row(record, embedding_column=embedding_column)
        if embedding is None:
            continue
        text = getattr(record, "text", None)
        if not isinstance(text, str) or not text.strip():
            continue

        source_path = _source_path(record)
        chunk_metadata = dict(metadata)
        if source_path:
            chunk_metadata.setdefault("source_path", source_path)
        page_number = _page_number(record)
        if page_number is not None:
            chunk_metadata.setdefault("page_number", page_number)

        rows.append(
            build_memory_row(
                text=text,
                embedding=embedding,
                memory_id=new_memory_id(),
                memory_type="semantic",
                namespace=namespace,
                agent_id=agent_id,
                user_id=user_id,
                event_type="document",
                # Reference material is durable but should not outrank a fact
                # the agent actually learned, so keep it mid-scale.
                importance=0.4,
                tags=tags,
                metadata=chunk_metadata,
            )
        )
    return rows


def _source_path(record: Any) -> Optional[str]:
    for field in ("path", "source_path"):
        value = getattr(record, field, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    meta = getattr(record, "metadata", None)
    if isinstance(meta, dict):
        value = meta.get("source_path")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _page_number(record: Any) -> Optional[int]:
    value = getattr(record, "page_number", None)
    try:
        page = int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return page if page is not None and page >= 0 else None
