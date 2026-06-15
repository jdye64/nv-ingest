# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import mimetypes
from pathlib import Path

import httpx

from nemo_retriever.search.models.responses import AgentHit
from nemo_retriever.search.services.documents import DocumentStore, StoredDocument


class DocumentNotFoundError(LookupError):
    """Raised when the original document for a hit cannot be resolved."""


def _lookup_keys(hit: AgentHit) -> list[str]:
    keys: list[str] = []
    for value in (
        hit.source,
        hit.provenance.source_id,
        hit.provenance.pdf_basename,
        hit.metadata.get("source_path"),
        hit.metadata.get("source_id"),
    ):
        if isinstance(value, str) and value.strip():
            keys.append(value.strip())
    source_meta = hit.metadata.get("source_metadata")
    if isinstance(source_meta, dict):
        for value in (source_meta.get("source_id"), source_meta.get("source_name")):
            if isinstance(value, str) and value.strip():
                keys.append(value.strip())
    deduped: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
        basename = Path(key).name
        if basename and basename not in seen:
            seen.add(basename)
            deduped.append(basename)
    return deduped


def _from_local_path(source_key: str) -> StoredDocument | None:
    path = Path(source_key).expanduser()
    if not path.is_file():
        return None
    data = path.read_bytes()
    filename = path.name
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return StoredDocument(filename=filename, data=data, content_type=content_type, stored_at=0.0)


async def _from_remote_url(url: str) -> StoredDocument | None:
    if not url.startswith(("http://", "https://")):
        return None
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        response = await client.get(url)
        response.raise_for_status()
    filename = Path(url.split("?", 1)[0]).name or "document"
    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip()
    if not content_type:
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return StoredDocument(
        filename=filename,
        data=response.content,
        content_type=content_type,
        stored_at=0.0,
    )


async def resolve_document(hit: AgentHit, store: DocumentStore | None) -> StoredDocument:
    keys = _lookup_keys(hit)
    for key in keys:
        if store is not None:
            doc = store.get(key)
            if doc is not None:
                return doc
        local = _from_local_path(key)
        if local is not None:
            return local
        try:
            remote = await _from_remote_url(key)
        except httpx.HTTPError:
            remote = None
        if remote is not None:
            return remote

    label = hit.source or hit.provenance.source_id or "unknown source"
    raise DocumentNotFoundError(
        f"Original document not available for {label!r}. "
        "Upload the file via the search UI, ensure the source path is readable by this server, "
        "or use an http(s) source URL."
    )
