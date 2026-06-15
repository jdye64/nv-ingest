# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import time
import uuid
from collections import OrderedDict
from threading import Lock
from typing import Any

from nemo_retriever.search.models.responses import (
    AgentHit,
    AgentHitExportLinks,
    AgentHitProvenance,
    SearchResponse,
)
from nemo_retriever.vdb.records import _normalize_hit

_PREVIEW_LEN = 200


class HitCache:
    """TTL cache mapping hit_id -> AgentHit for export endpoints."""

    def __init__(self, *, ttl_s: int = 3600, max_searches: int = 100) -> None:
        self._ttl_s = ttl_s
        self._max_searches = max_searches
        self._hits: dict[str, tuple[float, AgentHit]] = {}
        self._searches: OrderedDict[str, float] = OrderedDict()
        self._lock = Lock()

    def _evict_expired(self, now: float) -> None:
        stale_ids = [hid for hid, (_, ts) in self._hits.items() if now - ts > self._ttl_s]
        for hid in stale_ids:
            self._hits.pop(hid, None)
        stale_searches = [sid for sid, ts in self._searches.items() if now - ts > self._ttl_s]
        for sid in stale_searches:
            self._searches.pop(sid, None)

    def _trim_searches(self) -> None:
        while len(self._searches) > self._max_searches:
            old_search_id, _ = self._searches.popitem(last=False)
            prefix = f"{old_search_id}:"
            for hid in list(self._hits):
                if hid.startswith(prefix):
                    self._hits.pop(hid, None)

    def store(self, search_id: str, hits: list[AgentHit]) -> None:
        now = time.time()
        with self._lock:
            self._evict_expired(now)
            self._searches[search_id] = now
            self._searches.move_to_end(search_id)
            for hit in hits:
                self._hits[hit.hit_id] = (now, hit)
            self._trim_searches()

    def get(self, hit_id: str) -> AgentHit | None:
        now = time.time()
        with self._lock:
            entry = self._hits.get(hit_id)
            if entry is None:
                return None
            ts, hit = entry
            if now - ts > self._ttl_s:
                self._hits.pop(hit_id, None)
                return None
            return hit


def _content_type(metadata: dict[str, Any]) -> str | None:
    for key in ("type", "_content_type", "content_type"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _distance(hit: dict[str, Any]) -> float | None:
    for key in ("_distance", "_score", "distance", "score"):
        value = hit.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _preview(text: str, limit: int = _PREVIEW_LEN) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def build_agent_hit(raw_hit: dict[str, Any], *, search_id: str, rank: int) -> AgentHit:
    normalized = _normalize_hit(raw_hit)
    metadata = dict(normalized.get("metadata") or {})
    text = str(normalized.get("text") or "")
    source = str(normalized.get("source") or normalized.get("source_id") or "")
    page_number = normalized.get("page_number")
    hit_id = f"{search_id}:{rank}"
    base = f"/api/v1/hits/{hit_id}/export"
    doc_base = f"/api/v1/hits/{hit_id}/document"
    return AgentHit(
        rank=rank,
        hit_id=hit_id,
        text=text,
        text_preview=_preview(text),
        source=source,
        page_number=page_number if isinstance(page_number, int) else None,
        content_type=_content_type(metadata),
        distance=_distance(normalized),
        metadata=metadata,
        provenance=AgentHitProvenance(
            source_id=str(normalized.get("source_id") or source),
            pdf_basename=str(normalized.get("pdf_basename") or ""),
            pdf_page=str(normalized.get("pdf_page") or ""),
        ),
        export=AgentHitExportLinks(
            text_url=f"{base}?format=text",
            json_url=f"{base}?format=json",
            summary_url=f"{base}?format=summary",
            document_url=f"{doc_base}?download=1",
        ),
    )


def build_search_response(query: str, top_k: int, raw: dict[str, Any]) -> tuple[SearchResponse, list[AgentHit]]:
    search_id = uuid.uuid4().hex
    results = raw.get("results") or []
    raw_hits: list[dict[str, Any]] = []
    if results and isinstance(results[0], dict):
        raw_hits = list(results[0].get("hits") or [])

    hits = [build_agent_hit(h, search_id=search_id, rank=i + 1) for i, h in enumerate(raw_hits)]
    response = SearchResponse(
        search_id=search_id,
        query=query,
        top_k=top_k,
        hit_count=len(hits),
        hits=hits,
    )
    return response, hits


def agent_hit_to_json(hit: AgentHit) -> str:
    return json.dumps(hit.model_dump(), indent=2, ensure_ascii=False)
