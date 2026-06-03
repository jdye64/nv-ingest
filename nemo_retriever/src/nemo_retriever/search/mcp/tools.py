# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from nemo_retriever.search.config import SearchConfig
from nemo_retriever.search.mcp.registry import search_tool
from nemo_retriever.search.services.export import export_hit
from nemo_retriever.search.services.hits import HitCache, build_search_response
from nemo_retriever.search.services.ingest import ingest_file_paths
from nemo_retriever.search.services.service_client import ServiceClient, ServiceUnavailableError

_config: SearchConfig | None = None
_hit_cache: HitCache | None = None


def configure_mcp(config: SearchConfig, hit_cache: HitCache) -> None:
    global _config, _hit_cache
    _config = config
    _hit_cache = hit_cache


def _require_config() -> SearchConfig:
    if _config is None:
        raise RuntimeError("Search MCP is not configured.")
    return _config


def _require_cache() -> HitCache:
    if _hit_cache is None:
        raise RuntimeError("Search MCP hit cache is not configured.")
    return _hit_cache


@search_tool(
    name="get_corpus_status",
    description="Return retriever service and vectordb corpus readiness (row count, table exists).",
)
async def get_corpus_status() -> dict[str, Any]:
    cfg = _require_config()
    client = ServiceClient(cfg)
    try:
        health = await client.get_service_health()
    except ServiceUnavailableError as exc:
        return {"service_reachable": False, "message": str(exc)}

    vdb = await client.get_vectordb_health()
    return {
        "service_reachable": True,
        "service_mode": health.get("mode"),
        "vectordb_enabled": vdb is not None,
        "total_rows": int(vdb.get("total_rows") or 0) if vdb else 0,
        "table_exists": bool(vdb.get("table_exists")) if vdb else False,
        "service_url": cfg.service_url,
    }


@search_tool(
    name="search_corpus",
    description="Semantic search over ingested documents via the retriever service /v1/query endpoint.",
)
async def search_corpus(query: str, top_k: int = 10) -> dict[str, Any]:
    cfg = _require_config()
    cache = _require_cache()
    client = ServiceClient(cfg)
    raw = await client.query(query.strip(), top_k)
    response, hits = build_search_response(query.strip(), top_k, raw)
    cache.store(response.search_id, hits)
    return response.model_dump()


@search_tool(
    name="ingest_documents",
    description="Ingest one or more local file paths into the retriever service (server-side read).",
)
async def ingest_documents(paths: list[str], label: str | None = None) -> dict[str, Any]:
    cfg = _require_config()
    file_paths = [Path(p).expanduser().resolve() for p in paths]
    missing = [str(p) for p in file_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Files not found: {', '.join(missing)}")
    result = await ingest_file_paths(file_paths, cfg, label=label)
    return result.model_dump()


@search_tool(
    name="export_hit",
    description="Export a search hit as text, summary, or json by hit_id from a prior search_corpus call.",
)
async def export_hit_tool(hit_id: str, format: str = "text") -> str:
    cache = _require_cache()
    hit = cache.get(hit_id)
    if hit is None:
        raise KeyError(f"Hit {hit_id!r} not found or expired.")
    if format not in ("text", "summary", "json"):
        raise ValueError("format must be one of: text, summary, json")
    body, _ = export_hit(hit, format)  # type: ignore[arg-type]
    return body
