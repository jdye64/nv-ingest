# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastMCP integration for retriever service mode.

The MCP server is intentionally a thin facade over the public service HTTP
API.  This keeps local stdio usage and remote ``/mcp`` usage aligned with
the same routes that SDK and REST clients already exercise.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_retriever.service.config import MCPQueryMethods, ServiceConfig
from nemo_retriever.service.query_schema import QueryFormat

logger = logging.getLogger(__name__)


class MCPDocumentInput(BaseModel):
    """Document payload accepted by the MCP ``ingest_documents`` tool."""

    model_config = ConfigDict(extra="forbid")

    path: str | None = Field(
        default=None,
        description="Path to a document visible to the MCP server process.",
    )
    filename: str | None = Field(
        default=None,
        description="Filename to report to the service. Required for base64 uploads.",
    )
    content_base64: str | None = Field(
        default=None,
        description="Base64-encoded document bytes for remote agents.",
    )
    content_type: str | None = Field(
        default=None,
        description="Optional MIME type recorded in the upload metadata.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-document metadata forwarded in the service metadata field.",
    )

    @model_validator(mode="after")
    def _validate_source(self) -> "MCPDocumentInput":
        has_path = bool(self.path)
        has_content = bool(self.content_base64)
        if has_path == has_content:
            raise ValueError("Provide exactly one of path or content_base64.")
        if has_content and not self.filename:
            raise ValueError("filename is required when content_base64 is provided.")
        return self


@dataclass(frozen=True)
class ServiceMCPSettings:
    """Runtime settings for a service-mode MCP server."""

    base_url: str = "http://localhost:7670"
    api_token: str | None = None
    auth_header_name: str = "Authorization"
    max_concurrency: int = 8
    request_timeout_s: float = 60.0
    agentic_request_timeout_s: float = 1800.0
    ingest_timeout_s: float = 1800.0
    poll_interval_s: float = 2.0
    enable_write_tools: bool = True
    enable_memory_tools: bool = True
    memory_request_timeout_s: float = 300.0
    query_methods: MCPQueryMethods = "classic"

    @property
    def normalized_base_url(self) -> str:
        return self.base_url.rstrip("/")

    @property
    def auth_headers(self) -> dict[str, str]:
        token = (self.api_token or "").strip()
        if not token:
            return {}
        return {self.auth_header_name: f"Bearer {token}"}

    @property
    def enable_classic_query(self) -> bool:
        return self.query_methods in ("classic", "all")

    @property
    def enable_agentic_query(self) -> bool:
        return self.query_methods in ("agentic", "all")


def _effective_query_methods(
    configured: MCPQueryMethods,
    *,
    agentic_enabled: bool,
) -> MCPQueryMethods:
    """Drop agentic MCP tools when agentic retrieval is not configured."""
    if agentic_enabled:
        return configured
    if configured == "agentic":
        return "classic"
    if configured == "all":
        return "classic"
    return configured


def settings_from_service_config(config: ServiceConfig) -> ServiceMCPSettings:
    """Build MCP settings for the MCP app mounted inside ``retriever service start``."""
    mcp_cfg = config.mcp
    base_url = mcp_cfg.base_url
    if not base_url:
        host = config.server.host
        if host in {"", "0.0.0.0", "::"}:
            host = "127.0.0.1"
        if ":" in host and not host.startswith("[") and host.count(":") > 1:
            host = f"[{host}]"
        base_url = f"http://{host}:{config.server.port}"

    return ServiceMCPSettings(
        base_url=base_url,
        api_token=config.auth.api_token,
        auth_header_name=config.auth.header_name,
        max_concurrency=mcp_cfg.max_concurrency,
        request_timeout_s=mcp_cfg.request_timeout_s,
        agentic_request_timeout_s=config.agentic.request_timeout_s,
        ingest_timeout_s=mcp_cfg.ingest_timeout_s,
        poll_interval_s=mcp_cfg.poll_interval_s,
        enable_write_tools=mcp_cfg.enable_write_tools,
        enable_memory_tools=mcp_cfg.enable_memory_tools,
        query_methods=_effective_query_methods(
            mcp_cfg.query_methods,
            agentic_enabled=config.agentic.enabled,
        ),
    )


class ServiceMCPClient:
    """Small async client used by MCP tools."""

    def __init__(
        self,
        settings: ServiceMCPSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport

    def _client(self, *, timeout_s: float | None = None) -> httpx.AsyncClient:
        timeout_value = timeout_s if timeout_s is not None else self._settings.request_timeout_s
        return httpx.AsyncClient(
            base_url=self._settings.normalized_base_url,
            headers=self._settings.auth_headers,
            timeout=httpx.Timeout(timeout_value),
            transport=self._transport,
        )

    @staticmethod
    def _json_or_text(resp: httpx.Response) -> Any:
        try:
            return resp.json()
        except ValueError:
            return {"text": resp.text}

    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        detail = resp.text[:1000] if resp.text else "(empty)"
        raise RuntimeError(f"Retriever service returned HTTP {resp.status_code}: {detail}")

    async def health(self) -> dict[str, Any]:
        async with self._client() as client:
            resp = await client.get("/v1/health")
        self._raise_for_status(resp)
        return dict(self._json_or_text(resp))

    async def pipeline_config(self) -> dict[str, Any]:
        async with self._client() as client:
            resp = await client.get("/v1/ingest/pipeline-config")
        self._raise_for_status(resp)
        return dict(self._json_or_text(resp))

    async def get_job(self, job_id: str, *, include_documents: bool = False) -> dict[str, Any]:
        async with self._client() as client:
            resp = await client.get(
                f"/v1/ingest/job/{job_id}",
                params={"include_documents": str(include_documents).lower()},
            )
        self._raise_for_status(resp)
        return dict(self._json_or_text(resp))

    async def list_job_documents(
        self,
        job_id: str,
        *,
        status: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"offset": offset, "limit": limit}
        if status is not None:
            params["status"] = status
        async with self._client() as client:
            resp = await client.get(f"/v1/ingest/job/{job_id}/documents", params=params)
        self._raise_for_status(resp)
        return dict(self._json_or_text(resp))

    async def get_document(self, job_id: str, document_id: str) -> dict[str, Any]:
        async with self._client() as client:
            resp = await client.get(f"/v1/ingest/job/{job_id}/document/{document_id}")
        self._raise_for_status(resp)
        return dict(self._json_or_text(resp))

    async def query(
        self,
        query: str,
        *,
        top_k: int = 5,
        format: QueryFormat = "hits",
        rerank: bool = False,
        rerank_top_k: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = dict(payload or {})
        # Typed tool arguments are authoritative; payload is only for additional
        # service options such as filters. In particular, it must not enable
        # reranking when the explicit rerank argument is false.
        for key in ("query", "top_k", "format", "agentic", "rerank", "rerank_top_k"):
            body.pop(key, None)
        body.update({"query": query, "top_k": top_k, "format": format})
        if rerank:
            body["rerank"] = True
        if rerank_top_k is not None:
            body["rerank_top_k"] = rerank_top_k
        async with self._client() as client:
            resp = await client.post("/v1/query", json=body)
        self._raise_for_status(resp)
        return dict(self._json_or_text(resp))

    async def agentic_query(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Call ``POST /v1/query`` with ``agentic=true`` (long timeout)."""
        async with self._client(timeout_s=self._settings.agentic_request_timeout_s) as client:
            resp = await client.post(
                "/v1/query",
                json={
                    "query": query,
                    "top_k": top_k,
                    "format": "hits",
                    "agentic": True,
                },
            )
        self._raise_for_status(resp)
        return dict(self._json_or_text(resp))

    async def answer(
        self,
        query: str,
        *,
        top_k: int = 5,
        include_chunks: bool = False,
        include_metadata: bool = False,
        reasoning_enabled: bool | None = None,
        reference: str | None = None,
        judge: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "include_chunks": include_chunks,
            "include_metadata": include_metadata,
            "judge": judge,
        }
        if reasoning_enabled is not None:
            body["reasoning_enabled"] = reasoning_enabled
        if reference is not None:
            body["reference"] = reference
        async with self._client() as client:
            resp = await client.post("/v1/answer", json=body)
        self._raise_for_status(resp)
        return dict(self._json_or_text(resp))

    async def _memory_post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        if not self._settings.enable_memory_tools:
            raise RuntimeError("MCP memory tools are disabled for this service.")
        async with self._client(timeout_s=self._settings.memory_request_timeout_s) as client:
            resp = await client.post(path, json=body)
        self._raise_for_status(resp)
        return dict(self._json_or_text(resp))

    async def remember(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        return await self._memory_post("/v1/memory/remember", {"records": records})

    async def recall(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._memory_post("/v1/memory/recall", body)

    async def memory_timeline(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._memory_post("/v1/memory/timeline", body)

    async def forget(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._memory_post("/v1/memory/forget", body)

    async def consolidate(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._memory_post("/v1/memory/consolidate", body)

    async def memory_stats(self, namespace: str | None = None) -> dict[str, Any]:
        if not self._settings.enable_memory_tools:
            raise RuntimeError("MCP memory tools are disabled for this service.")
        params = {"namespace": namespace} if namespace else None
        async with self._client(timeout_s=self._settings.memory_request_timeout_s) as client:
            resp = await client.get("/v1/memory/stats", params=params)
        self._raise_for_status(resp)
        return dict(self._json_or_text(resp))

    async def ingest_documents(
        self,
        documents: list[MCPDocumentInput],
        *,
        label: str | None = None,
        job_metadata: dict[str, Any] | None = None,
        pipeline_spec: dict[str, Any] | None = None,
        retain_results: bool = False,
        include_result_data: bool = False,
    ) -> dict[str, Any]:
        if not self._settings.enable_write_tools:
            raise RuntimeError("MCP write tools are disabled for this service.")
        if not documents:
            raise ValueError("documents must contain at least one item.")

        timeout_at = time.monotonic() + self._settings.ingest_timeout_s
        async with self._client(timeout_s=None) as client:
            job_resp = await client.post(
                "/v1/ingest/job",
                json={
                    "expected_documents": len(documents),
                    "label": label,
                    "metadata": job_metadata or {},
                    "retain_results": retain_results or include_result_data,
                },
            )
            self._raise_for_status(job_resp)
            job = dict(self._json_or_text(job_resp))
            job_id = str(job["job_id"])

            upload_results = await self._upload_documents(
                client,
                job_id,
                documents,
                pipeline_spec=pipeline_spec,
            )
            document_ids = [
                item["document_id"] for item in upload_results if item.get("ok") and item.get("document_id")
            ]
            documents_page = await self._poll_documents_until_terminal(
                client,
                job_id,
                document_ids,
                timeout_at=timeout_at,
            )

            if include_result_data:
                for item in documents_page.get("items", []):
                    if item.get("status") in {"completed", "failed"}:
                        detail = await self._get_document_with_client(client, job_id, item["document_id"])
                        item["result_data"] = detail.get("result_data")

        upload_errors = [item for item in upload_results if not item.get("ok")]
        return {
            "job_id": job_id,
            "job": job,
            "uploads": upload_results,
            "upload_errors": upload_errors,
            "documents": documents_page,
            "timed_out": bool(documents_page.get("timed_out")),
        }

    async def _upload_documents(
        self,
        client: httpx.AsyncClient,
        job_id: str,
        documents: list[MCPDocumentInput],
        *,
        pipeline_spec: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        sem = asyncio.Semaphore(self._settings.max_concurrency)

        async def _upload_one(doc: MCPDocumentInput) -> dict[str, Any]:
            async with sem:
                try:
                    filename, content = _document_bytes(doc)
                    metadata = {
                        "filename": filename,
                        "content_type": doc.content_type,
                        "metadata": doc.metadata,
                    }
                    if pipeline_spec is not None:
                        metadata["pipeline"] = pipeline_spec
                    resp = await client.post(
                        f"/v1/ingest/job/{job_id}/document",
                        files={
                            "file": (
                                filename,
                                content,
                                doc.content_type or "application/octet-stream",
                            )
                        },
                        data={"metadata": json.dumps(metadata)},
                    )
                    self._raise_for_status(resp)
                    body = dict(self._json_or_text(resp))
                    return {"ok": True, "filename": filename, **body}
                except Exception as exc:  # noqa: BLE001 - MCP response should include every upload failure.
                    filename = doc.filename or (Path(doc.path).name if doc.path else "")
                    logger.warning("MCP upload failed for %s: %s", filename or "<unknown>", exc)
                    return {"ok": False, "filename": filename, "error": str(exc)}

        return await asyncio.gather(*(_upload_one(doc) for doc in documents))

    async def _poll_documents_until_terminal(
        self,
        client: httpx.AsyncClient,
        job_id: str,
        document_ids: list[str],
        *,
        timeout_at: float,
    ) -> dict[str, Any]:
        wanted = set(document_ids)
        latest: dict[str, Any] = {
            "job_id": job_id,
            "total": len(document_ids),
            "total_filtered": len(document_ids),
            "offset": 0,
            "limit": len(document_ids),
            "items": [],
        }
        if not wanted:
            latest["timed_out"] = False
            return latest

        while True:
            latest = await self._list_all_job_documents(client, job_id)
            items = latest.get("items", [])
            seen = {str(item.get("document_id")): item for item in items}
            terminal = {doc_id for doc_id in wanted if seen.get(doc_id, {}).get("status") in {"completed", "failed"}}
            if terminal == wanted:
                latest["timed_out"] = False
                return latest
            if time.monotonic() >= timeout_at:
                latest["timed_out"] = True
                return latest
            await asyncio.sleep(self._settings.poll_interval_s)

    async def _list_all_job_documents(self, client: httpx.AsyncClient, job_id: str) -> dict[str, Any]:
        limit = 1000
        offset = 0
        all_items: list[dict[str, Any]] = []
        first_page: dict[str, Any] | None = None
        while True:
            resp = await client.get(
                f"/v1/ingest/job/{job_id}/documents",
                params={"offset": offset, "limit": limit},
            )
            self._raise_for_status(resp)
            page = dict(self._json_or_text(resp))
            if first_page is None:
                first_page = dict(page)
            items = list(page.get("items") or [])
            all_items.extend(items)
            total = int(page.get("total_filtered", page.get("total", len(all_items))))
            offset += len(items)
            if not items or offset >= total:
                break
        result = first_page or {
            "job_id": job_id,
            "total": 0,
            "total_filtered": 0,
            "offset": 0,
            "limit": limit,
        }
        result["items"] = all_items
        result["offset"] = 0
        result["limit"] = len(all_items)
        return result

    async def _get_document_with_client(
        self,
        client: httpx.AsyncClient,
        job_id: str,
        document_id: str,
    ) -> dict[str, Any]:
        resp = await client.get(f"/v1/ingest/job/{job_id}/document/{document_id}")
        self._raise_for_status(resp)
        return dict(self._json_or_text(resp))


def _document_bytes(doc: MCPDocumentInput) -> tuple[str, bytes]:
    if doc.path:
        path = Path(doc.path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Document path does not exist or is not a file: {path}")
        return doc.filename or path.name, path.read_bytes()

    raw = doc.content_base64 or ""
    if "," in raw and raw.lstrip().startswith("data:"):
        raw = raw.split(",", 1)[1]
    raw = "".join(raw.split())
    try:
        content = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Invalid base64 content for {doc.filename!r}: {exc}") from exc
    return doc.filename or "document.bin", content


def build_mcp(settings: ServiceMCPSettings | None = None) -> FastMCP:
    """Create a FastMCP server for a retriever service endpoint."""
    settings = settings or ServiceMCPSettings()
    service = ServiceMCPClient(settings)

    mcp = FastMCP(
        "NeMo Retriever Service",
        instructions=(
            "Use these tools to interact with a running NVIDIA NeMo Retriever "
            "service. Ingest documents, check job status, query the configured "
            "VectorDB, run agentic retrieval when configured, and ask the "
            "configured answer-generation endpoint."
        ),
    )

    @mcp.tool(name="health", description="Check the retriever service health endpoint.")
    async def health() -> dict[str, Any]:
        return await service.health()

    @mcp.tool(
        name="pipeline_config",
        description="Return redacted live service pipeline configuration.",
    )
    async def pipeline_config() -> dict[str, Any]:
        return await service.pipeline_config()

    @mcp.tool(name="get_job", description="Fetch a service ingestion job by ID.")
    async def get_job(job_id: str, include_documents: bool = False) -> dict[str, Any]:
        return await service.get_job(job_id, include_documents=include_documents)

    @mcp.tool(
        name="list_job_documents",
        description="List document statuses for a service ingestion job.",
    )
    async def list_job_documents(
        job_id: str,
        status: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        return await service.list_job_documents(job_id, status=status, offset=offset, limit=limit)

    @mcp.tool(
        name="get_document",
        description="Fetch one document status/result from a service ingestion job.",
    )
    async def get_document(job_id: str, document_id: str) -> dict[str, Any]:
        return await service.get_document(job_id, document_id)

    if settings.enable_classic_query:

        @mcp.tool(
            name="query",
            description=(
                "Search ingested documents through the service VectorDB endpoint. "
                "format='hits' (default) returns raw retrieval hits; format='evidence' "
                "returns the fidelity-tagged, citation-ready {evidence, coverage} shape. "
                "Retrieval (dense vs hybrid) is auto-detected from the table's own indexes. "
                "Set rerank=true to rerank candidates through the configured service endpoint."
            ),
        )
        async def query(
            query: str,
            top_k: int = 5,
            format: QueryFormat = "hits",
            rerank: bool = False,
            rerank_top_k: int | None = None,
            payload: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            return await service.query(
                query,
                top_k=top_k,
                format=format,
                rerank=rerank,
                rerank_top_k=rerank_top_k,
                payload=payload,
            )

    if settings.enable_agentic_query:

        @mcp.tool(
            name="agentic_query",
            description=(
                "Run the configured agentic (ReAct) retrieval workflow over ingested "
                "documents via POST /v1/query with agentic=true. Returns the standard "
                "hits envelope with the same chunk-level fields as classic retrieval, "
                "plus top-level doc_id, rank, and result_source for the selecting "
                "stage; rank and result_source also remain under metadata for compatibility."
            ),
        )
        async def agentic_query(query: str, top_k: int = 5) -> dict[str, Any]:
            return await service.agentic_query(query, top_k=top_k)

    @mcp.tool(name="answer", description="Search ingested documents and generate an answer.")
    async def answer(
        query: str,
        top_k: int = 5,
        include_chunks: bool = False,
        include_metadata: bool = False,
        reasoning_enabled: bool | None = None,
        reference: str | None = None,
        judge: bool = False,
    ) -> dict[str, Any]:
        return await service.answer(
            query,
            top_k=top_k,
            include_chunks=include_chunks,
            include_metadata=include_metadata,
            reasoning_enabled=reasoning_enabled,
            reference=reference,
            judge=judge,
        )

    if settings.enable_write_tools:

        @mcp.tool(
            name="ingest_documents",
            description=(
                "Upload documents to the retriever service. Each document can "
                "reference a server-visible path or provide base64 content."
            ),
        )
        async def ingest_documents(
            documents: list[MCPDocumentInput],
            label: str | None = None,
            job_metadata: dict[str, Any] | None = None,
            pipeline_spec: dict[str, Any] | None = None,
            retain_results: bool = False,
            include_result_data: bool = False,
        ) -> dict[str, Any]:
            return await service.ingest_documents(
                documents,
                label=label,
                job_metadata=job_metadata,
                pipeline_spec=pipeline_spec,
                retain_results=retain_results,
                include_result_data=include_result_data,
            )

    if settings.enable_memory_tools:
        _register_memory_tools(mcp, service)

    return mcp


def _register_memory_tools(mcp: FastMCP, service: "ServiceMCPClient") -> None:
    """Register the agent-memory tools on an MCP server.

    These are the highest-leverage integration point for memory: any
    MCP-capable host gets episodic and semantic memory with no SDK work, over
    the transport the service already exposes.
    """

    @mcp.tool(
        name="remember",
        description=(
            "Store something worth recalling later. Use memory_type='episodic' for "
            "things that happened (default) and 'semantic' for durable facts such as "
            "a user preference. Pass session_id to keep a conversation's memories "
            "together so they can be replayed with the timeline tool."
        ),
    )
    async def remember(
        text: str,
        memory_type: str = "episodic",
        session_id: str | None = None,
        namespace: str = "default",
        agent_id: str | None = None,
        user_id: str | None = None,
        role: str | None = None,
        tags: list[str] | None = None,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = {
            "text": text,
            "memory_type": memory_type,
            "namespace": namespace,
            "session_id": session_id,
            "agent_id": agent_id,
            "user_id": user_id,
            "role": role,
            "importance": importance,
            "tags": tags or [],
            "metadata": metadata or {},
        }
        return await service.remember([record])

    @mcp.tool(
        name="recall",
        description=(
            "Search stored memories by meaning. Narrow the search with session_id, "
            "memory_type, tags, or a time window rather than filtering the results "
            "yourself. Set recency_halflife_seconds to favor recent memories when "
            "several are equally relevant."
        ),
    )
    async def recall(
        query: str,
        top_k: int = 10,
        session_id: str | None = None,
        namespace: str = "default",
        memory_type: str | None = None,
        agent_id: str | None = None,
        user_id: str | None = None,
        tags_any: list[str] | None = None,
        min_importance: float | None = None,
        occurred_after: str | None = None,
        occurred_before: str | None = None,
        recency_halflife_seconds: float | None = None,
    ) -> dict[str, Any]:
        return await service.recall(
            {
                "query": query,
                "top_k": top_k,
                "recency_halflife_seconds": recency_halflife_seconds,
                "filter": _memory_filter_body(
                    namespace=namespace,
                    session_id=session_id,
                    memory_type=memory_type,
                    agent_id=agent_id,
                    user_id=user_id,
                    tags_any=tags_any,
                    min_importance=min_importance,
                    occurred_after=occurred_after,
                    occurred_before=occurred_before,
                ),
            }
        )

    @mcp.tool(
        name="timeline",
        description=(
            "Replay one session's memories in the order they happened. Use this "
            "instead of recall when you need what came before or after something, "
            "rather than what is most similar to a query."
        ),
    )
    async def timeline(
        session_id: str,
        namespace: str = "default",
        since: str | None = None,
        until: str | None = None,
        limit: int = 100,
        ascending: bool = True,
    ) -> dict[str, Any]:
        return await service.memory_timeline(
            {
                "session_id": session_id,
                "namespace": namespace,
                "since": since,
                "until": until,
                "limit": limit,
                "ascending": ascending,
            }
        )

    @mcp.tool(
        name="forget",
        description=(
            "Retire memories that are wrong or no longer wanted. The default marks "
            "them no longer valid so they stop appearing in recall while staying "
            "auditable. Pass hard=true to delete the rows outright."
        ),
    )
    async def forget(
        memory_id: str | None = None,
        session_id: str | None = None,
        namespace: str = "default",
        memory_type: str | None = None,
        tags_any: list[str] | None = None,
        hard: bool = False,
    ) -> dict[str, Any]:
        if memory_id is None and session_id is None and not tags_any:
            raise ValueError("forget requires memory_id, session_id, or tags_any.")
        body: dict[str, Any] = {"hard": hard}
        if memory_id:
            body["memory_id"] = memory_id
        else:
            body["filter"] = _memory_filter_body(
                namespace=namespace,
                session_id=session_id,
                memory_type=memory_type,
                tags_any=tags_any,
            )
        return await service.forget(body)

    @mcp.tool(
        name="consolidate",
        description=(
            "Distill a session's episodic memories into durable facts, superseding "
            "any earlier fact they contradict. This makes a language model call, so "
            "run it at the end of a session rather than during one."
        ),
    )
    async def consolidate(
        session_id: str | None = None,
        namespace: str = "default",
        max_episodes: int = 200,
        since: str | None = None,
    ) -> dict[str, Any]:
        return await service.consolidate(
            {
                "session_id": session_id,
                "namespace": namespace,
                "max_episodes": max_episodes,
                "since": since,
            }
        )

    @mcp.tool(
        name="memory_stats",
        description="Report how many memories a namespace holds and the time span they cover.",
    )
    async def memory_stats(namespace: str = "default") -> dict[str, Any]:
        return await service.memory_stats(namespace)


def _memory_filter_body(**fields: Any) -> dict[str, Any]:
    """Drop unset fields so server-side filter defaults still apply."""
    body = {key: value for key, value in fields.items() if value not in (None, [], "")}
    body.setdefault("namespace", "default")
    return body


def build_mcp_app(settings: ServiceMCPSettings | None = None) -> Any:
    """Create the ASGI app mounted by ``retriever service start``."""
    return build_mcp(settings).http_app(path="/")
