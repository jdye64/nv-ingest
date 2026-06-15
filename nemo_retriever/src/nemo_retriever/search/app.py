# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI application for NeMo Retriever Search."""

from __future__ import annotations

import logging
import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from nemo_retriever.search.config import SearchConfig, load_config
from nemo_retriever.search.services.documents import DocumentStore
from nemo_retriever.search.services.hits import HitCache

logger = logging.getLogger(__name__)

mimetypes.add_type("text/javascript", ".jsx")

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    config: SearchConfig = app.state.config
    cache: HitCache = app.state.hit_cache
    document_store: DocumentStore = app.state.document_store
    try:
        from nemo_retriever.search.mcp.tools import configure_mcp

        configure_mcp(config, cache, document_store)
    except Exception:
        logger.exception("Failed to configure search MCP tools")
    yield


def create_app(config: SearchConfig | None = None) -> FastAPI:
    cfg = config or load_config()
    hit_cache = HitCache(ttl_s=cfg.hit_cache_ttl_s, max_searches=cfg.hit_cache_max_searches)
    document_store = DocumentStore(
        ttl_s=cfg.document_store_ttl_s,
        max_documents=cfg.document_store_max_documents,
    )

    combined_lifespan = _lifespan
    mcp_asgi_app = None
    try:
        from fastmcp.utilities.lifespan import combine_lifespans

        from nemo_retriever.search.mcp.server import build_mcp_app

        mcp_asgi_app = build_mcp_app()
        combined_lifespan = combine_lifespans(_lifespan, mcp_asgi_app.lifespan)
    except Exception:
        logger.warning("fastmcp not available — MCP server will not be mounted", exc_info=True)

    application = FastAPI(
        title="NeMo Retriever Search",
        description="Google-minimal search UI and agent-friendly API over the retriever service.",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=combined_lifespan,
    )
    application.state.config = cfg
    application.state.default_config = cfg
    application.state.hit_cache = hit_cache
    application.state.document_store = document_store

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, HTTPException):
            raise exc
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})

    from nemo_retriever.search.routers import hits, ingest, search, settings

    application.include_router(search.router, prefix="/api/v1")
    application.include_router(hits.router, prefix="/api/v1")
    application.include_router(ingest.router, prefix="/api/v1")
    application.include_router(settings.router, prefix="/api/v1")

    if mcp_asgi_app is not None:
        application.mount("/mcp", mcp_asgi_app)
        logger.info("MCP server mounted at /mcp")

    if STATIC_DIR.is_dir():
        application.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @application.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(str(STATIC_DIR / "index.html"))
    else:
        logger.warning("Static directory not found at %s — search UI will not be served", STATIC_DIR)

    return application


app = create_app()
