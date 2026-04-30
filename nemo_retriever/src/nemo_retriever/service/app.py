# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI application factory for the retriever service mode."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from typing import AsyncIterator

from fastapi import FastAPI

from nemo_retriever.service.config import ServiceConfig
from nemo_retriever.service.db.engine import DatabaseEngine
from nemo_retriever.service.db.repository import Repository
from nemo_retriever.service.event_bus import EventBus
from nemo_retriever.service.processing.pool import ProcessingPool

logger = logging.getLogger(__name__)


def _configure_logging(config: ServiceConfig) -> None:
    """Set up root logger with both console and rotating-file handlers."""
    root = logging.getLogger()
    root.setLevel(config.logging.level.upper())

    fmt = logging.Formatter(config.logging.format)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        config.logging.file,
        maxBytes=50 * 1024 * 1024,  # 50 MiB
        backupCount=5,
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    logger.info("Logging configured: level=%s file=%s", config.logging.level, config.logging.file)


def _apply_resource_limits(config: ServiceConfig) -> None:
    """Best-effort resource capping (Linux only for some features)."""
    res = config.resources

    if res.gpu_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(res.gpu_devices)
        logger.info("CUDA_VISIBLE_DEVICES set to %s", os.environ["CUDA_VISIBLE_DEVICES"])

    if res.max_cpu_cores is not None:
        try:
            cpus = set(range(res.max_cpu_cores))
            os.sched_setaffinity(0, cpus)
            logger.info("CPU affinity restricted to %d cores", res.max_cpu_cores)
        except (AttributeError, OSError) as exc:
            logger.warning("Could not set CPU affinity: %s", exc)

    if res.max_memory_mb is not None:
        try:
            import resource as _resource

            limit_bytes = res.max_memory_mb * 1024 * 1024
            _resource.setrlimit(_resource.RLIMIT_AS, (limit_bytes, limit_bytes))
            logger.info("Memory limit set to %d MiB", res.max_memory_mb)
        except (ImportError, ValueError, OSError) as exc:
            logger.warning("Could not set memory limit: %s", exc)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle for the service."""
    config: ServiceConfig = app.state.config

    db_engine = DatabaseEngine(config.database.path)
    db_engine.initialize()
    app.state.db_engine = db_engine
    app.state.repository = Repository(db_engine)

    event_bus = EventBus()
    app.state.event_bus = event_bus

    loop = asyncio.get_running_loop()
    pool = ProcessingPool(config, db_engine, event_bus, loop)
    pool.start()
    app.state.processing_pool = pool

    logger.info(
        "Retriever service started — host=%s port=%d workers=%d",
        config.server.host,
        config.server.port,
        config.processing.num_workers,
    )

    yield

    drain_timeout_s = float(getattr(getattr(config, "drain", None), "timeout_s", 60.0))
    logger.info(
        "Shutting down: draining pool (timeout=%.1fs, in_flight=%d)",
        drain_timeout_s,
        pool.in_flight_batches() if hasattr(pool, "in_flight_batches") else 0,
    )
    drained = await pool.drain(drain_timeout_s) if hasattr(pool, "drain") else True
    if not drained:
        logger.warning(
            "Drain incomplete after %.1fs; forcing executor shutdown — " "in-flight pages may have their results lost.",
            drain_timeout_s,
        )
    pool.shutdown()
    db_engine.close()
    logger.info("Retriever service stopped")


def create_app(config: ServiceConfig) -> FastAPI:
    """Build and return a fully-configured :class:`FastAPI` application."""
    _configure_logging(config)
    _apply_resource_limits(config)

    app = FastAPI(
        title="Retriever Service",
        description="Low-latency document ingestion service powered by nemo-retriever",
        version="1.0.0",
        docs_url="/docs",
        lifespan=_lifespan,
    )
    app.state.config = config

    if config.auth.api_token:
        from nemo_retriever.service.auth import BearerAuthMiddleware

        app.add_middleware(BearerAuthMiddleware, config=config.auth)
        logger.info(
            "Bearer-token authentication ENABLED (header=%s, bypass=%s)",
            config.auth.header_name,
            config.auth.bypass_paths,
        )
    else:
        logger.info("Bearer-token authentication DISABLED (no api_token configured)")

    from nemo_retriever.service.routers import ingest, internal, metrics, stream, system

    app.include_router(ingest.router, prefix="/v1")
    app.include_router(internal.router, prefix="/v1")
    app.include_router(metrics.router, prefix="/v1")
    app.include_router(stream.router, prefix="/v1")
    app.include_router(system.router, prefix="/v1")

    return app
