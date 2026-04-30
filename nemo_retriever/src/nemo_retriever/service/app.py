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

from pathlib import Path

from nemo_retriever.service.config import ServiceConfig
from nemo_retriever.service.db.engine import DatabaseEngine
from nemo_retriever.service.db.repository import Repository
from nemo_retriever.service.event_bus import EventBus
from nemo_retriever.service.metrics import (
    setup_instrumentation as setup_metrics_instrumentation,
    start_refresh_loop as start_metrics_refresh_loop,
)
from nemo_retriever.service.processing.pool import ProcessingPool
from nemo_retriever.service.spool import SpoolStore

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


def _resolve_spool_root(config: ServiceConfig) -> Path:
    """Pick the spool directory: explicit override, else ``<db_dir>/spool``."""
    if config.spool.path:
        return Path(config.spool.path)
    return Path(config.database.path).resolve().parent / "spool"


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle for the service."""
    config: ServiceConfig = app.state.config

    db_engine = DatabaseEngine(config.database.path)
    db_engine.initialize()
    app.state.db_engine = db_engine
    app.state.repository = Repository(db_engine)

    event_bus = EventBus(
        default_maxsize=config.event_bus.queue_maxsize,
        buffer_size=config.event_bus.replay_buffer_size,
        overflow_policy=config.event_bus.overflow_policy,
        publish_timeout_s=config.event_bus.publish_timeout_s,
    )
    app.state.event_bus = event_bus

    spool_store: SpoolStore | None = None
    if config.spool.enabled:
        spool_root = _resolve_spool_root(config)
        spool_store = SpoolStore(spool_root)
        app.state.spool_store = spool_store
        logger.info("Spool enabled at %s", spool_root)
    else:
        app.state.spool_store = None
        logger.warning(
            "Spool DISABLED — accepted pages live only in RAM until processed; "
            "a pod restart between accept and processing will lose them"
        )

    loop = asyncio.get_running_loop()
    pool = ProcessingPool(config, db_engine, event_bus, loop, spool_store=spool_store)
    pool.start()
    app.state.processing_pool = pool

    # Recover any pages that were spooled but never processed — must
    # happen BEFORE we start accepting new HTTP requests so recovery
    # doesn't race fresh ingest for the (limited) batch buffer.
    if spool_store is not None:
        try:
            recovered = await asyncio.to_thread(pool.recover_from_spool)
            if recovered:
                logger.info("Spool recovery re-enqueued %d page(s)", recovered)
        except Exception:  # noqa: BLE001
            logger.exception("Spool recovery failed — continuing with cold start")
        pool.start_spool_cleanup()

    metrics_task = start_metrics_refresh_loop(app)

    logger.info(
        "Retriever service started — host=%s port=%d workers=%d " "spool=%s event_bus.policy=%s",
        config.server.host,
        config.server.port,
        config.processing.num_workers,
        "on" if spool_store else "off",
        config.event_bus.overflow_policy,
    )

    yield

    metrics_task.cancel()
    try:
        await metrics_task
    except asyncio.CancelledError:
        pass

    drain_timeout_s = float(getattr(getattr(config, "drain", None), "timeout_s", 60.0))
    logger.info(
        "Shutting down: draining pool (timeout=%.1fs, in_flight=%d)",
        drain_timeout_s,
        pool.in_flight_batches() if hasattr(pool, "in_flight_batches") else 0,
    )
    drained = await pool.drain(drain_timeout_s) if hasattr(pool, "drain") else True
    if not drained:
        logger.warning(
            "Drain incomplete after %.1fs; forcing executor shutdown — "
            "in-flight pages remain durable in the spool and will be re-enqueued on next start.",
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

    # Prometheus instrumentation must be wired before the app starts —
    # the lifespan only kicks off the periodic gauge-refresh task.
    setup_metrics_instrumentation(app)

    return app
