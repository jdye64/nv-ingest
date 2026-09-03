# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cluster configuration endpoints.

Lets a server hold the configuration for an entire cluster. Workers, harnesses,
external tools, and web pages can ``GET /v1/config`` to fetch the authoritative
configuration and ``PUT /v1/config`` to update it. The introspection catalog and
JSON Schema are also served so UIs can render self-documenting config editors.

The store is in-memory (per gateway/standalone process). Until a config is
pushed, ``GET`` derives the unified object from the process's running
``ServiceConfig`` so the endpoint always returns something useful.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["config"])


class _ClusterConfigStore:
    """Process-local store for the cluster's authoritative configuration."""

    def __init__(self) -> None:
        self._config: dict[str, Any] | None = None
        self._version: int = 0

    def get(self) -> dict[str, Any] | None:
        return self._config

    def set(self, config: dict[str, Any]) -> int:
        self._config = config
        self._version += 1
        return self._version

    @property
    def version(self) -> int:
        return self._version


_STORE = _ClusterConfigStore()


def get_cluster_config_store() -> _ClusterConfigStore:
    return _STORE


class ConfigResponse(BaseModel):
    config: dict[str, Any]
    version: int = Field(description="Monotonic version; 0 means derived from the running service config.")
    source: str


class ConfigUpdate(BaseModel):
    """Body accepted by PUT /v1/config — the unified config (or its subset)."""

    config: dict[str, Any] = Field(default_factory=dict)


def _derive_from_service(request: Request) -> dict[str, Any]:
    """Build a unified config dict from the running ServiceConfig."""
    from nemo_retriever.config import build_config

    service_config = getattr(request.app.state, "config", None)
    service_data = service_config.model_dump(mode="json") if service_config is not None else {}
    unified = build_config(file_data={"service": service_data}, use_env=False)
    return unified.model_dump(mode="json")


@router.get("/config", response_model=ConfigResponse, summary="Fetch the cluster configuration")
async def get_config(request: Request) -> ConfigResponse:
    """Return the authoritative cluster configuration.

    If a configuration has been pushed via ``PUT``, that document is returned.
    Otherwise the config is derived from this process's running service config.
    """
    stored = _STORE.get()
    if stored is not None:
        return ConfigResponse(config=stored, version=_STORE.version, source="cluster-store")
    return ConfigResponse(config=_derive_from_service(request), version=0, source="running-service")


@router.put("/config", response_model=ConfigResponse, summary="Update the cluster configuration")
async def put_config(update: ConfigUpdate) -> ConfigResponse:
    """Validate and store a new cluster configuration.

    The body is validated against :class:`NeMoRetrieverConfig` before it is
    accepted, so invalid documents are rejected with HTTP 422.
    """
    from nemo_retriever.config import build_config

    payload = update.config or {}
    try:
        validated = build_config(file_data=payload, use_env=False)
    except Exception as exc:  # noqa: BLE001 - surface validation errors to client
        raise HTTPException(status_code=422, detail=f"Invalid configuration: {exc}") from exc

    version = _STORE.set(validated.model_dump(mode="json"))
    logger.info("Cluster configuration updated to version %d", version)
    return ConfigResponse(config=_STORE.get() or {}, version=version, source="cluster-store")


@router.get("/config/catalog", summary="Introspected, categorized configuration catalog")
async def get_catalog() -> dict[str, Any]:
    """Return the self-documenting catalog (sections, fields, categories, impact)."""
    from nemo_retriever.config import catalog_as_dict

    return catalog_as_dict()


@router.get("/config/schema", summary="JSON Schema for the unified configuration")
async def get_schema() -> dict[str, Any]:
    """Return the JSON Schema of :class:`NeMoRetrieverConfig`."""
    from nemo_retriever.config import NeMoRetrieverConfig

    return NeMoRetrieverConfig.model_json_schema()


__all__ = ["router", "get_cluster_config_store"]
