# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from fastapi import Request

from nemo_retriever.search.config import SearchConfig
from nemo_retriever.search.models.requests import SettingsUpdateRequest
from nemo_retriever.search.models.responses import (
    SettingsDefaults,
    SettingsPlaceholder,
    SettingsResponse,
)

_PLACEHOLDERS: list[SettingsPlaceholder] = [
    SettingsPlaceholder(
        id="embed_endpoint",
        label="Embed endpoint override",
        description="Direct NIM embedding URL (future — queries currently use the service /v1/query proxy).",
        enabled=False,
    ),
    SettingsPlaceholder(
        id="rerank",
        label="Reranking",
        description="Enable reranking on search results (future).",
        enabled=False,
    ),
    SettingsPlaceholder(
        id="content_types",
        label="Content-type filter",
        description="Limit hits to text, table, chart, or image chunks (future).",
        enabled=False,
    ),
]


def get_config(request: Request) -> SearchConfig:
    return request.app.state.config


def get_default_config(request: Request) -> SearchConfig:
    return request.app.state.default_config


def settings_response(request: Request) -> SettingsResponse:
    cfg = get_config(request)
    defaults = get_default_config(request)
    return SettingsResponse(
        service_url=cfg.service_url,
        vectordb_url=cfg.vectordb_url,
        api_token_set=bool(cfg.api_token),
        default_top_k=cfg.default_top_k,
        defaults=SettingsDefaults(
            service_url=defaults.service_url,
            vectordb_url=defaults.vectordb_url,
            default_top_k=defaults.default_top_k,
        ),
        placeholders=_PLACEHOLDERS,
    )


def apply_settings_update(request: Request, update: SettingsUpdateRequest) -> SearchConfig:
    cfg = get_config(request)
    patches: dict[str, object] = {}

    if update.service_url is not None:
        patches["service_url"] = update.service_url.rstrip("/")
    if update.vectordb_url is not None:
        patches["vectordb_url"] = update.vectordb_url.rstrip("/")
    if update.default_top_k is not None:
        patches["default_top_k"] = update.default_top_k
    if update.api_token is not None:
        token = update.api_token.strip()
        patches["api_token"] = token or None

    if not patches:
        return cfg

    new_cfg = cfg.with_updates(**patches)
    request.app.state.config = new_cfg

    try:
        from nemo_retriever.search.mcp.tools import configure_mcp

        configure_mcp(new_cfg, request.app.state.hit_cache)
    except Exception:
        pass

    return new_cfg
