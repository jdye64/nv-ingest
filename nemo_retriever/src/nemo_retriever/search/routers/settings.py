# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from fastapi import APIRouter, Request

from nemo_retriever.search.models.requests import SettingsUpdateRequest
from nemo_retriever.search.models.responses import SettingsResponse
from nemo_retriever.search.services.config_state import (
    apply_settings_update,
    get_default_config,
    settings_response,
)

router = APIRouter(tags=["settings"])


@router.get("/settings", response_model=SettingsResponse)
async def read_settings(request: Request) -> SettingsResponse:
    return settings_response(request)


@router.put("/settings", response_model=SettingsResponse)
async def update_settings(req: SettingsUpdateRequest, request: Request) -> SettingsResponse:
    apply_settings_update(request, req)
    return settings_response(request)


@router.post("/settings/reset", response_model=SettingsResponse)
async def reset_settings(request: Request) -> SettingsResponse:
    request.app.state.config = get_default_config(request)
    try:
        from nemo_retriever.search.mcp.tools import configure_mcp

        configure_mcp(request.app.state.config, request.app.state.hit_cache)
    except Exception:
        pass
    return settings_response(request)
