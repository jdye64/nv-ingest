# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decorator-driven registration of MCP-exposed search tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

_TOOL_REGISTRY: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class ToolMeta:
    name: str
    description: str
    tags: list[str] = field(default_factory=list)


def search_tool(*, name: str, description: str = "", tags: list[str] | None = None):
    def decorator(fn: Callable) -> Callable:
        key = f"{fn.__module__}.{fn.__qualname__}"
        _TOOL_REGISTRY[key] = {
            "fn": fn,
            "name": name,
            "description": description,
            "tags": tags or [],
        }
        fn._search_tool_meta = ToolMeta(name=name, description=description, tags=tags or [])  # type: ignore[attr-defined]
        return fn

    return decorator


def get_tool_registry() -> dict[str, dict[str, Any]]:
    return dict(_TOOL_REGISTRY)
