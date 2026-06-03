# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any

from fastmcp import FastMCP

from nemo_retriever.search.mcp.registry import get_tool_registry

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "NeMo Retriever Search",
    instructions=(
        "Search and ingest documents through the NeMo Retriever service. "
        "Use get_corpus_status before searching. Call search_corpus with a "
        "natural language query, then export_hit to retrieve hit text for "
        "downstream agent tools."
    ),
)


def register_tools_from_registry() -> int:
    registry = get_tool_registry()
    count = 0
    for _key, entry in registry.items():
        fn = entry["fn"]
        tool_name = entry["name"]
        tool_description = entry["description"]

        if asyncio.iscoroutinefunction(fn):

            async def _wrapper(_fn=fn, **kwargs: Any) -> Any:
                return await _fn(**kwargs)

        else:

            def _wrapper(_fn=fn, **kwargs: Any) -> Any:
                return _fn(**kwargs)

        _wrapper.__name__ = fn.__name__
        _wrapper.__doc__ = fn.__doc__
        sig = inspect.signature(fn)
        _wrapper.__signature__ = sig  # type: ignore[attr-defined]
        _wrapper.__annotations__ = fn.__annotations__.copy()

        mcp.tool(name=tool_name, description=tool_description)(_wrapper)
        count += 1

    logger.info("Registered %d NeMo Search MCP tools", count)
    return count


def build_mcp_app():
    import nemo_retriever.search.mcp.tools  # noqa: F401

    register_tools_from_registry()
    return mcp.http_app(path="/")
