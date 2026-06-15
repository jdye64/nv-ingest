# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the NeMo Retriever Search app."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SearchConfig:
    """Runtime configuration for ``retriever search start``."""

    service_url: str = "http://localhost:7670"
    vectordb_url: str = "http://localhost:7671"
    api_token: str | None = None
    host: str = "0.0.0.0"
    port: int = 8200
    hit_cache_ttl_s: int = 3600
    hit_cache_max_searches: int = 100
    document_store_ttl_s: int = 86400
    document_store_max_documents: int = 500
    ingest_max_concurrency: int = 8
    default_top_k: int = 10

    def with_updates(self, **kwargs: object) -> "SearchConfig":
        """Return a copy with the given fields replaced."""
        return replace(self, **kwargs)

    @property
    def auth_headers(self) -> dict[str, str]:
        if self.api_token:
            return {"Authorization": f"Bearer {self.api_token}"}
        return {}


def load_config(
    *,
    service_url: str | None = None,
    vectordb_url: str | None = None,
    api_token: str | None = None,
    host: str | None = None,
    port: int | None = None,
) -> SearchConfig:
    """Load config from explicit CLI overrides, then environment variables."""
    return SearchConfig(
        service_url=(service_url or os.environ.get("NEMO_RETRIEVER_SERVICE_URL") or "http://localhost:7670").rstrip(
            "/"
        ),
        vectordb_url=(vectordb_url or os.environ.get("NEMO_RETRIEVER_VECTORDB_URL") or "http://localhost:7671").rstrip(
            "/"
        ),
        api_token=(api_token or os.environ.get("NEMO_RETRIEVER_API_TOKEN") or None),
        host=host or os.environ.get("NEMO_RETRIEVER_SEARCH_HOST") or "0.0.0.0",
        port=port or int(os.environ.get("NEMO_RETRIEVER_SEARCH_PORT", "8200")),
    )
