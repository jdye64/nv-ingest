# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import httpx

from nemo_retriever.search.config import SearchConfig

SERVICE_START_HINT = "Start the retriever service: `retriever service start`"


class ServiceUnavailableError(RuntimeError):
    """Raised when the retriever service cannot be reached."""


class ServiceClient:
    """Thin async HTTP client for retriever service / vectordb endpoints."""

    def __init__(self, config: SearchConfig) -> None:
        self._config = config

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        headers.update(self._config.auth_headers)
        return headers

    async def get_service_health(self) -> dict:
        url = f"{self._config.service_url}/v1/health"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, headers=self._config.auth_headers)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPError as exc:
            raise ServiceUnavailableError(
                f"Retriever service unreachable at {self._config.service_url}. {SERVICE_START_HINT}"
            ) from exc

    async def get_vectordb_health(self) -> dict | None:
        url = f"{self._config.vectordb_url}/v1/health"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, headers=self._config.auth_headers)
                if resp.status_code >= 400:
                    return None
                return resp.json()
        except httpx.HTTPError:
            return None

    async def query(self, query: str, top_k: int) -> dict:
        url = f"{self._config.service_url}/v1/query"
        payload = {"query": query, "top_k": top_k}
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, json=payload, headers=self._headers())
        except httpx.HTTPError as exc:
            raise ServiceUnavailableError(
                f"Failed to reach retriever service at {url}. {SERVICE_START_HINT}"
            ) from exc

        if resp.status_code == 404:
            detail = resp.text[:500]
            raise RuntimeError(
                "VectorDB query is not available on this service pod. "
                "Ensure vectordb is enabled and use gateway/standalone mode. "
                f"Detail: {detail}"
            )
        if resp.status_code == 422:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise RuntimeError(str(detail))
        if resp.status_code >= 400:
            raise RuntimeError(f"Query failed with HTTP {resp.status_code}: {resp.text[:500]}")

        return resp.json()
