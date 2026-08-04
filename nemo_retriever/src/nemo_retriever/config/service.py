# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The configuration service: load, serialize, persist, and sync config.

:class:`ConfigService` is the operational entry point other components use
(harnesses, external tools, web pages, the FastAPI service). It layers config
from defaults, a local file, environment variables, a remote cluster endpoint,
and explicit overrides, then supports serializing it back out and pushing it to
a cluster config server.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

import yaml

from nemo_retriever.config.schema import (
    ENV_NESTED_DELIMITER,
    ENV_PREFIX,
    NeMoRetrieverConfig,
    build_config,
)

logger = logging.getLogger(__name__)

#: Environment variable that points at a unified config file.
CONFIG_FILE_ENV = "NEMO_RETRIEVER_CONFIG_FILE"
#: Default unified config file name looked up in the working directory.
DEFAULT_CONFIG_FILENAME = "nemo-retriever.yaml"
#: Path (relative to a service base URL) where the cluster config is served.
REMOTE_CONFIG_PATH = "/v1/config"

_SECRET_FIELD_HINTS = ("api_key", "api_token", "password", "secret", "auth_token")
SerializeFormat = Literal["yaml", "json"]


def _looks_secret(key: str) -> bool:
    lowered = key.lower()
    return any(hint in lowered for hint in _SECRET_FIELD_HINTS)


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("***" if _looks_secret(k) and v not in (None, "") else _redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


class ConfigService:
    """Load, serialize, persist, and synchronize :class:`NeMoRetrieverConfig`.

    Precedence (highest first):
    explicit overrides → remote cluster → environment → file → defaults.
    """

    def __init__(self, config: NeMoRetrieverConfig, *, source: str | None = None) -> None:
        self.config = config
        self.source = source or "defaults"

    # -- discovery ---------------------------------------------------------

    @staticmethod
    def discover_config_path(explicit: str | os.PathLike[str] | None = None) -> Path | None:
        """Find a unified config file using standard precedence.

        1. ``explicit`` path (raises if missing)
        2. ``$NEMO_RETRIEVER_CONFIG_FILE``
        3. ``./nemo-retriever.yaml``
        """
        if explicit is not None:
            path = Path(explicit)
            if not path.is_file():
                raise FileNotFoundError(f"Config file not found: {path}")
            return path

        env_path = os.environ.get(CONFIG_FILE_ENV)
        if env_path:
            path = Path(env_path)
            if not path.is_file():
                raise FileNotFoundError(f"{CONFIG_FILE_ENV} points to a missing file: {path}")
            return path

        cwd_candidate = Path.cwd() / DEFAULT_CONFIG_FILENAME
        if cwd_candidate.is_file():
            return cwd_candidate

        return None

    # -- loading -----------------------------------------------------------

    @classmethod
    def load(
        cls,
        config_path: str | os.PathLike[str] | None = None,
        *,
        remote_url: str | None = None,
        overrides: dict[str, Any] | None = None,
        use_env: bool = True,
    ) -> "ConfigService":
        """Load configuration from all layers into a :class:`ConfigService`."""
        path = cls.discover_config_path(config_path)
        file_data = cls._read_file(path) if path is not None else {}

        remote_data: dict[str, Any] = {}
        source_parts: list[str] = ["defaults"]
        if path is not None:
            source_parts.append(f"file={path}")
        if use_env:
            source_parts.append(f"env[{ENV_PREFIX}*]")
        if remote_url:
            remote_data = cls.fetch_remote(remote_url)
            source_parts.append(f"remote={remote_url}")
        if overrides:
            source_parts.append("overrides")

        config = build_config(
            file_data=file_data,
            remote_data=remote_data,
            overrides=overrides,
            use_env=use_env,
        )
        return cls(config, source=" < ".join(source_parts))

    @classmethod
    def from_service_yaml(cls, service_yaml_path: str | os.PathLike[str]) -> "ConfigService":
        """Import an existing ``retriever-service.yaml`` into the ``service`` section.

        Bridges the legacy flat service config file into the unified object so
        existing deployments can adopt the central config without a rewrite.
        """
        path = Path(service_yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"Service config file not found: {path}")
        service_raw = yaml.safe_load(path.read_text()) or {}
        config = build_config(file_data={"service": service_raw}, use_env=False)
        return cls(config, source=f"service_yaml={path}")

    @staticmethod
    def _read_file(path: Path) -> dict[str, Any]:
        text = path.read_text()
        if path.suffix in (".json",):
            data = json.loads(text)
        else:
            data = yaml.safe_load(text)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError(f"Config file {path} must contain a mapping at the top level")
        return data

    # -- serialization -----------------------------------------------------

    def to_dict(self, *, redact_secrets: bool = False, exclude_none: bool = False) -> dict[str, Any]:
        """Serialize the config to a plain dict."""
        data = self.config.model_dump(mode="json", exclude_none=exclude_none)
        if redact_secrets:
            data = _redact(data)
        return data

    def to_yaml(self, *, redact_secrets: bool = False) -> str:
        return yaml.safe_dump(self.to_dict(redact_secrets=redact_secrets), sort_keys=False, default_flow_style=False)

    def to_json(self, *, redact_secrets: bool = False, indent: int = 2) -> str:
        return json.dumps(self.to_dict(redact_secrets=redact_secrets), indent=indent)

    def serialize(self, fmt: SerializeFormat = "yaml", *, redact_secrets: bool = False) -> str:
        if fmt == "json":
            return self.to_json(redact_secrets=redact_secrets)
        return self.to_yaml(redact_secrets=redact_secrets)

    # -- persistence -------------------------------------------------------

    def save(
        self,
        path: str | os.PathLike[str],
        *,
        fmt: SerializeFormat | None = None,
        redact_secrets: bool = False,
    ) -> Path:
        """Persist the config to a local file (format inferred from suffix)."""
        out = Path(path)
        resolved_fmt: SerializeFormat = fmt or ("json" if out.suffix == ".json" else "yaml")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.serialize(resolved_fmt, redact_secrets=redact_secrets))
        logger.info("Wrote NeMo Retriever config to %s (%s)", out, resolved_fmt)
        return out

    # -- remote cluster sync ----------------------------------------------

    @staticmethod
    def _remote_endpoint(base_url: str) -> str:
        return base_url.rstrip("/") + REMOTE_CONFIG_PATH

    @classmethod
    def fetch_remote(cls, base_url: str, *, timeout: float = 30.0) -> dict[str, Any]:
        """GET the cluster configuration document from a remote service."""
        import httpx

        url = cls._remote_endpoint(base_url)
        resp = httpx.get(url, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, dict):
            raise ValueError(f"Remote config at {url} did not return a JSON object")
        return payload.get("config", payload)

    def push_remote(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        api_token: str | None = None,
    ) -> dict[str, Any]:
        """PUT this configuration to a remote cluster config server."""
        import httpx

        url = self._remote_endpoint(base_url)
        headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}
        resp = httpx.put(url, json=self.to_dict(), headers=headers, timeout=timeout)
        resp.raise_for_status()
        logger.info("Pushed NeMo Retriever config to %s", url)
        return resp.json() if resp.content else {}

    # -- convenience -------------------------------------------------------

    def __rich__(self):  # pragma: no cover - display only
        return self.config.__rich__() if hasattr(self.config, "__rich__") else repr(self.config)

    def __str__(self) -> str:
        return self.to_yaml(redact_secrets=True)


__all__ = [
    "ConfigService",
    "CONFIG_FILE_ENV",
    "DEFAULT_CONFIG_FILENAME",
    "REMOTE_CONFIG_PATH",
    "ENV_PREFIX",
    "ENV_NESTED_DELIMITER",
]
