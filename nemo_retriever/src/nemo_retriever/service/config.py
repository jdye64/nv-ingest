# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service-mode configuration backed by ``retriever-service.yaml``."""

from __future__ import annotations

from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "0.0.0.0"
    port: int = 8000


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str = "INFO"
    file: str = "retriever-service.log"
    format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


class DatabaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = "retriever-service.db"


class ProcessingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_pool_size: int = 32


class ResourceLimitsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_memory_mb: int | None = None
    max_cpu_cores: int | None = None
    gpu_devices: list[str] = Field(default_factory=list)


class ServiceConfig(BaseModel):
    """Top-level configuration for the retriever service mode.

    Every section has sensible defaults so a zero-config launch works out of
    the box.  Values can be overridden per-field from CLI flags.
    """

    model_config = ConfigDict(extra="forbid")

    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    resources: ResourceLimitsConfig = Field(default_factory=ResourceLimitsConfig)


def _bundled_yaml_path() -> Path:
    """Return the path to the default ``retriever-service.yaml`` shipped with the package."""
    ref = importlib_resources.files("nemo_retriever.service") / "retriever-service.yaml"
    return Path(str(ref))


def _discover_config_path(explicit: str | None = None) -> Path | None:
    """Locate a config file using the standard precedence rules.

    1. *explicit* path supplied via ``--config``
    2. ``./retriever-service.yaml`` in the current working directory
    3. Bundled default inside the package
    """
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")
        return p

    cwd_candidate = Path.cwd() / "retriever-service.yaml"
    if cwd_candidate.is_file():
        return cwd_candidate

    bundled = _bundled_yaml_path()
    if bundled.is_file():
        return bundled

    return None


def load_config(
    config_path: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> ServiceConfig:
    """Load a :class:`ServiceConfig` from YAML with optional CLI overrides.

    Parameters
    ----------
    config_path
        Explicit path to a YAML config file.  ``None`` triggers auto-discovery.
    overrides
        Flat ``section.key`` overrides (e.g. ``{"server.port": 9000}``).
    """
    path = _discover_config_path(config_path)
    if path is not None:
        raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    else:
        raw = {}

    if overrides:
        for dotted_key, value in overrides.items():
            if value is None:
                continue
            parts = dotted_key.split(".")
            target = raw
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value

    return ServiceConfig(**raw)
