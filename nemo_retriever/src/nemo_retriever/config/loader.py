# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and save ``RetrieverServiceConfig`` from YAML."""

from __future__ import annotations

from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from nemo_retriever.common.schemas.base import RichModel
from nemo_retriever.config.context import set_config
from nemo_retriever.config.models import RetrieverServiceConfig

_REDACTED_FIELDS = frozenset({"api_key", "api_token", "password", "secret"})

_CONFIG_FILENAMES = ("retriever-config.yaml", "retriever-service.yaml")


def _bundled_yaml_paths() -> tuple[Path, ...]:
    """Bundled defaults shipped with the package (config package first)."""
    paths: list[Path] = []
    for package in ("nemo_retriever.config", "nemo_retriever.service"):
        ref = importlib_resources.files(package)
        for name in _CONFIG_FILENAMES:
            candidate = ref / name
            if candidate.is_file():
                paths.append(Path(str(candidate)))
    return tuple(paths)


def discover_config_path(explicit: str | None = None) -> Path | None:
    """Locate a config file using standard precedence rules.

    1. *explicit* path from ``--config``
    2. ``./retriever-config.yaml`` then ``./retriever-service.yaml`` in CWD
    3. Bundled default inside the package
    """
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        return path

    for name in _CONFIG_FILENAMES:
        cwd_candidate = Path.cwd() / name
        if cwd_candidate.is_file():
            return cwd_candidate

    bundled = _bundled_yaml_paths()
    if bundled:
        return bundled[0]

    return None


def _apply_overrides(raw: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    if not overrides:
        return raw
    for dotted_key, value in overrides.items():
        if value is None:
            continue
        parts = dotted_key.split(".")
        target = raw
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return raw


def _redact_value(field_name: str, field_value: Any) -> str:
    if field_name in _REDACTED_FIELDS and field_value:
        return "****"
    return repr(field_value)


def print_config_tree(config: RetrieverServiceConfig, *, source: str | None = None) -> None:
    """Print a redacted Rich tree of *config* to stderr."""
    from rich.console import Console
    from rich.tree import Tree

    console = Console(stderr=True)
    tree = Tree(f"[bold]RetrieverServiceConfig[/bold]  (source: {source or 'defaults'})")
    for section_name, section_value in config:
        if isinstance(section_value, RichModel):
            branch = tree.add(f"[cyan]{section_name}[/cyan]")
            for field_name, field_value in section_value:
                display = _redact_value(field_name, field_value)
                branch.add(f"[dim]{field_name}[/dim] = [white]{display}[/white]")
        else:
            tree.add(f"[cyan]{section_name}[/cyan] = [white]{section_value!r}[/white]")
    console.print(tree)


def load_config(
    config_path: str | None = None,
    overrides: dict[str, Any] | None = None,
    *,
    set_active: bool = True,
    print_tree: bool = True,
) -> RetrieverServiceConfig:
    """Load a :class:`RetrieverServiceConfig` from YAML with optional overrides."""
    path = discover_config_path(config_path)
    if path is not None:
        raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    else:
        raw = {}

    raw = _apply_overrides(raw, overrides)
    config = RetrieverServiceConfig(**raw)

    if print_tree:
        print_config_tree(config, source=str(path) if path else "defaults")

    if set_active:
        set_config(config)

    return config


def _redact_model(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        redacted: dict[str, Any] = {}
        for name in type(obj).model_fields:
            value = getattr(obj, name)
            if name in _REDACTED_FIELDS and value:
                redacted[name] = "****"
            else:
                redacted[name] = _redact_model(value)
        return redacted
    if isinstance(obj, dict):
        return {key: _redact_model(val) for key, val in obj.items()}
    if isinstance(obj, list):
        return [_redact_model(item) for item in obj]
    return obj


def save_config(
    path: str | Path,
    config: RetrieverServiceConfig | None = None,
    *,
    redact_secrets: bool = True,
) -> Path:
    """Serialize the active (or supplied) config to YAML."""
    if config is None:
        from nemo_retriever.config.context import get_config

        config = get_config()

    payload = config.model_dump(mode="python")
    if redact_secrets:
        payload = _redact_model(payload)

    out = Path(path)
    out.write_text(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False))
    return out


def print_config() -> RetrieverServiceConfig:
    """Print and return the active configuration."""
    from nemo_retriever.config.context import get_config

    config = get_config()
    print_config_tree(config, source="active")
    return config


def get_config_dump(*, redacted: bool = True) -> dict[str, Any]:
    """Return the active configuration as a plain dict."""
    from nemo_retriever.config.context import get_config

    config = get_config()
    payload = config.model_dump(mode="python")
    if redacted:
        payload = _redact_model(payload)
    return payload
