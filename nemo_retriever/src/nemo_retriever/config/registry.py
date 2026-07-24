# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry of ``@configured`` sites for introspection and documentation generation."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from nemo_retriever.common.schemas.base import RichModel
from nemo_retriever.config.justification import ConfigJustification, normalize_justifications

_CONFIGURED_REGISTRY: list["ConfiguredEntry"] = []


class ConfiguredEntry(RichModel):
    """Metadata for a single ``@configured`` function."""

    qualname: str
    section: str
    model_name: str
    param_name: str | None = None
    variants: tuple[str, ...] = Field(default_factory=tuple)
    justification: tuple[ConfigJustification, ...] = Field(default_factory=tuple)
    rationale: str = ""
    run_modes: tuple[str, ...] = Field(default_factory=tuple)
    doc: str | None = None


def register_configured(entry: ConfiguredEntry) -> None:
    """Append *entry* to the global registry (deduplicated by qualname)."""
    for existing in _CONFIGURED_REGISTRY:
        if existing.qualname == entry.qualname:
            return
    _CONFIGURED_REGISTRY.append(entry)


def list_configured() -> list[ConfiguredEntry]:
    """Return all registered ``@configured`` sites."""
    return list(_CONFIGURED_REGISTRY)


def clear_configured_registry() -> None:
    """Clear the registry — test helper only."""
    _CONFIGURED_REGISTRY.clear()


def build_configured_entry(
    *,
    fn: Any,
    section: str,
    model: type[Any],
    param_name: str | None,
    variants: tuple[str, ...],
    justification: ConfigJustification | tuple[ConfigJustification, ...],
    rationale: str,
    run_modes: tuple[str, ...],
) -> ConfiguredEntry:
    """Construct a :class:`ConfiguredEntry` from decorator arguments."""
    return ConfiguredEntry(
        qualname=f"{fn.__module__}.{fn.__qualname__}",
        section=section,
        model_name=model.__name__,
        param_name=param_name,
        variants=variants,
        justification=normalize_justifications(justification),
        rationale=rationale,
        run_modes=run_modes,
        doc=(fn.__doc__ or "").strip() or None,
    )
