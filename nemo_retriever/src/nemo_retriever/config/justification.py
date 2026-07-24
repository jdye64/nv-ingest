# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Impact tags for configuration knobs — used by ``@configured`` and generated docs."""

from __future__ import annotations

from enum import Enum
from typing import Any, Sequence

from pydantic import Field
from pydantic.fields import FieldInfo


class ConfigJustification(str, Enum):
    """Why a configuration exists and what changing it affects."""

    PERFORMANCE = "performance"
    ACCURACY = "accuracy"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    MEMORY = "memory"
    COST = "cost"
    RELIABILITY = "reliability"
    SECURITY = "security"
    COMPATIBILITY = "compatibility"


def normalize_justifications(
    value: ConfigJustification | Sequence[ConfigJustification] | None,
) -> tuple[ConfigJustification, ...]:
    """Return a normalized tuple of justification tags."""
    if value is None:
        return ()
    if isinstance(value, ConfigJustification):
        return (value,)
    return tuple(value)


def justified_field(
    default: Any = ...,
    *,
    justification: ConfigJustification | Sequence[ConfigJustification] | None = None,
    rationale: str = "",
    **field_kwargs: Any,
) -> Any:
    """Like :func:`pydantic.Field` but attaches impact metadata for docs generation."""
    extra = dict(field_kwargs.pop("json_schema_extra", None) or {})
    tags = normalize_justifications(justification)
    if tags:
        extra["justification"] = [tag.value for tag in tags]
    if rationale:
        extra["rationale"] = rationale
    if extra:
        field_kwargs["json_schema_extra"] = extra
    if default is ...:
        return Field(**field_kwargs)
    return Field(default, **field_kwargs)


def field_justification_metadata(field_info: FieldInfo) -> tuple[tuple[ConfigJustification, ...], str]:
    """Extract justification tags and rationale from a Pydantic field."""
    extra = field_info.json_schema_extra or {}
    if not isinstance(extra, dict):
        return (), ""
    raw_tags = extra.get("justification", ())
    tags: list[ConfigJustification] = []
    for item in raw_tags if isinstance(raw_tags, (list, tuple)) else (raw_tags,):
        if isinstance(item, ConfigJustification):
            tags.append(item)
        elif isinstance(item, str):
            tags.append(ConfigJustification(item))
    rationale = extra.get("rationale", "")
    return tuple(tags), str(rationale)
