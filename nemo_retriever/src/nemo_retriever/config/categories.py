# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-labeling configuration primitives.

This module provides the vocabulary used to make NeMo Retriever configuration
*self documenting*.  Every tunable can be tagged with:

* one or more :class:`ConfigCategory` values describing **what** it changes
  (model selection, performance, throughput, accuracy, ...);
* a human-readable ``impact`` string describing **how** changing it affects the
  system; and
* an optional ``tuning_hint`` giving operators a starting point.

Two entry points attach that metadata:

* :func:`config_field` — a drop-in replacement for :func:`pydantic.Field` that
  records the metadata in ``json_schema_extra`` so it survives into JSON Schema
  and OpenAPI.
* :func:`config_section` — a class decorator that tags a whole Pydantic model,
  supplies a default category for its fields, and registers it so the docs
  generator can discover it.

For models that are defined elsewhere (e.g. the existing ``ServiceConfig`` and
``*Params`` classes) and that we do not want to rewrite yet,
:func:`annotate_fields` attaches the same metadata externally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

from pydantic import Field
from pydantic.fields import FieldInfo

#: Key under which per-field config metadata is stored in ``json_schema_extra``.
METADATA_KEY = "nemo_config"

#: Attribute set on a class by :func:`config_section`.
SECTION_ATTR = "__nemo_config_section__"


class ConfigCategory(str, Enum):
    """What a configuration value actually changes about the system.

    The categories are intentionally coarse so that operators can reason about
    trade-offs ("if I turn this up, do I trade accuracy for throughput?")
    without needing to understand every knob individually.
    """

    MODEL_SELECTION = "model_selection"
    """Which model/endpoint is used (SKU, provider, local vs remote NIM)."""

    PERFORMANCE = "performance"
    """Single-request latency and resource footprint (GPU memory, warmup)."""

    THROUGHPUT = "throughput"
    """Aggregate work per unit time (batch sizes, worker/queue sizing, concurrency)."""

    ACCURACY = "accuracy"
    """Quality of extraction / retrieval results (chunking, dedup, hybrid search)."""

    GENERAL = "general"
    """Operational plumbing with no direct quality/perf trade-off (paths, logging, topology)."""

    SECURITY = "security"
    """Trust boundary controls (auth, egress allowlists, override policy)."""

    @property
    def title(self) -> str:
        """Human-friendly title used in generated docs."""
        return _CATEGORY_TITLES[self]

    @property
    def summary(self) -> str:
        """One-line description used in generated docs."""
        return _CATEGORY_SUMMARIES[self]


_CATEGORY_TITLES: dict[ConfigCategory, str] = {
    ConfigCategory.MODEL_SELECTION: "Model Selection",
    ConfigCategory.PERFORMANCE: "Performance",
    ConfigCategory.THROUGHPUT: "Throughput",
    ConfigCategory.ACCURACY: "Accuracy",
    ConfigCategory.GENERAL: "General",
    ConfigCategory.SECURITY: "Security",
}

_CATEGORY_SUMMARIES: dict[ConfigCategory, str] = {
    ConfigCategory.MODEL_SELECTION: (
        "Choose which models and endpoints power each pipeline stage — local "
        "Hugging Face weights, remote NIM microservices, or hosted APIs."
    ),
    ConfigCategory.PERFORMANCE: (
        "Tune single-request latency and per-worker resource use, such as GPU "
        "memory utilization and model warmup."
    ),
    ConfigCategory.THROUGHPUT: (
        "Scale aggregate work per unit time through batch sizes, worker and "
        "queue sizing, and concurrency limits."
    ),
    ConfigCategory.ACCURACY: (
        "Control the quality of extracted content and retrieval results — "
        "chunking, deduplication, and hybrid search behaviour."
    ),
    ConfigCategory.GENERAL: (
        "Operational plumbing that does not directly trade off quality or "
        "performance: server topology, logging, and filesystem paths."
    ),
    ConfigCategory.SECURITY: (
        "Enforce trust boundaries: authentication, egress allowlists, and how "
        "permissively per-request pipeline overrides are accepted."
    ),
}


def _coerce_categories(
    categories: ConfigCategory | str | Sequence[ConfigCategory | str],
) -> tuple[ConfigCategory, ...]:
    if isinstance(categories, (ConfigCategory, str)):
        items: Iterable[ConfigCategory | str] = [categories]
    else:
        items = categories
    out: list[ConfigCategory] = []
    for item in items:
        out.append(item if isinstance(item, ConfigCategory) else ConfigCategory(item))
    if not out:
        raise ValueError("at least one ConfigCategory is required")
    return tuple(out)


@dataclass(frozen=True)
class ConfigMeta:
    """Structured metadata describing a single configuration value."""

    categories: tuple[ConfigCategory, ...]
    impact: str
    tuning_hint: str | None = None
    stability: str = "stable"
    since: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "categories": [c.value for c in self.categories],
            "impact": self.impact,
            "tuning_hint": self.tuning_hint,
            "stability": self.stability,
            "since": self.since,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ConfigMeta":
        return cls(
            categories=_coerce_categories(raw.get("categories") or [ConfigCategory.GENERAL]),
            impact=raw.get("impact", ""),
            tuning_hint=raw.get("tuning_hint"),
            stability=raw.get("stability", "stable"),
            since=raw.get("since"),
        )


@dataclass(frozen=True)
class SectionMeta:
    """Metadata attached to a whole config model by :func:`config_section`."""

    category: ConfigCategory
    title: str | None = None
    description: str | None = None
    group: str | None = None


def config_field(
    default: Any = ...,
    *,
    category: ConfigCategory | str | Sequence[ConfigCategory | str],
    impact: str,
    tuning_hint: str | None = None,
    stability: str = "stable",
    since: str | None = None,
    **field_kwargs: Any,
) -> Any:
    """Return a Pydantic :func:`Field` carrying config category metadata.

    Use this exactly like :func:`pydantic.Field`, adding ``category`` and
    ``impact``::

        max_tokens: int = config_field(
            1024,
            category=ConfigCategory.ACCURACY,
            impact="Larger chunks preserve more context per embedding but "
                   "reduce retrieval granularity.",
            tuning_hint="Start at 1024; lower to 512 for dense fact lookup.",
            ge=1,
        )

    The metadata is stored under ``json_schema_extra['nemo_config']`` so it is
    visible in the model's JSON Schema, OpenAPI, and to the docs generator.
    """
    meta = ConfigMeta(
        categories=_coerce_categories(category),
        impact=impact,
        tuning_hint=tuning_hint,
        stability=stability,
        since=since,
    )

    extra = field_kwargs.pop("json_schema_extra", None) or {}
    if not isinstance(extra, dict):
        raise TypeError("config_field only supports dict json_schema_extra")
    extra = {**extra, METADATA_KEY: meta.to_dict()}

    return Field(default, json_schema_extra=extra, **field_kwargs)


def config_section(
    *,
    category: ConfigCategory | str,
    title: str | None = None,
    description: str | None = None,
    group: str | None = None,
    register: bool = True,
):
    """Class decorator tagging a Pydantic model as a config section.

    * ``category`` becomes the default category for any field in the model that
      was not declared with :func:`config_field`.
    * ``title``/``description`` override the values shown in generated docs
      (defaults derive from the class name and docstring).
    * When ``register`` is true (default) the class is added to the global
      registry so the docs generator discovers it automatically.
    """

    resolved_category = category if isinstance(category, ConfigCategory) else ConfigCategory(category)

    def _decorator(cls):
        meta = SectionMeta(
            category=resolved_category,
            title=title,
            description=description,
            group=group,
        )
        setattr(cls, SECTION_ATTR, meta)
        if register:
            register_section(cls)
        return cls

    return _decorator


# ---------------------------------------------------------------------------
# Registry + external annotation
# ---------------------------------------------------------------------------

_SECTION_REGISTRY: list[type] = []

# Externally attached field metadata for models we do not own:
# {model_class: {field_name: ConfigMeta}}
_EXTERNAL_FIELD_META: dict[type, dict[str, ConfigMeta]] = {}

# Externally attached section metadata for models we do not own.
_EXTERNAL_SECTION_META: dict[type, SectionMeta] = {}


def register_section(cls: type) -> type:
    """Register a config model so the docs generator can discover it."""
    if cls not in _SECTION_REGISTRY:
        _SECTION_REGISTRY.append(cls)
    return cls


def registered_sections() -> list[type]:
    """Return all explicitly registered config model classes (in order)."""
    return list(_SECTION_REGISTRY)


def annotate_fields(
    model: type,
    mapping: dict[str, ConfigMeta | dict[str, Any]],
    *,
    section: SectionMeta | None = None,
) -> None:
    """Attach category metadata to a model that we do not want to edit.

    This lets the existing ``ServiceConfig`` sub-sections and ``*Params``
    classes participate in the self-documenting system without rewriting them.
    """
    resolved: dict[str, ConfigMeta] = {}
    for field_name, value in mapping.items():
        resolved[field_name] = value if isinstance(value, ConfigMeta) else ConfigMeta.from_dict(value)
    _EXTERNAL_FIELD_META.setdefault(model, {}).update(resolved)
    if section is not None:
        _EXTERNAL_SECTION_META[model] = section


def get_section_meta(model: type) -> SectionMeta | None:
    """Return section metadata for a model (decorator or external)."""
    meta = getattr(model, SECTION_ATTR, None)
    if isinstance(meta, SectionMeta):
        return meta
    return _EXTERNAL_SECTION_META.get(model)


def get_field_meta(model: type, field_name: str, field_info: FieldInfo | None = None) -> ConfigMeta:
    """Resolve metadata for a single field.

    Resolution order: inline :func:`config_field` metadata → external
    annotation → the owning section's default category.
    """
    if field_info is not None:
        extra = field_info.json_schema_extra
        if isinstance(extra, dict) and METADATA_KEY in extra:
            return ConfigMeta.from_dict(extra[METADATA_KEY])  # type: ignore[arg-type]

    external = _EXTERNAL_FIELD_META.get(model, {})
    if field_name in external:
        return external[field_name]

    section = get_section_meta(model)
    default_category = section.category if section else ConfigCategory.GENERAL
    return ConfigMeta(categories=(default_category,), impact="")


def _reset_registry_for_tests() -> None:
    """Clear registry state — used only by the test suite."""
    _SECTION_REGISTRY.clear()
    _EXTERNAL_FIELD_META.clear()
    _EXTERNAL_SECTION_META.clear()


__all__ = [
    "ConfigCategory",
    "ConfigMeta",
    "SectionMeta",
    "METADATA_KEY",
    "SECTION_ATTR",
    "config_field",
    "config_section",
    "register_section",
    "registered_sections",
    "annotate_fields",
    "get_section_meta",
    "get_field_meta",
]
