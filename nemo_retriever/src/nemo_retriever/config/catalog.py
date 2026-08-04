# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Introspection: turn a Pydantic config model into a documentable catalog.

The catalog walks a model recursively, and for every leaf (scalar) field records
its dotted path, the environment variable that overrides it, its type, default,
description, and resolved :class:`~nemo_retriever.config.categories.ConfigMeta`.

This single data structure powers both the docs generator and programmatic
introspection (``retriever config schema``).
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from nemo_retriever.config.categories import (
    ConfigCategory,
    ConfigMeta,
    get_field_meta,
    get_section_meta,
)

# Sensitive field names whose defaults must never be rendered in docs/dumps.
_SECRET_FIELD_HINTS = ("api_key", "api_token", "password", "secret", "auth_token")


def _is_model(annotation: Any) -> type[BaseModel] | None:
    """Return the BaseModel subclass for a field annotation, unwrapping Optionals."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for arg in typing.get_args(annotation):
        if isinstance(arg, type) and issubclass(arg, BaseModel):
            return arg
    return None


def _type_label(annotation: Any) -> str:
    """Render a readable type label for an annotation."""
    if annotation is None:
        return "None"
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def _is_secret(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in _SECRET_FIELD_HINTS)


@dataclass
class FieldDoc:
    """A single documentable leaf configuration field."""

    path: str
    env_var: str
    type_label: str
    default: Any
    required: bool
    description: str | None
    meta: ConfigMeta
    secret: bool = False

    @property
    def categories(self) -> tuple[ConfigCategory, ...]:
        return self.meta.categories

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "env_var": self.env_var,
            "type": self.type_label,
            "default": "***" if self.secret and self.default not in (None, "") else self.default,
            "required": self.required,
            "description": self.description,
            "categories": [c.value for c in self.meta.categories],
            "impact": self.meta.impact,
            "tuning_hint": self.meta.tuning_hint,
            "stability": self.meta.stability,
            "secret": self.secret,
        }


@dataclass
class SectionDoc:
    """A config model rendered as a titled section of fields."""

    model: type[BaseModel]
    path: str
    title: str
    description: str | None
    default_category: ConfigCategory
    fields: list[FieldDoc] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "title": self.title,
            "description": self.description,
            "default_category": self.default_category.value,
            "fields": [f.to_dict() for f in self.fields],
        }


def _section_title(model: type[BaseModel]) -> str:
    section = get_section_meta(model)
    if section and section.title:
        return section.title
    name = model.__name__
    for suffix in ("Config", "Params"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    # CamelCase -> spaced words
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i and (name[i - 1].islower() or (i + 1 < len(name) and name[i + 1].islower())):
            out.append(" ")
        out.append(ch)
    return "".join(out).strip()


def _section_description(model: type[BaseModel]) -> str | None:
    section = get_section_meta(model)
    if section and section.description:
        return section.description
    doc = (model.__doc__ or "").strip()
    return doc or None


def build_catalog(
    model: type[BaseModel],
    *,
    env_prefix: str = "NEMO_RETRIEVER_",
    env_delimiter: str = "__",
) -> list[SectionDoc]:
    """Walk ``model`` recursively and return a flat, ordered list of sections.

    Each nested BaseModel becomes its own :class:`SectionDoc`; scalar leaves are
    grouped under the nearest enclosing model. Environment variable names are
    derived from the dotted path using ``env_prefix`` + ``env_delimiter`` to
    match the pydantic-settings loader.
    """
    sections: list[SectionDoc] = []
    _visit(
        model,
        path_parts=[],
        env_parts=[],
        env_prefix=env_prefix,
        env_delimiter=env_delimiter,
        sections=sections,
        seen=set(),
    )
    return sections


def _visit(
    model: type[BaseModel],
    *,
    path_parts: list[str],
    env_parts: list[str],
    env_prefix: str,
    env_delimiter: str,
    sections: list[SectionDoc],
    seen: set[type],
) -> None:
    if model in seen:
        return
    seen = seen | {model}

    section = SectionDoc(
        model=model,
        path=".".join(path_parts) or "(root)",
        title=_section_title(model),
        description=_section_description(model),
        default_category=(get_section_meta(model).category if get_section_meta(model) else ConfigCategory.GENERAL),
    )
    sections.append(section)

    nested: list[tuple[str, type[BaseModel], FieldInfo]] = []

    for field_name, field_info in model.model_fields.items():
        annotation = field_info.annotation
        submodel = _is_model(annotation)
        child_path = path_parts + [field_name]
        child_env = env_parts + [field_name.upper()]

        if submodel is not None:
            nested.append((field_name, submodel, field_info))
            continue

        meta = get_field_meta(model, field_name, field_info)
        default = _resolve_default(field_info)
        secret = _is_secret(field_name)
        env_var = env_prefix + env_delimiter.join(child_env)

        section.fields.append(
            FieldDoc(
                path=".".join(child_path),
                env_var=env_var,
                type_label=_type_label(annotation),
                default=default,
                required=field_info.is_required(),
                description=field_info.description,
                meta=meta,
                secret=secret,
            )
        )

    for field_name, submodel, _field_info in nested:
        _visit(
            submodel,
            path_parts=path_parts + [field_name],
            env_parts=env_parts + [field_name.upper()],
            env_prefix=env_prefix,
            env_delimiter=env_delimiter,
            sections=sections,
            seen=seen,
        )


def _resolve_default(field_info: FieldInfo) -> Any:
    """Best-effort scalar default for display (factories rendered lazily)."""
    from pydantic_core import PydanticUndefined

    if field_info.default is not PydanticUndefined and field_info.default is not None:
        return field_info.default
    if field_info.default_factory is not None:
        try:
            return field_info.default_factory()  # type: ignore[call-arg]
        except Exception:
            return None
    if field_info.default is None:
        return None
    return None


def iter_fields(sections: list[SectionDoc]):
    """Yield every :class:`FieldDoc` across all sections."""
    for section in sections:
        yield from section.fields


def fields_by_category(sections: list[SectionDoc]) -> dict[ConfigCategory, list[FieldDoc]]:
    """Group all fields by category (a field may appear under several)."""
    grouped: dict[ConfigCategory, list[FieldDoc]] = {c: [] for c in ConfigCategory}
    for field_doc in iter_fields(sections):
        for category in field_doc.categories:
            grouped[category].append(field_doc)
    return grouped


__all__ = [
    "FieldDoc",
    "SectionDoc",
    "build_catalog",
    "iter_fields",
    "fields_by_category",
]
