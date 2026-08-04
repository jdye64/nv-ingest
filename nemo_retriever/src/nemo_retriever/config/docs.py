# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a single, extensive configuration reference page.

The generator walks :class:`~nemo_retriever.config.schema.NeMoRetrieverConfig`
via the :mod:`~nemo_retriever.config.catalog` introspection layer and emits a
self-contained Fern MDX page. Because the metadata (categories, impact, tuning
hints) travels with the fields, the docs never drift from the code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nemo_retriever.config.catalog import (
    FieldDoc,
    SectionDoc,
    build_catalog,
    fields_by_category,
)
from nemo_retriever.config.categories import ConfigCategory

DocFormat = str  # "fern" | "markdown"


def _fmt_default(field: FieldDoc) -> str:
    if field.secret and field.default not in (None, ""):
        return "`***`"
    default = field.default
    if default is None:
        return "`null`" if not field.required else "_required_"
    if isinstance(default, str) and default == "":
        return '`""`'
    if isinstance(default, (list, dict)) and not default:
        return "`[]`" if isinstance(default, list) else "`{}`"
    text = str(default)
    if len(text) > 60:
        text = text[:57] + "..."
    return f"`{text}`"


def _escape_cell(text: str | None) -> str:
    if not text:
        return ""
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _category_badges(field: FieldDoc) -> str:
    return " ".join(f"`{c.value}`" for c in field.categories)


def _field_rows(fields: list[FieldDoc]) -> list[str]:
    rows = ["| Setting | Env var | Type | Default | Impact |", "| --- | --- | --- | --- | --- |"]
    for f in fields:
        impact = _escape_cell(f.meta.impact or f.description)
        if f.meta.tuning_hint:
            impact = f"{impact} _Hint: {_escape_cell(f.meta.tuning_hint)}_" if impact else _escape_cell(
                f.meta.tuning_hint
            )
        rows.append(
            f"| `{f.path}` | `{f.env_var}` | {f.type_label} | {_fmt_default(f)} | {impact} |"
        )
    return rows


def generate_markdown(
    sections: list[SectionDoc] | None = None,
    *,
    fmt: DocFormat = "fern",
    title: str = "Configuration Reference",
    subtitle: str | None = None,
) -> str:
    """Render the full configuration reference as a single page.

    ``fmt='fern'`` emits MDX with Fern frontmatter and callout components;
    ``fmt='markdown'`` emits plain GitHub-flavoured Markdown.
    """
    if sections is None:
        from nemo_retriever.config.schema import NeMoRetrieverConfig

        sections = build_catalog(NeMoRetrieverConfig)

    subtitle = subtitle or (
        "Every NeMo Retriever setting in one place, grouped by what it changes. "
        "This page is generated from the code — categories, defaults, and env "
        "vars stay in sync automatically."
    )

    lines: list[str] = []
    is_fern = fmt == "fern"

    if is_fern:
        lines += ["---", f"title: {title}", f"subtitle: {subtitle}", "---", ""]
    else:
        lines += [f"# {title}", "", f"> {subtitle}", ""]

    lines += _render_intro(sections, is_fern)
    lines += _render_by_category(sections, is_fern)
    lines += _render_reference(sections, is_fern)

    return "\n".join(lines).rstrip() + "\n"


def _render_intro(sections: list[SectionDoc], is_fern: bool) -> list[str]:
    total_fields = sum(len(s.fields) for s in sections)
    grouped = fields_by_category(sections)
    lines: list[str] = ["## Overview", ""]
    lines.append(
        f"NeMo Retriever exposes **{total_fields}** documented settings across "
        f"**{len([s for s in sections if s.fields])}** sections. Each setting is "
        "tagged with one or more categories describing what it changes:"
    )
    lines.append("")
    if is_fern:
        lines.append("<CardGroup cols={2}>")
        for category in ConfigCategory:
            count = len(grouped.get(category, []))
            lines.append(
                f'  <Card title="{category.title}" icon="gear">'
            )
            lines.append(f"    {category.summary} ({count} settings)")
            lines.append("  </Card>")
        lines.append("</CardGroup>")
    else:
        for category in ConfigCategory:
            count = len(grouped.get(category, []))
            lines.append(f"- **{category.title}** (`{category.value}`, {count}) — {category.summary}")
    lines.append("")
    lines.append(
        "Settings can be provided (in ascending precedence) via defaults, a "
        "config file, environment variables, a remote cluster endpoint, or "
        "explicit overrides. Environment variable names are shown for every "
        "setting."
    )
    lines.append("")
    return lines


def _render_by_category(sections: list[SectionDoc], is_fern: bool) -> list[str]:
    grouped = fields_by_category(sections)
    lines: list[str] = ["## Settings by category", ""]
    for category in ConfigCategory:
        fields = grouped.get(category, [])
        if not fields:
            continue
        lines.append(f"### {category.title}")
        lines.append("")
        lines.append(category.summary)
        lines.append("")
        if is_fern:
            lines.append(f'<Callout intent="info">Category tag: <code>{category.value}</code></Callout>')
            lines.append("")
        lines += _field_rows(fields)
        lines.append("")
    return lines


def _render_reference(sections: list[SectionDoc], is_fern: bool) -> list[str]:
    lines: list[str] = ["## Full reference by section", ""]
    lines.append(
        "The same settings organized by the structure of the configuration "
        "object. Nested keys use dotted paths (e.g. `service.server.port`)."
    )
    lines.append("")
    for section in sections:
        if not section.fields:
            continue
        heading = section.title
        lines.append(f"### {heading}")
        lines.append("")
        if section.path != "(root)":
            lines.append(f"Config path: `{section.path}`")
            lines.append("")
        if section.description:
            lines.append(section.description)
            lines.append("")
        lines += _field_rows(section.fields)
        lines.append("")
    return lines


def generate_config_docs(
    output_path: str | Path,
    *,
    fmt: DocFormat = "fern",
    title: str = "Configuration Reference",
) -> Path:
    """Write the configuration reference page to ``output_path``."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(generate_markdown(fmt=fmt, title=title))
    return out


def catalog_as_dict() -> dict[str, Any]:
    """Return the full introspected catalog as JSON-serializable data."""
    from nemo_retriever.config.schema import NeMoRetrieverConfig

    sections = build_catalog(NeMoRetrieverConfig)
    return {"sections": [s.to_dict() for s in sections]}


__all__ = ["generate_markdown", "generate_config_docs", "catalog_as_dict"]
