# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_retriever.search.models.responses import AgentHit, HitExportFormat
from nemo_retriever.search.services.hits import agent_hit_to_json


def export_hit_text(hit: AgentHit) -> str:
    return hit.text or ""


def export_hit_summary(hit: AgentHit) -> str:
    lines = [
        f"rank: {hit.rank}",
        f"source: {hit.source or hit.provenance.source_id or '—'}",
        f"page_number: {hit.page_number if hit.page_number is not None else '—'}",
        f"content_type: {hit.content_type or '—'}",
        f"distance: {hit.distance if hit.distance is not None else '—'}",
        f"pdf_page: {hit.provenance.pdf_page or '—'}",
        "",
        hit.text or "",
    ]
    return "\n".join(lines)


def export_hit(hit: AgentHit, fmt: HitExportFormat) -> tuple[str, str]:
    """Return (body, media_type) for the requested export format."""
    if fmt == "text":
        return export_hit_text(hit), "text/plain; charset=utf-8"
    if fmt == "summary":
        return export_hit_summary(hit), "text/plain; charset=utf-8"
    return agent_hit_to_json(hit), "application/json"
