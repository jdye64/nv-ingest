# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public agent-memory wire models shared by the service and the Python SDK.

Clients describe what they want to recall with :class:`MemoryFilter` and never
send SQL. :func:`compile_memory_filter` turns a validated filter into a
predicate against the typed columns declared in
:mod:`nemo_retriever.common.vdb.memory_schema`. Keeping compilation on this
side of the boundary means a caller cannot reach columns, tables, or
expressions that the filter model does not name.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import Field, field_validator, model_validator

from nemo_retriever.common.schemas.base import RichModel

MemoryTypeName = Literal["episodic", "semantic", "procedural"]
MemoryRoleName = Literal["user", "assistant", "tool", "system"]
MemoryEventTypeName = Literal["message", "tool_call", "observation", "fact", "summary", "document"]
RecallStrategy = Literal["hybrid", "dense"]

#: Columns a compiled filter is allowed to reference. Anything outside this
#: set is unreachable from client input by construction.
_FILTERABLE_COLUMNS: frozenset[str] = frozenset(
    {
        "memory_id",
        "memory_type",
        "namespace",
        "session_id",
        "agent_id",
        "user_id",
        "role",
        "event_type",
        "occurred_at",
        "importance",
        "confidence",
        "tags",
        "valid_to",
    }
)


def _sql_string(value: str) -> str:
    """Return a single-quoted SQL string literal with quotes escaped."""
    return "'" + str(value).replace("'", "''") + "'"


def _sql_timestamp(value: datetime) -> str:
    moment = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return _sql_string(moment.astimezone(timezone.utc).isoformat())


def _column(name: str) -> str:
    if name not in _FILTERABLE_COLUMNS:
        raise ValueError(f"{name!r} is not a filterable memory column")
    return name


class MemoryRecord(RichModel):
    """One memory to write, or one already written.

    ``occurred_at`` is when the event happened, which is not always when it was
    ingested. Capture layers that replay a transcript should set it explicitly
    so the timeline reflects the conversation rather than the import.
    """

    text: str = Field(min_length=1)
    memory_id: str | None = None
    memory_type: MemoryTypeName = "episodic"
    namespace: str = Field(default="default", min_length=1, max_length=128)
    session_id: str | None = Field(default=None, max_length=128)
    agent_id: str | None = Field(default=None, max_length=128)
    user_id: str | None = Field(default=None, max_length=128)
    role: MemoryRoleName | None = None
    event_type: MemoryEventTypeName | None = None
    occurred_at: datetime | None = None
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    turn_index: int | None = Field(default=None, ge=0)
    source_memory_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def _reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("memory text must not be blank")
        return value


class MemoryHit(RichModel):
    """One recalled memory with its retrieval score."""

    memory_id: str
    text: str
    memory_type: str
    namespace: str = "default"
    session_id: str | None = None
    agent_id: str | None = None
    user_id: str | None = None
    role: str | None = None
    event_type: str | None = None
    occurred_at: str | None = None
    ingested_at: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    importance: float = 0.5
    confidence: float = 1.0
    turn_index: int | None = None
    source_memory_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float | None = Field(
        default=None,
        description="Final ranking score after any recency weighting; higher is better.",
    )
    distance: float | None = Field(
        default=None,
        description="Native dense-vector distance; lower values are more similar.",
    )


class MemoryFilter(RichModel):
    """Structured recall filter compiled server-side into a predicate."""

    namespace: str | None = Field(default=None, max_length=128)
    session_id: str | None = Field(default=None, max_length=128)
    agent_id: str | None = Field(default=None, max_length=128)
    user_id: str | None = Field(default=None, max_length=128)
    memory_type: MemoryTypeName | None = None
    memory_ids: list[str] = Field(default_factory=list)
    role: MemoryRoleName | None = None
    event_type: MemoryEventTypeName | None = None
    occurred_after: datetime | None = None
    occurred_before: datetime | None = None
    tags_any: list[str] = Field(default_factory=list)
    min_importance: float | None = Field(default=None, ge=0.0, le=1.0)
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    include_superseded: bool = False

    @model_validator(mode="after")
    def _validate_window(self) -> "MemoryFilter":
        if self.occurred_after is not None and self.occurred_before is not None:
            if self.occurred_after > self.occurred_before:
                raise ValueError("occurred_after must be earlier than occurred_before")
        return self

    @field_validator("memory_ids", "tags_any")
    @classmethod
    def _reject_blank_entries(cls, value: list[str]) -> list[str]:
        cleaned = [item for item in value if item and item.strip()]
        if len(cleaned) != len(value):
            raise ValueError("list entries must be non-empty strings")
        return cleaned

    @property
    def is_targeted(self) -> bool:
        """Whether this filter narrows beyond a whole namespace.

        Destructive operations require this. A namespace alone matches every
        memory a caller owns, which is almost never what a ``forget`` meant.
        """
        return any(
            (
                self.memory_ids,
                self.session_id,
                self.agent_id,
                self.user_id,
                self.memory_type,
                self.role,
                self.event_type,
                self.tags_any,
                self.occurred_after is not None,
                self.occurred_before is not None,
                self.min_importance is not None,
                self.min_confidence is not None,
            )
        )


def compile_memory_filter(
    memory_filter: MemoryFilter | None = None,
    *,
    namespace: str | None = None,
) -> str | None:
    """Compile a validated filter into a LanceDB ``where`` predicate.

    ``namespace`` is the caller's effective namespace. It is applied even when
    the filter omits one, so a recall cannot silently read across namespaces.
    Returns ``None`` when nothing constrains the scan.
    """
    memory_filter = memory_filter or MemoryFilter()
    clauses: list[str] = []

    effective_namespace = memory_filter.namespace or namespace
    if effective_namespace:
        clauses.append(f"{_column('namespace')} = {_sql_string(effective_namespace)}")

    for field in ("session_id", "agent_id", "user_id", "memory_type", "role", "event_type"):
        value = getattr(memory_filter, field)
        if value:
            clauses.append(f"{_column(field)} = {_sql_string(str(value))}")

    if memory_filter.memory_ids:
        rendered = ", ".join(_sql_string(item) for item in memory_filter.memory_ids)
        clauses.append(f"{_column('memory_id')} IN ({rendered})")

    if memory_filter.occurred_after is not None:
        clauses.append(f"{_column('occurred_at')} >= {_sql_timestamp(memory_filter.occurred_after)}")
    if memory_filter.occurred_before is not None:
        clauses.append(f"{_column('occurred_at')} <= {_sql_timestamp(memory_filter.occurred_before)}")

    if memory_filter.min_importance is not None:
        clauses.append(f"{_column('importance')} >= {float(memory_filter.min_importance)}")
    if memory_filter.min_confidence is not None:
        clauses.append(f"{_column('confidence')} >= {float(memory_filter.min_confidence)}")

    if memory_filter.tags_any:
        rendered = ", ".join(_sql_string(tag) for tag in memory_filter.tags_any)
        clauses.append(f"array_has_any({_column('tags')}, [{rendered}])")

    if not memory_filter.include_superseded:
        clauses.append(f"{_column('valid_to')} IS NULL")

    if not clauses:
        return None
    return " AND ".join(clauses)


class MemoryRememberRequest(RichModel):
    """Write one or more memories in a single call."""

    records: list[MemoryRecord] = Field(min_length=1)


class MemoryWriteResult(RichModel):
    """Identifiers assigned to a completed memory write."""

    memory_ids: list[str] = Field(default_factory=list)
    written: int = 0


class MemoryRecallRequest(RichModel):
    """Semantic recall over stored memories."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=1000)
    filter: MemoryFilter | None = None
    strategy: RecallStrategy = "hybrid"
    recency_halflife_seconds: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "When set, blend retrieval score with exponential recency decay so "
            "older memories rank lower. Omit for pure semantic ranking."
        ),
    )


class MemoryRecallResponse(RichModel):
    """Ranked recall hits."""

    hits: list[MemoryHit] = Field(default_factory=list)


class MemoryTimelineRequest(RichModel):
    """Chronological replay of a session with no embedding cost."""

    session_id: str = Field(min_length=1, max_length=128)
    since: datetime | None = None
    until: datetime | None = None
    limit: int = Field(default=100, ge=1, le=1000)
    ascending: bool = True
    namespace: str | None = Field(default=None, max_length=128)


class MemoryForgetRequest(RichModel):
    """Retire memories by identifier or by filter."""

    memory_id: str | None = None
    filter: MemoryFilter | None = None
    hard: bool = Field(
        default=False,
        description="Delete rows outright instead of stamping valid_to.",
    )

    @model_validator(mode="after")
    def _require_a_target(self) -> "MemoryForgetRequest":
        if self.memory_id is None and self.filter is None:
            raise ValueError("forget requires either memory_id or filter")
        return self


class MemoryForgetResult(RichModel):
    """Outcome of a forget request."""

    matched: int = 0
    forgotten: int = 0
    hard: bool = False


class MemoryConsolidateRequest(RichModel):
    """Distill episodic memories in a session into durable semantic facts."""

    session_id: str | None = Field(default=None, max_length=128)
    namespace: str | None = Field(default=None, max_length=128)
    max_episodes: int = Field(default=200, ge=1, le=2000)
    since: datetime | None = None


class MemoryConsolidateResult(RichModel):
    """Facts produced by one consolidation pass."""

    episodes_considered: int = 0
    facts_written: int = 0
    facts_superseded: int = 0
    memory_ids: list[str] = Field(default_factory=list)


class MemoryStats(RichModel):
    """Coarse counters describing a memory namespace."""

    namespace: str
    total: int = 0
    by_type: dict[str, int] = Field(default_factory=dict)
    sessions: int = 0
    oldest_occurred_at: Optional[str] = None
    newest_occurred_at: Optional[str] = None
