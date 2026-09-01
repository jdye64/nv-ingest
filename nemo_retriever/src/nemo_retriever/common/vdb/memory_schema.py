# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent-memory table schema and row construction for LanceDB.

The document tables built by :mod:`nemo_retriever.common.vdb.lancedb_schema`
keep every per-chunk attribute inside a single JSON ``metadata`` string, so
filtering degrades to SQL ``LIKE``. Agent memory needs session scoping,
time-range recall, and importance thresholds, so this module defines a
parallel table layout that promotes those attributes to typed columns.

``occurred_at`` and the validity columns are real timestamps rather than the
ISO strings used by ``lancedb_schema(collection_managed=True)``. That is what
makes range predicates and recency decay work in the storage engine instead of
in Python.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence

MemoryType = Literal["episodic", "semantic", "procedural"]
MemoryRole = Literal["user", "assistant", "tool", "system"]
MemoryEventType = Literal["message", "tool_call", "observation", "fact", "summary", "document"]

MEMORY_TYPES: frozenset[str] = frozenset({"episodic", "semantic", "procedural"})
MEMORY_ROLES: frozenset[str] = frozenset({"user", "assistant", "tool", "system"})
MEMORY_EVENT_TYPES: frozenset[str] = frozenset({"message", "tool_call", "observation", "fact", "summary", "document"})

#: Default physical table name for the agent-memory layout.
DEFAULT_MEMORY_TABLE = "nemo-retriever-memory"

#: Columns promoted out of the JSON blob so they can be filtered natively.
MEMORY_FILTER_COLUMNS: tuple[str, ...] = (
    "memory_id",
    "memory_type",
    "namespace",
    "session_id",
    "agent_id",
    "user_id",
    "role",
    "event_type",
    "occurred_at",
    "ingested_at",
    "valid_from",
    "valid_to",
    "importance",
    "confidence",
    "turn_index",
)

#: Columns returned by recall and timeline reads. ``vector`` is deliberately
#: excluded: callers never need the raw embedding and it dominates payload size.
MEMORY_RESULT_COLUMNS: tuple[str, ...] = (
    "text",
    *MEMORY_FILTER_COLUMNS,
    "source_memory_ids",
    "tags",
    "metadata",
)


def new_memory_id() -> str:
    """Return a fresh opaque identifier for one memory record."""
    return f"mem_{uuid.uuid4().hex}"


def utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def memory_schema(vector_dim: int = 2048) -> Any:
    """Return the PyArrow schema for the agent-memory table layout."""
    import pyarrow as pa  # type: ignore

    return pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), int(vector_dim))),
            pa.field("text", pa.string()),
            pa.field("memory_id", pa.string()),
            pa.field("memory_type", pa.string()),
            pa.field("namespace", pa.string()),
            pa.field("session_id", pa.string()),
            pa.field("agent_id", pa.string()),
            pa.field("user_id", pa.string()),
            pa.field("role", pa.string()),
            pa.field("event_type", pa.string()),
            pa.field("occurred_at", pa.timestamp("us", tz="UTC")),
            pa.field("ingested_at", pa.timestamp("us", tz="UTC")),
            pa.field("valid_from", pa.timestamp("us", tz="UTC")),
            # A null ``valid_to`` means the record is still true. Supersession
            # stamps this column instead of deleting rows, so consolidation
            # stays auditable and recall can replay historical belief state.
            pa.field("valid_to", pa.timestamp("us", tz="UTC")),
            pa.field("importance", pa.float32()),
            pa.field("confidence", pa.float32()),
            pa.field("turn_index", pa.int32()),
            pa.field("source_memory_ids", pa.list_(pa.string())),
            pa.field("tags", pa.list_(pa.string())),
            pa.field("metadata", pa.string()),
        ]
    )


def _coerce_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    raise TypeError(f"Cannot interpret {type(value).__name__} as a timestamp")


def _coerce_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Sequence):
        return [str(item) for item in value if item is not None and str(item).strip()]
    raise TypeError(f"Expected a string or sequence of strings, got {type(value).__name__}")


def _validate_choice(value: Any, allowed: frozenset[str], field: str) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    if text not in allowed:
        raise ValueError(f"{field} must be one of {sorted(allowed)}, got {text!r}")
    return text


def build_memory_row(
    *,
    text: str,
    embedding: Sequence[float],
    memory_id: Optional[str] = None,
    memory_type: str = "episodic",
    namespace: str = "default",
    session_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    user_id: Optional[str] = None,
    role: Optional[str] = None,
    event_type: Optional[str] = None,
    occurred_at: Any = None,
    ingested_at: Any = None,
    valid_from: Any = None,
    valid_to: Any = None,
    importance: float = 0.5,
    confidence: float = 1.0,
    turn_index: Optional[int] = None,
    source_memory_ids: Any = None,
    tags: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one LanceDB-ready memory row.

    ``occurred_at`` defaults to ``ingested_at`` so that records captured
    without an explicit event time still order correctly on the timeline.
    """
    _validate_choice(memory_type, MEMORY_TYPES, "memory_type")
    _validate_choice(role, MEMORY_ROLES, "role")
    _validate_choice(event_type, MEMORY_EVENT_TYPES, "event_type")

    vector = [float(component) for component in embedding]
    if not vector:
        raise ValueError("Memory rows require a non-empty embedding vector")

    ingested = _coerce_datetime(ingested_at) or utcnow()
    occurred = _coerce_datetime(occurred_at) or ingested
    start = _coerce_datetime(valid_from) or occurred

    return {
        "vector": vector,
        "text": str(text),
        "memory_id": memory_id or new_memory_id(),
        "memory_type": str(memory_type),
        "namespace": str(namespace),
        "session_id": str(session_id) if session_id else "",
        "agent_id": str(agent_id) if agent_id else "",
        "user_id": str(user_id) if user_id else "",
        "role": str(role) if role else "",
        "event_type": str(event_type) if event_type else "",
        "occurred_at": occurred,
        "ingested_at": ingested,
        "valid_from": start,
        "valid_to": _coerce_datetime(valid_to),
        "importance": float(importance),
        "confidence": float(confidence),
        "turn_index": int(turn_index) if turn_index is not None else -1,
        "source_memory_ids": _coerce_str_list(source_memory_ids),
        "tags": _coerce_str_list(tags),
        "metadata": json.dumps(metadata or {}, ensure_ascii=False, default=str),
    }


def infer_memory_vector_dim(rows: Iterable[Dict[str, Any]]) -> int:
    """Return the embedding width from the first row that carries a vector."""
    for row in rows:
        vector = row.get("vector")
        if isinstance(vector, list) and vector:
            return len(vector)
    return 0


def parse_memory_row(row: Any) -> Dict[str, Any]:
    """Normalize a raw LanceDB memory row into plain Python values.

    Decodes the ``metadata`` JSON escape hatch and renders timestamps as
    ISO 8601 strings so the result is directly JSON serializable for HTTP and
    Model Context Protocol (MCP) responses.
    """
    if not isinstance(row, dict):
        row = dict(row)

    out: Dict[str, Any] = {}
    for column in MEMORY_RESULT_COLUMNS:
        if column not in row:
            continue
        value = row[column]
        if column == "metadata":
            out[column] = _decode_metadata(value)
        elif column in {"occurred_at", "ingested_at", "valid_from", "valid_to"}:
            out[column] = _isoformat(value)
        elif column in {"source_memory_ids", "tags"}:
            out[column] = _coerce_str_list(_maybe_tolist(value))
        else:
            out[column] = value

    for score_field in ("_distance", "_score", "_relevance_score"):
        if score_field in row and row[score_field] is not None:
            out[score_field] = float(row[score_field])
    return out


def _maybe_tolist(value: Any) -> Any:
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else value


def _decode_metadata(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _isoformat(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        moment = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc).isoformat()
    to_pydatetime = getattr(value, "to_pydatetime", None)
    if callable(to_pydatetime):
        try:
            return _isoformat(to_pydatetime())
        except Exception:
            return None
    if isinstance(value, str):
        return value or None
    return None
