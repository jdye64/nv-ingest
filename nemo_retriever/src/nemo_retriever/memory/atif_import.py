# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import ATIF agent trajectories into episodic memory.

Agentic retrieval already writes ATIF traces under ``./agentic-traces`` (see
:mod:`nemo_retriever._agentic.nemo_agent.atif`). Replaying those files is the
cheapest way to fill a memory namespace with real agent traffic, and it
exercises the schema against production-shaped data before anyone turns on
live capture.

The trace's own ``session_id`` becomes the memory session, and each step's
timestamp becomes ``occurred_at``, so an imported trajectory replays through
:meth:`AgenticIngestor.timeline` in its original order.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from nemo_retriever.common.schemas.memory import MemoryRecord, MemoryWriteResult
from nemo_retriever.memory.salience import SalienceFilter, default_salience

logger = logging.getLogger(__name__)

#: ATIF step sources mapped onto memory roles.
_SOURCE_TO_ROLE = {
    "user": "user",
    "system": "system",
    "agent": "assistant",
    "tool": "tool",
}


def import_atif_trajectory(
    ingestor: Any,
    trace: Mapping[str, Any] | str | Path,
    *,
    namespace: Optional[str] = None,
    agent_id: Optional[str] = None,
    user_id: Optional[str] = None,
    salience: SalienceFilter = default_salience,
    include_observations: bool = True,
    tags: Sequence[str] = (),
) -> MemoryWriteResult:
    """Convert one ATIF trajectory into episodic memories and store them.

    ``trace`` is a loaded trajectory mapping or a path to a trace JSON file.
    Returns the write result after flushing, so the memories are durable when
    this returns.
    """
    payload = _load_trace(trace)
    records = atif_records(
        payload,
        namespace=namespace,
        agent_id=agent_id,
        user_id=user_id,
        salience=salience,
        include_observations=include_observations,
        tags=list(tags),
    )
    if not records:
        return MemoryWriteResult(memory_ids=[], written=0)

    ingestor.remember_many(records)
    return ingestor.flush()


def import_atif_directory(
    ingestor: Any,
    directory: str | Path,
    *,
    pattern: str = "*.json",
    **kwargs: Any,
) -> MemoryWriteResult:
    """Import every ATIF trace in a directory.

    One malformed file does not abort the import; it is logged and skipped, so
    a partially corrupt trace directory still yields usable memory.
    """
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    memory_ids: List[str] = []
    written = 0
    for path in sorted(root.glob(pattern)):
        try:
            result = import_atif_trajectory(ingestor, path, **kwargs)
        except Exception:
            logger.warning("Skipping unreadable ATIF trace %s", path, exc_info=True)
            continue
        memory_ids.extend(result.memory_ids)
        written += result.written
    return MemoryWriteResult(memory_ids=memory_ids, written=written)


def atif_records(
    trace: Mapping[str, Any],
    *,
    namespace: Optional[str] = None,
    agent_id: Optional[str] = None,
    user_id: Optional[str] = None,
    salience: SalienceFilter = default_salience,
    include_observations: bool = True,
    tags: Optional[List[str]] = None,
) -> List[MemoryRecord]:
    """Build episodic records from a trajectory without writing them."""
    session_id = str(trace.get("session_id") or "")[:128] or None
    agent = trace.get("agent") if isinstance(trace.get("agent"), Mapping) else {}
    resolved_agent_id = agent_id or str(agent.get("name") or "") or None
    model_name = str(agent.get("model_name") or "") or None
    base_tags = list(tags or [])

    records: List[MemoryRecord] = []
    for index, step in enumerate(_steps(trace)):
        occurred_at = _parse_timestamp(step.get("timestamp"))
        role = _SOURCE_TO_ROLE.get(str(step.get("source") or ""), "assistant")

        for text, event_type in _step_texts(step, include_observations=include_observations):
            verdict = salience(text, role=role, event_type=event_type)
            if not verdict.keep:
                continue
            records.append(
                MemoryRecord(
                    text=verdict.text,
                    memory_type="episodic",
                    namespace=namespace or "default",
                    session_id=session_id,
                    agent_id=resolved_agent_id,
                    user_id=user_id,
                    role=role,  # type: ignore[arg-type]
                    event_type=event_type,  # type: ignore[arg-type]
                    occurred_at=occurred_at,
                    importance=verdict.importance,
                    turn_index=index,
                    tags=base_tags,
                    metadata=_step_metadata(step, model_name=model_name),
                )
            )
    return records


def _load_trace(trace: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(trace, Mapping):
        return trace
    payload = json.loads(Path(trace).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"ATIF trace at {trace} is not a JSON object")
    return payload


def _steps(trace: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    steps = trace.get("steps")
    if not isinstance(steps, list):
        return []
    return [step for step in steps if isinstance(step, Mapping)]


def _step_texts(
    step: Mapping[str, Any],
    *,
    include_observations: bool,
) -> List[tuple[str, str]]:
    texts: List[tuple[str, str]] = []

    message = step.get("message")
    if isinstance(message, str) and message.strip():
        texts.append((message, "message"))

    tool_calls = step.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            rendered = _render_tool_call(call)
            if rendered:
                texts.append((rendered, "tool_call"))

    if include_observations:
        observation = step.get("observation")
        results = observation.get("results") if isinstance(observation, Mapping) else None
        if isinstance(results, list):
            for entry in results:
                if not isinstance(entry, Mapping):
                    continue
                content = entry.get("content")
                if isinstance(content, str) and content.strip():
                    texts.append((content, "observation"))

    return texts


def _render_tool_call(call: Any) -> Optional[str]:
    if not isinstance(call, Mapping):
        return None
    function = call.get("function") if isinstance(call.get("function"), Mapping) else call
    name = function.get("name") if isinstance(function, Mapping) else None
    if not name:
        return None
    arguments = function.get("arguments") if isinstance(function, Mapping) else None
    if isinstance(arguments, (dict, list)):
        rendered = json.dumps(arguments, ensure_ascii=False, default=str)
    else:
        rendered = str(arguments or "")
    return f"Called tool {name} with {rendered}".strip()


def _step_metadata(step: Mapping[str, Any], *, model_name: Optional[str]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {"atif_step_id": step.get("step_id")}
    if model_name:
        metadata["model_name"] = model_name
    extra = step.get("extra")
    if isinstance(extra, Mapping):
        stage = extra.get("stage")
        if stage:
            metadata["stage"] = str(stage)
    return metadata


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
