# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distill episodic memories into durable semantic facts.

Episodes are cheap to write and expensive to search: a long session buries the
few things worth remembering under hundreds of turns. Consolidation reads a
window of episodes, asks a language model for the durable facts, and writes
those back as ``memory_type="semantic"`` records that carry
``source_memory_ids`` back to the episodes they came from.

Facts are never overwritten. A new fact that contradicts an old one supersedes
it by stamping ``valid_to`` on the old row, so the record of what the agent
once believed survives.

This costs an LLM call per session, so it is explicitly triggered. Nothing in
this package schedules it.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

from nemo_retriever.common.schemas.memory import (
    MemoryConsolidateRequest,
    MemoryConsolidateResult,
    MemoryFilter,
    MemoryForgetRequest,
    MemoryHit,
    MemoryRecord,
)

logger = logging.getLogger(__name__)

DEFAULT_CONSOLIDATION_MODEL = "gpt-4o-mini"

_SYSTEM_PROMPT = """\
You distill an agent's conversation history into durable facts.

Return JSON only, matching this shape:
{"facts": [{"text": "...", "importance": 0.0-1.0, "confidence": 0.0-1.0,
            "tags": ["..."], "supersedes": ["memory_id", ...]}]}

Rules:
- A fact must still be true after this conversation ends. Skip anything
  transient, such as what a tool returned once or what step the agent was on.
- Write each fact as one self-contained sentence. It will be read later with
  no surrounding conversation.
- Do not restate the same fact twice.
- Set "supersedes" to the ids of any listed EXISTING FACTS this one replaces.
- Return {"facts": []} when nothing is worth keeping.
"""


def consolidate_session(
    backend: Any,
    request: MemoryConsolidateRequest,
    *,
    llm: Any = None,
    model: str = DEFAULT_CONSOLIDATION_MODEL,
) -> MemoryConsolidateResult:
    """Run one consolidation pass over a session's episodic memories.

    Parameters
    ----------
    backend
        A memory backend, normally
        :class:`~nemo_retriever.memory.backends.LocalMemoryBackend`.
    request
        Session, namespace, and window to consolidate.
    llm
        A prepared LLM backend. When omitted, a LiteLLM backend is built for
        ``model``. Pass a
        :class:`~nemo_retriever.\\_agentic.nemo_agent.llm.CallableLLMBackend`
        to reuse a client the caller already owns.
    """
    episodes = _load_episodes(backend, request)
    if not episodes:
        return MemoryConsolidateResult(episodes_considered=0)

    existing = _load_existing_facts(backend, request)
    client = llm or _default_llm(model)
    facts = _extract_facts(client, episodes=episodes, existing=existing)
    if not facts:
        return MemoryConsolidateResult(episodes_considered=len(episodes))

    superseded = _supersede(backend, facts, known_ids={fact.memory_id for fact in existing})
    written = backend.write([_to_record(fact, request=request, episodes=episodes) for fact in facts])
    return MemoryConsolidateResult(
        episodes_considered=len(episodes),
        facts_written=written.written,
        facts_superseded=superseded,
        memory_ids=written.memory_ids,
    )


def _load_episodes(backend: Any, request: MemoryConsolidateRequest) -> List[MemoryHit]:
    from nemo_retriever.common.schemas.memory import MemoryTimelineRequest

    if request.session_id:
        return list(
            backend.timeline(
                MemoryTimelineRequest(
                    session_id=request.session_id,
                    since=request.since,
                    limit=request.max_episodes,
                    namespace=request.namespace,
                )
            )
        )

    # Without a session there is no timeline to replay, so fall back to a broad
    # recall over the namespace's episodic memories.
    from nemo_retriever.common.schemas.memory import MemoryRecallRequest

    return list(
        backend.recall(
            MemoryRecallRequest(
                query="notable events, decisions, and stated preferences",
                top_k=request.max_episodes,
                filter=MemoryFilter(
                    namespace=request.namespace,
                    memory_type="episodic",
                    occurred_after=request.since,
                ),
            )
        )
    )


def _load_existing_facts(backend: Any, request: MemoryConsolidateRequest) -> List[MemoryHit]:
    from nemo_retriever.common.schemas.memory import MemoryRecallRequest

    try:
        return list(
            backend.recall(
                MemoryRecallRequest(
                    query="established facts and preferences",
                    top_k=50,
                    filter=MemoryFilter(namespace=request.namespace, memory_type="semantic"),
                )
            )
        )
    except Exception:
        # A namespace with no semantic memories yet is the normal case on the
        # first pass; supersession simply has nothing to target.
        logger.debug("Could not load existing semantic memories for consolidation", exc_info=True)
        return []


def _default_llm(model: str) -> Any:
    from nemo_retriever._agentic.nemo_agent.llm import create_llm, create_llm_config

    return create_llm(create_llm_config("litellm", model=model, temperature=0.0))


def _extract_facts(
    llm: Any,
    *,
    episodes: Sequence[MemoryHit],
    existing: Sequence[MemoryHit],
) -> List["_Fact"]:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _render_prompt(episodes, existing)},
    ]
    try:
        result = llm.completion(messages)
    except Exception:
        logger.exception("Consolidation LLM call failed; no facts written")
        return []

    content = (result.message or {}).get("content")
    return _parse_facts(content)


def _render_prompt(episodes: Sequence[MemoryHit], existing: Sequence[MemoryHit]) -> str:
    lines: List[str] = []
    if existing:
        lines.append("EXISTING FACTS")
        for fact in existing:
            lines.append(f"- [{fact.memory_id}] {fact.text}")
        lines.append("")
    lines.append("CONVERSATION")
    for episode in episodes:
        speaker = episode.role or episode.event_type or "event"
        when = episode.occurred_at or ""
        lines.append(f"- ({when}) {speaker}: {episode.text}")
    return "\n".join(lines)


class _Fact:
    """One parsed fact from the consolidation response."""

    __slots__ = ("text", "importance", "confidence", "tags", "supersedes")

    def __init__(
        self,
        text: str,
        *,
        importance: float = 0.7,
        confidence: float = 0.8,
        tags: Optional[List[str]] = None,
        supersedes: Optional[List[str]] = None,
    ) -> None:
        self.text = text
        self.importance = importance
        self.confidence = confidence
        self.tags = tags or []
        self.supersedes = supersedes or []


def _parse_facts(content: Any) -> List[_Fact]:
    if not isinstance(content, str) or not content.strip():
        return []
    payload = _load_json_object(content)
    if payload is None:
        logger.warning("Consolidation response was not valid JSON; no facts written")
        return []

    facts: List[_Fact] = []
    for entry in payload.get("facts", []):
        if not isinstance(entry, dict):
            continue
        text = entry.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        facts.append(
            _Fact(
                text.strip(),
                importance=_clamp(entry.get("importance"), default=0.7),
                confidence=_clamp(entry.get("confidence"), default=0.8),
                tags=[str(tag) for tag in entry.get("tags", []) if str(tag).strip()],
                supersedes=[str(item) for item in entry.get("supersedes", []) if str(item).strip()],
            )
        )
    return facts


def _load_json_object(content: str) -> Optional[Dict[str, Any]]:
    text = content.strip()
    if text.startswith("```"):
        # Models routinely wrap JSON in a fenced block despite the instruction.
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _clamp(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(max(number, 0.0), 1.0)


def _supersede(backend: Any, facts: Sequence[_Fact], *, known_ids: set[str]) -> int:
    # Only retire ids the model was actually shown. A hallucinated identifier
    # must never widen into a predicate that matches unrelated memories.
    targets = sorted({item for fact in facts for item in fact.supersedes} & known_ids)
    if not targets:
        return 0
    result = backend.forget(MemoryForgetRequest(filter=MemoryFilter(memory_ids=targets, memory_type="semantic")))
    return result.forgotten


def _to_record(
    fact: _Fact,
    *,
    request: MemoryConsolidateRequest,
    episodes: Sequence[MemoryHit],
) -> MemoryRecord:
    return MemoryRecord(
        text=fact.text,
        memory_type="semantic",
        namespace=request.namespace or "default",
        session_id=request.session_id,
        event_type="fact",
        importance=fact.importance,
        confidence=fact.confidence,
        tags=fact.tags,
        source_memory_ids=[episode.memory_id for episode in episodes],
        metadata={"consolidated_from_episodes": len(episodes)},
    )
