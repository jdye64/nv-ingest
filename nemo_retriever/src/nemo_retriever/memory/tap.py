# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Passive capture of agent turns into episodic memory.

An agent that has to call ``remember()`` will forget to. :class:`MemoryTap`
wraps the chat callable the agent already uses and mirrors each turn into
memory on the way past, so capture is a wiring change rather than a change to
the agent's reasoning.

The wrapped callable follows the OpenAI chat-completions shape, which is the
same contract
:class:`~nemo_retriever._agentic.nemo_agent.llm.CallableLLMBackend` expects.
Any framework that lets you inject a client can therefore be tapped::

    tap = MemoryTap(memory, completion_fn=my_client, session_id="thread-42")
    backend = create_llm(config, completion_fn=tap)

Capture never blocks the agent. Turns are queued on the ingestor and embedded
by its batched flush, and a capture failure is logged rather than raised: an
agent should not fail its turn because remembering it failed.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from nemo_retriever.common.schemas.memory import MemoryRecord
from nemo_retriever.memory.salience import SalienceFilter, default_salience

logger = logging.getLogger(__name__)


class MemoryTap:
    """Mirror an OpenAI-compatible chat callable's traffic into episodic memory.

    Parameters
    ----------
    memory
        An :class:`~nemo_retriever.ingestor.agentic_ingestor.AgenticIngestor`.
    completion_fn
        The chat callable to wrap. It is invoked unchanged and its result is
        returned to the caller untouched.
    session_id
        Session stamped on captured turns. Defaults to the ingestor's session.
    agent_id, user_id
        Identity stamped on captured turns.
    salience
        Filter deciding which turns are worth embedding. Replace it to capture
        everything or to apply a domain-specific rule.
    capture_prompt
        Whether to capture inbound messages. The last user message is captured
        by default; set ``False`` when the caller already writes user turns.
    capture_tool_calls
        Whether to capture the assistant's tool calls as separate memories.
    """

    def __init__(
        self,
        memory: Any,
        completion_fn: Callable[..., Dict[str, Any]],
        *,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
        salience: SalienceFilter = default_salience,
        capture_prompt: bool = True,
        capture_tool_calls: bool = True,
    ) -> None:
        self._memory = memory
        self._completion_fn = completion_fn
        self._session_id = session_id
        self._agent_id = agent_id
        self._user_id = user_id
        self._salience = salience
        self._capture_prompt = capture_prompt
        self._capture_tool_calls = capture_tool_calls
        self._turn_index = 0

    def __call__(self, *, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        """Invoke the wrapped callable and mirror the turn into memory."""
        response = self._completion_fn(messages=messages, **kwargs)
        try:
            self._capture(messages, response)
        except Exception:
            logger.warning("Memory capture failed for this turn; continuing", exc_info=True)
        return response

    async def acall(self, *, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        """Async variant for callables that return an awaitable."""
        response = await self._completion_fn(messages=messages, **kwargs)  # type: ignore[misc]
        try:
            self._capture(messages, response)
        except Exception:
            logger.warning("Memory capture failed for this turn; continuing", exc_info=True)
        return response

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def _capture(self, messages: Sequence[Mapping[str, Any]], response: Any) -> None:
        records: List[MemoryRecord] = []
        if self._capture_prompt:
            records.extend(self._prompt_records(messages))
        records.extend(self._response_records(response))
        if records:
            self._memory.remember_many(records)
        self._turn_index += 1

    def _prompt_records(self, messages: Sequence[Mapping[str, Any]]) -> List[MemoryRecord]:
        # Only the newest user message is new information. Everything earlier
        # in the history was captured on a previous turn.
        latest = next(
            (message for message in reversed(list(messages)) if message.get("role") == "user"),
            None,
        )
        if latest is None:
            return []
        return self._records_for(_content_text(latest.get("content")), role="user", event_type="message")

    def _response_records(self, response: Any) -> List[MemoryRecord]:
        message = _assistant_message(response)
        if message is None:
            return []

        records = self._records_for(
            _content_text(message.get("content")),
            role="assistant",
            event_type="message",
        )
        if not self._capture_tool_calls:
            return records

        for call in message.get("tool_calls") or []:
            rendered = _render_tool_call(call)
            if rendered:
                records.extend(self._records_for(rendered, role="assistant", event_type="tool_call"))
        return records

    def _records_for(self, text: str, *, role: str, event_type: str) -> List[MemoryRecord]:
        verdict = self._salience(text, role=role, event_type=event_type)
        if not verdict.keep:
            return []
        return [
            MemoryRecord(
                text=verdict.text,
                memory_type="episodic",
                namespace=self._memory.namespace,
                session_id=self._session_id,
                agent_id=self._agent_id,
                user_id=self._user_id,
                role=role,  # type: ignore[arg-type]
                event_type=event_type,  # type: ignore[arg-type]
                importance=verdict.importance,
                turn_index=self._turn_index,
            )
        ]


def _assistant_message(response: Any) -> Optional[Dict[str, Any]]:
    """Pull the assistant message out of an OpenAI-shaped response."""
    if response is None:
        return None
    if isinstance(response, Mapping):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, Mapping):
                message = first.get("message")
                if isinstance(message, Mapping):
                    return dict(message)
        message = response.get("message")
        if isinstance(message, Mapping):
            return dict(message)
        return None

    # Also accept a CompletionResult, so a tap can wrap a backend result
    # directly rather than only a raw client response.
    message = getattr(response, "message", None)
    return dict(message) if isinstance(message, Mapping) else None


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, Mapping):
                text = block.get("text")
                if text is not None:
                    parts.append(str(text))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _render_tool_call(call: Any) -> Optional[str]:
    if not isinstance(call, Mapping):
        return None
    function = call.get("function") if isinstance(call.get("function"), Mapping) else call
    name = function.get("name") if isinstance(function, Mapping) else None
    if not name:
        return None
    arguments = function.get("arguments") if isinstance(function, Mapping) else None
    return f"Called tool {name} with {arguments if arguments else '{}'}"
