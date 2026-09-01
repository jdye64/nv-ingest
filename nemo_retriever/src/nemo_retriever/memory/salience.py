# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decide which agent output is worth remembering.

Capture layers see every message, tool call, and observation. Embedding all of
it costs real money and, worse, buries the useful memories under transcript
noise so recall gets less accurate as the agent runs longer.

This is deliberately a cheap heuristic filter with no model call. It is a
default, not a policy: pass a custom callable to any capture layer to replace
it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Protocol

#: Below this many characters, text is almost always an acknowledgement.
MIN_SALIENT_CHARS = 24

#: Above this, store a prefix. Long observations are usually a dump whose first
#: page carries the signal.
MAX_SALIENT_CHARS = 4_000

_LOW_SIGNAL_PATTERNS = (
    r"^(ok(ay)?|sure|got it|thanks?|thank you|done|yes|no|sounds good)[.!]?$",
    r"^(i'?ll|let me|now i'?ll) (check|look|search|try|see)\b",
    r"^(here (is|are) )?the (results?|output)[:.]?$",
    r"^\s*$",
)
_LOW_SIGNAL_RE = re.compile("|".join(_LOW_SIGNAL_PATTERNS), re.IGNORECASE)

#: Phrases that mark durable user intent, which should survive the length and
#: boilerplate checks even in a short message.
_HIGH_SIGNAL_RE = re.compile(
    r"\b(prefer|always|never|remember|my name is|i am|i'?m using|"
    r"the (api|token|key|endpoint|repo|project) is|instead of|from now on|"
    r"deadline|due|requirement|must|should not)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SalienceVerdict:
    """Whether to store a candidate memory, and how strongly to weight it."""

    keep: bool
    text: str = ""
    importance: float = 0.5
    reason: str = ""


class SalienceFilter(Protocol):
    """Callable deciding whether a candidate memory is worth embedding."""

    def __call__(
        self,
        text: str,
        *,
        role: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> SalienceVerdict: ...


def default_salience(
    text: str,
    *,
    role: Optional[str] = None,
    event_type: Optional[str] = None,
) -> SalienceVerdict:
    """Score a candidate memory with the built-in heuristics."""
    cleaned = (text or "").strip()
    if not cleaned:
        return SalienceVerdict(keep=False, reason="empty")

    high_signal = bool(_HIGH_SIGNAL_RE.search(cleaned))

    if not high_signal and _LOW_SIGNAL_RE.match(cleaned):
        return SalienceVerdict(keep=False, reason="boilerplate")
    if not high_signal and len(cleaned) < MIN_SALIENT_CHARS:
        return SalienceVerdict(keep=False, reason="too short")

    stored = cleaned if len(cleaned) <= MAX_SALIENT_CHARS else _truncate(cleaned)
    return SalienceVerdict(
        keep=True,
        text=stored,
        importance=_importance(role=role, event_type=event_type, high_signal=high_signal),
        reason="high signal" if high_signal else "retained",
    )


def keep_everything(
    text: str,
    *,
    role: Optional[str] = None,
    event_type: Optional[str] = None,
) -> SalienceVerdict:
    """Store every non-empty candidate. Useful for replaying a full transcript."""
    cleaned = (text or "").strip()
    if not cleaned:
        return SalienceVerdict(keep=False, reason="empty")
    stored = cleaned if len(cleaned) <= MAX_SALIENT_CHARS else _truncate(cleaned)
    return SalienceVerdict(keep=True, text=stored, importance=0.5, reason="unfiltered")


def _truncate(text: str) -> str:
    omitted = len(text) - MAX_SALIENT_CHARS
    return f"{text[:MAX_SALIENT_CHARS].rstrip()}\n...[{omitted} characters omitted]"


def _importance(*, role: Optional[str], event_type: Optional[str], high_signal: bool) -> float:
    if high_signal:
        return 0.9
    # What the user said outranks what the agent said about it, which in turn
    # outranks whatever a tool happened to return.
    if role == "user":
        return 0.7
    if event_type == "observation" or role == "tool":
        return 0.3
    return 0.5
