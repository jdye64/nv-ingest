# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight in-memory pub/sub for SSE streaming.

Each document can have zero or more subscriber queues.  When an event is
published for a document, it is fanned out to all connected queues.  If no
subscribers exist the event is silently discarded.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class EventBus:
    """Per-document fan-out event bus backed by ``asyncio.Queue``."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {}

    def subscribe(self, document_id: str) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers.setdefault(document_id, []).append(queue)
        return queue

    def unsubscribe(self, document_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        queues = self._subscribers.get(document_id)
        if queues:
            try:
                queues.remove(queue)
            except ValueError:
                pass
            if not queues:
                del self._subscribers[document_id]

    async def publish(self, document_id: str, event: dict[str, Any]) -> None:
        queues = self._subscribers.get(document_id, [])
        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("SSE queue full for document %s — dropping event", document_id)

    def has_subscribers(self, document_id: str) -> bool:
        return bool(self._subscribers.get(document_id))
