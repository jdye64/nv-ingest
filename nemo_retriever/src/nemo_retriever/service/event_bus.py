# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight in-memory pub/sub for SSE streaming.

Each document (or job) can have zero or more subscriber queues.  When an
event is published it is fanned out to all connected queues.  If no
subscribers exist the event is silently discarded.

Backpressure model
------------------
Each subscription is a **bounded** ``asyncio.Queue`` (default 1024 events).
``publish()`` always uses ``put_nowait`` so a slow subscriber can never
block worker processes from completing batches.  When a queue overflows
the subscription is marked overflowed, the queue is drained, and a single
sentinel event ``{"event": "stream_overflow"}`` is injected so the SSE
generator can emit it to the client and close the stream cleanly.  The
client is expected to reconnect or fall back to polling the REST status
endpoints.

Supports both per-document and multi-document (session) subscriptions so a
single SSE connection can receive events for many documents at once.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


_DEFAULT_MAXSIZE = 1024
_OVERFLOW_EVENT = {"event": "stream_overflow", "reason": "subscriber_too_slow"}


class _Subscription:
    """A single subscriber's bounded queue with overflow detection.

    The public ``queue`` attribute is a vanilla ``asyncio.Queue`` so that
    SSE generators can ``await queue.get()`` exactly as before.  All writes
    go through :meth:`put` which performs the overflow handling.
    """

    __slots__ = ("queue", "maxsize", "_overflowed")

    def __init__(self, maxsize: int = _DEFAULT_MAXSIZE) -> None:
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=maxsize)
        self.maxsize = maxsize
        self._overflowed = False

    @property
    def overflowed(self) -> bool:
        return self._overflowed

    def put(self, event: dict[str, Any]) -> bool:
        """Try to enqueue ``event``.  Returns ``False`` if the subscription
        has already overflowed (or just overflowed on this call)."""
        if self._overflowed:
            return False
        try:
            self.queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            self._overflowed = True
            self._drain_and_signal()
            return False

    def _drain_and_signal(self) -> None:
        """Empty the queue and inject the overflow sentinel.

        After this runs the consumer will pull the sentinel as the next
        event and is expected to terminate the SSE stream.
        """
        try:
            while True:
                self.queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            self.queue.put_nowait(dict(_OVERFLOW_EVENT))
        except asyncio.QueueFull:
            # Re-raising would mask the original overflow; the subscriber is
            # already going to be dropped so this is a noop in practice.
            pass


class EventBus:
    """Per-key fan-out event bus backed by bounded ``asyncio.Queue`` instances.

    Keys are usually ``document_id`` or ``job_id`` strings.  The bus does not
    care which — events published under a key go to every subscription for
    that key.
    """

    def __init__(self, *, default_maxsize: int = _DEFAULT_MAXSIZE) -> None:
        self._subscribers: dict[str, list[_Subscription]] = {}
        self._default_maxsize = default_maxsize

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    def subscribe(
        self,
        document_id: str,
        *,
        maxsize: int | None = None,
    ) -> asyncio.Queue[dict[str, Any]]:
        """Subscribe to events for a single document/job.

        Returns the underlying ``asyncio.Queue`` so existing SSE generators
        keep working unchanged.
        """
        sub = _Subscription(maxsize=maxsize or self._default_maxsize)
        self._subscribers.setdefault(document_id, []).append(sub)
        return sub.queue

    def subscribe_many(
        self,
        document_ids: list[str],
        *,
        maxsize: int | None = None,
    ) -> asyncio.Queue[dict[str, Any]]:
        """Subscribe to events for multiple documents on a single queue.

        Every event for any of the listed documents is delivered to the same
        queue, tagged with its ``document_id`` (or ``job_id``) in the payload.
        """
        sub = _Subscription(maxsize=maxsize or self._default_maxsize)
        for doc_id in document_ids:
            self._subscribers.setdefault(doc_id, []).append(sub)
        return sub.queue

    def unsubscribe(self, document_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        subs = self._subscribers.get(document_id)
        if not subs:
            return
        self._subscribers[document_id] = [s for s in subs if s.queue is not queue]
        if not self._subscribers[document_id]:
            del self._subscribers[document_id]

    def unsubscribe_many(
        self,
        document_ids: list[str],
        queue: asyncio.Queue[dict[str, Any]],
    ) -> None:
        for doc_id in document_ids:
            self.unsubscribe(doc_id, queue)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def publish(self, document_id: str, event: dict[str, Any]) -> None:
        """Fan ``event`` out to every subscriber of ``document_id``.

        Never blocks the publisher — a saturated subscriber is dropped via
        the overflow path described in the module docstring.
        """
        subs = self._subscribers.get(document_id, [])
        if not subs:
            return

        for sub in subs:
            ok = sub.put(event)
            if not ok:
                logger.warning(
                    "SSE subscription for %s overflowed (maxsize=%d) — "
                    "dropping subscriber and signalling stream_overflow",
                    document_id,
                    sub.maxsize,
                )

    def has_subscribers(self, document_id: str) -> bool:
        return bool(self._subscribers.get(document_id))
