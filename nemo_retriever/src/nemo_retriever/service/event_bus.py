# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight in-memory pub/sub for SSE streaming.

Each document (or job) can have zero or more subscriber queues.  When an
event is published it is fanned out to all connected queues and stored in
a per-key ring buffer so reconnecting clients can replay missed events.
If no subscribers exist when the event is published the event is still
buffered (so a late subscriber gets it) but not delivered to anyone.

Backpressure model
------------------
Each subscription is a **bounded** ``asyncio.Queue`` (default 1024 events).
``publish()`` always uses ``put_nowait`` so a slow subscriber can never
block worker processes from completing batches.  When a queue overflows
the subscription is marked overflowed, the queue is drained, and a single
sentinel event ``{"event": "stream_overflow"}`` is injected so the SSE
generator can emit it to the client and close the stream cleanly.  The
client is expected to reconnect (with ``Last-Event-ID``) or fall back to
polling the REST status endpoints.

Resumable streams
-----------------
Every published event is annotated with a monotonic ``seq`` integer
(unique within this process) and stored in a fixed-size per-key ring
buffer (default 256 events).  When a client reconnects with the
``Last-Event-ID`` header the SSE generator drains the buffer and replays
all events with ``seq > last_event_id`` before pulling new ones from the
queue.

Supports both per-document and multi-document (session) subscriptions so
a single SSE connection can receive events for many documents at once.
"""

from __future__ import annotations

import asyncio
import collections
import itertools
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


_DEFAULT_MAXSIZE = 1024
_DEFAULT_BUFFER_SIZE = 256
_OVERFLOW_EVENT = {"event": "stream_overflow", "reason": "subscriber_too_slow"}


# Process-wide monotonic event sequence number.  ``itertools.count`` is
# thread-safe in CPython for ``next()`` calls (single bytecode op).
_seq_counter = itertools.count(1)


def _next_seq() -> int:
    return next(_seq_counter)


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
        """Empty the queue and inject the overflow sentinel."""
        try:
            while True:
                self.queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            self.queue.put_nowait(dict(_OVERFLOW_EVENT))
        except asyncio.QueueFull:
            pass


class EventBus:
    """Per-key fan-out event bus backed by bounded ``asyncio.Queue`` instances.

    Keys are usually ``document_id`` or ``job_id`` strings.  The bus does not
    care which — events published under a key go to every subscription for
    that key.
    """

    def __init__(
        self,
        *,
        default_maxsize: int = _DEFAULT_MAXSIZE,
        buffer_size: int = _DEFAULT_BUFFER_SIZE,
    ) -> None:
        self._subscribers: dict[str, list[_Subscription]] = {}
        self._buffers: dict[str, collections.deque[dict[str, Any]]] = {}
        self._buffers_lock = threading.Lock()
        self._default_maxsize = default_maxsize
        self._buffer_size = buffer_size

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    def subscribe(
        self,
        document_id: str,
        *,
        maxsize: int | None = None,
    ) -> asyncio.Queue[dict[str, Any]]:
        """Subscribe to events for a single document/job."""
        sub = _Subscription(maxsize=maxsize or self._default_maxsize)
        self._subscribers.setdefault(document_id, []).append(sub)
        return sub.queue

    def subscribe_many(
        self,
        document_ids: list[str],
        *,
        maxsize: int | None = None,
    ) -> asyncio.Queue[dict[str, Any]]:
        """Subscribe to events for multiple documents on a single queue."""
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
    # Replay buffer
    # ------------------------------------------------------------------

    def replay(self, document_ids: list[str], *, after_seq: int) -> list[dict[str, Any]]:
        """Return buffered events for *document_ids* whose ``seq > after_seq``.

        Returned events are sorted by ``seq`` so a multi-key replay still
        produces a coherent stream.  Events too old to be in the buffer are
        silently lost; the client should fall back to polling the REST
        endpoints in that case.
        """
        out: list[dict[str, Any]] = []
        with self._buffers_lock:
            for key in document_ids:
                buf = self._buffers.get(key)
                if not buf:
                    continue
                for evt in buf:
                    seq = evt.get("seq")
                    if isinstance(seq, int) and seq > after_seq:
                        out.append(evt)
        out.sort(key=lambda e: e.get("seq", 0))
        return out

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def publish(self, document_id: str, event: dict[str, Any]) -> None:
        """Fan ``event`` out to every subscriber and store it for replay.

        Mutates ``event`` to add a ``seq`` field if not already present.
        """
        if "seq" not in event:
            event["seq"] = _next_seq()

        # Buffer first so even an event with no subscribers is replayable.
        with self._buffers_lock:
            buf = self._buffers.get(document_id)
            if buf is None:
                buf = collections.deque(maxlen=self._buffer_size)
                self._buffers[document_id] = buf
            buf.append(event)

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
