# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock


@dataclass(frozen=True, slots=True)
class StoredDocument:
    filename: str
    data: bytes
    content_type: str
    stored_at: float


def _source_keys(name: str) -> list[str]:
    keys: list[str] = []
    if not name or not str(name).strip():
        return keys
    name = str(name).strip()
    keys.append(name)
    path = Path(name)
    if path.name and path.name not in keys:
        keys.append(path.name)
    try:
        resolved = str(path.expanduser().resolve())
        if resolved not in keys:
            keys.append(resolved)
    except OSError:
        pass
    return keys


class DocumentStore:
    """In-memory store of uploaded document bytes keyed by source identifiers."""

    def __init__(self, *, ttl_s: int = 86400, max_documents: int = 500) -> None:
        self._ttl_s = ttl_s
        self._max_documents = max_documents
        self._by_key: dict[str, StoredDocument] = {}
        self._lock = Lock()

    def _evict_expired(self, now: float) -> None:
        stale = [key for key, doc in self._by_key.items() if now - doc.stored_at > self._ttl_s]
        for key in stale:
            self._by_key.pop(key, None)

    def _trim(self) -> None:
        if len(self._by_key) <= self._max_documents:
            return
        ranked = sorted(self._by_key.items(), key=lambda item: item[1].stored_at)
        for key, _ in ranked[: len(self._by_key) - self._max_documents]:
            self._by_key.pop(key, None)

    def register_file(
        self,
        source_name: str,
        data: bytes,
        *,
        display_name: str | None = None,
    ) -> None:
        if not data:
            return
        filename = display_name or Path(source_name).name or "document"
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        now = time.time()
        doc = StoredDocument(filename=filename, data=data, content_type=content_type, stored_at=now)
        with self._lock:
            self._evict_expired(now)
            for key in _source_keys(source_name):
                self._by_key[key] = doc
            if filename not in self._by_key:
                self._by_key[filename] = doc
            self._trim()

    def get(self, source_key: str) -> StoredDocument | None:
        if not source_key or not str(source_key).strip():
            return None
        now = time.time()
        with self._lock:
            self._evict_expired(now)
            doc = self._by_key.get(str(source_key).strip())
            if doc is not None and now - doc.stored_at <= self._ttl_s:
                return doc
            basename = Path(source_key).name
            if basename and basename != source_key:
                doc = self._by_key.get(basename)
                if doc is not None and now - doc.stored_at <= self._ttl_s:
                    return doc
        return None
