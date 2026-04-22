# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thread-safe SQLite connection pool using ``threading.local``."""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS documents (
    id              TEXT PRIMARY KEY,
    filename        TEXT NOT NULL,
    content_type    TEXT NOT NULL DEFAULT 'application/octet-stream',
    content_sha256  TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    total_pages     INTEGER,
    pages_received  INTEGER NOT NULL DEFAULT 0,
    processing_status TEXT NOT NULL DEFAULT 'queued',
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_sha256 ON documents(content_sha256);

CREATE TABLE IF NOT EXISTS page_results (
    id          TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id),
    page_number INTEGER NOT NULL,
    content_json TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_page_results_doc ON page_results(document_id);

CREATE TABLE IF NOT EXISTS processing_metrics (
    id               TEXT PRIMARY KEY,
    document_id      TEXT NOT NULL REFERENCES documents(id),
    model_name       TEXT NOT NULL,
    invocation_count INTEGER NOT NULL DEFAULT 0,
    pages_processed  INTEGER NOT NULL DEFAULT 0,
    detections_count INTEGER NOT NULL DEFAULT 0,
    duration_ms      REAL NOT NULL DEFAULT 0.0,
    created_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_metrics_doc ON processing_metrics(document_id);
"""


class DatabaseEngine:
    """Manages thread-local SQLite connections to a single database file."""

    def __init__(self, db_path: str) -> None:
        self._db_path = str(Path(db_path).resolve())
        self._local = threading.local()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @property
    def connection(self) -> sqlite3.Connection:
        """Return (or create) the thread-local connection."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    def initialize(self) -> None:
        """Create tables and indexes if they do not exist."""
        conn = self.connection
        conn.executescript(_DDL)
        conn.commit()
        logger.info("SQLite database initialized at %s", self._db_path)

    def close(self) -> None:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
