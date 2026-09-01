# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent memory: episodic capture, semantic recall, and consolidation.

The public entry point is
:class:`~nemo_retriever.ingestor.agentic_ingestor.AgenticIngestor`. This
package holds the pieces it composes: the storage-facing backends, the
embedder, the consolidation pass, and the capture helpers.

Imports here stay lazy because the local backend pulls in the embedding stack
while the service backend only needs an HTTP client.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "MemoryBackend",
    "LocalMemoryBackend",
    "ServiceMemoryBackend",
    "MemoryEmbedder",
    "MemoryTap",
    "consolidate_session",
    "import_atif_trajectory",
    "import_atif_directory",
]


def __getattr__(name: str) -> Any:
    if name in {"MemoryBackend", "LocalMemoryBackend", "ServiceMemoryBackend"}:
        from nemo_retriever.memory import backends

        return getattr(backends, name)
    if name == "MemoryEmbedder":
        from nemo_retriever.memory.embedding import MemoryEmbedder

        return MemoryEmbedder
    if name == "MemoryTap":
        from nemo_retriever.memory.tap import MemoryTap

        return MemoryTap
    if name == "consolidate_session":
        from nemo_retriever.memory.consolidation import consolidate_session

        return consolidate_session
    if name in {"import_atif_trajectory", "import_atif_directory"}:
        from nemo_retriever.memory import atif_import

        return getattr(atif_import, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
