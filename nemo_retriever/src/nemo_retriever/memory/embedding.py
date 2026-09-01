# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text embedding for memory writes and recall queries.

Reuses the same batch embed actors the retrieval path uses, so memories are
embedded by the model that will later search them. Actors are cached per
``input_type`` because constructing one can load a local model.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Literal, Optional, Sequence

logger = logging.getLogger(__name__)

EmbedRunMode = Literal["local", "service"]


class MemoryEmbedder:
    """Embed memory text and recall queries with a shared model configuration.

    Parameters
    ----------
    run_mode
        ``"local"`` resolves an on-box embedder; ``"service"`` requires an HTTP
        embedding endpoint in ``embed_kwargs``.
    embed_kwargs
        Overrides merged onto the defaults and validated as
        :class:`~nemo_retriever.common.params.EmbedParams`.
    """

    def __init__(
        self,
        *,
        run_mode: EmbedRunMode = "local",
        embed_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if run_mode not in ("local", "service"):
            raise ValueError(f"run_mode must be 'local' or 'service', got {run_mode!r}")
        self.run_mode = run_mode
        self.embed_kwargs = dict(embed_kwargs or {})
        self._actors: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def _build_params(self, input_type: str) -> Any:
        from nemo_retriever.graph.retriever import Retriever

        # Delegate to the retrieval path's own defaulting so memory vectors and
        # query vectors are always produced by the same resolved model.
        probe = Retriever(run_mode=self.run_mode, embed_kwargs=self.embed_kwargs)
        return probe._merge_embed_params({"input_type": input_type})

    def _actor(self, input_type: str) -> Any:
        with self._lock:
            actor = self._actors.get(input_type)
            if actor is not None:
                return actor

            params = self._build_params(input_type)
            if self.run_mode == "service":
                from nemo_retriever.operators.embed.cpu_operator import _BatchEmbedCPUActor

                actor = _BatchEmbedCPUActor(params=params)
            else:
                from nemo_retriever.operators.embed.operators import _BatchEmbedActor

                actor = _BatchEmbedActor(params=params)
            self._actors[input_type] = actor
            return actor

    def embed(self, texts: Sequence[str], *, input_type: str = "passage") -> List[List[float]]:
        """Return one embedding vector per input text."""
        values = [str(text) for text in texts]
        if not values:
            return []

        import pandas as pd

        from nemo_retriever.operators.vdb import query_vectors_from_embedded_dataframe

        frame = pd.DataFrame({"text": values})
        embedded = self._actor(input_type).process(frame)
        vectors = query_vectors_from_embedded_dataframe(embedded)
        if len(vectors) != len(values):
            raise RuntimeError(f"Embedder returned {len(vectors)} vectors for {len(values)} inputs")
        return vectors

    def embed_one(self, text: str, *, input_type: str = "query") -> List[float]:
        """Return the embedding for a single string."""
        return self.embed([text], input_type=input_type)[0]

    @property
    def vector_dim(self) -> Optional[int]:
        """Return the configured embedding width when the model declares one."""
        dimensions = self.embed_kwargs.get("dimensions")
        return int(dimensions) if dimensions else None
