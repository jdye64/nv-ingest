# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Ray Data adapter for .html: HtmlSplitActor turns bytes+path batches into chunk rows.
"""

from __future__ import annotations

from typing import Any, List

import time
import pandas as pd

from retriever.params import HtmlChunkParams
from retriever.utils.metrics_actor import emit_actor_metrics, estimate_batch_rows

from .convert import html_bytes_to_chunks_df


class HtmlSplitActor:
    """
    Ray Data map_batches callable: DataFrame with bytes, path -> DataFrame of chunks.

    Each output row has: text, path, page_number, metadata (same shape as html_file_to_chunks_df).
    """

    def __init__(
        self,
        params: HtmlChunkParams | None = None,
        metrics_actor: Any = None,
        stage_name: str = "html_split",
    ) -> None:
        self._params = params or HtmlChunkParams()
        self._metrics_actor = metrics_actor
        self._stage_name = str(stage_name)

    def __call__(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        t0 = time.perf_counter()
        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            out = pd.DataFrame(columns=["text", "path", "page_number", "metadata"])
            emit_actor_metrics(
                self._metrics_actor,
                stage=self._stage_name,
                duration_sec=(time.perf_counter() - t0),
                input_rows=estimate_batch_rows(batch_df),
                output_rows=estimate_batch_rows(out),
                ok=True,
            )
            return out

        params = self._params
        out_dfs: List[pd.DataFrame] = []
        for _, row in batch_df.iterrows():
            raw = row.get("bytes")
            path = row.get("path")
            if raw is None or path is None:
                continue
            path_str = str(path) if path is not None else ""
            try:
                chunk_df = html_bytes_to_chunks_df(raw, path_str, params=params)
                if not chunk_df.empty:
                    out_dfs.append(chunk_df)
            except Exception:
                continue
        if not out_dfs:
            out = pd.DataFrame(columns=["text", "path", "page_number", "metadata"])
            emit_actor_metrics(
                self._metrics_actor,
                stage=self._stage_name,
                duration_sec=(time.perf_counter() - t0),
                input_rows=estimate_batch_rows(batch_df),
                output_rows=estimate_batch_rows(out),
                ok=True,
            )
            return out
        out = pd.concat(out_dfs, ignore_index=True)
        emit_actor_metrics(
            self._metrics_actor,
            stage=self._stage_name,
            duration_sec=(time.perf_counter() - t0),
            input_rows=estimate_batch_rows(batch_df),
            output_rows=estimate_batch_rows(out),
            ok=True,
        )
        return out
