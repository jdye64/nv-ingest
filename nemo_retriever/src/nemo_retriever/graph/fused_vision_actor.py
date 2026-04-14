# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused vision actor that runs page-element detection, table structure,
graphic elements, and OCR in a single Ray Data ``map_batches`` stage.

Eliminates 3 intermediate object-store serialization boundaries that the
separate-stage pipeline would incur — the page image data is read from
the object store once and stays in-process memory for all 4 model stages.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.gpu_operator import GPUOperator

logger = logging.getLogger(__name__)


class FusedVisionActor(AbstractOperator, GPUOperator):
    """Single GPU actor that runs the full vision pipeline in-process.

    Loads all four vision sub-actors once in ``__init__`` and delegates
    to them sequentially in ``process()``.  The DataFrame never leaves
    process memory between stages, avoiding Ray object-store
    serialization of page images between each model.
    """

    def __init__(
        self,
        *,
        # --- PageElementDetection ---
        page_elements_trt_engine_path: Optional[str] = None,
        page_elements_invoke_url: Optional[str] = None,
        inference_batch_size: int = 32,
        # --- TableStructure ---
        use_table_structure: bool = True,
        extract_tables: bool = True,
        table_structure_trt_engine_path: Optional[str] = None,
        table_structure_invoke_url: Optional[str] = None,
        table_output_format: Optional[str] = None,
        # --- GraphicElements ---
        use_graphic_elements: bool = True,
        extract_charts: bool = True,
        graphic_elements_trt_engine_path: Optional[str] = None,
        graphic_elements_invoke_url: Optional[str] = None,
        # --- OCR ---
        extract_text: bool = False,
        extract_infographics: bool = False,
        ocr_trt_engine_path: Optional[str] = None,
        ocr_invoke_url: Optional[str] = None,
        # --- Common ---
        api_key: Optional[str] = None,
        request_timeout_s: float = 120.0,
        remote_max_pool_workers: int = 16,
        remote_max_retries: int = 10,
        remote_max_429_retries: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            page_elements_trt_engine_path=page_elements_trt_engine_path,
            page_elements_invoke_url=page_elements_invoke_url,
            inference_batch_size=inference_batch_size,
            use_table_structure=use_table_structure,
            extract_tables=extract_tables,
            table_structure_trt_engine_path=table_structure_trt_engine_path,
            table_structure_invoke_url=table_structure_invoke_url,
            table_output_format=table_output_format,
            use_graphic_elements=use_graphic_elements,
            extract_charts=extract_charts,
            graphic_elements_trt_engine_path=graphic_elements_trt_engine_path,
            graphic_elements_invoke_url=graphic_elements_invoke_url,
            extract_text=extract_text,
            extract_infographics=extract_infographics,
            ocr_trt_engine_path=ocr_trt_engine_path,
            ocr_invoke_url=ocr_invoke_url,
            api_key=api_key,
            request_timeout_s=request_timeout_s,
            remote_max_pool_workers=remote_max_pool_workers,
            remote_max_retries=remote_max_retries,
            remote_max_429_retries=remote_max_429_retries,
            **kwargs,
        )

        self._use_table_structure = use_table_structure and extract_tables
        self._use_graphic_elements = use_graphic_elements and extract_charts

        _remote_common = {
            "api_key": api_key,
            "request_timeout_s": request_timeout_s,
            "remote_max_pool_workers": remote_max_pool_workers,
            "remote_max_retries": remote_max_retries,
            "remote_max_429_retries": remote_max_429_retries,
        }

        # ---- 1. Page Element Detection ------------------------------------
        from nemo_retriever.page_elements.gpu_actor import (
            PageElementDetectionActor as _PEDActor,
        )

        detect_kw: dict[str, Any] = {
            "inference_batch_size": inference_batch_size,
        }
        if page_elements_trt_engine_path:
            detect_kw["page_elements_trt_engine_path"] = page_elements_trt_engine_path
        if page_elements_invoke_url:
            detect_kw["page_elements_invoke_url"] = page_elements_invoke_url
        if api_key:
            detect_kw["api_key"] = api_key
        self._detect = _PEDActor(**detect_kw)
        logger.info("FusedVisionActor: PageElementDetection sub-actor loaded.")

        # ---- 2. Table Structure + OCR (conditional) -----------------------
        self._table: Optional[Any] = None
        if self._use_table_structure:
            from nemo_retriever.table.gpu_actor import (
                TableStructureActor as _TSActor,
            )

            table_kw: dict[str, Any] = {**_remote_common}
            if table_structure_trt_engine_path:
                table_kw["table_structure_trt_engine_path"] = table_structure_trt_engine_path
            if table_structure_invoke_url:
                table_kw["table_structure_invoke_url"] = table_structure_invoke_url
            if table_output_format:
                table_kw["table_output_format"] = table_output_format
            if ocr_trt_engine_path:
                table_kw["ocr_trt_engine_path"] = ocr_trt_engine_path
            if ocr_invoke_url:
                table_kw["ocr_invoke_url"] = ocr_invoke_url
            self._table = _TSActor(**table_kw)
            logger.info("FusedVisionActor: TableStructure sub-actor loaded.")

        # ---- 3. Graphic Elements + OCR (conditional) ----------------------
        self._graphic: Optional[Any] = None
        if self._use_graphic_elements:
            from nemo_retriever.chart.gpu_actor import (
                GraphicElementsActor as _GEActor,
            )

            graphic_kw: dict[str, Any] = {**_remote_common}
            if graphic_elements_trt_engine_path:
                graphic_kw["graphic_elements_trt_engine_path"] = graphic_elements_trt_engine_path
            if graphic_elements_invoke_url:
                graphic_kw["graphic_elements_invoke_url"] = graphic_elements_invoke_url
            if ocr_trt_engine_path:
                graphic_kw["ocr_trt_engine_path"] = ocr_trt_engine_path
            if ocr_invoke_url:
                graphic_kw["ocr_invoke_url"] = ocr_invoke_url
            self._graphic = _GEActor(**graphic_kw)
            logger.info("FusedVisionActor: GraphicElements sub-actor loaded.")

        # ---- 4. OCR (conditional — for text / non-table/chart crops) ------
        self._ocr: Optional[Any] = None
        self._needs_ocr = any([extract_text, extract_infographics,
                               (extract_tables and not use_table_structure),
                               (extract_charts and not use_graphic_elements)])
        if self._needs_ocr:
            from nemo_retriever.ocr.gpu_ocr import OCRActor as _OCRActor

            ocr_kw: dict[str, Any] = {
                "extract_text": extract_text,
                "extract_tables": extract_tables and not use_table_structure,
                "extract_charts": extract_charts and not use_graphic_elements,
                "extract_infographics": extract_infographics,
                "use_graphic_elements": use_graphic_elements,
                "inference_batch_size": inference_batch_size,
                **_remote_common,
            }
            if ocr_trt_engine_path:
                ocr_kw["ocr_trt_engine_path"] = ocr_trt_engine_path
            if ocr_invoke_url:
                ocr_kw["ocr_invoke_url"] = ocr_invoke_url
            self._ocr = _OCRActor(**ocr_kw)
            logger.info("FusedVisionActor: OCR sub-actor loaded.")

        logger.info(
            "FusedVisionActor ready: detect=True, table=%s, graphic=%s, ocr=%s",
            self._use_table_structure,
            self._use_graphic_elements,
            self._needs_ocr,
        )

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, batch_df: Any, **kwargs: Any) -> Any:
        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return batch_df

        t0 = time.perf_counter()

        batch_df = self._detect.run(batch_df)

        if self._table is not None:
            batch_df = self._table.run(batch_df)

        if self._graphic is not None:
            batch_df = self._graphic.run(batch_df)

        if self._ocr is not None:
            batch_df = self._ocr.run(batch_df)

        elapsed = time.perf_counter() - t0
        logger.debug(
            "FusedVisionActor: processed %d rows in %.2fs",
            len(batch_df), elapsed,
        )
        return batch_df

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def __call__(self, batch_df: Any, **override_kwargs: Any) -> Any:
        try:
            return self.run(batch_df, **override_kwargs)
        except BaseException as exc:
            logger.exception("FusedVisionActor: unhandled error in run()")
            if isinstance(batch_df, pd.DataFrame):
                out = batch_df.copy()
                payload = {
                    "timing": None,
                    "error": {
                        "stage": "fused_vision_actor_call",
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                }
                n = len(out.index)
                from nemo_retriever.page_elements.shared import _error_payload
                out["page_elements_v3"] = [_error_payload(stage="fused_actor", exc=exc) for _ in range(n)]
                out["page_elements_v3_num_detections"] = [0] * n
                out["page_elements_v3_counts_by_label"] = [{}] * n
                return out
            raise
