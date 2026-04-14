# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused vision actor — all vision models in one Ray Data stage.

Restructured pipeline:

1. **Overlapped detection + JPEG decode** — DALI NVJPEG detects on GPU
   while cv2 decodes JPEGs to numpy on CPU threads concurrently.
2. **Table-structure TRT** — batched GPU inference (no OCR yet).
3. **Graphic-elements TRT** — batched GPU inference (no OCR yet).
   Runs sequentially after table-structure since both share the same
   GPU; threading them only adds contention.
4. **Unified OCR** — ALL crops (table + chart) are collected and run
   through the OCR model in a single batched pass, eliminating the
   per-crop GPU syncs that were the main bottleneck.
5. **Join + stitch** — structure/GE detections are joined with OCR
   results per-crop and written to the DataFrame.
6. **Compact** — decoded pixels are swapped back to JPEG bytes for
   compact object-store serialization.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.gpu_operator import GPUOperator

logger = logging.getLogger(__name__)

_DECODE_THREADS = 8
_OCR_BATCH_SIZE = 8


def _decode_one(pair: Tuple[int, bytes]) -> Tuple[int, "np.ndarray | None", bytes]:
    """Decode a single JPEG buffer. Runs in a thread (cv2 releases GIL)."""
    idx, jpeg = pair
    arr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    return idx, arr, jpeg


class FusedVisionActor(AbstractOperator, GPUOperator):
    """Single GPU actor that runs the full vision pipeline in-process.

    Loads all vision sub-actors once in ``__init__`` and orchestrates
    a restructured pipeline that batches ALL OCR crops into a single
    pass instead of running per-crop OCR inside each model stage.
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

        self._pool = ThreadPoolExecutor(
            max_workers=max(4, _DECODE_THREADS),
            thread_name_prefix="fused_vision",
        )

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

        # ---- 2. Table Structure (conditional) ------------------------------
        self._table: Optional[Any] = None
        if self._use_table_structure:
            from nemo_retriever.table.gpu_actor import (
                TableStructureActor as _TSActor,
            )

            table_kw: dict[str, Any] = {
                **_remote_common,
                "inference_batch_size": inference_batch_size,
            }
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

        # ---- 3. Graphic Elements (conditional) -----------------------------
        self._graphic: Optional[Any] = None
        if self._use_graphic_elements:
            from nemo_retriever.chart.gpu_actor import (
                GraphicElementsActor as _GEActor,
            )

            graphic_kw: dict[str, Any] = {
                **_remote_common,
                "inference_batch_size": inference_batch_size,
            }
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

        # ---- 4. OCR standalone (for text / infographic crops) ---------------
        self._ocr: Optional[Any] = None
        self._needs_standalone_ocr = any([
            extract_text, extract_infographics,
            (extract_tables and not use_table_structure),
            (extract_charts and not use_graphic_elements),
        ])
        if self._needs_standalone_ocr:
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

        # ---- Extract a single OCR model reference for unified OCR pass -----
        self._ocr_model: Optional[Any] = None
        if self._table is not None:
            self._ocr_model = getattr(self._table, "_ocr_model", None)
        if self._ocr_model is None and self._graphic is not None:
            self._ocr_model = getattr(self._graphic, "_ocr_model", None)

        self._has_deferred_ocr = (
            self._ocr_model is not None
            and (self._use_table_structure or self._use_graphic_elements)
        )

        logger.info(
            "FusedVisionActor ready: detect=True, table=%s, graphic=%s, "
            "standalone_ocr=%s, deferred_ocr=%s",
            self._use_table_structure,
            self._use_graphic_elements,
            self._needs_standalone_ocr,
            self._has_deferred_ocr,
        )

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    # ------------------------------------------------------------------
    # JPEG decode helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _restore_jpeg_bytes(
        batch_df: pd.DataFrame, saved: Dict[int, bytes],
    ) -> None:
        """Swap decoded pixels back to JPEG bytes for compact serialization."""
        for idx, jpeg in saved.items():
            pi = batch_df.at[idx, "page_image"]
            if isinstance(pi, dict):
                pi.pop("pixels", None)
                pi["jpeg_bytes"] = jpeg

    # ------------------------------------------------------------------
    # Unified OCR pass
    # ------------------------------------------------------------------

    def _run_deferred_ocr(self, batch_df: pd.DataFrame) -> None:
        """Collect ALL pending OCR crops, run OCR once, join & stitch."""
        from nemo_retriever.ocr.ocr import (
            _blocks_to_pseudo_markdown,
            _blocks_to_text,
            _parse_ocr_result,
        )
        from nemo_retriever.utils.table_and_chart import (
            join_graphic_elements_and_ocr_output,
            join_table_structure_and_ocr_output,
        )

        has_table = "_table_pending_ocr" in batch_df.columns
        has_chart = "_chart_pending_ocr" in batch_df.columns

        if not has_table and not has_chart:
            return

        # --- Flatten all crops into one list ---
        all_crop_arrays: List[np.ndarray] = []
        # (source_type, df_idx, crop_idx_within_row)
        crop_sources: List[Tuple[str, Any, int]] = []

        if has_table:
            for idx in batch_df.index:
                for ci, p in enumerate(batch_df.at[idx, "_table_pending_ocr"]):
                    all_crop_arrays.append(p["crop_array"])
                    crop_sources.append(("table", idx, ci))

        if has_chart:
            for idx in batch_df.index:
                for ci, p in enumerate(batch_df.at[idx, "_chart_pending_ocr"]):
                    all_crop_arrays.append(p["crop_array"])
                    crop_sources.append(("chart", idx, ci))

        n_total = len(all_crop_arrays)
        if n_total == 0:
            if has_table:
                batch_df.drop(columns=["_table_pending_ocr"], inplace=True)
            if has_chart:
                batch_df.drop(columns=["_chart_pending_ocr"], inplace=True)
            return

        # --- Single batched OCR pass ---
        ocr_results: List[Any] = [None] * n_total
        for start in range(0, n_total, _OCR_BATCH_SIZE):
            batch_slice = all_crop_arrays[start : start + _OCR_BATCH_SIZE]
            try:
                batch_preds = self._ocr_model.invoke(
                    batch_slice, merge_level="word",
                )
                if isinstance(batch_preds, list) and len(batch_preds) == len(batch_slice):
                    for j, pred in enumerate(batch_preds):
                        ocr_results[start + j] = pred
                    continue
            except Exception:
                pass
            for j, crop in enumerate(batch_slice):
                try:
                    ocr_results[start + j] = self._ocr_model.invoke(
                        crop, merge_level="word",
                    )
                except Exception:
                    pass

        # --- Stitch table results ---
        if has_table:
            all_table: List[List[Dict[str, Any]]] = batch_df["table"].tolist()
            for oci, (src_type, idx, ci) in enumerate(crop_sources):
                if src_type != "table":
                    continue
                p = batch_df.at[idx, "_table_pending_ocr"][ci]
                crop_hw = (int(p["crop_array"].shape[0]), int(p["crop_array"].shape[1]))
                ocr_preds = ocr_results[oci]

                markdown = join_table_structure_and_ocr_output(
                    p["structure_dets"], ocr_preds, crop_hw,
                )
                if not markdown:
                    blocks = _parse_ocr_result(ocr_preds)
                    markdown = (
                        _blocks_to_pseudo_markdown(blocks, crop_hw=crop_hw)
                        or _blocks_to_text(blocks)
                    )

                row_pos = batch_df.index.get_loc(idx)
                all_table[row_pos].append(
                    {"bbox_xyxy_norm": p["bbox"], "text": markdown},
                )
            batch_df["table"] = all_table
            batch_df.drop(columns=["_table_pending_ocr"], inplace=True)

        # --- Stitch chart results ---
        if has_chart:
            all_chart: List[List[Dict[str, Any]]] = batch_df["chart"].tolist()
            for oci, (src_type, idx, ci) in enumerate(crop_sources):
                if src_type != "chart":
                    continue
                p = batch_df.at[idx, "_chart_pending_ocr"][ci]
                crop_hw = (int(p["crop_array"].shape[0]), int(p["crop_array"].shape[1]))
                ocr_preds = ocr_results[oci]

                text = join_graphic_elements_and_ocr_output(
                    p["ge_dets"], ocr_preds, crop_hw,
                )
                if not text:
                    blocks = _parse_ocr_result(ocr_preds)
                    text = _blocks_to_text(blocks)

                row_pos = batch_df.index.get_loc(idx)
                all_chart[row_pos].append(
                    {"bbox_xyxy_norm": p["bbox"], "text": text},
                )
            batch_df["chart"] = all_chart
            batch_df.drop(columns=["_chart_pending_ocr"], inplace=True)

    # ------------------------------------------------------------------
    # Main processing pipeline
    # ------------------------------------------------------------------

    def process(self, batch_df: Any, **kwargs: Any) -> Any:
        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return batch_df

        batch_df = batch_df.copy()

        t0 = time.perf_counter()

        # ===== Phase 1: Overlapped GPU detection + CPU JPEG decode =====
        # Submit individual decode tasks to self._pool (no temp pool).
        # Detection reads jpeg_bytes via DALI NVJPEG on GPU while cv2
        # decodes the same bytes on CPU threads concurrently.
        decode_futures: List[Future] = []
        saved_jpegs: Dict[int, bytes] = {}

        if "page_image" in batch_df.columns:
            for idx in batch_df.index:
                pi = batch_df.at[idx, "page_image"]
                if isinstance(pi, dict) and pi.get("jpeg_bytes") is not None:
                    decode_futures.append(
                        self._pool.submit(_decode_one, (idx, pi["jpeg_bytes"])),
                    )

        batch_df = self._detect.process(batch_df, _inplace=True)

        for fut in decode_futures:
            r_idx, arr, jpeg = fut.result()
            if arr is not None:
                pi = batch_df.at[r_idx, "page_image"]
                pi["pixels"] = arr
                saved_jpegs[r_idx] = jpeg

        for r_idx in saved_jpegs:
            pi = batch_df.at[r_idx, "page_image"]
            pi.pop("jpeg_bytes", None)

        t_detect = time.perf_counter() - t0

        # ===== Phase 2: Table structure (skip OCR) — sequential GPU =====
        t1 = time.perf_counter()
        if self._table is not None:
            batch_df = self._table.process(
                batch_df, _inplace=True, _skip_ocr=True,
            )
        t_ts = time.perf_counter() - t1

        # ===== Phase 3: Graphic elements (skip OCR) — sequential GPU ====
        t2 = time.perf_counter()
        if self._graphic is not None:
            batch_df = self._graphic.process(
                batch_df, _inplace=True, _skip_ocr=True,
            )
        t_ge = time.perf_counter() - t2

        # ===== Phase 4: Unified OCR on ALL crops ========================
        t3 = time.perf_counter()
        if self._has_deferred_ocr:
            self._run_deferred_ocr(batch_df)
        t_ocr = time.perf_counter() - t3

        # ===== Phase 5: Standalone OCR (text / infographic only) ========
        t4 = time.perf_counter()
        if self._ocr is not None:
            batch_df = self._ocr.process(batch_df, _inplace=True)
        t_standalone_ocr = time.perf_counter() - t4

        # ===== Phase 6: Compact for serialization =======================
        t5 = time.perf_counter()
        self._restore_jpeg_bytes(batch_df, saved_jpegs)
        t_ser = time.perf_counter() - t5

        elapsed = time.perf_counter() - t0

        n_det = 0
        for idx in batch_df.index:
            pe = batch_df.at[idx, "page_elements_v3"] if "page_elements_v3" in batch_df.columns else None
            if isinstance(pe, dict):
                n_det += len(pe.get("detections") or [])

        n_table = 0
        n_chart = 0
        if "table" in batch_df.columns:
            for idx in batch_df.index:
                tl = batch_df.at[idx, "table"]
                if isinstance(tl, list):
                    n_table += len(tl)
        if "chart" in batch_df.columns:
            for idx in batch_df.index:
                cl = batch_df.at[idx, "chart"]
                if isinstance(cl, list):
                    n_chart += len(cl)

        logger.info(
            "FusedVisionActor: %d rows, %d dets, %d table, %d chart "
            "in %.2fs (detect=%.2fs, ts=%.2fs, ge=%.2fs, "
            "ocr=%.2fs, standalone_ocr=%.2fs, ser=%.2fs)",
            len(batch_df), n_det, n_table, n_chart,
            elapsed, t_detect, t_ts, t_ge,
            t_ocr, t_standalone_ocr, t_ser,
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
                n = len(out.index)
                from nemo_retriever.page_elements.shared import _error_payload
                out["page_elements_v3"] = [_error_payload(stage="fused_actor", exc=exc) for _ in range(n)]
                out["page_elements_v3_num_detections"] = [0] * n
                out["page_elements_v3_counts_by_label"] = [{}] * n
                return out
            raise
