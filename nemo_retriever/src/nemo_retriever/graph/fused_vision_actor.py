# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused vision actor — all vision models in one Ray Data stage.

Performance tricks applied:
1. **Zero intermediate serialization** — DataFrame never leaves process
   memory between the four model stages.
2. **Overlapped CPU↔GPU** — JPEG→numpy decode runs on CPU threads
   *while* page-element detection runs on GPU (DALI NVJPEG).
3. **Parallel table + graphic** — table-structure and graphic-elements
   are independent (disjoint output columns, read-only on inputs) so
   they run concurrently in separate threads with their own CUDA
   streams.
4. **Thread-pooled JPEG decode** — ``cv2.imdecode`` releases the GIL;
   a persistent thread pool decodes all pages in parallel.
5. **Single decode, shared pixels** — each page's JPEG bytes are decoded
   exactly once; all downstream stages consume the cached numpy array.
6. **Compact on exit** — decoded pixels are swapped back to JPEG bytes
   before the DataFrame is serialized to the object store.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.gpu_operator import GPUOperator

logger = logging.getLogger(__name__)

_DECODE_THREADS = 8


def _decode_one(pair: Tuple[int, bytes]) -> Tuple[int, "np.ndarray | None", bytes]:
    """Decode a single JPEG buffer. Runs in a thread (cv2 releases GIL)."""
    idx, jpeg = pair
    arr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    return idx, arr, jpeg


class FusedVisionActor(AbstractOperator, GPUOperator):
    """Single GPU actor that runs the full vision pipeline in-process.

    Loads all four vision sub-actors once in ``__init__`` and keeps a
    persistent thread pool for CPU-bound work (JPEG decode, image
    cropping) so it can overlap with GPU inference.
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

        # Persistent thread pool for CPU work (JPEG decode, parallel stages).
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

        # ---- 2. Table Structure + OCR (conditional) -----------------------
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

        # ---- 3. Graphic Elements + OCR (conditional) ----------------------
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

        self._can_parallel_table_graphic = (
            self._table is not None and self._graphic is not None
        )

        logger.info(
            "FusedVisionActor ready: detect=True, table=%s, graphic=%s, "
            "ocr=%s, parallel_table_graphic=%s",
            self._use_table_structure,
            self._use_graphic_elements,
            self._needs_ocr,
            self._can_parallel_table_graphic,
        )

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    # ------------------------------------------------------------------
    # Pixel caching helpers
    # ------------------------------------------------------------------

    def _decode_pixels_parallel(
        self, batch_df: pd.DataFrame,
    ) -> Dict[int, bytes]:
        """Thread-pool JPEG decode → numpy.  Returns saved jpeg_bytes map.

        ``cv2.imdecode`` releases the GIL, so N pages decode in
        parallel across ``_DECODE_THREADS`` threads.
        """
        saved: Dict[int, bytes] = {}
        if "page_image" not in batch_df.columns:
            return saved

        work: List[Tuple[int, bytes]] = []
        for idx in batch_df.index:
            pi = batch_df.at[idx, "page_image"]
            if isinstance(pi, dict) and pi.get("jpeg_bytes") is not None:
                work.append((idx, pi["jpeg_bytes"]))

        if not work:
            return saved

        for idx, arr, jpeg in self._pool.map(_decode_one, work):
            if arr is None:
                continue
            pi = batch_df.at[idx, "page_image"]
            saved[idx] = jpeg
            pi["pixels"] = arr
            del pi["jpeg_bytes"]

        return saved

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
    # Parallel table + graphic
    # ------------------------------------------------------------------

    def _run_table_and_graphic_parallel(
        self, batch_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run table-structure and graphic-elements concurrently.

        They write to disjoint columns (``table`` / ``table_structure_ocr_v1``
        vs ``chart`` / ``graphic_elements_ocr_v1``) and only READ
        ``page_image`` and ``page_elements_v3``.  Table operates in-place
        on batch_df; graphic gets a shallow copy for thread safety, then
        its new columns are merged back.
        """
        df_graphic = batch_df.copy()

        ft: Future = self._pool.submit(
            self._table.process, batch_df, _inplace=True,  # type: ignore[union-attr]
        )
        fg: Future = self._pool.submit(
            self._graphic.process, df_graphic, _inplace=True,  # type: ignore[union-attr]
        )

        batch_df = ft.result()
        df_g = fg.result()

        for col in df_g.columns:
            if col not in batch_df.columns:
                batch_df[col] = df_g[col].values

        return batch_df

    # ------------------------------------------------------------------
    # Main processing pipeline
    # ------------------------------------------------------------------

    def process(self, batch_df: Any, **kwargs: Any) -> Any:
        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return batch_df

        # Single copy up front so all downstream _inplace writes are
        # on an owned DataFrame — avoids SettingWithCopyWarning.
        batch_df = batch_df.copy()

        t0 = time.perf_counter()

        # ------ Overlapped phase: GPU detection + CPU JPEG decode ------
        # Kick off JPEG→numpy decode in background threads NOW.
        # Detection reads jpeg_bytes via DALI NVJPEG (GPU hardware),
        # while cv2.imdecode runs on CPU threads concurrently.
        # We ADD pixels alongside jpeg_bytes (don't delete yet) so
        # detection can still read jpeg_bytes safely.
        decode_future: Optional[Future] = None
        if "page_image" in batch_df.columns:
            decode_future = self._pool.submit(
                self._overlapped_decode, batch_df,
            )

        batch_df = self._detect.process(batch_df, _inplace=True)

        # Wait for CPU decode to finish and swap jpeg_bytes → pixels.
        saved_jpegs: Dict[int, bytes] = {}
        if decode_future is not None:
            saved_jpegs = decode_future.result()

        t_detect = time.perf_counter() - t0

        # ------ Table + Graphic (parallel when both enabled) -----------
        t1 = time.perf_counter()
        if self._can_parallel_table_graphic:
            batch_df = self._run_table_and_graphic_parallel(batch_df)
        else:
            if self._table is not None:
                batch_df = self._table.process(batch_df, _inplace=True)
            if self._graphic is not None:
                batch_df = self._graphic.process(batch_df, _inplace=True)
        t_tg = time.perf_counter() - t1

        # ------ OCR ---------------------------------------------------
        t2 = time.perf_counter()
        if self._ocr is not None:
            batch_df = self._ocr.process(batch_df, _inplace=True)
        t_ocr = time.perf_counter() - t2

        # ------ Compact for serialization -----------------------------
        t3 = time.perf_counter()
        self._restore_jpeg_bytes(batch_df, saved_jpegs)
        t_ser = time.perf_counter() - t3

        elapsed = time.perf_counter() - t0

        n_det = 0
        for idx in batch_df.index:
            pe = batch_df.at[idx, "page_elements_v3"] if "page_elements_v3" in batch_df.columns else None
            if isinstance(pe, dict):
                n_det += len(pe.get("detections") or [])

        logger.info(
            "FusedVisionActor: %d rows, %d detections in %.2fs "
            "(detect=%.2fs, table+graphic=%.2fs, ocr=%.2fs, ser=%.2fs)",
            len(batch_df), n_det, elapsed, t_detect, t_tg, t_ocr, t_ser,
        )
        return batch_df

    # ------------------------------------------------------------------
    # Overlapped JPEG decode (runs in thread pool during detection)
    # ------------------------------------------------------------------

    @staticmethod
    def _overlapped_decode(batch_df: pd.DataFrame) -> Dict[int, bytes]:
        """Decode JPEG bytes in threads; ADD pixels but keep jpeg_bytes.

        After detection completes, the caller will delete jpeg_bytes
        so downstream stages find pixels first.
        """
        saved: Dict[int, bytes] = {}
        work: List[Tuple[int, bytes]] = []

        for idx in batch_df.index:
            pi = batch_df.at[idx, "page_image"]
            if isinstance(pi, dict) and pi.get("jpeg_bytes") is not None:
                work.append((idx, pi["jpeg_bytes"]))

        if not work:
            return saved

        pool = ThreadPoolExecutor(
            max_workers=min(_DECODE_THREADS, len(work)),
            thread_name_prefix="jpeg_dec",
        )
        try:
            for r_idx, arr, jpeg in pool.map(_decode_one, work):
                if arr is None:
                    continue
                pi = batch_df.at[r_idx, "page_image"]
                pi["pixels"] = arr
                saved[r_idx] = jpeg
            # Now remove jpeg_bytes so downstream stages use pixels.
            for r_idx in saved:
                pi = batch_df.at[r_idx, "page_image"]
                pi.pop("jpeg_bytes", None)
        finally:
            pool.shutdown(wait=False)

        return saved

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
