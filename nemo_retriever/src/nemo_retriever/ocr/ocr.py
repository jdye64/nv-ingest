# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""
Crop-based OCR stage — GPU-throughput-optimized.

Runs Nemotron OCR v1 on table / chart / infographic regions detected by
PageElements v3.  The key design decisions for throughput are:

1. **Cross-row batching** — crops are collected from *all* rows in the
   DataFrame first, grouped by merge-level, then fed to the model in
   maximally-packed batches.  This keeps the GPU saturated even when
   individual pages have few detections.

2. **Zero unnecessary copies** — crops use ``np.ascontiguousarray`` only
   when the slice is non-contiguous; the output DataFrame is mutated
   in-place rather than copied.

3. **GPU-warm actor** — ``OCRActor.__init__`` casts submodules to fp16,
   warms up the CUDA caching allocator, and sets ``expandable_segments``
   so the hot path triggers no new CUDA mallocs.
"""

from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import time
import traceback

import numpy as np
import pandas as pd
from nemo_retriever.params import RemoteRetryParams
from nemo_retriever.nim.nim import invoke_image_inference_batches
from nemo_retriever.util.image import (
    crop_all_from_page,
    get_page_image_array,
    np_rgb_to_b64_png,
)
from nemo_retriever.util.ocr_parsing import (
    blocks_to_pseudo_markdown,
    blocks_to_text,
    extract_remote_ocr_item,
    parse_ocr_result,
)
from nemo_retriever.util.table_and_chart import join_graphic_elements_and_ocr_output

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Named tuples for the cross-row batching pipeline
# ---------------------------------------------------------------------------


class _CropJob(NamedTuple):
    row_idx: int
    label: str
    bbox: List[float]
    crop: np.ndarray
    crop_hw: Tuple[int, int]


# ---------------------------------------------------------------------------
# Lightweight helpers
# ---------------------------------------------------------------------------


def _error_payload(*, stage: str, exc: BaseException) -> Dict[str, Any]:
    return {
        "timing": None,
        "error": {
            "stage": str(stage),
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        },
    }


# ---------------------------------------------------------------------------
# Graphic-elements integration
# ---------------------------------------------------------------------------


def _bboxes_close(a: Sequence[float], b: Sequence[float], tol: float = 1e-4) -> bool:
    if len(a) != 4 or len(b) != 4:
        return False
    return all(abs(float(a[i]) - float(b[i])) < tol for i in range(4))


def _find_ge_detections_for_bbox(
    row: Any,
    chart_bbox: Sequence[float],
) -> Optional[List[Dict[str, Any]]]:
    """Find graphic-element detections whose bbox matches *chart_bbox*."""
    ge_col = getattr(row, "graphic_elements_v1", None)
    if not isinstance(ge_col, dict):
        return None
    regions = ge_col.get("regions")
    if not isinstance(regions, list):
        return None

    for region in regions:
        if not isinstance(region, dict):
            continue
        region_bbox = region.get("bbox_xyxy_norm")
        if not isinstance(region_bbox, (list, tuple)) or len(region_bbox) != 4:
            continue
        if _bboxes_close(chart_bbox, region_bbox):
            dets = region.get("detections")
            if isinstance(dets, list) and dets:
                return dets
    return None


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------


def _format_ocr_result(
    label: str,
    preds: Any,
    crop_hw: Tuple[int, int],
    bbox: List[float],
    *,
    use_graphic_elements: bool,
    row: Any,
) -> Optional[Dict[str, Any]]:
    """Convert raw OCR predictions into a ``{"bbox_xyxy_norm", "text"}`` entry."""
    if label == "chart" and use_graphic_elements:
        ge_dets = _find_ge_detections_for_bbox(row, bbox)
        if ge_dets:
            text = join_graphic_elements_and_ocr_output(ge_dets, preds, crop_hw)
            if text:
                return {"bbox_xyxy_norm": bbox, "text": text}

    blocks = parse_ocr_result(preds)
    if label == "table":
        text = blocks_to_pseudo_markdown(blocks) or blocks_to_text(blocks)
    else:
        text = blocks_to_text(blocks)

    return {"bbox_xyxy_norm": bbox, "text": text}


# ---------------------------------------------------------------------------
# Cross-row crop collection
# ---------------------------------------------------------------------------


def _collect_crop_jobs(
    batch_df: pd.DataFrame,
    wanted_labels: set,
) -> Tuple[List[_CropJob], List[Any]]:
    """Collect all crop jobs from every row in *batch_df*."""
    jobs: List[_CropJob] = []
    rows: List[Any] = []

    for row_idx, row in enumerate(batch_df.itertuples(index=False)):
        rows.append(row)

        pe = getattr(row, "page_elements_v3", None)
        dets: List[Dict[str, Any]] = []
        if isinstance(pe, dict):
            dets = pe.get("detections") or []
        if not isinstance(dets, list):
            dets = []

        page_image = getattr(row, "page_image", None)
        crops = crop_all_from_page(page_image, dets, wanted_labels)

        for label_name, bbox, crop_array in crops:
            jobs.append(
                _CropJob(
                    row_idx=row_idx,
                    label=label_name,
                    bbox=bbox,
                    crop=crop_array,
                    crop_hw=(crop_array.shape[0], crop_array.shape[1]),
                )
            )

    return jobs, rows


# ---------------------------------------------------------------------------
# Local batch inference
# ---------------------------------------------------------------------------


def _invoke_local_batched(
    jobs: List[_CropJob],
    *,
    model: Any,
    merge_level: str,
    batch_size: int,
) -> List[Any]:
    """Invoke the local model on *jobs* in batches, returning aligned predictions."""
    all_preds: List[Any] = [None] * len(jobs)

    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start : start + batch_size]
        crops = [j.crop for j in chunk]

        try:
            batch_preds = model.invoke(crops, merge_level=merge_level)
        except Exception:
            batch_preds = None

        if isinstance(batch_preds, list) and len(batch_preds) == len(chunk):
            for i, pred in enumerate(batch_preds):
                all_preds[start + i] = pred
        else:
            for i, j in enumerate(chunk):
                try:
                    all_preds[start + i] = model.invoke(j.crop, merge_level=merge_level)
                except Exception:
                    all_preds[start + i] = None

    return all_preds


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def ocr_page_elements(
    batch_df: Any,
    *,
    model: Any = None,
    invoke_url: Optional[str] = None,
    api_key: Optional[str] = None,
    request_timeout_s: float = 120.0,
    extract_tables: bool = False,
    extract_charts: bool = False,
    extract_infographics: bool = False,
    use_graphic_elements: bool = False,
    inference_batch_size: int = 8,
    remote_retry: RemoteRetryParams | None = None,
    **kwargs: Any,
) -> Any:
    """Run Nemotron OCR v1 on cropped regions detected by PageElements v3.

    **Local inference path** (the GPU hot-path):

    1. Collect all crops from all rows in a single pass.
    2. Group by merge-level (``"word"`` for tables, ``"paragraph"`` otherwise).
    3. Feed maximally-packed batches to the model across all rows.
    4. Scatter results back to per-row accumulators.

    This cross-row batching keeps the GPU saturated even when individual
    pages have few detections.
    """
    retry = remote_retry or RemoteRetryParams(
        remote_max_pool_workers=int(kwargs.get("remote_max_pool_workers", 16)),
        remote_max_retries=int(kwargs.get("remote_max_retries", 10)),
        remote_max_429_retries=int(kwargs.get("remote_max_429_retries", 5)),
    )
    if not isinstance(batch_df, pd.DataFrame):
        raise NotImplementedError("ocr_page_elements currently only supports pandas.DataFrame input.")

    invoke_url = (invoke_url or kwargs.get("ocr_invoke_url") or "").strip()
    use_remote = bool(invoke_url)
    if not use_remote and model is None:
        raise ValueError("A local `model` is required when `invoke_url` is not provided.")

    wanted_labels: set[str] = set()
    if extract_tables:
        wanted_labels.add("table")
    if extract_charts:
        wanted_labels.add("chart")
    if extract_infographics:
        wanted_labels.add("infographic")

    n_rows = len(batch_df)
    row_table: List[List[Dict[str, Any]]] = [[] for _ in range(n_rows)]
    row_chart: List[List[Dict[str, Any]]] = [[] for _ in range(n_rows)]
    row_infographic: List[List[Dict[str, Any]]] = [[] for _ in range(n_rows)]
    row_error: List[Any] = [None] * n_rows

    t0 = time.perf_counter()

    try:
        if use_remote:
            _run_remote(
                batch_df,
                wanted_labels=wanted_labels,
                invoke_url=invoke_url,
                api_key=api_key,
                request_timeout_s=request_timeout_s,
                use_graphic_elements=use_graphic_elements,
                inference_batch_size=inference_batch_size,
                retry=retry,
                row_table=row_table,
                row_chart=row_chart,
                row_infographic=row_infographic,
                row_error=row_error,
                **kwargs,
            )
        else:
            _run_local(
                batch_df,
                model=model,
                wanted_labels=wanted_labels,
                use_graphic_elements=use_graphic_elements,
                inference_batch_size=inference_batch_size,
                row_table=row_table,
                row_chart=row_chart,
                row_infographic=row_infographic,
                row_error=row_error,
            )
    except BaseException as e:
        print(f"Warning: OCR failed: {type(e).__name__}: {e}")
        err = {
            "stage": "ocr_page_elements",
            "type": e.__class__.__name__,
            "message": str(e),
            "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
        }
        for i in range(n_rows):
            if row_error[i] is None:
                row_error[i] = err

    elapsed = time.perf_counter() - t0
    timing = {"seconds": float(elapsed)}

    all_ocr_meta = [{"timing": timing, "error": row_error[i]} for i in range(n_rows)]

    out = batch_df.copy()
    if extract_tables or "table" not in out.columns:
        out["table"] = row_table
    if extract_charts or "chart" not in out.columns:
        out["chart"] = row_chart
    if extract_infographics or "infographic" not in out.columns:
        out["infographic"] = row_infographic
    out["ocr_v1"] = all_ocr_meta
    return out


# ---------------------------------------------------------------------------
# Local inference path (GPU hot-path)
# ---------------------------------------------------------------------------


def _run_local(
    batch_df: pd.DataFrame,
    *,
    model: Any,
    wanted_labels: set,
    use_graphic_elements: bool,
    inference_batch_size: int,
    row_table: List[List[Dict[str, Any]]],
    row_chart: List[List[Dict[str, Any]]],
    row_infographic: List[List[Dict[str, Any]]],
    row_error: List[Any],
) -> None:
    """Cross-row batched local inference."""
    if inference_batch_size is None or inference_batch_size < 1:
        raise ValueError(f"inference_batch_size must be >= 1, got {inference_batch_size}")

    batch_size = max(1, int(inference_batch_size))

    jobs, rows = _collect_crop_jobs(batch_df, wanted_labels)
    if not jobs:
        return

    word_jobs: List[Tuple[int, _CropJob]] = []
    para_jobs: List[Tuple[int, _CropJob]] = []
    for idx, job in enumerate(jobs):
        if job.label == "table":
            word_jobs.append((idx, job))
        else:
            para_jobs.append((idx, job))

    all_preds: List[Any] = [None] * len(jobs)

    for merge_level, indexed_jobs in (("word", word_jobs), ("paragraph", para_jobs)):
        if not indexed_jobs:
            continue
        plain_jobs = [j for _, j in indexed_jobs]
        preds = _invoke_local_batched(plain_jobs, model=model, merge_level=merge_level, batch_size=batch_size)
        for (global_idx, _), pred in zip(indexed_jobs, preds):
            all_preds[global_idx] = pred

    for job, preds in zip(jobs, all_preds):
        entry = _format_ocr_result(
            job.label,
            preds,
            job.crop_hw,
            job.bbox,
            use_graphic_elements=use_graphic_elements,
            row=rows[job.row_idx],
        )
        if entry is None:
            continue
        if job.label == "table":
            row_table[job.row_idx].append(entry)
        elif job.label == "chart":
            row_chart[job.row_idx].append(entry)
        elif job.label == "infographic":
            row_infographic[job.row_idx].append(entry)


# ---------------------------------------------------------------------------
# Remote inference path
# ---------------------------------------------------------------------------


def _run_remote(
    batch_df: pd.DataFrame,
    *,
    wanted_labels: set,
    invoke_url: str,
    api_key: Optional[str],
    request_timeout_s: float,
    use_graphic_elements: bool,
    inference_batch_size: int,
    retry: RemoteRetryParams,
    row_table: List[List[Dict[str, Any]]],
    row_chart: List[List[Dict[str, Any]]],
    row_infographic: List[List[Dict[str, Any]]],
    row_error: List[Any],
    **kwargs: Any,
) -> None:
    """Per-row remote OCR inference."""
    for row_idx, row in enumerate(batch_df.itertuples(index=False)):
        try:
            pe = getattr(row, "page_elements_v3", None)
            dets: List[Dict[str, Any]] = []
            if isinstance(pe, dict):
                dets = pe.get("detections") or []
            if not isinstance(dets, list):
                dets = []

            page_image = getattr(row, "page_image", None)
            page_arr = get_page_image_array(page_image)
            if page_arr is None:
                continue

            crops = crop_all_from_page(page_arr, dets, wanted_labels)
            if not crops:
                continue

            crop_b64s: List[str] = []
            crop_meta: List[Tuple[str, List[float], Tuple[int, int]]] = []
            for label_name, bbox, crop_array in crops:
                crop_b64s.append(np_rgb_to_b64_png(crop_array))
                crop_meta.append((label_name, bbox, (crop_array.shape[0], crop_array.shape[1])))

            response_items = invoke_image_inference_batches(
                invoke_url=invoke_url,
                image_b64_list=crop_b64s,
                api_key=api_key,
                timeout_s=float(request_timeout_s),
                max_batch_size=int(kwargs.get("inference_batch_size", inference_batch_size)),
                max_pool_workers=int(retry.remote_max_pool_workers),
                max_retries=int(retry.remote_max_retries),
                max_429_retries=int(retry.remote_max_429_retries),
            )
            if len(response_items) != len(crop_meta):
                raise RuntimeError(f"Expected {len(crop_meta)} OCR responses, got {len(response_items)}")

            for i, (label_name, bbox, crop_hw) in enumerate(crop_meta):
                preds = extract_remote_ocr_item(response_items[i])
                entry = _format_ocr_result(
                    label_name,
                    preds,
                    crop_hw,
                    bbox,
                    use_graphic_elements=use_graphic_elements,
                    row=row,
                )
                if entry is None:
                    continue
                if label_name == "table":
                    row_table[row_idx].append(entry)
                elif label_name == "chart":
                    row_chart[row_idx].append(entry)
                elif label_name == "infographic":
                    row_infographic[row_idx].append(entry)

        except BaseException as e:
            print(f"Warning: OCR failed for row {row_idx}: {type(e).__name__}: {e}")
            row_error[row_idx] = {
                "stage": "ocr_page_elements",
                "type": e.__class__.__name__,
                "message": str(e),
                "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
            }


# ---------------------------------------------------------------------------
# Ray Actor
# ---------------------------------------------------------------------------

_IMAGE_COLUMNS = ("page_image", "images")


def _drop_image_columns(df: pd.DataFrame) -> None:
    """Drop heavy image payload columns in-place to shrink downstream Arrow blocks."""
    for col in _IMAGE_COLUMNS:
        if col in df.columns:
            df.drop(columns=col, inplace=True)


class OCRActor:
    """Ray-friendly callable that initializes Nemotron OCR v1 once per actor.

    The constructor casts model sub-components to fp16, warms up the CUDA
    caching allocator, and enables ``expandable_segments`` so the hot path
    triggers no new CUDA mallocs.

    Usage with Ray Data::

        ds = ds.map_batches(
            OCRActor,
            batch_size=16, batch_format="pandas", num_cpus=4, num_gpus=1,
            compute=ray.data.ActorPoolStrategy(size=8),
            fn_constructor_kwargs={
                "extract_tables": True,
                "extract_charts": True,
                "extract_infographics": False,
            },
        )
    """

    __slots__ = ("ocr_kwargs", "_model", "_remote_retry", "_drop_page_image")

    def __init__(self, **ocr_kwargs: Any) -> None:
        import warnings

        if Image is not None:
            warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

        self._drop_page_image = bool(ocr_kwargs.pop("drop_page_image", True))

        self.ocr_kwargs = dict(ocr_kwargs)
        invoke_url = str(self.ocr_kwargs.get("ocr_invoke_url") or self.ocr_kwargs.get("invoke_url") or "").strip()
        if invoke_url and "invoke_url" not in self.ocr_kwargs:
            self.ocr_kwargs["invoke_url"] = invoke_url

        self.ocr_kwargs["extract_tables"] = bool(self.ocr_kwargs.get("extract_tables", False))
        self.ocr_kwargs["extract_charts"] = bool(self.ocr_kwargs.get("extract_charts", False))
        self.ocr_kwargs["extract_infographics"] = bool(self.ocr_kwargs.get("extract_infographics", False))
        self.ocr_kwargs["use_graphic_elements"] = bool(self.ocr_kwargs.get("use_graphic_elements", False))
        self.ocr_kwargs["request_timeout_s"] = float(self.ocr_kwargs.get("request_timeout_s", 120.0))
        self.ocr_kwargs["inference_batch_size"] = int(self.ocr_kwargs.get("inference_batch_size", 8))

        self._remote_retry = RemoteRetryParams(
            remote_max_pool_workers=int(self.ocr_kwargs.get("remote_max_pool_workers", 16)),
            remote_max_retries=int(self.ocr_kwargs.get("remote_max_retries", 10)),
            remote_max_429_retries=int(self.ocr_kwargs.get("remote_max_429_retries", 5)),
        )
        if invoke_url:
            self._model = None
        else:
            from nemo_retriever.model.local import NemotronOCRV1

            max_bs = int(self.ocr_kwargs.get("inference_batch_size", 8))
            self._model = NemotronOCRV1(max_batch_size=max_bs)

    def __call__(self, batch_df: Any, **override_kwargs: Any) -> Any:
        try:
            result = ocr_page_elements(
                batch_df,
                model=self._model,
                remote_retry=self._remote_retry,
                **self.ocr_kwargs,
                **override_kwargs,
            )
            if self._drop_page_image and isinstance(result, pd.DataFrame):
                _drop_image_columns(result)
            return result
        except BaseException as e:
            if isinstance(batch_df, pd.DataFrame):
                out = batch_df.copy()
                payload = _error_payload(stage="actor_call", exc=e)
                n = len(out.index)
                out["table"] = [[] for _ in range(n)]
                out["chart"] = [[] for _ in range(n)]
                out["infographic"] = [[] for _ in range(n)]
                out["ocr_v1"] = [payload for _ in range(n)]
                return out
            return [{"ocr_v1": _error_payload(stage="actor_call", exc=e)}]


# ---------------------------------------------------------------------------
# Nemotron Parse v1.2
# ---------------------------------------------------------------------------


def _extract_parse_text(response_item: Any) -> str:
    if response_item is None:
        return ""
    if isinstance(response_item, str):
        return response_item.strip()
    if isinstance(response_item, dict):
        for key in ("generated_text", "text", "output_text", "prediction", "output", "data"):
            value = response_item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list) and value:
                first = value[0]
                if isinstance(first, str) and first.strip():
                    return first.strip()
                if isinstance(first, dict):
                    inner = _extract_parse_text(first)
                    if inner:
                        return inner
    if isinstance(response_item, list):
        for item in response_item:
            text = _extract_parse_text(item)
            if text:
                return text
    try:
        return str(response_item).strip()
    except Exception:
        return ""


def nemotron_parse_page_elements(
    batch_df: Any,
    *,
    model: Any = None,
    invoke_url: Optional[str] = None,
    api_key: Optional[str] = None,
    request_timeout_s: float = 120.0,
    extract_tables: bool = False,
    extract_charts: bool = False,
    extract_infographics: bool = False,
    task_prompt: str = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>",
    remote_retry: RemoteRetryParams | None = None,
    **kwargs: Any,
) -> Any:
    """Run Nemotron Parse v1.2 on cropped page elements."""
    retry = remote_retry or RemoteRetryParams(
        remote_max_pool_workers=int(kwargs.get("remote_max_pool_workers", 16)),
        remote_max_retries=int(kwargs.get("remote_max_retries", 10)),
        remote_max_429_retries=int(kwargs.get("remote_max_429_retries", 5)),
    )
    if not isinstance(batch_df, pd.DataFrame):
        raise NotImplementedError("nemotron_parse_page_elements currently only supports pandas.DataFrame input.")

    invoke_url = (invoke_url or kwargs.get("nemotron_parse_invoke_url") or "").strip()
    use_remote = bool(invoke_url)
    if not use_remote and model is None:
        raise ValueError("A local `model` is required when `invoke_url` is not provided.")

    wanted_labels: set[str] = set()
    if extract_tables:
        wanted_labels.add("table")
    if extract_charts:
        wanted_labels.add("chart")
    if extract_infographics:
        wanted_labels.add("infographic")

    all_table: List[List[Dict[str, Any]]] = []
    all_chart: List[List[Dict[str, Any]]] = []
    all_infographic: List[List[Dict[str, Any]]] = []
    all_meta: List[Dict[str, Any]] = []

    t0_total = time.perf_counter()

    for row in batch_df.itertuples(index=False):
        table_items: List[Dict[str, Any]] = []
        chart_items: List[Dict[str, Any]] = []
        infographic_items: List[Dict[str, Any]] = []
        row_err: Any = None

        try:
            pe = getattr(row, "page_elements_v3", None)
            dets: List[Dict[str, Any]] = []
            if isinstance(pe, dict):
                dets = pe.get("detections") or []
            if not isinstance(dets, list):
                dets = []

            page_image = getattr(row, "page_image", None)
            page_arr = get_page_image_array(page_image)
            if page_arr is None:
                all_table.append(table_items)
                all_chart.append(chart_items)
                all_infographic.append(infographic_items)
                all_meta.append({"timing": None, "error": None})
                continue

            crops = crop_all_from_page(page_arr, dets, wanted_labels)
            if not crops and wanted_labels:
                crops = [("full_page", [0.0, 0.0, 1.0, 1.0], page_arr.copy())]

            if use_remote:
                crop_b64s: List[str] = []
                crop_meta: List[Tuple[str, List[float]]] = []
                for label_name, bbox, crop_array in crops:
                    crop_b64s.append(np_rgb_to_b64_png(crop_array))
                    crop_meta.append((label_name, bbox))

                if crop_b64s:
                    response_items = invoke_image_inference_batches(
                        invoke_url=invoke_url,
                        image_b64_list=crop_b64s,
                        api_key=api_key,
                        timeout_s=float(request_timeout_s),
                        max_batch_size=int(kwargs.get("inference_batch_size", 8)),
                        max_pool_workers=int(retry.remote_max_pool_workers),
                        max_retries=int(retry.remote_max_retries),
                        max_429_retries=int(retry.remote_max_429_retries),
                    )
                    if len(response_items) != len(crop_meta):
                        raise RuntimeError(f"Expected {len(crop_meta)} Parse responses, got {len(response_items)}")

                    for i, (label_name, bbox) in enumerate(crop_meta):
                        text = _extract_parse_text(response_items[i])
                        entry = {"bbox_xyxy_norm": bbox, "text": text}
                        if label_name == "table":
                            table_items.append(entry)
                        elif label_name == "chart":
                            chart_items.append(entry)
                        elif label_name == "infographic":
                            infographic_items.append(entry)
                        elif label_name == "full_page":
                            if extract_tables:
                                table_items.append(dict(entry))
                            if extract_charts:
                                chart_items.append(dict(entry))
                            if extract_infographics:
                                infographic_items.append(dict(entry))
            else:
                for label_name, bbox, crop_array in crops:
                    text = str(model.invoke(crop_array, task_prompt=task_prompt) or "").strip()
                    entry = {"bbox_xyxy_norm": bbox, "text": text}
                    if label_name == "table":
                        table_items.append(entry)
                    elif label_name == "chart":
                        chart_items.append(entry)
                    elif label_name == "infographic":
                        infographic_items.append(entry)
                    elif label_name == "full_page":
                        if extract_tables:
                            table_items.append(dict(entry))
                        if extract_charts:
                            chart_items.append(dict(entry))
                        if extract_infographics:
                            infographic_items.append(dict(entry))

        except BaseException as e:
            print(f"Warning: Nemotron Parse failed: {type(e).__name__}: {e}")
            row_err = {
                "stage": "nemotron_parse_page_elements",
                "type": e.__class__.__name__,
                "message": str(e),
                "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
            }

        all_table.append(table_items)
        all_chart.append(chart_items)
        all_infographic.append(infographic_items)
        all_meta.append({"timing": None, "error": row_err})

    elapsed = time.perf_counter() - t0_total
    for meta in all_meta:
        meta["timing"] = {"seconds": float(elapsed)}

    out = batch_df.copy()
    out["table"] = all_table
    out["chart"] = all_chart
    out["infographic"] = all_infographic
    out["table_parse"] = all_table
    out["chart_parse"] = all_chart
    out["infographic_parse"] = all_infographic
    out["nemotron_parse_v1_2"] = all_meta
    return out


class NemotronParseActor:
    """Ray-friendly callable that initializes Nemotron Parse v1.2 once per actor."""

    __slots__ = (
        "_model",
        "_extract_tables",
        "_extract_charts",
        "_extract_infographics",
        "_invoke_url",
        "_api_key",
        "_request_timeout_s",
        "_task_prompt",
        "_remote_retry",
        "_drop_page_image",
    )

    def __init__(
        self,
        *,
        extract_tables: bool = False,
        extract_charts: bool = False,
        extract_infographics: bool = False,
        nemotron_parse_invoke_url: Optional[str] = None,
        invoke_url: Optional[str] = None,
        api_key: Optional[str] = None,
        request_timeout_s: float = 120.0,
        task_prompt: str = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>",
        remote_max_pool_workers: int = 16,
        remote_max_retries: int = 10,
        remote_max_429_retries: int = 5,
        drop_page_image: bool = True,
    ) -> None:
        self._drop_page_image = bool(drop_page_image)
        self._invoke_url = (nemotron_parse_invoke_url or invoke_url or "").strip()
        self._api_key = api_key
        self._request_timeout_s = float(request_timeout_s)
        self._task_prompt = str(task_prompt)
        self._remote_retry = RemoteRetryParams(
            remote_max_pool_workers=int(remote_max_pool_workers),
            remote_max_retries=int(remote_max_retries),
            remote_max_429_retries=int(remote_max_429_retries),
        )
        if self._invoke_url:
            self._model = None
        else:
            from nemo_retriever.model.local import NemotronParseV12

            self._model = NemotronParseV12(task_prompt=self._task_prompt)
        self._extract_tables = bool(extract_tables)
        self._extract_charts = bool(extract_charts)
        self._extract_infographics = bool(extract_infographics)

    def __call__(self, batch_df: Any, **override_kwargs: Any) -> Any:
        try:
            result = nemotron_parse_page_elements(
                batch_df,
                model=self._model,
                invoke_url=self._invoke_url,
                api_key=self._api_key,
                request_timeout_s=self._request_timeout_s,
                task_prompt=self._task_prompt,
                extract_tables=self._extract_tables,
                extract_charts=self._extract_charts,
                extract_infographics=self._extract_infographics,
                remote_retry=self._remote_retry,
                **override_kwargs,
            )
            if self._drop_page_image and isinstance(result, pd.DataFrame):
                _drop_image_columns(result)
            return result
        except BaseException as e:
            if isinstance(batch_df, pd.DataFrame):
                out = batch_df.copy()
                payload = _error_payload(stage="nemotron_parse_actor_call", exc=e)
                n = len(out.index)
                out["table"] = [[] for _ in range(n)]
                out["chart"] = [[] for _ in range(n)]
                out["infographic"] = [[] for _ in range(n)]
                out["table_parse"] = [[] for _ in range(n)]
                out["chart_parse"] = [[] for _ in range(n)]
                out["infographic_parse"] = [[] for _ in range(n)]
                out["nemotron_parse_v1_2"] = [payload for _ in range(n)]
                return out
            return [{"nemotron_parse_v1_2": _error_payload(stage="nemotron_parse_actor_call", exc=e)}]
