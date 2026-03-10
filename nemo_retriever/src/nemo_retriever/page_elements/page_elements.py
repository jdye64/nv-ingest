# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import base64
import io
import time
import traceback

from nemotron_page_elements_v3.utils import postprocess_preds_page_element
import pandas as pd
from nemo_retriever.params import RemoteRetryParams

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]

try:
    from nv_ingest_api.internal.primitives.nim.model_interface.yolox import (
        postprocess_page_elements_v3,
        YOLOX_PAGE_V3_CLASS_LABELS,
    )
except ImportError:
    postprocess_page_elements_v3 = None  # type: ignore[assignment,misc]
    YOLOX_PAGE_V3_CLASS_LABELS = None  # type: ignore[assignment]

from nemo_retriever.nim.nim import invoke_page_elements_batches


# ---------------------------------------------------------------------------
# Lightweight helpers
# ---------------------------------------------------------------------------


def _error_payload(*, stage: str, exc: BaseException) -> Dict[str, Any]:
    return {
        "detections": [],
        "error": {
            "stage": str(stage),
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        },
    }


def _decode_b64_image_to_np_array(image_b64: str) -> Tuple["np.ndarray", Tuple[int, int]]:
    if Image is None or np is None:  # pragma: no cover
        raise ImportError("page element detection requires pillow and numpy.")

    raw = base64.b64decode(image_b64)
    with Image.open(io.BytesIO(raw)) as im0:
        im = im0.convert("RGB")
        w, h = im.size
        arr = np.array(im)

    return arr, (int(h), int(w))


def _labels_from_model(_model: Any) -> List[str]:
    return ["table", "chart", "title", "infographic", "text", "header_footer"]


def _counts_by_label(detections: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for d in detections:
        if not isinstance(d, dict):
            continue
        name = d.get("label_name")
        if not isinstance(name, str) or not name.strip():
            name = f"label_{d.get('label')}"
        k = str(name)
        out[k] = int(out.get(k, 0) + 1)
    return out


# ---------------------------------------------------------------------------
# Postprocessing: model tensors -> list-of-dicts
# ---------------------------------------------------------------------------


def _postprocess_to_per_image_detections(
    *,
    boxes: Any,
    labels: Any,
    scores: Any,
    batch_size: int,
    label_names: List[str],
) -> List[List[Dict[str, Any]]]:
    """Convert model postprocess outputs into per-image detection dicts.

    Uses bulk ``.tolist()`` on each tensor to avoid per-element Python overhead.
    """
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for page element detection postprocess.")

    def _as_list(x: Any) -> List[Any]:
        return x if isinstance(x, list) else [x]

    if isinstance(boxes, torch.Tensor) and boxes.ndim == 3:
        boxes_list = [boxes[i] for i in range(boxes.shape[0])]
    else:
        boxes_list = _as_list(boxes)

    if isinstance(labels, torch.Tensor) and labels.ndim == 2:
        labels_list = [labels[i] for i in range(labels.shape[0])]
    else:
        labels_list = _as_list(labels)

    if isinstance(scores, torch.Tensor) and scores.ndim == 2:
        scores_list = [scores[i] for i in range(scores.shape[0])]
    else:
        scores_list = _as_list(scores)

    n = min(len(boxes_list), len(labels_list), len(scores_list), int(batch_size))
    num_labels = len(label_names)
    out: List[List[Dict[str, Any]]] = []

    for i in range(n):
        bi = boxes_list[i]
        li = labels_list[i]
        si = scores_list[i]

        if not isinstance(bi, torch.Tensor) or not isinstance(li, torch.Tensor) or not isinstance(si, torch.Tensor):
            out.append([])
            continue

        bi = bi.detach().cpu()
        li = li.detach().cpu()
        si = si.detach().cpu()

        if bi.ndim != 2 or bi.shape[-1] != 4:
            out.append([])
            continue

        # Bulk convert to Python lists -- single C++ boundary crossing each.
        boxes_py = bi.tolist()
        labels_py = li.tolist()
        scores_py = si.tolist()

        dets: List[Dict[str, Any]] = []
        for bbox, label_val, score_val in zip(boxes_py, labels_py, scores_py):
            label_i = int(label_val)
            label_name = label_names[label_i] if 0 <= label_i < num_labels else f"label_{label_i}"
            dets.append(
                {
                    "bbox_xyxy_norm": bbox,
                    "label": label_i,
                    "label_name": label_name,
                    "score": float(score_val),
                }
            )
        out.append(dets)

    while len(out) < int(batch_size):
        out.append([])
    return out[: int(batch_size)]


# ---------------------------------------------------------------------------
# Label mapping between retriever ("text") and API ("paragraph")
# ---------------------------------------------------------------------------

_RETRIEVER_LABEL_NAMES = ["table", "chart", "title", "infographic", "text", "header_footer"]
_RETRIEVER_TO_API = {"text": "paragraph"}
_API_TO_RETRIEVER = {"paragraph": "text"}


def _detections_to_annotation_dict(
    dets: List[Dict[str, Any]],
) -> Dict[str, List[List[float]]]:
    """Convert detection dicts to the annotation_dict format expected by
    ``postprocess_page_elements_v3``.
    """
    ann: Dict[str, List[List[float]]] = {}
    for d in dets:
        name = _RETRIEVER_TO_API.get(d["label_name"], d["label_name"])
        bbox = list(d["bbox_xyxy_norm"])
        bbox.append(float(d["score"]) if d["score"] is not None else 0.0)
        ann.setdefault(name, []).append(bbox)
    return ann


def _annotation_dict_to_detections(
    ann_dict: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Convert an annotation_dict back into detection dicts."""
    dets: List[Dict[str, Any]] = []
    for api_name, entries in ann_dict.items():
        retriever_name = _API_TO_RETRIEVER.get(api_name, api_name)
        try:
            label_id = _RETRIEVER_LABEL_NAMES.index(retriever_name)
        except ValueError:
            label_id = None
        for entry in entries:
            dets.append(
                {
                    "bbox_xyxy_norm": list(entry[:4]),
                    "label": label_id,
                    "label_name": retriever_name,
                    "score": float(entry[4]) if len(entry) > 4 else 0.0,
                }
            )
    return dets


def _bounding_boxes_to_detections(
    bb_dict: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Convert a bounding_boxes dict (NIM API format) to detection dicts."""
    dets: List[Dict[str, Any]] = []
    for api_name, entries in bb_dict.items():
        retriever_name = _API_TO_RETRIEVER.get(api_name, api_name)
        try:
            label_id = _RETRIEVER_LABEL_NAMES.index(retriever_name)
        except ValueError:
            label_id = None
        for entry in entries:
            dets.append(
                {
                    "bbox_xyxy_norm": [
                        float(entry["x_min"]),
                        float(entry["y_min"]),
                        float(entry["x_max"]),
                        float(entry["y_max"]),
                    ],
                    "label": label_id,
                    "label_name": retriever_name,
                    "score": float(entry.get("confidence", 0.0)),
                }
            )
    return dets


def _apply_page_elements_v3_postprocess(
    dets: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Apply ``postprocess_page_elements_v3`` (box fusion, title matching,
    expansion, overlap removal) to a single image's detection list.
    """
    if postprocess_page_elements_v3 is None or not dets:
        return dets
    try:
        ann_dict = _detections_to_annotation_dict(dets)
        labels = YOLOX_PAGE_V3_CLASS_LABELS if YOLOX_PAGE_V3_CLASS_LABELS is not None else list(ann_dict.keys())
        result = postprocess_page_elements_v3(ann_dict, labels=labels)
        return _annotation_dict_to_detections(result)
    except Exception:
        return dets


# ---------------------------------------------------------------------------
# Remote response parsing (unchanged)
# ---------------------------------------------------------------------------


def _remote_response_to_detections(
    *,
    response_json: Dict[str, Any],
    label_names: List[str],
    thresholds_per_class: Sequence[float],
) -> List[Dict[str, Any]]:
    candidates: List[Any] = [response_json]
    data_list = response_json.get("data")
    if isinstance(data_list, list) and data_list:
        candidates.append(data_list[0])
    output_list = response_json.get("output")
    if isinstance(output_list, list) and output_list:
        candidates.append(output_list[0])
    pred_list = response_json.get("predictions")
    if isinstance(pred_list, list) and pred_list:
        candidates.append(pred_list[0])

    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        try:
            boxes, labels, scores = postprocess_preds_page_element(cand, list(thresholds_per_class), label_names)
            dets = _postprocess_to_per_image_detections(
                boxes=[boxes],
                labels=[labels],
                scores=[scores],
                batch_size=1,
                label_names=label_names,
            )[0]
            return _apply_page_elements_v3_postprocess(dets)
        except Exception:
            pass

    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        bb = cand.get("bounding_boxes")
        if isinstance(bb, dict):
            try:
                dets = _bounding_boxes_to_detections(bb)
                return _apply_page_elements_v3_postprocess(dets)
            except Exception:
                pass

    for cand in candidates:
        if not isinstance(cand, dict) or not cand:
            continue
        if all(isinstance(v, list) for v in cand.values()):
            try:
                dets = _annotation_dict_to_detections(cand)  # type: ignore[arg-type]
                return _apply_page_elements_v3_postprocess(dets)
            except Exception:
                pass

    raise RuntimeError(f"Unsupported remote response format (keys={list(response_json.keys())!r})")


# ---------------------------------------------------------------------------
# Row-level image decoding (CPU, unavoidable)
# ---------------------------------------------------------------------------


def _decode_row_image(row: Any) -> Tuple["np.ndarray", Tuple[int, int]]:
    """Extract an HWC numpy array and (h, w) shape from a DataFrame row.

    Tries, in order: raw PNG bytes, pre-decoded numpy array, base64 string.
    """
    page_image = row.get("page_image")
    if not isinstance(page_image, dict):
        raise ValueError("No usable image data found in row.")

    png = page_image.get("image_png")
    if isinstance(png, (bytes, bytearray)) and png and Image is not None:
        with Image.open(io.BytesIO(png)) as im0:
            arr = np.array(im0.convert("RGB"))
        return arr, (int(arr.shape[0]), int(arr.shape[1]))

    arr_val = page_image.get("image_array")
    if isinstance(arr_val, np.ndarray):
        return arr_val, (int(arr_val.shape[0]), int(arr_val.shape[1]))

    b64_val = page_image.get("image_b64")
    if isinstance(b64_val, str) and b64_val:
        return _decode_b64_image_to_np_array(b64_val)

    raise ValueError("No usable image data found in row.")


def _decode_row_b64(row: Any) -> str:
    """Extract a base64-encoded image string from a DataFrame row."""
    page_image = row.get("page_image")
    if not isinstance(page_image, dict):
        raise ValueError("No usable image data found in row.")

    png = page_image.get("image_png")
    if isinstance(png, (bytes, bytearray)) and png:
        return base64.b64encode(png).decode("ascii")

    b64_val = page_image.get("image_b64")
    if isinstance(b64_val, str) and b64_val:
        return b64_val

    raise ValueError("No usable image data found in row.")


# ---------------------------------------------------------------------------
# Local inference for one chunk (GPU hot path)
# ---------------------------------------------------------------------------


def _run_local_chunk(
    *,
    model: Any,
    images: List["np.ndarray"],
    orig_shapes: List[Tuple[int, int]],
    label_names: List[str],
) -> List[List[Dict[str, Any]]]:
    """Preprocess, infer, and postprocess a single batch on the GPU.

    ``model`` must expose ``preprocess_batch(images) -> Tensor`` and
    ``postprocess(preds) -> (boxes, labels, scores)``.
    """
    batch = model.preprocess_batch(images)

    with torch.inference_mode():
        with torch.autocast(device_type="cuda"):
            preds = model(batch, orig_shapes)

    if isinstance(preds, dict):
        preds = [preds]

    boxes, labels, scores = model.postprocess(preds)

    per_image = _postprocess_to_per_image_detections(
        boxes=boxes,
        labels=labels,
        scores=scores,
        batch_size=len(images),
        label_names=label_names,
    )
    return [_apply_page_elements_v3_postprocess(d) for d in per_image]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def detect_page_elements_v3(
    pages_df: Any,
    *,
    model: Any = None,
    invoke_url: Optional[str] = None,
    api_key: Optional[str] = None,
    request_timeout_s: float = 120.0,
    inference_batch_size: int = 8,
    output_column: str = "page_elements_v3",
    num_detections_column: str = "page_elements_v3_num_detections",
    counts_by_label_column: str = "page_elements_v3_counts_by_label",
    remote_retry: RemoteRetryParams | None = None,
    **kwargs: Any,
) -> Any:
    """Run Nemotron Page Elements v3 on a pandas batch.

    For **local** inference the hot path is:
      numpy arrays -> ``model.preprocess_batch`` (pinned H2D, resize on GPU)
      -> 4D BCHW float16 tensor -> ``model.forward`` -> postprocess

    No CPU-side tensor ops occur between decode and final dict conversion.
    """
    retry = remote_retry or RemoteRetryParams(
        remote_max_pool_workers=int(kwargs.get("remote_max_pool_workers", 16)),
        remote_max_retries=int(kwargs.get("remote_max_retries", 10)),
        remote_max_429_retries=int(kwargs.get("remote_max_429_retries", 5)),
    )

    if not isinstance(pages_df, pd.DataFrame):
        raise NotImplementedError("detect_page_elements_v3 currently only supports pandas.DataFrame input.")
    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be > 0")

    invoke_url = (invoke_url or kwargs.get("page_elements_invoke_url") or "").strip()
    use_remote = bool(invoke_url)

    if not use_remote and model is None:
        raise ValueError("A local `model` is required when `invoke_url` is not provided.")

    # ------------------------------------------------------------------
    # 1. Decode images from every row (CPU, unavoidable)
    # ------------------------------------------------------------------
    row_images: List[Optional["np.ndarray"]] = []
    row_shapes: List[Optional[Tuple[int, int]]] = []
    row_b64: List[Optional[str]] = []
    row_payloads: List[Dict[str, Any]] = []

    label_names = _labels_from_model(model) if model is not None else list(_RETRIEVER_LABEL_NAMES)
    if model is not None and hasattr(model, "thresholds_per_class"):
        thresholds_per_class = getattr(model, "thresholds_per_class")
    else:
        thresholds_per_class = [0.0 for _ in label_names]

    for _, row in pages_df.iterrows():
        try:
            if use_remote:
                row_b64.append(_decode_row_b64(row))
                row_images.append(None)
                row_shapes.append(None)
            else:
                arr, shape = _decode_row_image(row)
                row_images.append(arr)
                row_shapes.append(shape)
                row_b64.append(None)
            row_payloads.append({"detections": []})
        except BaseException as e:
            row_images.append(None)
            row_shapes.append(None)
            row_b64.append(None)
            row_payloads.append(_error_payload(stage="decode_image", exc=e))

    # ------------------------------------------------------------------
    # 2. Identify valid rows
    # ------------------------------------------------------------------
    if use_remote:
        valid_indices = [i for i, b64 in enumerate(row_b64) if b64]
    else:
        valid_indices = [i for i, img in enumerate(row_images) if img is not None and row_shapes[i] is not None]

    if (not use_remote) and valid_indices and torch is None:  # pragma: no cover
        raise ImportError("torch is required for page element detection.")

    # ------------------------------------------------------------------
    # 3a. Remote inference path (unchanged)
    # ------------------------------------------------------------------
    if use_remote and valid_indices:
        valid_b64 = [row_b64[i] for i in valid_indices if row_b64[i]]

        t0 = time.perf_counter()
        try:
            response_items = invoke_page_elements_batches(
                invoke_url=invoke_url,
                image_b64_list=valid_b64,
                api_key=api_key,
                timeout_s=float(request_timeout_s),
                max_batch_size=int(inference_batch_size),
                max_pool_workers=int(retry.remote_max_pool_workers),
                max_retries=int(retry.remote_max_retries),
                max_429_retries=int(retry.remote_max_429_retries),
            )
            elapsed = time.perf_counter() - t0

            if len(response_items) != len(valid_indices):
                raise RuntimeError(
                    f"Remote response count mismatch: expected {len(valid_indices)}, got {len(response_items)}"
                )

            for local_i, row_i in enumerate(valid_indices):
                dets = _remote_response_to_detections(
                    response_json=response_items[local_i],
                    label_names=label_names,
                    thresholds_per_class=thresholds_per_class,
                )
                row_payloads[row_i] = {
                    "detections": dets,
                    "timing": {"seconds": float(elapsed)},
                    "error": None,
                }
        except BaseException as e:
            elapsed = time.perf_counter() - t0
            print(f"Warning: page_elements remote inference failed: {type(e).__name__}: {e}")
            for row_i in valid_indices:
                row_payloads[row_i] = _error_payload(stage="remote_inference", exc=e) | {
                    "timing": {"seconds": float(elapsed)}
                }

    # ------------------------------------------------------------------
    # 3b. Local inference path (GPU-optimized)
    # ------------------------------------------------------------------
    if not use_remote:
        for chunk_start in range(0, len(valid_indices), int(inference_batch_size)):
            chunk_idx = valid_indices[chunk_start : chunk_start + int(inference_batch_size)]
            if not chunk_idx:
                continue

            chunk_images: List["np.ndarray"] = []
            chunk_shapes: List[Tuple[int, int]] = []
            for i in chunk_idx:
                chunk_images.append(row_images[i])  # type: ignore[arg-type]
                chunk_shapes.append(row_shapes[i])  # type: ignore[arg-type]

            t0 = time.perf_counter()
            try:
                per_image_dets = _run_local_chunk(
                    model=model,
                    images=chunk_images,
                    orig_shapes=chunk_shapes,
                    label_names=label_names,
                )
                elapsed = time.perf_counter() - t0

                for local_i, row_i in enumerate(chunk_idx):
                    dets = per_image_dets[local_i] if local_i < len(per_image_dets) else []
                    row_payloads[row_i] = {
                        "detections": dets,
                        "timing": {"seconds": float(elapsed)},
                        "error": None,
                    }
            except BaseException as e:
                elapsed = time.perf_counter() - t0
                for row_i in chunk_idx:
                    row_payloads[row_i] = _error_payload(stage="local_inference", exc=e) | {
                        "timing": {"seconds": float(elapsed)}
                    }

    # ------------------------------------------------------------------
    # 4. Attach results to the DataFrame
    # ------------------------------------------------------------------
    out = pages_df.copy()
    out[output_column] = row_payloads
    out[num_detections_column] = [
        int(len(p.get("detections") or [])) if isinstance(p, dict) else 0 for p in row_payloads
    ]
    out[counts_by_label_column] = [
        _counts_by_label(p.get("detections") or []) if isinstance(p, dict) else {} for p in row_payloads
    ]
    return out


# ---------------------------------------------------------------------------
# Ray actor
# ---------------------------------------------------------------------------


class PageElementDetectionActor:
    """Ray-friendly callable that initializes Nemotron Page Elements v3 once.

    Use with Ray Data::

        ds = ds.map_batches(
            PageElementDetectionActor,
            fn_constructor_kwargs={...},
            batch_format="pandas",
        )
    """

    __slots__ = ("detect_kwargs", "_model")

    def __init__(self, **detect_kwargs: Any) -> None:
        self.detect_kwargs = dict(detect_kwargs)
        invoke_url = str(
            self.detect_kwargs.get("page_elements_invoke_url") or self.detect_kwargs.get("invoke_url") or ""
        ).strip()
        if invoke_url and "invoke_url" not in self.detect_kwargs:
            self.detect_kwargs["invoke_url"] = invoke_url
        if invoke_url:
            self._model = None
        else:
            from nemo_retriever.model.local import NemotronPageElementsV3

            max_bs = int(self.detect_kwargs.get("inference_batch_size", 16))
            self._model = NemotronPageElementsV3(max_batch_size=max_bs)

    def __call__(self, pages_df: Any, **override_kwargs: Any) -> Any:
        try:
            return detect_page_elements_v3(
                pages_df,
                model=self._model,
                **self.detect_kwargs,
                **override_kwargs,
            )
        except Exception as e:
            if isinstance(pages_df, pd.DataFrame):
                out = pages_df.copy()
                payload = _error_payload(stage="actor_call", exc=e)
                out["page_elements_v3"] = [payload for _ in range(len(out.index))]
                out["page_elements_v3_num_detections"] = [0 for _ in range(len(out.index))]
                out["page_elements_v3_counts_by_label"] = [{} for _ in range(len(out.index))]
                return out
            return [{"page_elements_v3": _error_payload(stage="actor_call", exc=e)}]
