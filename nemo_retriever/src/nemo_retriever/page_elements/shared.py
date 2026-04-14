# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import base64
import io
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from nemo_retriever.nim.nim import NIMClient, invoke_page_elements_batches
from nemo_retriever.params import RemoteRetryParams
from nemo_retriever.page_elements.local import (
    YOLOX_PAGE_V3_CLASS_LABELS,
    YOLOX_PAGE_V3_FINAL_SCORE,
    postprocess_page_elements_v3,
    postprocess_preds_page_element,
)

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

TensorOrArray = Union["torch.Tensor", "np.ndarray"]


def _ensure_chw_float_tensor(x: TensorOrArray) -> "torch.Tensor":
    """
    Normalize a single image into a CHW float32 torch.Tensor suitable for batching.

    Accepts either:
    - torch.Tensor in CHW or 1xCHW (or CHW-like) formats
    - np.ndarray in CHW or HWC (RGB) formats (optionally with leading batch dim=1)
    """
    if torch is None or np is None:  # pragma: no cover
        raise ImportError("page element detection requires torch and numpy.")

    if isinstance(x, torch.Tensor):
        t = x
    elif isinstance(x, np.ndarray):
        arr = x
        # Squeeze trivial batch dimension if present.
        if arr.ndim == 4 and int(arr.shape[0]) == 1:
            arr = arr[0]
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D image array, got shape {getattr(arr, 'shape', None)}")

        # Heuristic: HWC (RGB) -> CHW; otherwise assume already CHW-like.
        if int(arr.shape[-1]) == 3 and int(arr.shape[0]) != 3:
            t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
        else:
            t = torch.from_numpy(np.ascontiguousarray(arr))
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(x)!r}")

    # Squeeze trivial batch dimension if present.
    if t.ndim == 4 and int(t.shape[0]) == 1:
        t = t[0]
    if t.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(t.shape)}")

    # Keep 0-255 range: resize_pad pads with 114.0 (designed for 0-255),
    # and YoloXWrapper.forward() handles the 0-255 → model-input conversion.
    t = t.to(dtype=torch.float32)

    return t.contiguous()


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


def _extract_page_pixels(page_image: Dict[str, Any]) -> Tuple["np.ndarray", Tuple[int, int]]:
    """Extract raw pixel array from a page_image dict.

    Supports three formats (checked in priority order):
    - **JPEG bytes** (``"jpeg_bytes"`` key): cv2 decode — fast, small memory.
    - **Raw pixels** (``"pixels"`` key): zero-copy, no decode needed.
    - **Base64 legacy** (``"image_b64"`` key): base64 decode → PIL → numpy.

    Returns ``(BGR_HWC_uint8_array, (H, W))``.
    """
    if np is None:  # pragma: no cover
        raise ImportError("page element detection requires numpy.")

    # Preferred path: JPEG-compressed bytes (small in-memory footprint).
    jpeg = page_image.get("jpeg_bytes")
    if jpeg is not None:
        import cv2 as _cv2
        arr = _cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), _cv2.IMREAD_COLOR)
        if arr is not None:
            h, w = int(arr.shape[0]), int(arr.shape[1])
            return arr, (h, w)

    # Raw numpy pixels (legacy from earlier raw-pixel pipeline).
    pixels = page_image.get("pixels")
    if pixels is not None:
        arr = np.ascontiguousarray(pixels)
        h, w = int(arr.shape[0]), int(arr.shape[1])
        return arr, (h, w)

    # Base64-encoded JPEG/PNG.
    image_b64 = page_image.get("image_b64")
    if image_b64:
        if Image is None:  # pragma: no cover
            raise ImportError("page element detection requires pillow for base64 images.")
        raw = base64.b64decode(image_b64)
        with Image.open(io.BytesIO(raw)) as im0:
            im = im0.convert("RGB")
            w, h = im.size
            arr = np.array(im)
            arr = arr[:, :, ::-1].copy()  # RGB → BGR
        return arr, (int(h), int(w))

    raise ValueError("page_image has no 'jpeg_bytes', 'pixels', or 'image_b64' key.")


def _labels_from_model(_model: Any) -> List[str]:
    return [
        "table",
        "chart",
        "title",
        "infographic",
        "text",
        "header_footer",
    ]


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


def _postprocess_to_per_image_detections(
    *,
    boxes: Any,
    labels: Any,
    scores: Any,
    batch_size: int,
    label_names: List[str],
) -> List[List[Dict[str, Any]]]:
    """
    Convert model postprocess outputs into a list of per-image detection dicts.

    Expected detection format matches the "stage2 page_elements_v3 json" used by `nemo_retriever.utils.image.render`.
    """
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for page element detection postprocess.")

    # Normalize to per-image tensors.
    def _as_list(x: Any) -> List[Any]:
        if isinstance(x, list):
            return x
        return [x]

    # If tensors include a batch dimension, split them.
    if isinstance(boxes, torch.Tensor) and boxes.ndim == 3:
        boxes_list = [boxes[i] for i in range(int(boxes.shape[0]))]
    else:
        boxes_list = _as_list(boxes)

    if isinstance(labels, torch.Tensor) and labels.ndim == 2:
        labels_list = [labels[i] for i in range(int(labels.shape[0]))]
    else:
        labels_list = _as_list(labels)

    if isinstance(scores, torch.Tensor) and scores.ndim == 2:
        scores_list = [scores[i] for i in range(int(scores.shape[0]))]
    else:
        scores_list = _as_list(scores)

    n = min(len(boxes_list), len(labels_list), len(scores_list), int(batch_size))
    out: List[List[Dict[str, Any]]] = []
    for i in range(n):
        bi = boxes_list[i]
        li = labels_list[i]
        si = scores_list[i]

        if not isinstance(bi, torch.Tensor) or not isinstance(li, torch.Tensor) or not isinstance(si, torch.Tensor):
            out.append([])
            continue

        # Move to CPU for safe conversion.
        bi = bi.detach().cpu()
        li = li.detach().cpu()
        si = si.detach().cpu()

        # Common shapes:
        # - boxes: (N,4)
        # - labels: (N,)
        # - scores: (N,)
        if bi.ndim != 2 or bi.shape[-1] != 4:
            out.append([])
            continue

        n_det = int(bi.shape[0])
        dets: List[Dict[str, Any]] = []
        for j in range(n_det):
            try:
                x1, y1, x2, y2 = [float(x) for x in bi[j].tolist()]
            except Exception:
                continue

            label_i: Optional[int]
            try:
                label_i = int(li[j].item())
            except Exception:
                label_i = None

            score_f: Optional[float]
            try:
                score_f = float(si[j].item())
            except Exception:
                score_f = None

            label_name = None
            if label_i is not None and 0 <= label_i < len(label_names):
                label_name = label_names[label_i]
            if not label_name:
                label_name = f"label_{label_i}" if label_i is not None else "unknown"

            dets.append(
                {
                    "bbox_xyxy_norm": [x1, y1, x2, y2],
                    "label": label_i,
                    "label_name": str(label_name),
                    "score": score_f,
                }
            )
        out.append(dets)

    # If model returned fewer splits than requested, pad.
    while len(out) < int(batch_size):
        out.append([])
    return out[: int(batch_size)]


# -- Label mapping between retriever ("text") and API ("paragraph") --
_RETRIEVER_LABEL_NAMES = ["table", "chart", "title", "infographic", "text", "header_footer"]
_RETRIEVER_TO_API = {"text": "paragraph"}
_API_TO_RETRIEVER = {"paragraph": "text"}


def _detections_to_annotation_dict(
    dets: List[Dict[str, Any]],
) -> Dict[str, List[List[float]]]:
    """Convert a list of detection dicts into the annotation_dict format expected by
    ``postprocess_page_elements_v3``.

    Each detection dict has keys ``bbox_xyxy_norm``, ``label_name``, ``score``.
    The annotation_dict maps label names (using API naming, i.e. "paragraph") to
    ``[[x0, y0, x1, y1, confidence], ...]``.
    """
    ann: Dict[str, List[List[float]]] = {}
    for d in dets:
        name = _RETRIEVER_TO_API.get(d["label_name"], d["label_name"])
        bbox = list(d["bbox_xyxy_norm"])  # [x0, y0, x1, y1]
        bbox.append(float(d["score"]) if d["score"] is not None else 0.0)
        ann.setdefault(name, []).append(bbox)
    return ann


def _annotation_dict_to_detections(
    ann_dict: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Convert an annotation_dict back into a list of detection dicts.

    Maps API label names back to retriever names (e.g. "paragraph" -> "text")
    and assigns integer label IDs from the retriever label order.
    """
    dets: List[Dict[str, Any]] = []
    for api_name, entries in ann_dict.items():
        retriever_name = _API_TO_RETRIEVER.get(api_name, api_name)
        try:
            label_id = _RETRIEVER_LABEL_NAMES.index(retriever_name)
        except ValueError:
            label_id = None
        for entry in entries:
            # entry is [x0, y0, x1, y1, confidence]
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
    """Convert a bounding_boxes dict (NIM API format) to detection dicts.

    Input format: {"label": [{"x_min": ..., "y_min": ..., "x_max": ..., "y_max": ..., "confidence": ...}, ...]}
    """
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


def _apply_final_score_filter(
    dets: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Filter detections by per-class final score thresholds (YOLOX_PAGE_V3_FINAL_SCORE).

    This should be applied **after** WBF post-processing to match the NIM pipeline ordering.
    Maps retriever label "text" to API label "paragraph" for threshold lookup.
    """
    if not YOLOX_PAGE_V3_FINAL_SCORE or not dets:
        return dets
    filtered: List[Dict[str, Any]] = []
    for d in dets:
        api_name = _RETRIEVER_TO_API.get(d["label_name"], d["label_name"])
        threshold = YOLOX_PAGE_V3_FINAL_SCORE.get(api_name, 0.0)
        if d.get("score") is not None and d["score"] >= threshold:
            filtered.append(d)
    return filtered


def _apply_page_elements_v3_postprocess(
    dets: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Apply ``postprocess_page_elements_v3`` (box fusion, title matching,
    expansion, overlap removal) to a single image's detection list.

    Returns the original detections unchanged if the API function is unavailable.
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


def _remote_response_to_detections(
    *,
    response_json: Dict[str, Any],
    label_names: List[str],
    thresholds_per_class: Sequence[float],
) -> List[Dict[str, Any]]:
    # Try direct model-pred style payload first (or common wrappers around it).
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

    # NIM bounding_boxes format:
    # {"index": 0, "bounding_boxes": {"title": [{"x_min": ..., "y_min": ..., ...}]}}
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

    # Fall back to API-style annotation dict:
    # {"table": [[x0,y0,x1,y1,conf], ...], "paragraph": [...]}
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
    nim_client: NIMClient | None = None,
    **kwargs: Any,
) -> Any:
    retry = remote_retry or RemoteRetryParams(
        remote_max_pool_workers=int(kwargs.get("remote_max_pool_workers", 16)),
        remote_max_retries=int(kwargs.get("remote_max_retries", 10)),
        remote_max_429_retries=int(kwargs.get("remote_max_429_retries", 5)),
    )
    """
    Run Nemotron Page Elements v3 on a pandas batch.

    Input:
      - `pages_df`: pandas.DataFrame (typical Ray Data `batch_format="pandas"`)
        Must contain an image base64 source either in `image_b64` or one of
        `images`/`tables`/`charts`/`infographics` (each as list[{"image_b64": ...}]).

    Output:
      - returns a pandas.DataFrame with original columns preserved, plus:
        - `output_column`: dict payload {"detections": [...], "timing": {...}, "error": {...?}}
        - `num_detections_column`: int
        - `counts_by_label_column`: dict[str,int]

    Notes:
      - This function internally batches model invocations in chunks of `inference_batch_size`
        to enforce batch=8 even if Ray provides larger `map_batches` frames.
    """
    if not isinstance(pages_df, pd.DataFrame):
        raise NotImplementedError("detect_page_elements_v3 currently only supports pandas.DataFrame input.")

    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be > 0")

    # Working snippet for single image inference and debugging
    # breakpoint()
    # first_page = pages_df.iloc[0]
    # b64 = first_page.get("page_image")["image_b64"]

    # t, orig_shape = _decode_b64_image_to_np_array(b64)

    # # Inference
    # with torch.inference_mode():
    #     x = model.preprocess(t)
    #     preds = model(x, orig_shape)[0]

    # print(preds)
    # breakpoint()

    invoke_url = (invoke_url or kwargs.get("page_elements_invoke_url") or "").strip()
    use_remote = bool(invoke_url)

    if not use_remote and model is None:
        raise ValueError("A local `model` is required when `invoke_url` is not provided.")

    # Prepare per-row decode artifacts (local mode), raw base64 (remote mode),
    # and placeholders for missing/errored rows.
    row_tensors: List[Optional[TensorOrArray]] = []
    row_shapes: List[Optional[Tuple[int, int]]] = []
    row_b64: List[Optional[str]] = []
    row_payloads: List[Dict[str, Any]] = []

    label_names = _labels_from_model(model) if model is not None else list(_RETRIEVER_LABEL_NAMES)
    if model is not None and hasattr(model, "thresholds_per_class"):
        thresholds_per_class = getattr(model, "thresholds_per_class")
    else:
        # Use the same per-class thresholds as the yolox pipeline.
        # label_names uses "text" where yolox uses "paragraph"; _RETRIEVER_TO_API maps between them.
        thresholds_per_class = [
            YOLOX_PAGE_V3_FINAL_SCORE.get(_RETRIEVER_TO_API.get(name, name), 0.0) for name in label_names
        ]

    # ---- Extract page_image dicts and decode in parallel ----
    page_image_dicts: List[Optional[Dict[str, Any]]] = []
    for _, row in pages_df.iterrows():
        try:
            pi = row.get("page_image")
            if pi is None or not isinstance(pi, dict):
                raise ValueError("No usable page_image found in row.")
            if "jpeg_bytes" not in pi and "pixels" not in pi and "image_b64" not in pi:
                raise ValueError("page_image has no 'jpeg_bytes', 'pixels', or 'image_b64'.")
            page_image_dicts.append(pi)
        except BaseException:
            page_image_dicts.append(None)

    def _safe_extract(pi: Optional[Dict[str, Any]]) -> Tuple[Optional[TensorOrArray], Optional[Tuple[int, int]], Optional[BaseException]]:
        if pi is None:
            return None, None, ValueError("No usable page_image found in row.")
        try:
            arr, orig_shape = _extract_page_pixels(pi)
            return arr, orig_shape, None
        except BaseException as e:
            return None, None, e

    _DECODE_WORKERS = min(16, max(1, len(page_image_dicts)))

    # When jpeg_bytes are available AND the model has GPU preprocess
    # (DALI NVJPEG), skip CPU decode — pass bytes straight through.
    _has_gpu_pp = hasattr(model, "preprocess_batch_gpu") if model is not None else False
    _all_jpeg = all(
        pi is not None and "jpeg_bytes" in pi
        for pi in page_image_dicts
    ) if page_image_dicts else False

    if not use_remote and page_image_dicts:
        if _all_jpeg and _has_gpu_pp:
            # JPEG bytes will be decoded on GPU by DALI — no CPU decode needed.
            # Store the raw bytes as the "tensor" and extract shape from the dict.
            decode_results = []
            for pi in page_image_dicts:
                if pi is None:
                    decode_results.append((None, None, ValueError("No page_image")))
                else:
                    sh = pi.get("orig_shape_hw", (0, 0))
                    decode_results.append((pi["jpeg_bytes"], (int(sh[0]), int(sh[1])), None))
        else:
            has_raw = any(pi is not None and ("pixels" in pi or "jpeg_bytes" in pi)
                          for pi in page_image_dicts)
            if has_raw:
                decode_results = [_safe_extract(pi) for pi in page_image_dicts]
            else:
                with ThreadPoolExecutor(max_workers=_DECODE_WORKERS) as pool:
                    decode_results = list(pool.map(_safe_extract, page_image_dicts))
    else:
        decode_results = [(None, None, None)] * len(page_image_dicts)

    for idx, pi in enumerate(page_image_dicts):
        if pi is None:
            row_b64.append(None)
            row_tensors.append(None)
            row_shapes.append(None)
            row_payloads.append(_error_payload(stage="decode_image", exc=ValueError("No page_image")))
            continue
        b64_val = pi.get("image_b64")
        if b64_val is None and use_remote:
            if "jpeg_bytes" in pi:
                b64_val = base64.b64encode(pi["jpeg_bytes"]).decode("ascii")
            elif "pixels" in pi:
                from nemo_retriever.ocr.ocr import _np_rgb_to_b64_png
                b64_val = _np_rgb_to_b64_png(pi["pixels"])
        row_b64.append(b64_val)
        if use_remote:
            row_tensors.append(None)
            row_shapes.append(None)
            row_payloads.append({"detections": []})
        else:
            t, orig_shape, err = decode_results[idx]
            if err is not None:
                row_tensors.append(None)
                row_shapes.append(None)
                row_payloads.append(_error_payload(stage="decode_image", exc=err))
            else:
                row_tensors.append(t)
                row_shapes.append(orig_shape)
                row_payloads.append({"detections": []})

    # Run inference over only valid rows, but write results back in original order.
    if use_remote:
        valid_indices = [i for i, b64 in enumerate(row_b64) if b64]
    else:
        valid_indices = [i for i, t in enumerate(row_tensors) if t is not None and row_shapes[i] is not None]

    if (not use_remote) and valid_indices and torch is None:  # pragma: no cover
        raise ImportError("torch is required for page element detection.")

    if use_remote and valid_indices:
        valid_b64: List[str] = []
        for row_i in valid_indices:
            b64 = row_b64[row_i]
            if b64:
                valid_b64.append(b64)

        t0 = time.perf_counter()
        try:
            _invoke_kw = dict(
                invoke_url=invoke_url,
                image_b64_list=valid_b64,
                api_key=api_key,
                timeout_s=float(request_timeout_s),
                max_batch_size=int(inference_batch_size),
                max_retries=int(retry.remote_max_retries),
                max_429_retries=int(retry.remote_max_429_retries),
            )
            if nim_client is not None:
                response_items = nim_client.invoke_page_elements_batches(**_invoke_kw)
            else:
                response_items = invoke_page_elements_batches(
                    **_invoke_kw,
                    max_pool_workers=int(retry.remote_max_pool_workers),
                )
            elapsed = time.perf_counter() - t0

            if len(response_items) != len(valid_indices):
                raise RuntimeError(
                    "Remote response count mismatch: " f"expected {len(valid_indices)}, got {len(response_items)}"
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

    def _finalize_chunk_preds(
        preds: Any,
        chunk_idx: List[int],
        num_images: int,
        elapsed: float,
    ) -> None:
        """Postprocess model predictions and write results into *row_payloads*."""
        if isinstance(preds, dict):
            preds_list2 = [preds]
        elif isinstance(preds, list):
            preds_list2 = preds
        else:
            preds_list2 = [preds]

        try:
            if hasattr(model, "postprocess"):
                boxes, labels, scores = model.postprocess(preds_list2)  # type: ignore[attr-defined]
            else:
                boxes_list_: List["torch.Tensor"] = []
                labels_list_: List["torch.Tensor"] = []
                scores_list_: List["torch.Tensor"] = []
                for p in preds_list2:
                    if not isinstance(p, dict):
                        boxes_list_.append(torch.empty((0, 4), dtype=torch.float32))
                        labels_list_.append(torch.empty((0,), dtype=torch.int64))
                        scores_list_.append(torch.empty((0,), dtype=torch.float32))
                        continue
                    b_np, l_np, s_np = postprocess_preds_page_element(
                        p,
                        thresholds_per_class,
                        label_names,
                    )
                    boxes_list_.append(torch.as_tensor(b_np, dtype=torch.float32))
                    labels_list_.append(torch.as_tensor(l_np, dtype=torch.int64))
                    scores_list_.append(torch.as_tensor(s_np, dtype=torch.float32))
                boxes, labels, scores = boxes_list_, labels_list_, scores_list_

            per_image_dets = _postprocess_to_per_image_detections(
                boxes=boxes,
                labels=labels,
                scores=scores,
                batch_size=num_images,
                label_names=label_names,
            )
            per_image_dets = [_apply_page_elements_v3_postprocess(dets) for dets in per_image_dets]
            per_image_dets = [_apply_final_score_filter(dets) for dets in per_image_dets]
            for local_i, row_i in enumerate(chunk_idx):
                dets = per_image_dets[local_i] if local_i < len(per_image_dets) else []
                row_payloads[row_i] = {
                    "detections": dets,
                    "timing": {"seconds": float(elapsed)},
                    "error": None,
                }
        except BaseException as e:
            for row_i in chunk_idx:
                row_payloads[row_i] = _error_payload(stage="postprocess", exc=e) | {
                    "timing": {"seconds": float(elapsed)}
                }

    _has_gpu_preprocess = hasattr(model, "preprocess_batch_gpu") if model is not None else False

    def _preprocess_one(i: int) -> Optional[Tuple[Tuple[int, int], TensorOrArray]]:
        """Preprocess a single image (safe to call from a thread)."""
        t = row_tensors[i]
        sh = row_shapes[i]
        if t is None or sh is None:
            return None
        try:
            pre = model.preprocess(t)  # type: ignore[arg-type]
            if isinstance(pre, torch.Tensor):
                if pre.ndim == 4 and int(pre.shape[0]) == 1:
                    pre = pre[0]
            elif isinstance(pre, np.ndarray):
                if pre.ndim == 4 and int(pre.shape[0]) == 1:
                    pre = pre[0]
            else:
                pre = t
            return sh, pre
        except Exception:
            return sh, t

    def _preprocess_chunk(chunk_idx: List[int]) -> Optional[Tuple[List[TensorOrArray], List[Tuple[int, int]], "torch.Tensor"]]:
        """Preprocess and stack a chunk of images.

        When the model exposes ``preprocess_batch_gpu``, raw decoded
        images are transferred to GPU and letterboxed there — much
        faster than per-image CPU resize.  Otherwise preprocessing
        runs in parallel threads on CPU.
        """
        valid_images: List[TensorOrArray] = []
        valid_shapes: List[Tuple[int, int]] = []
        for i in chunk_idx:
            t = row_tensors[i]
            sh = row_shapes[i]
            if t is None or sh is None:
                continue
            valid_images.append(t)
            valid_shapes.append(sh)

        if not valid_images:
            return None

        # Fast path: GPU letterbox (avoids per-image CPU resize_pad).
        if _has_gpu_preprocess:
            try:
                batch, orig_shapes = model.preprocess_batch_gpu(  # type: ignore[union-attr]
                    valid_images, known_shapes=valid_shapes,
                )
                return valid_images, orig_shapes, batch
            except Exception:
                pass  # fall through to CPU path

        # CPU path: per-image preprocess in parallel threads.
        def _do_preprocess(img: TensorOrArray) -> TensorOrArray:
            try:
                pre = model.preprocess(img)  # type: ignore[union-attr]
                if isinstance(pre, torch.Tensor) and pre.ndim == 4 and int(pre.shape[0]) == 1:
                    return pre[0]
                if isinstance(pre, np.ndarray) and pre.ndim == 4 and int(pre.shape[0]) == 1:
                    return pre[0]
                return pre
            except Exception:
                return img

        n_workers = min(8, max(1, len(valid_images)))
        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                pre_list = list(pool.map(_do_preprocess, valid_images))
        else:
            pre_list = [_do_preprocess(img) for img in valid_images]

        batch = torch.stack([_ensure_chw_float_tensor(x) for x in pre_list], dim=0)
        return pre_list, valid_shapes, batch

    _use_async = (
        not use_remote
        and model is not None
        and hasattr(model, "invoke_async")
        and hasattr(model, "flush")
    )

    # State for the double-buffered async path: metadata for the in-flight batch.
    _pending_chunk_idx: Optional[List[int]] = None
    _pending_num_images: Optional[int] = None
    _pending_t0: Optional[float] = None

    for chunk_start in range(0, len(valid_indices), int(inference_batch_size)):
        chunk_idx = valid_indices[chunk_start : chunk_start + int(inference_batch_size)]
        if not chunk_idx:
            continue

        if use_remote:
            continue

        prep = _preprocess_chunk(chunk_idx)
        if prep is None:
            continue
        pre_list, orig_shapes, batch = prep

        if _use_async:
            t0 = time.perf_counter()
            with torch.inference_mode():
                with torch.autocast(device_type="cuda"):
                    prev_preds = model.invoke_async(
                        batch, orig_shapes if len(pre_list) > 1 else orig_shapes[0],
                    )

            if prev_preds is not None and _pending_chunk_idx is not None:
                _finalize_chunk_preds(
                    prev_preds,
                    _pending_chunk_idx,
                    _pending_num_images,  # type: ignore[arg-type]
                    time.perf_counter() - _pending_t0,  # type: ignore[operator]
                )

            _pending_chunk_idx = chunk_idx
            _pending_num_images = len(pre_list)
            _pending_t0 = t0
        else:
            t0 = time.perf_counter()
            try:
                with torch.inference_mode():
                    with torch.autocast(device_type="cuda"):
                        preds = model(batch, orig_shapes) if len(pre_list) > 1 else model(batch, orig_shapes[0])
                if len(pre_list) > 1:
                    if isinstance(preds, dict):
                        raise RuntimeError("Model returned a single pred dict for batched input.")
                    if isinstance(preds, list) and len(preds) != len(pre_list):
                        raise RuntimeError(
                            f"Model returned {len(preds)} preds for batch size {len(pre_list)}; falling back to per-image."
                        )
            except Exception as ex:
                print(f"Error invoking model: {ex}")
                preds_list_fb: List[Any] = []
                for j in range(int(batch.shape[0])):
                    preds_list_fb.append(model(batch[j : j + 1], orig_shapes[j]))
                preds = preds_list_fb
            elapsed = time.perf_counter() - t0
            _finalize_chunk_preds(preds, chunk_idx, len(pre_list), elapsed)

    # Drain the last in-flight batch (double-buffer flush).
    if _use_async and _pending_chunk_idx is not None:
        with torch.inference_mode():
            with torch.autocast(device_type="cuda"):
                last_preds = model.flush()
        if last_preds is not None:
            _finalize_chunk_preds(
                last_preds,
                _pending_chunk_idx,
                _pending_num_images,  # type: ignore[arg-type]
                time.perf_counter() - _pending_t0,  # type: ignore[operator]
            )

    # ---- Decode-once: write decoded pixels back so downstream stages
    # (table-structure, graphic-elements, OCR) skip redundant JPEG decode.
    if not use_remote:
        target_df = pages_df if kwargs.get("_inplace") else None
        for idx, pi in enumerate(page_image_dicts):
            if pi is None:
                continue
            t = row_tensors[idx]
            if t is None:
                continue
            if "jpeg_bytes" not in pi:
                continue
            if isinstance(t, torch.Tensor):
                arr_hwc = t.cpu().numpy()
                if arr_hwc.ndim == 3 and int(arr_hwc.shape[0]) == 3:
                    arr_hwc = np.ascontiguousarray(arr_hwc.transpose(1, 2, 0))
            elif isinstance(t, np.ndarray):
                arr_hwc = t
            else:
                continue
            pi["pixels"] = arr_hwc
            pi.pop("jpeg_bytes", None)

    if kwargs.get("_inplace"):
        pages_df[output_column] = row_payloads
        pages_df[num_detections_column] = [
            int(len(p.get("detections") or [])) if isinstance(p, dict) else 0 for p in row_payloads
        ]
        pages_df[counts_by_label_column] = [
            _counts_by_label(p.get("detections") or []) if isinstance(p, dict) else {} for p in row_payloads
        ]
        return pages_df

    out = pages_df.copy()
    out[output_column] = row_payloads
    out[num_detections_column] = [
        int(len(p.get("detections") or [])) if isinstance(p, dict) else 0 for p in row_payloads
    ]
    out[counts_by_label_column] = [
        _counts_by_label(p.get("detections") or []) if isinstance(p, dict) else {} for p in row_payloads
    ]
    return out
