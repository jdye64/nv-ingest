# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Page-image encoding, decoding, and cropping utilities.

Shared by the OCR, chart-detection, table-detection, and inprocess stages.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import base64
import io

import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def clamp_int(v: float, lo: int, hi: int) -> int:
    """Clamp *v* to ``[lo, hi]``, treating NaN as *lo*."""
    if v != v:  # NaN
        return lo
    return int(min(max(v, float(lo)), float(hi)))


def np_rgb_to_b64_png(arr: np.ndarray) -> str:
    """Encode an HWC uint8 RGB numpy array to a base64-encoded PNG string."""
    if Image is None:  # pragma: no cover
        raise ImportError("Pillow is required for image encoding.")
    img = Image.fromarray(arr.astype(np.uint8, copy=False), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Page-image accessors
# ---------------------------------------------------------------------------


def get_page_image_array(page_image: Any) -> Optional[np.ndarray]:
    """Extract or lazily decode a HWC uint8 RGB array from *page_image*.

    Accepts (in priority order):
    * A raw ``np.ndarray`` directly.
    * A ``dict`` with ``image_array`` (zero-copy).
    * A ``dict`` with ``image_png`` (PIL-decode PNG bytes).
    * A ``dict`` with ``image_b64`` (base64-decode + PIL, legacy fallback).
    """
    if isinstance(page_image, np.ndarray):
        return page_image
    if not isinstance(page_image, dict):
        return None
    arr = page_image.get("image_array")
    if isinstance(arr, np.ndarray):
        return arr
    png = page_image.get("image_png")
    if isinstance(png, (bytes, bytearray)) and png and Image is not None:
        try:
            with Image.open(io.BytesIO(png)) as im0:
                return np.asarray(im0.convert("RGB"), dtype=np.uint8).copy()
        except Exception:
            pass
    b64 = page_image.get("image_b64")
    if isinstance(b64, str) and b64 and Image is not None:
        try:
            raw = base64.b64decode(b64)
            with Image.open(io.BytesIO(raw)) as im0:
                return np.asarray(im0.convert("RGB"), dtype=np.uint8).copy()
        except Exception:
            pass
    return None


def get_page_image_b64(page_image: Any) -> Optional[str]:
    """Extract or lazily encode the page image as a base64-encoded PNG string."""
    if not isinstance(page_image, dict):
        return None
    b64 = page_image.get("image_b64")
    if isinstance(b64, str) and b64:
        return b64
    png = page_image.get("image_png")
    if isinstance(png, (bytes, bytearray)) and png:
        return base64.b64encode(png).decode("ascii")
    arr = page_image.get("image_array")
    if isinstance(arr, np.ndarray):
        return np_rgb_to_b64_png(arr)
    return None


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------


def crop_b64_image_by_norm_bbox(
    page_image: Any,
    *,
    bbox_xyxy_norm: Sequence[float],
    image_format: str = "png",
) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
    """Crop a page image by a normalized xyxy bbox, returning base64 PNG.

    Returns ``(cropped_b64, (h, w))`` or ``(None, None)`` on failure.
    """
    arr = get_page_image_array(page_image)
    if arr is None:
        return None, None

    try:
        x1n, y1n, x2n, y2n = (float(x) for x in bbox_xyxy_norm)
    except Exception:
        return None, None

    h, w = arr.shape[:2]
    if w <= 1 or h <= 1:
        return None, None

    x1 = clamp_int(x1n * w, 0, w)
    x2 = clamp_int(x2n * w, 0, w)
    y1 = clamp_int(y1n * h, 0, h)
    y2 = clamp_int(y2n * h, 0, h)
    if x2 <= x1 or y2 <= y1:
        return None, None

    crop = arr[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    if cw <= 1 or ch <= 1:
        return None, None

    return np_rgb_to_b64_png(crop), (ch, cw)


def crop_all_from_page(
    page_image: Any,
    detections: List[Dict[str, Any]],
    wanted_labels: set,
) -> List[Tuple[str, List[float], np.ndarray]]:
    """Crop all matching detections from *page_image* using pure numpy slicing.

    Returns a list of ``(label_name, bbox_xyxy_norm, crop_array)`` tuples.
    Uses ``np.ascontiguousarray`` to ensure each crop is contiguous.
    """
    arr = get_page_image_array(page_image)
    if arr is None:
        return []

    h, w = arr.shape[:2]
    if w <= 1 or h <= 1:
        return []

    results: List[Tuple[str, List[float], np.ndarray]] = []
    for det in detections:
        if not isinstance(det, dict):
            continue
        label_name = str(det.get("label_name") or "").strip()
        if label_name not in wanted_labels:
            continue

        bbox = det.get("bbox_xyxy_norm")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue

        try:
            x1n, y1n, x2n, y2n = (float(x) for x in bbox)
        except Exception:
            continue

        x1 = clamp_int(x1n * w, 0, w)
        x2 = clamp_int(x2n * w, 0, w)
        y1 = clamp_int(y1n * h, 0, h)
        y2 = clamp_int(y2n * h, 0, h)

        if x2 <= x1 or y2 <= y1:
            continue

        crop_array = np.ascontiguousarray(arr[y1:y2, x1:x2])
        ch, cw = crop_array.shape[:2]
        if cw <= 1 or ch <= 1:
            continue

        results.append((label_name, [float(x) for x in bbox], crop_array))

    return results
