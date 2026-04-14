# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistence helpers for stage-by-stage pipeline inspection.

``save_stage`` writes a pickled DataFrame (for faithful replay) plus
human-inspectable sidecar assets (images as files, per-row JSON).

``load_stage`` reads the pickle back so the next stage can consume it.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_PICKLE_NAME = "dataframe.pkl"
_META_NAME = "stage_meta.json"
_ASSETS_DIR = "assets"

# DataFrame columns that carry heavyweight binary payloads.
_BINARY_COLUMNS = frozenset({"bytes", "page_image", "images"})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_stage(
    df: pd.DataFrame,
    stage_dir: str | Path,
    stage_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist a stage's output DataFrame and human-inspectable assets.

    Parameters
    ----------
    df : pd.DataFrame
        The stage output.
    stage_dir : path-like
        Directory to write into (created if absent).
    stage_meta : dict, optional
        Extra metadata (timing, actor kwargs, etc.) saved to ``stage_meta.json``.

    Returns
    -------
    Path
        The *stage_dir* that was written to.
    """
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = stage_dir / _ASSETS_DIR
    assets_dir.mkdir(exist_ok=True)

    # 1. Pickle — full-fidelity round-trip for stage replay.
    df.to_pickle(stage_dir / _PICKLE_NAME)

    # 2. Human-inspectable sidecar assets.
    _export_assets(df, assets_dir)

    # 3. Stage metadata JSON.
    meta: Dict[str, Any] = {
        "row_count": len(df),
        "columns": list(df.columns),
    }
    if stage_meta:
        meta.update(stage_meta)
    (stage_dir / _META_NAME).write_text(json.dumps(meta, indent=2, default=str))

    return stage_dir


def load_stage(stage_dir: str | Path) -> pd.DataFrame:
    """Read a previously-persisted stage output."""
    pkl = Path(stage_dir) / _PICKLE_NAME
    if not pkl.exists():
        raise FileNotFoundError(f"No {_PICKLE_NAME} found in {stage_dir}")
    return pd.read_pickle(pkl)


# ---------------------------------------------------------------------------
# Asset export helpers
# ---------------------------------------------------------------------------

def _doc_stem(path_val: Any) -> str:
    """Derive a filesystem-safe stem from a row's ``path`` value."""
    if not path_val or not isinstance(path_val, str):
        return "unknown"
    return Path(path_val).stem


def _strip_binary(val: Any) -> Any:
    """Replace known heavy binary payloads with lightweight summaries."""
    if isinstance(val, (bytes, bytearray, memoryview)):
        return f"<{len(val)} bytes>"
    if isinstance(val, dict):
        out: Dict[str, Any] = {}
        for k, v in val.items():
            if k == "jpeg_bytes" and isinstance(v, (bytes, bytearray)):
                out[k] = f"<{len(v)} bytes>"
            elif k == "pixels":
                out[k] = f"<array>" if v is not None else None
            elif k == "image_b64" and isinstance(v, str) and len(v) > 200:
                out[k] = f"<base64 len={len(v)}>"
            else:
                out[k] = _strip_binary(v)
        return out
    if isinstance(val, list):
        return [_strip_binary(item) for item in val]
    # numpy arrays
    if hasattr(val, "shape"):
        return f"<ndarray shape={val.shape} dtype={val.dtype}>"
    return val


def _row_to_json(row: pd.Series, columns: pd.Index) -> Dict[str, Any]:
    """Convert a row to a JSON-safe dict, stripping binary blobs."""
    return {col: _strip_binary(row[col]) for col in columns}


def _export_assets(df: pd.DataFrame, assets_dir: Path) -> None:
    """Write human-inspectable files for each row in *df*."""
    has_page_image = "page_image" in df.columns
    has_images = "images" in df.columns

    for _, row in df.iterrows():
        stem = _doc_stem(row.get("path"))
        page = int(row["page_number"]) if "page_number" in df.columns else 0

        # Page image (JPEG produced by _render_page_to_base64).
        if has_page_image:
            pi = row["page_image"]
            if isinstance(pi, dict):
                jpeg_bytes = pi.get("jpeg_bytes")
                if jpeg_bytes:
                    (assets_dir / f"{stem}_{page}.jpeg").write_bytes(jpeg_bytes)

        # Embedded image crops (base64-encoded PNG from pdfium).
        if has_images:
            imgs = row["images"]
            if isinstance(imgs, list):
                for idx, img_obj in enumerate(imgs):
                    if not isinstance(img_obj, dict):
                        continue
                    b64 = img_obj.get("image_b64")
                    if not b64:
                        continue
                    try:
                        (assets_dir / f"{stem}_{page}_crop_{idx}.png").write_bytes(
                            base64.b64decode(b64)
                        )
                    except Exception:
                        logger.debug("Failed to decode image crop %s p%d #%d", stem, page, idx)

        # Per-row metadata JSON (binary blobs replaced with summaries).
        json_path = assets_dir / f"{stem}_{page}.json"
        try:
            json_path.write_text(
                json.dumps(_row_to_json(row, df.columns), indent=2, default=str)
            )
        except Exception:
            logger.debug("Failed to write JSON for %s p%d", stem, page)
