# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
OCR result parsing and text-block formatting utilities.

Shared by the OCR, chart-detection, and table-detection stages.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


# ---------------------------------------------------------------------------
# Remote response unwrapping
# ---------------------------------------------------------------------------


def extract_remote_ocr_item(response_item: Any) -> Any:
    """Unwrap a single NIM OCR response into raw predictions."""
    if isinstance(response_item, dict):
        td = response_item.get("text_detections")
        if isinstance(td, list) and td:
            return td
        for k in ("prediction", "predictions", "output", "outputs", "data"):
            v = response_item.get(k)
            if isinstance(v, list) and v:
                return v[0]
            if v is not None:
                return v
    return response_item


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def parse_ocr_result(preds: Any) -> List[Dict[str, Any]]:
    """Parse ``NemotronOCRV1.invoke()`` output into a flat list of text blocks.

    Each block is ``{"text": str, "sort_y": float, "sort_x": float}``.
    """
    blocks: List[Dict[str, Any]] = []

    # ---- dict form: {"boxes": [...], "texts": [...]} ----
    if isinstance(preds, dict):
        pb = preds.get("boxes") or preds.get("bboxes") or preds.get("bounding_boxes")
        pt = preds.get("texts") or preds.get("text_predictions") or preds.get("text")
        if isinstance(pb, list) and isinstance(pt, list):
            for b, txt in zip(pb, pt):
                if not isinstance(txt, str) or not txt.strip():
                    continue
                sort_y, sort_x = 0.0, 0.0
                if isinstance(b, list):
                    if len(b) == 4 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in b):
                        sort_y = float(b[0][1])
                        sort_x = float(b[0][0])
                    elif len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
                        sort_y = float(b[1])
                        sort_x = float(b[0])
                blocks.append({"text": txt.strip(), "sort_y": sort_y, "sort_x": sort_x})
        return blocks

    # ---- list form: list[dict] with various key conventions ----
    if isinstance(preds, list):
        for item in preds:
            if isinstance(item, str):
                if item.strip():
                    blocks.append({"text": item.strip(), "sort_y": 0.0, "sort_x": 0.0})
                continue
            if not isinstance(item, dict):
                continue

            tp = item.get("text_prediction")
            if isinstance(tp, dict):
                txt0 = str(tp.get("text") or "").strip()
                if txt0 and txt0 != "nan":
                    sort_y, sort_x = 0.0, 0.0
                    bb = item.get("bounding_box")
                    if isinstance(bb, dict):
                        pts = bb.get("points")
                        if isinstance(pts, list) and pts:
                            try:
                                sort_x = float(pts[0].get("x", 0.0))
                                sort_y = float(pts[0].get("y", 0.0))
                            except Exception:
                                pass
                    blocks.append({"text": txt0, "sort_y": sort_y, "sort_x": sort_x})
                continue

            if all(k in item for k in ("left", "right", "upper", "lower")) and isinstance(item.get("text"), str):
                txt0 = str(item.get("text") or "").strip()
                if not txt0 or txt0 == "nan":
                    continue
                try:
                    sort_x = float(item["left"])
                    sort_y = float(item["lower"])
                except Exception:
                    sort_x, sort_y = 0.0, 0.0
                blocks.append({"text": txt0, "sort_y": sort_y, "sort_x": sort_x})
                continue

            txt = item.get("text") or item.get("ocr_text") or item.get("generated_text") or item.get("output_text")
            if not isinstance(txt, str) or not txt.strip():
                continue
            sort_y, sort_x = 0.0, 0.0
            b = item.get("box") or item.get("bbox") or item.get("bounding_box") or item.get("bbox_points")
            if isinstance(b, list):
                if len(b) == 4 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in b):
                    sort_y = float(b[0][1])
                    sort_x = float(b[0][0])
                elif len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
                    sort_y = float(b[1])
                    sort_x = float(b[0])
            blocks.append({"text": txt.strip(), "sort_y": sort_y, "sort_x": sort_x})

    # ---- last-resort stringify ----
    if not blocks and preds is not None:
        s = ""
        try:
            s = str(preds).strip()
        except Exception:
            s = ""
        if s and s.lower() not in {"none", "null", "[]", "{}"}:
            blocks.append({"text": s, "sort_y": 0.0, "sort_x": 0.0})

    return blocks


# ---------------------------------------------------------------------------
# Text-block formatting
# ---------------------------------------------------------------------------


def blocks_to_text(blocks: List[Dict[str, Any]]) -> str:
    """Sort text blocks by reading order (y then x) and join with newlines."""
    blocks.sort(key=lambda b: (b.get("sort_y", 0.0), b.get("sort_x", 0.0)))
    return "\n".join(b["text"] for b in blocks if b.get("text"))


def blocks_to_pseudo_markdown(blocks: List[Dict[str, Any]]) -> str:
    """Convert OCR text blocks into pseudo-markdown table format.

    Uses DBSCAN clustering on y-coordinates to identify rows, then
    sorts within each row by x-coordinate and joins with pipe separators.
    """
    if not blocks:
        return ""

    valid = [b for b in blocks if b.get("text")]
    if not valid:
        return ""

    from sklearn.cluster import DBSCAN

    df = pd.DataFrame(valid)

    y_vals = df["sort_y"].values
    y_range = y_vals.max() - y_vals.min()
    if y_range > 0:
        y_norm = (y_vals - y_vals.min()) / y_range
        eps = 0.03
    else:
        y_norm = y_vals
        eps = 0.1

    dbscan = DBSCAN(eps=eps, min_samples=1)
    dbscan.fit(y_norm.reshape(-1, 1))
    df["cluster"] = dbscan.labels_
    df = df.sort_values(["cluster", "sort_x"])

    rows = []
    for _, grp in df.groupby("cluster", sort=True):
        rows.append("| " + " | ".join(grp["text"].tolist()) + " |")
    return "\n".join(rows)
