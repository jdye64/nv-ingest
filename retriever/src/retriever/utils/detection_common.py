from __future__ import annotations

import base64
import io
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def detection_error_payload(*, stage: str, exc: BaseException) -> Dict[str, Any]:
    return {
        "detections": [],
        "error": {
            "stage": str(stage),
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        },
    }


def decode_b64_image_to_chw_tensor(
    image_b64: str,
    *,
    import_error_message: str = "detection requires torch, pillow, and numpy.",
) -> Tuple["torch.Tensor", Tuple[int, int]]:
    if torch is None or Image is None or np is None:  # pragma: no cover
        raise ImportError(import_error_message)

    raw = base64.b64decode(image_b64)
    with Image.open(io.BytesIO(raw)) as im0:
        im = im0.convert("RGB")
        w, h = im.size
        arr = np.array(im, dtype=np.uint8)  # (H,W,3)

    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,H,W) uint8
    t = t.to(dtype=torch.float32) / 255.0
    return t, (int(h), int(w))


def crop_b64_image_by_norm_bbox(
    page_image_b64: str,
    *,
    bbox_xyxy_norm: Sequence[float],
    image_format: str = "png",
) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
    if Image is None:  # pragma: no cover
        raise ImportError("Cropping requires pillow.")
    if not isinstance(page_image_b64, str) or not page_image_b64:
        return None, None
    try:
        x1n, y1n, x2n, y2n = [float(x) for x in bbox_xyxy_norm]
    except Exception:
        return None, None

    try:
        raw = base64.b64decode(page_image_b64)
        with Image.open(io.BytesIO(raw)) as im0:
            im = im0.convert("RGB")
            w, h = im.size
            if w <= 1 or h <= 1:
                return None, None

            def _clamp_int(v: float, lo: int, hi: int) -> int:
                if v != v:  # NaN
                    return lo
                return int(min(max(v, float(lo)), float(hi)))

            x1 = _clamp_int(x1n * w, 0, w)
            x2 = _clamp_int(x2n * w, 0, w)
            y1 = _clamp_int(y1n * h, 0, h)
            y2 = _clamp_int(y2n * h, 0, h)

            if x2 <= x1 or y2 <= y1:
                return None, None

            crop = im.crop((x1, y1, x2, y2))
            cw, ch = crop.size
            if cw <= 1 or ch <= 1:
                return None, None

            buf = io.BytesIO()
            fmt = str(image_format or "png").lower()
            if fmt not in {"png"}:
                fmt = "png"
            crop.save(buf, format=fmt.upper())
            return base64.b64encode(buf.getvalue()).decode("ascii"), (int(ch), int(cw))
    except Exception:
        return None, None


def labels_from_model(model: Any) -> List[str]:
    try:
        labels = getattr(getattr(model, "_model", None), "labels", None)
        if isinstance(labels, (list, tuple)) and all(isinstance(x, str) for x in labels):
            return [str(x) for x in labels]
    except Exception:
        pass

    try:
        out = getattr(model, "output", None)
        if isinstance(out, dict):
            classes = out.get("classes")
            if isinstance(classes, (list, tuple)) and all(isinstance(x, str) for x in classes):
                return [str(x) for x in classes]
    except Exception:
        pass

    return []


def prediction_to_detections(pred: Any, *, label_names: List[str]) -> List[Dict[str, Any]]:
    if torch is None:  # pragma: no cover
        raise ImportError("torch required for prediction parsing.")

    boxes = labels = scores = None
    if isinstance(pred, dict):

        def _get_any(d: Dict[str, Any], *keys: str) -> Any:
            for k in keys:
                if k in d:
                    v = d.get(k)
                    if v is not None:
                        return v
            return None

        boxes = _get_any(pred, "boxes", "bboxes", "bbox", "box")
        labels = _get_any(pred, "labels", "classes", "class_ids", "class")
        scores = _get_any(pred, "scores", "conf", "confidences", "score")
    elif isinstance(pred, (list, tuple)) and len(pred) >= 3:
        boxes, labels, scores = pred[0], pred[1], pred[2]

    if boxes is None or labels is None:
        return []

    def _to_tensor(x: Any) -> Optional["torch.Tensor"]:
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        try:
            return torch.as_tensor(x).detach().cpu()
        except Exception:
            return None

    b = _to_tensor(boxes)
    labels_t = _to_tensor(labels)
    s = _to_tensor(scores) if scores is not None else None
    if b is None or labels_t is None:
        return []

    if b.ndim != 2 or int(b.shape[-1]) != 4:
        return []
    if labels_t.ndim == 2 and int(labels_t.shape[-1]) == 1:
        labels_t = labels_t.squeeze(-1)
    if labels_t.ndim != 1:
        return []

    n = int(min(b.shape[0], labels_t.shape[0]))
    dets: List[Dict[str, Any]] = []
    for i in range(n):
        try:
            x1, y1, x2, y2 = [float(x) for x in b[i].tolist()]
        except Exception:
            continue

        label_i: Optional[int]
        try:
            label_i = int(labels_t[i].item())
        except Exception:
            label_i = None

        score_f: Optional[float]
        if s is not None and s.ndim >= 1 and int(s.shape[0]) > i:
            try:
                score_f = float(s[i].item())
            except Exception:
                score_f = None
        else:
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
    return dets


def extract_remote_pred_item(response_item: Any) -> Any:
    if isinstance(response_item, dict):
        for k in ("prediction", "predictions", "output", "outputs", "data"):
            v = response_item.get(k)
            if isinstance(v, list) and v:
                return v[0]
            if v is not None:
                return v
    return response_item


def counts_by_label(detections: Sequence[Dict[str, Any]]) -> Dict[str, int]:
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
