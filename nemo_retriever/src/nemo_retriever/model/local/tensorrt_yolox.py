# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-backed YOLOX model wrapper.

Drop-in replacement for NemotronPageElementsV3 / NemotronGraphicElementsV1 /
NemotronTableStructureV1 that runs inference through a pre-compiled TensorRT
engine instead of PyTorch.

Build engines with:
    retriever-build-engines          # all three models
    retriever-build-engines -m page_elements_v3  # just one
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from ..model import BaseModel, RunMode

logger = logging.getLogger(__name__)

DEFAULT_ENGINE_DIR = Path.home() / ".cache" / "nemo-retriever" / "engines"

ENGINE_REGISTRY: dict[str, str] = {
    "page_elements_v3": "page_elements_v3.engine",
    "graphic_elements_v1": "graphic_elements_v1.engine",
    "table_structure_v1": "table_structure_v1.engine",
}

_INPUT_HW = (1024, 1024)
_PAD_VALUE = 114.0


def find_engine(model_slug: str, engine_dir: Optional[Path] = None) -> Optional[Path]:
    """Return the engine path if a compiled ``.engine`` file exists, else ``None``."""
    filename = ENGINE_REGISTRY.get(model_slug)
    if filename is None:
        return None
    path = (engine_dir or DEFAULT_ENGINE_DIR) / filename
    if path.is_file():
        return path
    return None


# ---------------------------------------------------------------------------
# Low-level TensorRT engine wrapper
# ---------------------------------------------------------------------------


class _TRTEngine:
    """Manages a serialised TensorRT engine: device memory, async copy, execute."""

    def __init__(self, engine_path: str) -> None:
        import tensorrt as trt

        try:
            from cuda.bindings import runtime as cudart
        except ImportError:
            from cuda import cudart  # type: ignore[no-redef]

        self._cudart = cudart

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self._input_name: str = ""
        self._output_name: str = ""
        self._output_shape_template: tuple[int, ...] = ()
        self._output_dtype = np.float32

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_name = name
            else:
                self._output_name = name
                self._output_shape_template = tuple(self.engine.get_tensor_shape(name))
                self._output_dtype = trt.nptype(self.engine.get_tensor_dtype(name))

        self.max_batch = 1
        static_input_shape = list(self.engine.get_tensor_shape(self._input_name))
        if static_input_shape[0] == -1 and self.engine.num_optimization_profiles > 0:
            ps = self.engine.get_tensor_profile_shape(self._input_name, 0)
            self.max_batch = ps[2][0]
        elif static_input_shape[0] > 0:
            self.max_batch = static_input_shape[0]

        input_shape = list(static_input_shape)
        input_shape[0] = self.max_batch if input_shape[0] == -1 else input_shape[0]
        in_bytes = int(np.prod(input_shape)) * np.dtype(np.float32).itemsize

        output_shape = list(self._output_shape_template)
        output_shape[0] = self.max_batch if output_shape[0] == -1 else output_shape[0]
        out_bytes = int(np.prod(output_shape)) * np.dtype(self._output_dtype).itemsize

        err, self._d_in = cudart.cudaMalloc(in_bytes)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMalloc (input) failed: {err}")
        err, self._d_out = cudart.cudaMalloc(out_bytes)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMalloc (output) failed: {err}")
        err, self._stream = cudart.cudaStreamCreate()
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaStreamCreate failed: {err}")

    def infer(self, tensor: torch.Tensor) -> np.ndarray:
        """Run inference on a contiguous BCHW float32 tensor (CPU or CUDA)."""
        cudart = self._cudart
        tensor = tensor.contiguous().float()
        batch_size = tensor.shape[0]
        self.context.set_input_shape(self._input_name, list(tensor.shape))

        nbytes = tensor.nelement() * tensor.element_size()
        kind = (
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
            if tensor.is_cuda
            else cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        )
        cudart.cudaMemcpyAsync(self._d_in, tensor.data_ptr(), nbytes, kind, self._stream)

        self.context.set_tensor_address(self._input_name, self._d_in)
        self.context.set_tensor_address(self._output_name, self._d_out)
        self.context.execute_async_v3(stream_handle=self._stream)

        out_shape = list(self._output_shape_template)
        out_shape[0] = batch_size
        host_buf = np.empty(out_shape, dtype=self._output_dtype)
        cudart.cudaMemcpyAsync(
            host_buf.ctypes.data,
            self._d_out,
            host_buf.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            self._stream,
        )
        cudart.cudaStreamSynchronize(self._stream)
        return host_buf

    def __del__(self) -> None:
        try:
            cudart = getattr(self, "_cudart", None)
            if cudart is None:
                return
            free = getattr(cudart, "cudaFree", None)
            if free is not None:
                for attr in ("_d_in", "_d_out"):
                    ptr = getattr(self, attr, None)
                    if ptr is not None:
                        free(ptr)
            destroy = getattr(cudart, "cudaStreamDestroy", None)
            stream = getattr(self, "_stream", None)
            if destroy is not None and stream is not None:
                destroy(stream)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# YOLOX postprocess (decode + NMS) — operates on raw backbone output
# ---------------------------------------------------------------------------


def _nms_single_class(
    boxes: np.ndarray, scores: np.ndarray, iou_thresh: float
) -> list[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while len(order) > 0:
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep


def _yolox_decode(
    raw: np.ndarray,
    num_classes: int,
    scale: float,
    orig_hw: tuple[int, int],
    conf_thresh: float,
    iou_thresh: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode one image's raw YOLOX output → (boxes_xyxy_norm, class_ids, scores).

    Returned boxes are normalised to [0, 1] w.r.t. the original image.
    """
    if raw.ndim == 3:
        raw = raw[0]

    ncols = raw.shape[-1]
    if ncols == 5 + num_classes:
        cxcywh, obj, cls = raw[:, :4], raw[:, 4], raw[:, 5:]
    elif ncols == 4 + num_classes:
        cxcywh, obj, cls = raw[:, :4], np.ones(raw.shape[0], np.float32), raw[:, 4:]
    else:
        return (
            np.empty((0, 4), np.float32),
            np.empty(0, np.int64),
            np.empty(0, np.float32),
        )

    scores_all = obj[:, None] * cls
    class_ids = scores_all.argmax(axis=1)
    max_scores = scores_all[np.arange(len(scores_all)), class_ids]

    mask = max_scores >= conf_thresh
    cxcywh, max_scores, class_ids = cxcywh[mask], max_scores[mask], class_ids[mask]
    if len(cxcywh) == 0:
        return (
            np.empty((0, 4), np.float32),
            np.empty(0, np.int64),
            np.empty(0, np.float32),
        )

    # cxcywh → xyxy (still in 1024×1024 feature space)
    boxes = np.empty_like(cxcywh)
    boxes[:, 0] = cxcywh[:, 0] - cxcywh[:, 2] / 2
    boxes[:, 1] = cxcywh[:, 1] - cxcywh[:, 3] / 2
    boxes[:, 2] = cxcywh[:, 0] + cxcywh[:, 2] / 2
    boxes[:, 3] = cxcywh[:, 1] + cxcywh[:, 3] / 2

    # Per-class NMS
    keep_all: list[int] = []
    for cid in np.unique(class_ids):
        m = class_ids == cid
        idx = np.where(m)[0]
        k = _nms_single_class(boxes[m], max_scores[m], iou_thresh)
        keep_all.extend(idx[k].tolist())
    keep = np.array(sorted(keep_all), dtype=np.int64)
    boxes, class_ids, max_scores = boxes[keep], class_ids[keep], max_scores[keep]

    # Rescale from 1024-space → original image, then normalise to [0, 1]
    orig_h, orig_w = orig_hw
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] / scale / orig_w, 0.0, 1.0)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] / scale / orig_h, 0.0, 1.0)

    return boxes.astype(np.float32), class_ids.astype(np.int64), max_scores.astype(np.float32)


# ---------------------------------------------------------------------------
# Letterbox preprocessing (pure torch, no nemotron-package dependency)
# ---------------------------------------------------------------------------


def _letterbox_chw(chw: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Resize-with-aspect-ratio + pad to *target_hw*, fill value 114.

    Input: CHW float32 in [0, 255].
    Output: CHW float32 in [0, 255], quantised to uint8 precision (matches NIM).
    """
    _, h, w = chw.shape
    th, tw = target_hw
    scale = min(th / h, tw / w)
    nh, nw = int(h * scale), int(w * scale)

    resized = F.interpolate(
        chw.unsqueeze(0), size=(nh, nw), mode="bilinear", align_corners=False
    ).squeeze(0)

    padded = torch.full((chw.shape[0], th, tw), _PAD_VALUE, dtype=torch.float32)
    padded[:, :nh, :nw] = resized
    padded = torch.clamp(padded, 0, 255).to(torch.uint8).float()
    return padded


# ---------------------------------------------------------------------------
# Public model wrapper
# ---------------------------------------------------------------------------


class TensorRTYOLOXModel(BaseModel):
    """TensorRT-accelerated YOLOX model.

    Exposes the same ``preprocess`` / ``invoke`` / ``postprocess`` /
    ``__call__`` interface as the PyTorch-backed Nemotron model wrappers,
    so it can be used as a transparent drop-in replacement.
    """

    def __init__(
        self,
        engine_path: str,
        class_labels: list[str],
        *,
        conf_thresh: float = 0.01,
        iou_thresh: float = 0.5,
    ) -> None:
        super().__init__()
        self._engine = _TRTEngine(engine_path)
        self._class_labels = list(class_labels)
        self._conf_thresh = conf_thresh
        self._iou_thresh = iou_thresh
        self._input_shape = _INPUT_HW
        self._max_batch = self._engine.max_batch
        logger.info(
            "Loaded TensorRT YOLOX engine: %s (%d classes, max_batch=%d)",
            engine_path, len(class_labels), self._max_batch,
        )

    # -- Preprocessing -------------------------------------------------------

    def preprocess(
        self,
        tensor: Union[torch.Tensor, np.ndarray],
        *_args: Any,
    ) -> torch.Tensor:
        """Letterbox resize + pad to 1024×1024 (matches YOLOX NIM preprocessing).

        Accepts CHW or BCHW in [0, 255] float32 (or uint8 ndarray). Extra
        positional args (e.g. ``orig_shape`` from the table-structure path)
        are accepted and ignored so the signature is compatible with all three
        model call sites.
        """
        if isinstance(tensor, np.ndarray):
            arr = tensor
            if arr.ndim == 4 and int(arr.shape[0]) == 1:
                arr = arr[0]
            if arr.ndim == 3 and int(arr.shape[-1]) == 3 and int(arr.shape[0]) != 3:
                tensor = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float()
            else:
                tensor = torch.from_numpy(np.ascontiguousarray(arr)).float()

        x = tensor.float()
        if x.ndim == 4 and x.shape[0] == 1:
            return _letterbox_chw(x[0], self._input_shape).unsqueeze(0)
        if x.ndim == 4:
            return torch.stack([_letterbox_chw(x[i], self._input_shape) for i in range(x.shape[0])])
        if x.ndim == 3:
            return _letterbox_chw(x, self._input_shape).unsqueeze(0)
        raise ValueError(f"Expected CHW or BCHW tensor, got shape {tuple(x.shape)}")

    # -- Inference + decode --------------------------------------------------

    def _infer_chunk(self, chunk: torch.Tensor) -> np.ndarray:
        """Run TRT on a chunk that fits within the engine's max batch."""
        return self._engine.infer(chunk)

    def invoke(
        self,
        input_data: torch.Tensor,
        orig_shape: Union[Tuple[int, int], Sequence[Tuple[int, int]]],
    ) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """Run TRT inference → YOLOX decode → NMS.

        Returns one prediction dict per image with keys ``boxes`` (N, 4) in
        normalised [0, 1] xyxy, ``labels`` (N,), and ``scores`` (N,).

        Automatically splits input into sub-batches if the engine's max batch
        size is smaller than the incoming batch.
        """
        batch_size = input_data.shape[0]
        max_b = self._engine.max_batch
        num_classes = len(self._class_labels)

        # Run TRT in chunks that fit the engine's compiled max batch size
        raw_chunks: list[np.ndarray] = []
        for start in range(0, batch_size, max_b):
            end = min(start + max_b, batch_size)
            raw_chunks.append(self._infer_chunk(input_data[start:end]))
        raw_output = np.concatenate(raw_chunks, axis=0)

        results: list[Dict[str, torch.Tensor]] = []
        for b in range(batch_size):
            if isinstance(orig_shape, (list, tuple)) and len(orig_shape) > 0 and isinstance(orig_shape[0], (list, tuple)):
                oh, ow = orig_shape[b]
            else:
                oh, ow = orig_shape  # type: ignore[misc]
            scale = min(_INPUT_HW[0] / oh, _INPUT_HW[1] / ow)

            boxes, labels, scores = _yolox_decode(
                raw_output[b: b + 1],
                num_classes,
                scale,
                (oh, ow),
                self._conf_thresh,
                self._iou_thresh,
            )
            results.append(
                {
                    "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                    "labels": torch.as_tensor(labels, dtype=torch.int64),
                    "scores": torch.as_tensor(scores, dtype=torch.float32),
                }
            )

        if batch_size == 1:
            return results[0]
        return results

    def __call__(
        self,
        input_data: torch.Tensor,
        orig_shape: Union[Tuple[int, int], Sequence[Tuple[int, int]]],
    ) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        return self.invoke(input_data, orig_shape)

    # -- Postprocess (page-elements compat) ----------------------------------

    def postprocess(
        self,
        preds: Union[Dict[str, torch.Tensor], Sequence[Dict[str, torch.Tensor]]],
    ) -> Tuple[
        Union[torch.Tensor, List[torch.Tensor]],
        Union[torch.Tensor, List[torch.Tensor]],
        Union[torch.Tensor, List[torch.Tensor]],
    ]:
        """Extract ``(boxes, labels, scores)`` from prediction dict(s).

        Compatible with the ``NemotronPageElementsV3.postprocess`` call site
        in ``page_elements/shared.py``.
        """
        if isinstance(preds, dict):
            return preds["boxes"], preds["labels"], preds["scores"]

        return (
            [p["boxes"] for p in preds],
            [p["labels"] for p in preds],
            [p["scores"] for p in preds],
        )

    # -- BaseModel metadata --------------------------------------------------

    @property
    def labels(self) -> list[str]:
        return self._class_labels

    @property
    def thresholds_per_class(self) -> dict[str, float]:
        return {label: 0.0 for label in self._class_labels}

    @property
    def model_name(self) -> str:
        return "tensorrt_yolox"

    @property
    def model_type(self) -> str:
        return "object-detection"

    @property
    def model_runmode(self) -> RunMode:
        return "local"

    @property
    def input(self) -> Any:
        return {"type": "image", "format": "RGB", "dimensions": "1024x1024"}

    @property
    def output(self) -> Any:
        return {
            "type": "detection",
            "format": "dict",
            "classes": self._class_labels,
        }

    @property
    def input_batch_size(self) -> int:
        return self._max_batch

    @property
    def input_shape(self) -> Tuple[int, int]:
        return self._input_shape
