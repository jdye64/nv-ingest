# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT engine wrappers for local GPU inference.

Provides drop-in replacements for HuggingFace-based models that load
pre-built ``.trt`` / ``.engine`` / ``.plan`` files and run inference
via the TensorRT Python API.  GPU memory is managed through PyTorch's
CUDA allocator (no ``pycuda`` dependency), which integrates correctly
with Ray's per-actor ``CUDA_VISIBLE_DEVICES`` assignment.

**Detection models** (YOLOX family)::

    from nemo_retriever.model.local.trt_engine import TRTYoloxEngine

    model = TRTYoloxEngine("/models/page_elements.engine")
    preprocessed = model.preprocess(chw_tensor)
    preds = model.invoke(preprocessed, orig_shape)

**Embedding models** (transformer encoder)::

    from nemo_retriever.model.local.trt_engine import TRTEmbedEngine

    model = TRTEmbedEngine("/models/embed.engine")
    vectors = model.embed(["hello world", "another sentence"])
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

_NP_TO_TORCH_DTYPE = {
    np.dtype("float32"): torch.float32,
    np.dtype("float16"): torch.float16,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("int8"): torch.int8,
    np.dtype("bool"): torch.bool,
}


def _load_engine(engine_path: str) -> Any:
    """Deserialize a TensorRT engine from disk.

    Enables ``engine_host_code_allowed`` on the runtime so that engines
    built with the ``VERSION_COMPATIBLE`` flag can be loaded across
    compatible TRT versions.
    """
    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(trt_logger)
        if hasattr(runtime, "engine_host_code_allowed"):
            runtime.engine_host_code_allowed = True
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(
            f"Failed to deserialize TensorRT engine from {engine_path}. "
            f"This usually means the .plan was compiled with a different "
            f"TensorRT version (installed: {trt.__version__}). Rebuild the "
            f"engine with your current TRT version or install the matching one."
        )
    return engine


class TRTYoloxEngine:
    """TensorRT engine wrapper that mimics the ``invoke`` / ``preprocess`` API
    used by ``NemotronPageElementsV3``, ``NemotronTableStructureV1``, and
    ``NemotronGraphicElementsV1``.

    The engine is expected to accept a single NCHW ``float32`` input and produce
    one or more output tensors matching the YOLOX detection head format.

    Parameters
    ----------
    engine_path
        Path to the serialized ``.trt`` / ``.engine`` / ``.plan`` file.
    input_shape
        Spatial (H, W) the engine was built for.  Defaults to ``(1024, 1024)``.
    labels
        Class label list used by post-processing.  When ``None``, a generic
        list is inferred from the output tensor shape.
    """

    def __init__(
        self,
        engine_path: str,
        *,
        input_shape: Tuple[int, int] = (1024, 1024),
        labels: Optional[List[str]] = None,
        conf_thresh: float = 0.01,
        iou_thresh: float = 0.5,
        min_bbox_size: int = 0,
    ) -> None:
        import tensorrt as trt

        self._device = torch.device("cuda")
        torch.cuda.init()

        self._engine_path = engine_path
        self._input_shape = input_shape
        self._labels = labels
        self._conf_thresh = conf_thresh
        self._iou_thresh = iou_thresh
        self._min_bbox_size = min_bbox_size

        engine = _load_engine(engine_path)
        self._engine = engine
        self._context = engine.create_execution_context()
        self._trt = trt
        self._stream = torch.cuda.Stream(self._device)

        self._input_name: Optional[str] = None
        self._output_names: List[str] = []
        self._input_dtype: Any = None
        self._output_dtypes: List[Any] = []
        self._output_shapes: List[Tuple[int, ...]] = []

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self._input_name = name
                self._input_dtype = trt.nptype(engine.get_tensor_dtype(name))
            else:
                self._output_names.append(name)
                self._output_dtypes.append(trt.nptype(engine.get_tensor_dtype(name)))
                self._output_shapes.append(tuple(engine.get_tensor_shape(name)))

        if self._input_name is None:
            raise RuntimeError(f"TRT engine {engine_path} has no input tensor")

        logger.info(
            "TRTYoloxEngine loaded: path=%s, input=%s, outputs=%s, spatial=%s, device=%s",
            engine_path,
            self._input_name,
            self._output_names,
            input_shape,
            self._device,
        )

    # ------------------------------------------------------------------
    # Public interface matching NemotronPageElementsV3 / TableStructureV1
    # ------------------------------------------------------------------

    def preprocess(self, tensor: Any, orig_shape: Any = None) -> torch.Tensor:
        """Resize / pad to engine input shape using the same aspect-preserving
        letterbox as the HuggingFace ``resize_pad``.

        Accepts CHW or BCHW torch tensors, or HWC / CHW numpy arrays.
        Returns BCHW ``float32`` on CPU ready for ``invoke``.
        """
        from nemotron_page_elements_v3.model import resize_pad as resize_pad_page_elements

        if isinstance(tensor, np.ndarray):
            arr = tensor
            if arr.ndim == 4 and int(arr.shape[0]) == 1:
                arr = arr[0]
            if arr.ndim != 3:
                raise ValueError(f"Expected 3D image array, got shape {getattr(arr, 'shape', None)}")
            if int(arr.shape[-1]) == 3 and int(arr.shape[0]) != 3:
                x = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
            else:
                x = torch.from_numpy(np.ascontiguousarray(arr))
            x = x.to(dtype=torch.float32)
        elif isinstance(tensor, torch.Tensor):
            x = tensor.float()
        else:
            raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(tensor)!r}")

        if x.ndim == 4:
            outs = []
            for i in range(int(x.shape[0])):
                y = resize_pad_page_elements(x[i], self._input_shape)
                y = torch.clamp(y, 0, 255).to(torch.uint8).float()
                outs.append(y)
            return torch.stack(outs, dim=0).contiguous()

        if x.ndim == 3:
            y = resize_pad_page_elements(x, self._input_shape)
            y = torch.clamp(y, 0, 255).to(torch.uint8).float()
            return y.unsqueeze(0).contiguous()

        raise ValueError(f"Expected CHW or BCHW tensor, got shape {tuple(x.shape)}")

    def invoke(
        self,
        input_tensor: torch.Tensor,
        orig_shape: Union[Tuple[int, int], List[Tuple[int, int]]],
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Run TRT inference and return results in the same dict format as
        the HuggingFace model wrappers.

        Boxes are returned **normalised to [0, 1]** relative to
        ``orig_shape``, matching the HuggingFace ``YoloXWrapper``.
        """
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)

        batch_size = int(input_tensor.shape[0])

        if isinstance(orig_shape, (list, tuple)) and orig_shape and isinstance(orig_shape[0], (list, tuple)):
            shapes = list(orig_shape)
        else:
            shapes = [orig_shape] * batch_size

        results: List[Dict[str, Any]] = []

        for b in range(batch_size):
            single = input_tensor[b : b + 1]
            raw_outputs = self._infer_single(single)
            sh = shapes[b] if b < len(shapes) else shapes[-1]
            pred = self._raw_to_pred_dict(raw_outputs, orig_shape=sh)
            results.append(pred)

        if batch_size == 1:
            return results[0]
        return results

    def __call__(
        self,
        input_data: torch.Tensor,
        orig_shape: Union[Tuple[int, int], List[Tuple[int, int]]],
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        return self.invoke(input_data, orig_shape)

    # ------------------------------------------------------------------
    # Metadata (matches BaseModel-style properties)
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return f"TRT:{self._engine_path}"

    @property
    def thresholds_per_class(self) -> Dict[str, float]:
        if self._labels:
            return {label: 0.0 for label in self._labels}
        return {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer_single(self, input_nchw: torch.Tensor) -> List[np.ndarray]:
        """Execute inference for a single B=1 input."""
        torch_dtype = _NP_TO_TORCH_DTYPE.get(np.dtype(self._input_dtype), torch.float32)
        d_input = input_nchw.to(device=self._device, dtype=torch_dtype).contiguous()
        self._context.set_tensor_address(self._input_name, d_input.data_ptr())

        d_outputs: List[torch.Tensor] = []
        for shape, np_dtype in zip(self._output_shapes, self._output_dtypes):
            effective_shape = (1, *shape[1:]) if shape[0] == -1 else shape
            t_dtype = _NP_TO_TORCH_DTYPE.get(np.dtype(np_dtype), torch.float32)
            d_outputs.append(torch.empty(effective_shape, dtype=t_dtype, device=self._device))

        for name, d_out in zip(self._output_names, d_outputs):
            self._context.set_tensor_address(name, d_out.data_ptr())

        with torch.cuda.stream(self._stream):
            self._context.execute_async_v3(stream_handle=self._stream.cuda_stream)

        self._stream.synchronize()

        return [d_out.cpu().numpy() for d_out in d_outputs]

    def _raw_to_pred_dict(
        self,
        raw_outputs: List[np.ndarray],
        *,
        orig_shape: Tuple[int, int],
    ) -> Dict[str, Any]:
        """Convert raw TRT output arrays into the prediction dict expected by
        the shared post-processing code.

        Replicates the full ``YoloXWrapper.forward()`` pipeline from
        ``nemotron_page_elements_v3``:
          1. NMS (conf threshold + torchvision NMS)
          2. Scale boxes from model input space → original image pixel space
          3. Clip to original image bounds & remove tiny boxes
          4. Normalise to ``[0, 1]``

        Returns a dict with normalised ``boxes``, ``labels``, ``scores``.
        """
        empty = {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.int64),
            "scores": torch.empty((0,), dtype=torch.float32),
        }

        if len(raw_outputs) == 1:
            out = raw_outputs[0]
            if out.ndim == 2:
                out = out[np.newaxis]  # (N, 5+C) → (1, N, 5+C)

            prediction = torch.from_numpy(out).float()
            num_classes = prediction.shape[-1] - 5

            from nemotron_page_elements_v3.yolox.boxes import postprocess as yolox_nms

            nms_results = yolox_nms(
                prediction,
                num_classes,
                conf_thre=self._conf_thresh,
                nms_thre=self._iou_thresh,
                class_agnostic=True,
            )

            p = nms_results[0]  # single image
            if p is None or p.numel() == 0:
                return empty

            p = p.view(-1, p.size(-1))

            orig_h, orig_w = orig_shape
            ratio = min(
                self._input_shape[0] / orig_h,
                self._input_shape[1] / orig_w,
            )
            boxes = p[:, :4] / ratio

            boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, orig_w)
            boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, orig_h)

            kept = (
                (boxes[:, 2] - boxes[:, 0] > self._min_bbox_size)
                & (boxes[:, 3] - boxes[:, 1] > self._min_bbox_size)
            )
            boxes = boxes[kept]
            p = p[kept]

            boxes[:, [0, 2]] /= orig_w
            boxes[:, [1, 3]] /= orig_h

            scores = p[:, 4] * p[:, 5]
            labels = p[:, 6]

            return {
                "boxes": boxes.float(),
                "labels": labels.long(),
                "scores": scores.float(),
            }

        if len(raw_outputs) >= 3:
            boxes_np = raw_outputs[0].squeeze(0)
            boxes = torch.from_numpy(boxes_np.copy()).float()

            orig_h, orig_w = orig_shape
            ratio = min(
                self._input_shape[0] / orig_h,
                self._input_shape[1] / orig_w,
            )
            boxes /= ratio
            boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, orig_w)
            boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, orig_h)
            boxes[:, [0, 2]] /= orig_w
            boxes[:, [1, 3]] /= orig_h

            return {
                "boxes": boxes,
                "labels": torch.from_numpy(raw_outputs[1].squeeze(0).copy()).long(),
                "scores": torch.from_numpy(raw_outputs[2].squeeze(0).copy()).float(),
            }

        return empty


# ======================================================================
# TRTEmbedEngine – transformer encoder embedding via TensorRT
# ======================================================================


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = x.float()
    denom = x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    return x / denom


class TRTEmbedEngine:
    """TensorRT engine wrapper that mimics the ``embed()`` API of
    ``LlamaNemotronEmbed1BV2Embedder``.

    The engine is expected to accept ``input_ids`` and ``attention_mask``
    tensors (int32, shape ``[B, S]``) and produce a last-hidden-state
    output of shape ``[B, S, D]``.  Mean-pooling and optional L2
    normalisation are applied on top.

    Parameters
    ----------
    engine_path
        Path to the serialized ``.trt`` / ``.engine`` file.
    tokenizer_name
        HuggingFace tokenizer to use for text→token conversion.
        Defaults to ``"nvidia/llama-nemotron-embed-1b-v2"``.
    max_length
        Maximum sequence length passed to the tokenizer.
    normalize
        Whether to L2-normalise the output vectors.
    """

    def __init__(
        self,
        engine_path: str,
        *,
        tokenizer_name: str = "nvidia/llama-nemotron-embed-1b-v2",
        max_length: int = 8192,
        normalize: bool = True,
        hf_cache_dir: Optional[str] = None,
    ) -> None:
        import tensorrt as trt

        from transformers import AutoTokenizer
        from nemo_retriever.utils.hf_cache import configure_global_hf_cache_base
        from nemo_retriever.utils.hf_model_registry import get_hf_revision

        self._device = torch.device("cuda")
        torch.cuda.init()

        self._engine_path = engine_path
        self._max_length = max_length
        self._normalize = normalize

        cache_dir = configure_global_hf_cache_base(hf_cache_dir)
        revision = get_hf_revision(tokenizer_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            revision=revision,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        engine = _load_engine(engine_path)
        self._engine = engine
        self._context = engine.create_execution_context()
        self._trt = trt
        self._stream = torch.cuda.Stream(self._device)

        self._io_info: Dict[str, Dict[str, Any]] = {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            self._io_info[name] = {
                "mode": mode,
                "dtype": trt.nptype(engine.get_tensor_dtype(name)),
                "shape": tuple(engine.get_tensor_shape(name)),
                "is_input": mode == trt.TensorIOMode.INPUT,
            }

        logger.info(
            "TRTEmbedEngine loaded: path=%s, tensors=%s, max_length=%d, normalize=%s, device=%s",
            engine_path,
            list(self._io_info.keys()),
            max_length,
            normalize,
            self._device,
        )

    @property
    def is_remote(self) -> bool:
        return False

    @property
    def model_name(self) -> str:
        return f"TRT:{self._engine_path}"

    def embed(self, texts: "Sequence[str]", *, batch_size: int = 64) -> torch.Tensor:
        """Return a CPU tensor of shape ``[N, D]``, matching the HF embedder API."""
        from typing import Sequence as _Seq  # noqa: F811

        texts_list = [str(t) for t in texts if str(t).strip()]
        if not texts_list:
            return torch.empty((0, 0), dtype=torch.float32)

        bs = max(1, int(batch_size))
        all_vecs: List[torch.Tensor] = []

        for start in range(0, len(texts_list), bs):
            chunk = texts_list[start : start + bs]
            tokens = self._tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max(1, self._max_length),
                return_tensors="np",
            )
            input_ids = tokens["input_ids"].astype(np.int32)
            attention_mask = tokens["attention_mask"].astype(np.int32)

            hidden = self._infer_batch(input_ids, attention_mask)

            mask_f = torch.from_numpy(attention_mask.astype(np.float32)).unsqueeze(-1)  # [B, S, 1]
            pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1)  # [B, D]
            if self._normalize:
                pooled = _l2_normalize(pooled)
            all_vecs.append(pooled)

        return torch.cat(all_vecs, dim=0) if all_vecs else torch.empty((0, 0), dtype=torch.float32)

    def _infer_batch(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> torch.Tensor:
        """Run TRT inference and return the last-hidden-state as a float32 CPU tensor."""
        inputs_np = {
            "input_ids": np.ascontiguousarray(input_ids),
            "attention_mask": np.ascontiguousarray(attention_mask),
        }

        d_inputs: Dict[str, torch.Tensor] = {}
        for name, arr in inputs_np.items():
            if name in self._io_info:
                self._context.set_input_shape(name, arr.shape)
                np_dtype = np.dtype(arr.dtype)
                t_dtype = _NP_TO_TORCH_DTYPE.get(np_dtype, torch.int32)
                d_buf = torch.from_numpy(arr).to(device=self._device, dtype=t_dtype).contiguous()
                self._context.set_tensor_address(name, d_buf.data_ptr())
                d_inputs[name] = d_buf

        output_name: Optional[str] = None
        for name, info in self._io_info.items():
            if not info["is_input"]:
                output_name = name
                break
        if output_name is None:
            raise RuntimeError("TRT embed engine has no output tensor")

        out_shape = tuple(self._context.get_tensor_shape(output_name))
        out_np_dtype = np.dtype(self._io_info[output_name]["dtype"])
        out_t_dtype = _NP_TO_TORCH_DTYPE.get(out_np_dtype, torch.float32)
        d_output = torch.empty(out_shape, dtype=out_t_dtype, device=self._device)
        self._context.set_tensor_address(output_name, d_output.data_ptr())

        with torch.cuda.stream(self._stream):
            self._context.execute_async_v3(stream_handle=self._stream.cuda_stream)

        self._stream.synchronize()

        return d_output.cpu().float()
