# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT engine wrappers for local GPU inference.

Provides drop-in replacements for HuggingFace-based models that load
pre-built ``.trt`` / ``.engine`` files and run inference via the TensorRT
Python API with ``pycuda``.

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
        Path to the serialized ``.trt`` / ``.engine`` file.
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
    ) -> None:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401 – initializes CUDA context

        self._engine_path = engine_path
        self._input_shape = input_shape
        self._labels = labels

        engine = _load_engine(engine_path)
        self._engine = engine
        self._context = engine.create_execution_context()
        self._trt = trt
        self._cuda = cuda

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

        self._stream = cuda.Stream()

        logger.info(
            "TRTYoloxEngine loaded: path=%s, input=%s, outputs=%s, spatial=%s",
            engine_path,
            self._input_name,
            self._output_names,
            input_shape,
        )

    # ------------------------------------------------------------------
    # Public interface matching NemotronPageElementsV3 / TableStructureV1
    # ------------------------------------------------------------------

    def preprocess(self, tensor: torch.Tensor, orig_shape: Any = None) -> torch.Tensor:
        """Resize / pad to engine input shape.

        Accepts CHW or BCHW tensors.  Returns BCHW ``float32`` on CPU
        ready for ``invoke``.
        """
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError(f"Expected CHW or BCHW tensor, got shape {tuple(tensor.shape)}")

        h, w = self._input_shape
        _, c, th, tw = tensor.shape
        if th != h or tw != w:
            tensor = torch.nn.functional.interpolate(
                tensor.float(), size=(h, w), mode="bilinear", align_corners=False
            )

        return tensor.to(dtype=torch.float32).contiguous()

    def invoke(
        self,
        input_tensor: torch.Tensor,
        orig_shape: Union[Tuple[int, int], List[Tuple[int, int]]],
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Run TRT inference and return results in the same dict format as
        the HuggingFace model wrappers."""
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)

        batch_size = int(input_tensor.shape[0])
        results: List[Dict[str, Any]] = []

        for b in range(batch_size):
            single = input_tensor[b : b + 1]
            raw_outputs = self._infer_single(single)
            pred = self._raw_to_pred_dict(raw_outputs)
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
        cuda = self._cuda

        h_input = np.ascontiguousarray(input_nchw.cpu().numpy().astype(self._input_dtype))

        d_input = cuda.mem_alloc(h_input.nbytes)
        cuda.memcpy_htod_async(d_input, h_input, self._stream)
        self._context.set_tensor_address(self._input_name, int(d_input))

        h_outputs: List[np.ndarray] = []
        d_outputs = []
        for shape, dtype in zip(self._output_shapes, self._output_dtypes):
            effective_shape = (1, *shape[1:]) if shape[0] == -1 else shape
            h_out = np.empty(effective_shape, dtype=dtype)
            d_out = cuda.mem_alloc(h_out.nbytes)
            h_outputs.append(h_out)
            d_outputs.append(d_out)

        for name, d_out in zip(self._output_names, d_outputs):
            self._context.set_tensor_address(name, int(d_out))

        self._context.execute_async_v3(stream_handle=self._stream.handle)

        for h_out, d_out in zip(h_outputs, d_outputs):
            cuda.memcpy_dtoh_async(h_out, d_out, self._stream)

        self._stream.synchronize()

        d_input.free()
        for d_out in d_outputs:
            d_out.free()

        return h_outputs

    def _raw_to_pred_dict(self, raw_outputs: List[np.ndarray]) -> Dict[str, Any]:
        """Convert raw TRT output arrays into the prediction dict format
        expected by the shared post-processing code.

        The YOLOX detection head typically outputs either:
        - A single tensor of shape ``(1, N, 5+C)`` (boxes + objectness + class scores), or
        - Multiple tensors for boxes, labels, scores separately.

        This method handles both and returns a dict with ``boxes``, ``labels``,
        ``scores`` keys as torch tensors.
        """
        if len(raw_outputs) == 1:
            out = raw_outputs[0]
            if out.ndim == 3:
                out = out[0]
            return {
                "boxes": torch.from_numpy(out[:, :4].copy()).float(),
                "labels": torch.from_numpy(out[:, 5:].argmax(axis=-1).copy()).long(),
                "scores": torch.from_numpy((out[:, 4:5] * out[:, 5:]).max(axis=-1).copy()).float(),
            }

        if len(raw_outputs) >= 3:
            boxes = torch.from_numpy(raw_outputs[0].squeeze(0).copy()).float()
            labels = torch.from_numpy(raw_outputs[1].squeeze(0).copy()).long()
            scores = torch.from_numpy(raw_outputs[2].squeeze(0).copy()).float()
            return {"boxes": boxes, "labels": labels, "scores": scores}

        return {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.int64),
            "scores": torch.empty((0,), dtype=torch.float32),
        }


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
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401

        from transformers import AutoTokenizer
        from nemo_retriever.utils.hf_cache import configure_global_hf_cache_base
        from nemo_retriever.utils.hf_model_registry import get_hf_revision

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
        self._cuda = cuda

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

        self._stream = cuda.Stream()

        logger.info(
            "TRTEmbedEngine loaded: path=%s, tensors=%s, max_length=%d, normalize=%s",
            engine_path,
            list(self._io_info.keys()),
            max_length,
            normalize,
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
        cuda = self._cuda
        trt = self._trt

        B, S = input_ids.shape

        inputs_np = {
            "input_ids": np.ascontiguousarray(input_ids),
            "attention_mask": np.ascontiguousarray(attention_mask),
        }

        d_inputs: Dict[str, Any] = {}
        for name, arr in inputs_np.items():
            if name in self._io_info:
                self._context.set_input_shape(name, arr.shape)
                d_buf = cuda.mem_alloc(arr.nbytes)
                cuda.memcpy_htod_async(d_buf, arr, self._stream)
                self._context.set_tensor_address(name, int(d_buf))
                d_inputs[name] = d_buf

        output_name: Optional[str] = None
        for name, info in self._io_info.items():
            if not info["is_input"]:
                output_name = name
                break
        if output_name is None:
            raise RuntimeError("TRT embed engine has no output tensor")

        out_shape = tuple(self._context.get_tensor_shape(output_name))
        out_dtype = self._io_info[output_name]["dtype"]
        h_output = np.empty(out_shape, dtype=out_dtype)
        d_output = cuda.mem_alloc(h_output.nbytes)
        self._context.set_tensor_address(output_name, int(d_output))

        self._context.execute_async_v3(stream_handle=self._stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, self._stream)
        self._stream.synchronize()

        for d_buf in d_inputs.values():
            d_buf.free()
        d_output.free()

        return torch.from_numpy(h_output.copy()).float()
