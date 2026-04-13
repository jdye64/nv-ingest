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

    Performance characteristics
    --------------------------
    * **Batched inference** – the entire BCHW batch is executed in a single
      ``execute_async_v3`` call instead of per-image.
    * **GPU-resident pipeline** – TRT output stays on GPU through NMS and
      box normalisation; only the final prediction dicts are copied to CPU.
    * **Buffer reuse** – output device buffers are allocated once and reused
      across calls of the same batch size, eliminating per-call ``cudaMalloc``.

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

        # Detect dynamic batch dimension.
        self._has_dynamic_batch = (
            tuple(engine.get_tensor_shape(self._input_name))[0] == -1
        )

        # Cache for reusable GPU output buffers keyed by resolved output shapes.
        self._cached_out_shapes: Optional[List[Tuple[int, ...]]] = None
        self._cached_out_bufs: List[torch.Tensor] = []

        # Pre-import the upstream NMS function once to avoid per-call overhead.
        from nemotron_page_elements_v3.yolox.boxes import postprocess as _yolox_nms
        self._yolox_nms = _yolox_nms

        logger.info(
            "TRTYoloxEngine loaded: path=%s, input=%s, outputs=%s, "
            "spatial=%s, dynamic_batch=%s, device=%s",
            engine_path,
            self._input_name,
            self._output_names,
            input_shape,
            self._has_dynamic_batch,
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

        The full batch is executed in **one** TRT call.  NMS and box
        normalisation run on GPU; only the final prediction dicts are
        moved to CPU.
        """
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)

        batch_size = int(input_tensor.shape[0])

        if isinstance(orig_shape, (list, tuple)) and orig_shape and isinstance(orig_shape[0], (list, tuple)):
            shapes = list(orig_shape)
        else:
            shapes = [orig_shape] * batch_size

        gpu_outputs = self._infer_batched(input_tensor)
        results = self._postprocess_batch(gpu_outputs, shapes)

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
    # Batched GPU inference
    # ------------------------------------------------------------------

    def _infer_batched(self, input_nchw: torch.Tensor) -> List[torch.Tensor]:
        """Execute batched TRT inference.  Returns **GPU-resident** tensors.

        Output buffers are cached and reused across calls of the same
        effective shape, eliminating per-call ``cudaMalloc`` overhead.
        """
        torch_dtype = _NP_TO_TORCH_DTYPE.get(np.dtype(self._input_dtype), torch.float32)
        d_input = input_nchw.to(device=self._device, dtype=torch_dtype).contiguous()

        if self._has_dynamic_batch:
            self._context.set_input_shape(self._input_name, tuple(d_input.shape))

        self._context.set_tensor_address(self._input_name, d_input.data_ptr())

        # Resolve actual output shapes (batch dim may be dynamic).
        needed_shapes: List[Tuple[int, ...]] = [
            tuple(self._context.get_tensor_shape(name))
            for name in self._output_names
        ]

        # Allocate or reuse output buffers.
        if needed_shapes != self._cached_out_shapes:
            self._cached_out_bufs = []
            for shape, np_dtype in zip(needed_shapes, self._output_dtypes):
                t_dtype = _NP_TO_TORCH_DTYPE.get(np.dtype(np_dtype), torch.float32)
                self._cached_out_bufs.append(
                    torch.empty(shape, dtype=t_dtype, device=self._device)
                )
            self._cached_out_shapes = needed_shapes

        for name, d_out in zip(self._output_names, self._cached_out_bufs):
            self._context.set_tensor_address(name, d_out.data_ptr())

        with torch.cuda.stream(self._stream):
            self._context.execute_async_v3(stream_handle=self._stream.cuda_stream)
        self._stream.synchronize()

        return self._cached_out_bufs

    # ------------------------------------------------------------------
    # GPU-resident post-processing (NMS + normalisation)
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_pred() -> Dict[str, Any]:
        return {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.int64),
            "scores": torch.empty((0,), dtype=torch.float32),
        }

    def _postprocess_batch(
        self,
        gpu_outputs: List[torch.Tensor],
        shapes: List[Tuple[int, int]],
    ) -> List[Dict[str, Any]]:
        """NMS + box scaling + normalisation, all on GPU.

        Replicates ``YoloXWrapper.forward()`` post-processing from
        ``nemotron_page_elements_v3`` but operates on the full batch
        at once.  Results are moved to CPU only at the end.
        """
        B = len(shapes)

        if len(gpu_outputs) == 1:
            raw = gpu_outputs[0]  # (B, N, 5+C) on GPU
            if raw.ndim == 2:
                raw = raw.unsqueeze(0)

            num_classes = raw.shape[-1] - 5

            # NMS runs on GPU tensors (torchvision.ops.nms supports CUDA).
            # The upstream postprocess modifies the tensor in-place
            # (cxcywh → xyxy), which is fine because TRT overwrites
            # the buffer on the next call.
            nms_results = self._yolox_nms(
                raw,
                num_classes,
                conf_thre=self._conf_thresh,
                nms_thre=self._iou_thresh,
                class_agnostic=True,
            )

            results: List[Dict[str, Any]] = []
            for b in range(B):
                p = nms_results[b]
                if p is None or p.numel() == 0:
                    results.append(self._empty_pred())
                    continue

                p = p.view(-1, p.size(-1))

                orig_h, orig_w = shapes[b]
                ratio = min(
                    self._input_shape[0] / orig_h,
                    self._input_shape[1] / orig_w,
                )
                boxes = p[:, :4] / ratio

                boxes[:, [0, 2]].clamp_(0, orig_w)
                boxes[:, [1, 3]].clamp_(0, orig_h)

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

                results.append({
                    "boxes": boxes.cpu().float(),
                    "labels": labels.cpu().long(),
                    "scores": scores.cpu().float(),
                })
            return results

        if len(gpu_outputs) >= 3:
            all_boxes = gpu_outputs[0]   # (B, N, 4) on GPU
            all_labels = gpu_outputs[1]  # (B, N) on GPU
            all_scores = gpu_outputs[2]  # (B, N) on GPU

            results = []
            for b in range(B):
                boxes = all_boxes[b].clone()
                orig_h, orig_w = shapes[b]
                ratio = min(
                    self._input_shape[0] / orig_h,
                    self._input_shape[1] / orig_w,
                )
                boxes /= ratio
                boxes[:, [0, 2]].clamp_(0, orig_w)
                boxes[:, [1, 3]].clamp_(0, orig_h)
                boxes[:, [0, 2]] /= orig_w
                boxes[:, [1, 3]] /= orig_h

                results.append({
                    "boxes": boxes.cpu().float(),
                    "labels": all_labels[b].cpu().long(),
                    "scores": all_scores[b].cpu().float(),
                })
            return results

        return [self._empty_pred() for _ in range(B)]


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

        # Parse optimization profiles: list of (profile_idx, max_batch, max_seq)
        # sorted by max_seq ascending so we pick the tightest fit first.
        self._profiles: List[Tuple[int, int, int]] = []
        input_tensor_name = next(
            (n for n, info in self._io_info.items() if info["is_input"]),
            "input_ids",
        )
        num_profiles = engine.num_optimization_profiles
        for p_idx in range(num_profiles):
            _min_shape, _opt_shape, max_shape = engine.get_tensor_profile_shape(
                input_tensor_name, p_idx
            )
            max_b, max_s = int(max_shape[0]), int(max_shape[1])
            self._profiles.append((p_idx, max_b, max_s))

        self._profiles.sort(key=lambda t: (t[2], t[1]))

        # Clamp max_length to the absolute max sequence any profile supports.
        abs_max_seq = max(p[2] for p in self._profiles) if self._profiles else max_length
        self._max_length = min(max_length, abs_max_seq)

        logger.info(
            "TRTEmbedEngine loaded: path=%s, tensors=%s, max_length=%d, "
            "normalize=%s, profiles=%s, device=%s",
            engine_path,
            list(self._io_info.keys()),
            self._max_length,
            normalize,
            self._profiles,
            self._device,
        )

    @property
    def is_remote(self) -> bool:
        return False

    @property
    def model_name(self) -> str:
        return f"TRT:{self._engine_path}"

    def embed(self, texts: "Sequence[str]", *, batch_size: int = 64) -> torch.Tensor:
        """Return a CPU tensor of shape ``[N, D]``, matching the HF embedder API.

        Tokenised inputs are automatically split into sub-batches that
        respect the engine's optimisation profile limits on
        ``(batch, seq_len)``.  Mean-pooling and L2 normalisation run on
        GPU; only the final vectors are copied to CPU.
        """
        texts_list = [str(t) for t in texts if str(t).strip()]
        if not texts_list:
            return torch.empty((0, 0), dtype=torch.float32)

        tokens_per_text = self._tokenizer(
            texts_list,
            padding=False,
            truncation=True,
            max_length=max(1, self._max_length),
        )
        lengths = [len(ids) for ids in tokens_per_text["input_ids"]]

        sub_batches = self._plan_sub_batches(lengths, max_batch=max(1, int(batch_size)))

        result_vecs: List[Optional[torch.Tensor]] = [None] * len(texts_list)

        for indices in sub_batches:
            chunk = [texts_list[i] for i in indices]
            max_seq = max(lengths[i] for i in indices)
            tokens = self._tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_seq,
                return_tensors="np",
            )
            input_ids = tokens["input_ids"].astype(np.int32)
            attention_mask = tokens["attention_mask"].astype(np.int32)

            # hidden stays on GPU
            hidden = self._infer_batch(input_ids, attention_mask)

            # Pooling and normalisation on GPU
            mask_gpu = (
                torch.from_numpy(attention_mask.astype(np.float32))
                .to(device=self._device)
                .unsqueeze(-1)
            )
            pooled = (hidden * mask_gpu).sum(dim=1) / mask_gpu.sum(dim=1)
            if self._normalize:
                pooled = _l2_normalize(pooled)

            # Single CPU transfer for the whole sub-batch
            pooled_cpu = pooled.cpu()
            for local_i, orig_i in enumerate(indices):
                result_vecs[orig_i] = pooled_cpu[local_i]

        filled = [v for v in result_vecs if v is not None]
        return torch.stack(filled, dim=0) if filled else torch.empty((0, 0), dtype=torch.float32)

    # ------------------------------------------------------------------
    # Profile-aware batching helpers
    # ------------------------------------------------------------------

    def _best_profile(self, batch: int, seq_len: int) -> Optional[int]:
        """Return the profile index whose limits fit ``(batch, seq_len)``.

        Prefers the tightest ``max_seq`` that still fits, breaking ties
        by tightest ``max_batch``.  Returns ``None`` if nothing fits.
        """
        for p_idx, max_b, max_s in self._profiles:
            if batch <= max_b and seq_len <= max_s:
                return p_idx
        return None

    def _max_batch_for_seq(self, seq_len: int) -> int:
        """Largest batch size any profile can handle for *seq_len*."""
        best = 0
        for _, max_b, max_s in self._profiles:
            if seq_len <= max_s and max_b > best:
                best = max_b
        return best

    def _plan_sub_batches(
        self,
        lengths: List[int],
        *,
        max_batch: int,
    ) -> List[List[int]]:
        """Partition text indices into sub-batches that each fit within an
        optimisation profile.

        Texts are sorted by token length so that texts within a sub-batch
        are similarly sized, minimising padding waste and maximising the
        batch size allowed by the profile.
        """
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])

        sub_batches: List[List[int]] = []
        pos = 0
        while pos < len(order):
            # The longest text in this candidate sub-batch determines
            # the required seq_len.  Start with one text and grow.
            end = pos + 1
            while end < len(order):
                candidate_seq = lengths[order[end]]
                candidate_batch = end - pos + 1
                if candidate_batch > max_batch:
                    break
                if self._best_profile(candidate_batch, candidate_seq) is None:
                    break
                end += 1

            # Validate the sub-batch we formed.
            sub = order[pos:end]
            max_seq = max(lengths[i] for i in sub)
            if self._best_profile(len(sub), max_seq) is None:
                # Even a single text doesn't fit (shouldn't happen if
                # max_length was clamped, but be defensive).
                sub_batches.append([sub[0]])
                pos += 1
            else:
                sub_batches.append(sub)
                pos = end

        return sub_batches

    # ------------------------------------------------------------------
    # TRT inference
    # ------------------------------------------------------------------

    def _infer_batch(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> torch.Tensor:
        """Run TRT inference and return the last-hidden-state as a **GPU-resident**
        float32 tensor.

        Automatically selects the best optimisation profile for the
        input shape ``(B, S)``.
        """
        B, S = input_ids.shape
        profile_idx = self._best_profile(B, S)
        if profile_idx is None:
            raise RuntimeError(
                f"No TRT optimization profile supports shape ({B}, {S}). "
                f"Available profiles: {self._profiles}"
            )

        if self._context.active_optimization_profile != profile_idx:
            self._context.set_optimization_profile_async(
                profile_idx, self._stream.cuda_stream
            )
            self._stream.synchronize()

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

        return d_output.float()
