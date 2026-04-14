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
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional DALI accelerated preprocessing
# ---------------------------------------------------------------------------

_DALI_AVAILABLE = False
try:
    import nvidia.dali.fn as _dali_fn
    import nvidia.dali.types as _dali_types
    from nvidia.dali import pipeline_def as _dali_pipeline_def

    _DALI_AVAILABLE = True
except ImportError:
    pass


class _DaliLetterboxPipeline:
    """DALI pipeline that letterbox-resizes a variable-size image batch on GPU.

    The pipeline mirrors ``_gpu_letterbox`` but executes entirely inside
    DALI's GPU kernels — no per-image Python loops, no intermediate copies.

    After ``build()``, call ``run(images)`` with a list of HWC uint8 numpy
    arrays and get back a ``(BCHW_float32_torch_tensor, orig_shapes)`` tuple,
    both on GPU.
    """

    def __init__(self, target_h: int, target_w: int, device_id: int = 0, max_batch_size: int = 64) -> None:
        if not _DALI_AVAILABLE:
            raise ImportError("nvidia-dali is required for DaliLetterboxPipeline.")

        self._target_h = target_h
        self._target_w = target_w
        self._device_id = device_id
        self._max_batch_size = max_batch_size
        self._pipe = self._build_pipeline()
        self._pipe.build()

    def _build_pipeline(self) -> Any:
        @_dali_pipeline_def(
            batch_size=self._max_batch_size,
            num_threads=4,
            device_id=self._device_id,
        )
        def _pipe():
            images = _dali_fn.external_source(device="cpu", name="images")
            images = images.gpu()
            images = _dali_fn.cast(images, dtype=_dali_types.FLOAT)
            images = _dali_fn.resize(
                images,
                resize_x=self._target_w,
                resize_y=self._target_h,
                mode="not_larger",
                interp_type=_dali_types.INTERP_LINEAR,
            )
            images = _dali_fn.crop(
                images,
                crop=(self._target_h, self._target_w),
                out_of_bounds_policy="pad",
                fill_values=114.0,
            )
            images = _dali_fn.transpose(images, perm=[2, 0, 1])
            return images

        return _pipe()

    def run(self, images: "list[np.ndarray]") -> Tuple[torch.Tensor, "list[Tuple[int, int]]"]:
        """Feed a batch of HWC uint8 numpy arrays, return (BCHW float32 GPU tensor, orig_shapes)."""
        orig_shapes = [(int(img.shape[0]), int(img.shape[1])) for img in images]

        self._pipe.feed_input("images", images)
        (output,) = self._pipe.run()

        # Zero-copy DALI → PyTorch via DLPack.
        t = torch.from_dlpack(output.as_tensor()).contiguous()

        return t, orig_shapes


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


def _gpu_letterbox(
    img: torch.Tensor,
    target_hw: Tuple[int, int],
    pad_value: float = 114.0,
) -> torch.Tensor:
    """Aspect-preserving resize + pad for a CHW or BCHW tensor (any device).

    Equivalent to ``nemotron_page_elements_v3.model.resize_pad`` but
    implemented with pure ``torch`` ops so it works on CUDA tensors.
    """
    if img.ndim == 3:
        img = img.unsqueeze(0)
    B, C, H, W = img.shape
    tgt_h, tgt_w = target_hw
    ratio = min(tgt_h / H, tgt_w / W)
    new_h, new_w = int(round(H * ratio)), int(round(W * ratio))

    resized = F.interpolate(
        img.float(), size=(new_h, new_w), mode="bilinear", align_corners=False,
    )

    pad_bottom = tgt_h - new_h
    pad_right = tgt_w - new_w
    if pad_bottom > 0 or pad_right > 0:
        resized = F.pad(resized, (0, pad_right, 0, pad_bottom), value=pad_value)

    return resized


class _InferSlot:
    """Resources for one side of the double-buffer: execution context,
    CUDA stream, and reusable output buffers."""

    __slots__ = ("ctx", "stream", "cached_shapes", "bufs", "_d_input_ref", "_d_input_pinned_ref")

    def __init__(self, engine: Any, device: torch.device) -> None:
        self.ctx = engine.create_execution_context()
        self.stream = torch.cuda.Stream(device)
        self.cached_shapes: Optional[List[Tuple[int, ...]]] = None
        self.bufs: List[torch.Tensor] = []
        self._d_input_ref: Optional[torch.Tensor] = None
        self._d_input_pinned_ref: Optional[torch.Tensor] = None


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
    * **Double-buffered pipelining** – two execution contexts alternate so
      GPU inference of batch N+1 overlaps with CPU post-processing of
      batch N.  Use ``invoke_async`` / ``flush`` to opt in.

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
        self._trt = trt

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

        self._has_dynamic_batch = (
            tuple(engine.get_tensor_shape(self._input_name))[0] == -1
        )

        # Double-buffer: two slots with independent contexts / streams / buffers.
        self._slots = [_InferSlot(engine, self._device) for _ in range(2)]
        self._active_slot = 0
        # (slot_index, orig_shapes, batch_size) for the in-flight batch, or None.
        self._inflight: Optional[Tuple[int, List[Tuple[int, int]], int]] = None

        from nemotron_page_elements_v3.yolox.boxes import postprocess as _yolox_nms
        self._yolox_nms = _yolox_nms

        # DALI-accelerated letterbox preprocessing (optional).
        self._dali_pipe: Optional[_DaliLetterboxPipeline] = None
        if _DALI_AVAILABLE:
            try:
                self._dali_pipe = _DaliLetterboxPipeline(
                    target_h=input_shape[0],
                    target_w=input_shape[1],
                    device_id=self._device.index or 0,
                    max_batch_size=64,
                )
                logger.info("DALI letterbox pipeline initialized for TRTYoloxEngine.")
            except Exception:
                logger.warning("DALI pipeline init failed; falling back to torch letterbox.", exc_info=True)
                self._dali_pipe = None

        logger.info(
            "TRTYoloxEngine loaded: path=%s, input=%s, outputs=%s, "
            "spatial=%s, dynamic_batch=%s, double_buffer=True, dali=%s, device=%s",
            engine_path,
            self._input_name,
            self._output_names,
            input_shape,
            self._has_dynamic_batch,
            self._dali_pipe is not None,
            self._device,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def preprocess(self, tensor: Any, orig_shape: Any = None) -> torch.Tensor:
        """Resize / pad to engine input shape using the same aspect-preserving
        letterbox as the HuggingFace ``resize_pad``.

        Accepts CHW or BCHW torch tensors, or HWC / CHW numpy arrays.
        Returns BCHW ``float32`` on CPU ready for ``invoke``.
        """
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
            y = _gpu_letterbox(x, self._input_shape)
            y = torch.clamp(y, 0, 255).to(torch.uint8).float()
            return y.contiguous()

        if x.ndim == 3:
            y = _gpu_letterbox(x.unsqueeze(0), self._input_shape)
            y = torch.clamp(y, 0, 255).to(torch.uint8).float()
            return y.contiguous()

        raise ValueError(f"Expected CHW or BCHW tensor, got shape {tuple(x.shape)}")

    def preprocess_batch_gpu(
        self,
        images: List[Any],
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """Letterbox a list of variable-sized raw images on GPU.

        Each element can be a HWC numpy array or a CHW torch tensor.
        Returns ``(batch_BCHW_on_GPU, orig_shapes)``.  The output batch
        is pre-allocated on GPU and filled with the pad value, so images
        of different sizes are handled without CPU resize.

        Uses DALI when available for fully GPU-accelerated preprocessing;
        falls back to per-image torch ops otherwise.
        """
        # ---- DALI fast path: whole batch in one GPU kernel launch ----
        if self._dali_pipe is not None:
            hwc_arrays: list[np.ndarray] = []
            for img in images:
                if isinstance(img, torch.Tensor):
                    arr = img.cpu().numpy()
                    if arr.ndim == 4:
                        arr = arr[0]
                    if arr.shape[0] == 3 and arr.ndim == 3:
                        arr = np.ascontiguousarray(arr.transpose(1, 2, 0))
                else:
                    arr = np.ascontiguousarray(img)
                    if arr.ndim == 4:
                        arr = arr[0]
                hwc_arrays.append(arr)
            try:
                return self._dali_pipe.run(hwc_arrays)
            except Exception:
                logger.warning("DALI preprocess failed; falling back to torch.", exc_info=True)

        # ---- Torch fallback: per-image letterbox on GPU ----
        tgt_h, tgt_w = self._input_shape
        B = len(images)

        d_batch = torch.full(
            (B, 3, tgt_h, tgt_w), 114.0,
            dtype=torch.float32, device=self._device,
        )

        orig_shapes: List[Tuple[int, int]] = []
        for i, img in enumerate(images):
            if isinstance(img, np.ndarray):
                arr = img
                if arr.ndim == 4 and int(arr.shape[0]) == 1:
                    arr = arr[0]
                if int(arr.shape[-1]) == 3 and int(arr.shape[0]) != 3:
                    t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
                else:
                    t = torch.from_numpy(np.ascontiguousarray(arr))
            elif isinstance(img, torch.Tensor):
                t = img
                if t.ndim == 4 and int(t.shape[0]) == 1:
                    t = t[0]
            else:
                raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(img)!r}")

            C, H, W = int(t.shape[0]), int(t.shape[1]), int(t.shape[2])
            orig_shapes.append((H, W))

            d_img = t.float().contiguous().pin_memory().to(
                device=self._device, non_blocking=True
            )
            ratio = min(tgt_h / H, tgt_w / W)
            new_h, new_w = int(round(H * ratio)), int(round(W * ratio))

            d_resized = F.interpolate(
                d_img.unsqueeze(0), size=(new_h, new_w),
                mode="bilinear", align_corners=False,
            )[0]
            d_batch[i, :C, :new_h, :new_w] = d_resized

        d_batch = torch.clamp(d_batch, 0, 255).to(torch.uint8).float()
        return d_batch.contiguous(), orig_shapes

    def invoke(
        self,
        input_tensor: torch.Tensor,
        orig_shape: Union[Tuple[int, int], List[Tuple[int, int]]],
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Synchronous inference — submit then immediately flush."""
        self.invoke_async(input_tensor, orig_shape)
        return self.flush()  # type: ignore[return-value]

    def invoke_async(
        self,
        input_tensor: torch.Tensor,
        orig_shape: Union[Tuple[int, int], List[Tuple[int, int]]],
    ) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """Launch TRT inference **asynchronously** and return the
        **previous** batch's post-processed results (or ``None`` on the
        first call).

        Use with :meth:`flush` to drain the final in-flight batch::

            for batch, shapes in chunks:
                prev = model.invoke_async(batch, shapes)
                if prev is not None:
                    handle(prev)
            last = model.flush()
            if last is not None:
                handle(last)
        """
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)

        batch_size = int(input_tensor.shape[0])

        if isinstance(orig_shape, (list, tuple)) and orig_shape and isinstance(orig_shape[0], (list, tuple)):
            shapes = list(orig_shape)
        else:
            shapes = [orig_shape] * batch_size

        # ---- collect previous in-flight results (if any) ----
        prev_results: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
        if self._inflight is not None:
            prev_slot_idx, prev_shapes, prev_bs = self._inflight
            prev_slot = self._slots[prev_slot_idx]
            prev_slot.stream.synchronize()
            preds = self._postprocess_batch(prev_slot.bufs, prev_shapes)
            prev_results = preds[0] if prev_bs == 1 else preds

        # ---- launch new batch on the current slot (no sync) ----
        slot = self._slots[self._active_slot]
        self._launch_on_slot(slot, input_tensor)
        self._inflight = (self._active_slot, shapes, batch_size)
        self._active_slot = 1 - self._active_slot

        return prev_results

    def flush(self) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """Synchronise and return the last in-flight batch's results.

        Returns ``None`` if nothing is in flight.
        """
        if self._inflight is None:
            return None
        slot_idx, shapes, bs = self._inflight
        slot = self._slots[slot_idx]
        slot.stream.synchronize()
        preds = self._postprocess_batch(slot.bufs, shapes)
        self._inflight = None
        return preds[0] if bs == 1 else preds

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
    # Double-buffered GPU inference
    # ------------------------------------------------------------------

    def _launch_on_slot(self, slot: _InferSlot, input_nchw: torch.Tensor) -> None:
        """Transfer *input_nchw* to GPU on *slot*'s stream, configure the
        execution context, and launch ``execute_async_v3`` **without**
        synchronising.  The caller is responsible for later calling
        ``slot.stream.synchronize()`` before reading from ``slot.bufs``.
        """
        torch_dtype = _NP_TO_TORCH_DTYPE.get(np.dtype(self._input_dtype), torch.float32)

        pinned_ref: Optional[torch.Tensor] = None
        if input_nchw.is_cuda:
            d_input = input_nchw.to(dtype=torch_dtype).contiguous()
        else:
            pinned_ref = input_nchw.to(dtype=torch_dtype).contiguous().pin_memory()
            with torch.cuda.stream(slot.stream):
                d_input = pinned_ref.to(device=self._device, non_blocking=True)

        ctx = slot.ctx
        if self._has_dynamic_batch:
            ctx.set_input_shape(self._input_name, tuple(d_input.shape))

        ctx.set_tensor_address(self._input_name, d_input.data_ptr())

        needed_shapes: List[Tuple[int, ...]] = [
            tuple(ctx.get_tensor_shape(name))
            for name in self._output_names
        ]

        if needed_shapes != slot.cached_shapes:
            slot.bufs = []
            for shape, np_dtype in zip(needed_shapes, self._output_dtypes):
                t_dtype = _NP_TO_TORCH_DTYPE.get(np.dtype(np_dtype), torch.float32)
                slot.bufs.append(
                    torch.empty(shape, dtype=t_dtype, device=self._device)
                )
            slot.cached_shapes = needed_shapes

        for name, d_out in zip(self._output_names, slot.bufs):
            ctx.set_tensor_address(name, d_out.data_ptr())

        # Keep d_input and pinned source alive until execution completes.
        slot._d_input_ref = d_input  # type: ignore[attr-defined]
        slot._d_input_pinned_ref = pinned_ref  # type: ignore[attr-defined]

        with torch.cuda.stream(slot.stream):
            ctx.execute_async_v3(stream_handle=slot.stream.cuda_stream)

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
            raw = gpu_outputs[0]
            if raw.ndim == 2:
                raw = raw.unsqueeze(0)

            num_classes = raw.shape[-1] - 5

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
            all_boxes = gpu_outputs[0]
            all_labels = gpu_outputs[1]
            all_scores = gpu_outputs[2]

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


class _EmbedSlot:
    """Resources for one side of the embed double-buffer."""

    __slots__ = ("ctx", "stream", "d_output", "d_inputs")

    def __init__(self, engine: Any, device: torch.device) -> None:
        self.ctx = engine.create_execution_context()
        self.stream = torch.cuda.Stream(device)
        self.d_output: Optional[torch.Tensor] = None
        self.d_inputs: Dict[str, torch.Tensor] = {}


class TRTEmbedEngine:
    """TensorRT engine wrapper that mimics the ``embed()`` API of
    ``LlamaNemotronEmbed1BV2Embedder``.

    The engine is expected to accept ``input_ids`` and ``attention_mask``
    tensors (int32, shape ``[B, S]``) and produce a last-hidden-state
    output of shape ``[B, S, D]``.  Mean-pooling and optional L2
    normalisation are applied on top.

    Performance characteristics
    --------------------------
    * **Double-buffered pipelining** – two execution contexts alternate
      across sub-batches so GPU inference of sub-batch N+1 overlaps with
      CPU-side pooling, normalisation, and ``cpu()`` transfer of sub-batch N.

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
        self._trt = trt

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

        # Resolve the output tensor name once.
        self._output_name: Optional[str] = None
        for name, info in self._io_info.items():
            if not info["is_input"]:
                self._output_name = name
                break
        if self._output_name is None:
            raise RuntimeError("TRT embed engine has no output tensor")

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

        abs_max_seq = max(p[2] for p in self._profiles) if self._profiles else max_length
        self._max_length = min(max_length, abs_max_seq)

        # Catalogue extra input tensors (beyond input_ids / attention_mask).
        _PRIMARY_INPUTS = {"input_ids", "attention_mask"}
        self._extra_input_names: List[str] = []
        for name, info in self._io_info.items():
            if info["is_input"] and name not in _PRIMARY_INPUTS:
                self._extra_input_names.append(name)
                logger.info(
                    "TRTEmbedEngine: extra input tensor '%s' shape=%s dtype=%s",
                    name, info["shape"], info["dtype"],
                )

        # llama-nemotron-embed-1b-v2 supports Matryoshka dims
        # {384, 512, 768, 1024, 2048}; always request the full 2048.
        self._embed_dim: int = 2048

        # Write tensor info to a temp file AND stdout for diagnostics.
        _diag_lines = []
        for name, info in self._io_info.items():
            profile_shapes = []
            if info["is_input"]:
                for p_idx in range(num_profiles):
                    try:
                        mn, opt, mx = engine.get_tensor_profile_shape(name, p_idx)
                        profile_shapes.append(
                            f"p{p_idx}:[{tuple(mn)}..{tuple(mx)}]"
                        )
                    except Exception:
                        pass
            raw_trt_dtype = engine.get_tensor_dtype(name)
            line = (
                f"[TRTEmbedEngine] tensor: name={name}, is_input={info['is_input']}, "
                f"trt_dtype={raw_trt_dtype}, np_dtype={info['dtype']}, "
                f"shape={info['shape']}, profiles=[{', '.join(profile_shapes)}]"
            )
            print(line, flush=True)
            _diag_lines.append(line)
        try:
            import os
            _diag_path = "/tmp/nemo_retriever_debug/trt_embed_tensors.txt"
            os.makedirs(os.path.dirname(_diag_path), exist_ok=True)
            with open(_diag_path, "w") as _f:
                _f.write("\n".join(_diag_lines) + "\n")
        except Exception:
            pass

        # Double-buffer: two slots with independent contexts / streams.
        self._slots = [_EmbedSlot(engine, self._device) for _ in range(2)]
        self._active_slot = 0

        logger.info(
            "TRTEmbedEngine loaded: path=%s, tensors=%s, max_length=%d, "
            "normalize=%s, profiles=%s, embed_dim=%d, extra_inputs=%s, "
            "double_buffer=True, device=%s",
            engine_path,
            list(self._io_info.keys()),
            self._max_length,
            normalize,
            self._profiles,
            self._embed_dim,
            self._extra_input_names,
            self._device,
        )

    def _bind_extra_inputs(self, ctx: Any, slot: Any, B: int) -> None:
        """Set shapes and addresses for all extra input tensors on *ctx*.

        The shape is derived from the engine metadata ndim, with the first
        (batch) dimension set to *B* so it matches the primary inputs.
        """
        for name in self._extra_input_names:
            info = self._io_info[name]
            np_dtype = np.dtype(info["dtype"])
            t_dtype = _NP_TO_TORCH_DTYPE.get(np_dtype, torch.int64)
            ndim = len(info["shape"])

            fill = self._embed_dim if name == "dimensions" else 0

            if ndim == 0:
                shape: Tuple[int, ...] = ()
                d_extra = torch.full((), fill, dtype=t_dtype, device=self._device)
            elif ndim == 1:
                shape = (B,)
                d_extra = torch.full((B,), fill, dtype=t_dtype, device=self._device)
            else:
                shape = (B, 1)
                d_extra = torch.full((B, 1), fill, dtype=t_dtype, device=self._device)

            ctx.set_input_shape(name, shape if len(shape) > 0 else (1,))
            ctx.set_tensor_address(name, d_extra.data_ptr())
            slot.d_inputs[f"_extra_{name}"] = d_extra

            if not hasattr(self, "_extra_logged"):
                logger.info(
                    "Binding extra input '%s': fill=%d, dtype=%s (torch=%s), "
                    "shape=%s, ndim=%d",
                    name, fill, np_dtype, t_dtype, shape, ndim,
                )
        self._extra_logged = True

    @property
    def is_remote(self) -> bool:
        return False

    @property
    def model_name(self) -> str:
        return f"TRT:{self._engine_path}"

    def embed(self, texts: "Sequence[str]", *, batch_size: int = 64) -> torch.Tensor:
        """Return a CPU tensor of shape ``[N, D]``, matching the HF embedder API.

        Tokenised inputs are split into sub-batches that respect the
        engine's optimisation profile limits.  Sub-batches are
        **double-buffered**: GPU inference of sub-batch N+1 overlaps
        with CPU-side pooling / normalisation / transfer of sub-batch N.
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

        # Double-buffer state for the previous in-flight sub-batch.
        pending_slot_idx: Optional[int] = None
        pending_mask: Optional[torch.Tensor] = None
        pending_indices: Optional[List[int]] = None

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
            input_ids = tokens["input_ids"].astype(np.int64)
            attention_mask = tokens["attention_mask"].astype(np.int64)

            # Launch this sub-batch's TRT inference async on the current slot.
            slot = self._slots[self._active_slot]
            self._launch_embed_on_slot(slot, input_ids, attention_mask)
            launched_slot_idx = self._active_slot
            self._active_slot = 1 - self._active_slot

            # While GPU executes, drain the previous sub-batch (if any).
            if pending_slot_idx is not None:
                self._drain_embed_slot(
                    pending_slot_idx, pending_mask, pending_indices, result_vecs,  # type: ignore[arg-type]
                )

            # Record this sub-batch as pending.
            pending_slot_idx = launched_slot_idx
            pending_mask = (
                torch.from_numpy(attention_mask.astype(np.float32, copy=False))
                .to(device=self._device)
                .unsqueeze(-1)
            )
            pending_indices = indices

        # Flush the last in-flight sub-batch.
        if pending_slot_idx is not None:
            self._drain_embed_slot(
                pending_slot_idx, pending_mask, pending_indices, result_vecs,  # type: ignore[arg-type]
            )

        filled = [v for v in result_vecs if v is not None]
        return torch.stack(filled, dim=0) if filled else torch.empty((0, 0), dtype=torch.float32)

    # ------------------------------------------------------------------
    # Double-buffer helpers
    # ------------------------------------------------------------------

    def _launch_embed_on_slot(
        self,
        slot: _EmbedSlot,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
    ) -> None:
        """Configure the slot's context and launch ``execute_async_v3``
        **without** synchronising.
        """
        B, S = input_ids.shape
        profile_idx = self._best_profile(B, S)
        if profile_idx is None:
            raise RuntimeError(
                f"No TRT optimization profile supports shape ({B}, {S}). "
                f"Available profiles: {self._profiles}"
            )

        ctx = slot.ctx

        # Drain any in-flight work before touching shapes or profiles.
        slot.stream.synchronize()
        torch.cuda.current_stream(self._device).synchronize()

        if ctx.active_optimization_profile != profile_idx:
            ctx.set_optimization_profile_async(profile_idx, slot.stream.cuda_stream)
            slot.stream.synchronize()

        inputs_np = {
            "input_ids": np.ascontiguousarray(input_ids),
            "attention_mask": np.ascontiguousarray(attention_mask),
        }

        # Allocate input GPU tensors, set shapes and addresses — TRT 10.x
        # needs valid tensor addresses before output shape can resolve.
        # Use the ENGINE's expected dtype (from _io_info), not the numpy
        # array's dtype, to guarantee the buffer element width matches
        # what TRT will read (e.g. INT64 requires 8 bytes per element).
        slot.d_inputs = {}
        for name, arr in inputs_np.items():
            if name in self._io_info:
                ctx.set_input_shape(name, arr.shape)
                engine_np_dtype = np.dtype(self._io_info[name]["dtype"])
                t_dtype = _NP_TO_TORCH_DTYPE.get(engine_np_dtype, torch.int64)
                cpu_buf = torch.tensor(
                    arr.astype(engine_np_dtype), dtype=t_dtype,
                ).contiguous().pin_memory()
                with torch.cuda.stream(slot.stream):
                    d_buf = cpu_buf.to(device=self._device, non_blocking=True)
                ctx.set_tensor_address(name, d_buf.data_ptr())
                slot.d_inputs[name] = d_buf
                slot.d_inputs[f"_pin_{name}"] = cpu_buf

        # Bind any extra input tensors (e.g. 'dimensions' for Matryoshka).
        self._bind_extra_inputs(ctx, slot, B)

        # Explicitly trigger shape inference now that all inputs are bound.
        try:
            ctx.infer_shapes()
        except Exception:
            pass

        try:
            out_shape = tuple(ctx.get_tensor_shape(self._output_name))
            shape_ok = all(d > 0 for d in out_shape)
        except Exception:
            out_shape = ()
            shape_ok = False

        if not shape_ok:
            D = self._embed_dim
            static_ndim = len(self._io_info[self._output_name]["shape"])
            out_shape = (B, S, D) if static_ndim >= 3 else (B, D)
            logger.debug("Used embed_dim=%d → output shape %s", D, out_shape)
        out_np_dtype = np.dtype(self._io_info[self._output_name]["dtype"])
        out_t_dtype = _NP_TO_TORCH_DTYPE.get(out_np_dtype, torch.float32)
        slot.d_output = torch.empty(out_shape, dtype=out_t_dtype, device=self._device)
        ctx.set_tensor_address(self._output_name, slot.d_output.data_ptr())

        with torch.cuda.stream(slot.stream):
            ok = ctx.execute_async_v3(stream_handle=slot.stream.cuda_stream)
        if not ok:
            logger.error(
                "execute_async_v3 FAILED for embed input (%d, %d) on profile %d. "
                "Extra inputs: %s. Output shape: %s",
                B, S, profile_idx,
                {n: slot.d_inputs.get(f"_extra_{n}") for n in self._extra_input_names},
                out_shape,
            )

    def _drain_embed_slot(
        self,
        slot_idx: int,
        mask_gpu: torch.Tensor,
        indices: List[int],
        result_vecs: List[Optional[torch.Tensor]],
    ) -> None:
        """Synchronise a slot, pool + normalise its output, and scatter
        the resulting CPU vectors into *result_vecs*.
        """
        slot = self._slots[slot_idx]
        slot.stream.synchronize()

        hidden = slot.d_output.float()  # type: ignore[union-attr]
        if hidden.ndim == 3:
            pooled = (hidden * mask_gpu).sum(dim=1) / mask_gpu.sum(dim=1)
        else:
            pooled = hidden
        if self._normalize:
            pooled = _l2_normalize(pooled)

        pooled_cpu = pooled.cpu()
        for local_i, orig_i in enumerate(indices):
            result_vecs[orig_i] = pooled_cpu[local_i]

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
            end = pos + 1
            while end < len(order):
                candidate_seq = lengths[order[end]]
                candidate_batch = end - pos + 1
                if candidate_batch > max_batch:
                    break
                if self._best_profile(candidate_batch, candidate_seq) is None:
                    break
                end += 1

            sub = order[pos:end]
            max_seq = max(lengths[i] for i in sub)
            if self._best_profile(len(sub), max_seq) is None:
                sub_batches.append([sub[0]])
                pos += 1
            else:
                sub_batches.append(sub)
                pos = end

        return sub_batches
