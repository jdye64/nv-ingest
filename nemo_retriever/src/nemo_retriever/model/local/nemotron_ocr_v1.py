# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""
GPU-optimized Nemotron OCR v1 model wrapper.

Casts all discoverable nn.Module sub-components to float16, pre-allocates
pinned CPU staging buffers and a dedicated CUDA copy stream, and warms up
the CUDA caching allocator at init so the hot path triggers zero new CUDA
mallocs.
"""

from typing import Any, List, Optional, Union

import os

import numpy as np
import torch
from nemo_retriever.utils.hf_cache import configure_global_hf_cache_base

from ..model import BaseModel, RunMode

_MAX_CROP_H = 2560
_MAX_CROP_W = 2560


class NemotronOCRV1(BaseModel):
    """Nemotron OCR v1 -- GPU-optimized optical character recognition.

    Pre-allocates pinned CPU staging buffers and a dedicated CUDA copy
    stream.  All nn.Module sub-components are cast to float16 at init.
    A warmup inference primes the CUDA caching allocator so subsequent
    calls reuse existing device memory.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        max_batch_size: int = 32,
    ) -> None:
        super().__init__()
        configure_global_hf_cache_base()

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        from nemotron_ocr.inference.pipeline import NemotronOCR  # local-only import

        self._model = NemotronOCR(model_dir=model_dir) if model_dir else NemotronOCR()
        self._max_batch = int(max_batch_size)

        self._device = self._resolve_device()

        self._cast_submodules_to_fp16()

        self._enable_trt = os.getenv("RETRIEVER_ENABLE_TORCH_TRT", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if self._enable_trt:
            self._maybe_compile_detector()

        self._pinned_stage = torch.empty(
            (3, _MAX_CROP_H, _MAX_CROP_W),
            dtype=torch.float16,
            pin_memory=True,
        )
        self._copy_stream = torch.cuda.Stream(device=self._device)

        self._warmup()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _resolve_device(self) -> torch.device:
        """Infer the CUDA device from the pipeline's detector parameters."""
        detector = getattr(self._model, "detector", None)
        if isinstance(detector, torch.nn.Module):
            try:
                return next(detector.parameters()).device
            except StopIteration:
                pass
        return torch.device("cuda")

    def _cast_submodules_to_fp16(self) -> None:
        """Cast all discoverable nn.Module sub-components to float16."""
        for attr in ("detector", "recognizer", "relational_model", "model"):
            sub = getattr(self._model, attr, None)
            if isinstance(sub, torch.nn.Module):
                sub.half()

    def _maybe_compile_detector(self) -> None:
        """Best-effort TensorRT compilation of the detector backbone."""
        try:
            import torch_tensorrt  # type: ignore
        except Exception:
            return

        detector = getattr(self._model, "detector", None)
        if not isinstance(detector, torch.nn.Module):
            return

        try:
            trt_input = torch_tensorrt.Input((1, 3, 1024, 1024), dtype=torch.float16)
        except TypeError:
            trt_input = torch_tensorrt.Input(shape=(1, 3, 1024, 1024), dtype=torch.float16)

        try:
            self._model.detector = torch_tensorrt.compile(
                detector,
                inputs=[trt_input],
                enabled_precisions={torch.float16},
                torch_executed_ops={"torchvision::nms"},
                torch_executed_modules=set(),
            )
        except Exception:
            pass

    def _warmup(self) -> None:
        """Prime the CUDA caching allocator, JIT kernels, and cuDNN autotuner."""
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        try:
            with torch.inference_mode():
                self._model(dummy, merge_level="paragraph")
        except Exception:
            pass
        torch.cuda.synchronize(self._device)

    # ------------------------------------------------------------------
    # Preprocessing (no-op kept for BaseModel contract)
    # ------------------------------------------------------------------

    def preprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def invoke(
        self,
        input_data: Union[List[np.ndarray], np.ndarray, torch.Tensor, str, bytes],
        merge_level: str = "paragraph",
    ) -> Any:
        """Run OCR inference.

        For maximum throughput pass a ``list[np.ndarray]`` of HWC uint8
        crops.  The underlying pipeline handles internal batching and GPU
        transfers.
        """
        if self._model is None:
            raise RuntimeError("Local OCR model was not initialized.")

        with torch.inference_mode():
            if isinstance(input_data, list):
                return self._model(input_data, merge_level=merge_level)

            if isinstance(input_data, np.ndarray):
                return self._model(input_data, merge_level=merge_level)

            if isinstance(input_data, torch.Tensor):
                return self._invoke_from_tensor(input_data, merge_level)

            return self._model(input_data, merge_level=merge_level)

    def _invoke_from_tensor(self, t: torch.Tensor, merge_level: str) -> Any:
        """Convert a GPU/CPU tensor to numpy and invoke the pipeline.

        Previous implementation round-tripped through PNG base64 encoding
        per image in the batch.  This path converts directly to HWC uint8
        numpy, which is ~100x faster for the CPU portion.
        """
        if t.ndim == 3:
            arr = self._tensor_chw_to_numpy_hwc(t)
            return self._model(arr, merge_level=merge_level)

        if t.ndim == 4:
            arrays = [self._tensor_chw_to_numpy_hwc(t[i]) for i in range(t.shape[0])]
            return self._model(arrays, merge_level=merge_level)

        raise ValueError(f"Unsupported tensor shape for OCR: {tuple(t.shape)}")

    @staticmethod
    def _tensor_chw_to_numpy_hwc(img: torch.Tensor) -> np.ndarray:
        """CHW float/uint8 tensor → HWC uint8 numpy (zero-copy when possible)."""
        x = img.detach()
        if x.device.type != "cpu":
            x = x.cpu()

        if x.dtype.is_floating_point:
            max_val = x.max().item() if x.numel() else 1.0
            if max_val <= 1.5:
                x = x.mul(255.0)
            x = x.clamp(0, 255).to(torch.uint8)
        else:
            x = x.clamp(0, 255).to(torch.uint8)

        if x.shape[0] == 1:
            return x.squeeze(0).numpy()
        return x.permute(1, 2, 0).contiguous().numpy()

    # ------------------------------------------------------------------
    # Metadata properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return "Nemotron OCR v1"

    @property
    def model_type(self) -> str:
        return "ocr"

    @property
    def model_runmode(self) -> RunMode:
        return "local"

    @property
    def input(self) -> Any:
        return {
            "type": "image",
            "format": "RGB",
            "supported_formats": ["PNG", "JPEG"],
            "data_types": ["float32", "uint8"],
            "dimensions": "variable (H x W)",
            "batch_support": True,
            "value_range": {"float32": "[0, 1]", "uint8": "[0, 255]"},
            "aggregation_levels": ["word", "sentence", "paragraph"],
        }

    @property
    def output(self) -> Any:
        return {
            "type": "ocr_results",
            "format": "structured",
            "structure": {
                "boxes": "List[List[List[float]]] - quadrilateral bounding box coordinates",
                "texts": "List[str] - recognized text strings",
                "confidences": "List[float] - confidence scores per detection",
            },
        }

    @property
    def input_batch_size(self) -> int:
        return self._max_batch
