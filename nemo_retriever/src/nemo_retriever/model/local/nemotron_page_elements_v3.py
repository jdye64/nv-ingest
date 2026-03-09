# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Sequence, Tuple, Union

from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from nemo_retriever.utils.hf_cache import configure_global_hf_cache_base
from ..model import HuggingFaceModel, RunMode

from nemotron_page_elements_v3.model import define_model as define_model_page_elements
from nemotron_page_elements_v3.utils import postprocess_preds_page_element

_TARGET_H = 1024
_TARGET_W = 1024
_PAD_VALUE = 114.0
_MAX_BATCH = 32
# Typical document pages at 200 DPI are ~1700x2200; 2560 covers up to ~300 DPI.
_MAX_SRC_H = 2560
_MAX_SRC_W = 2560


class NemotronPageElementsV3(HuggingFaceModel):
    """Nemotron Page Elements v3 -- GPU-optimized YOLOX object detector.

    Pre-allocates pinned CPU and GPU buffers so the hot path does zero
    CUDA allocations.  All preprocessing (resize, pad) runs on the GPU
    in float16.
    """

    def __init__(self, max_batch_size: int = _MAX_BATCH) -> None:
        super().__init__(self.model_name)
        configure_global_hf_cache_base()

        self._model = define_model_page_elements(self.model_name)
        self._page_elements_input_shape = (_TARGET_H, _TARGET_W)
        self._max_batch = int(max_batch_size)

        device = self._model.device

        # Cast model weights to fp16 so the forward pass stays in half precision
        # without relying on autocast for the bulk of the compute.
        self._model.half()

        # Pre-allocated GPU output buffer (reused every call).
        self._gpu_batch = torch.full(
            (self._max_batch, 3, _TARGET_H, _TARGET_W),
            _PAD_VALUE,
            dtype=torch.float16,
            device=device,
        )

        # Pinned CPU staging buffer for fast async H2D copies.
        self._pinned_stage = torch.empty(
            (3, _MAX_SRC_H, _MAX_SRC_W),
            dtype=torch.float16,
            pin_memory=True,
        )

        # Dedicated stream so H2D copies can overlap with prior-batch compute.
        self._copy_stream = torch.cuda.Stream(device=device)

    # ------------------------------------------------------------------
    # Batch preprocessing -- the only path used by the Ray actor
    # ------------------------------------------------------------------

    def preprocess_batch(self, images: List[np.ndarray]) -> torch.Tensor:
        """Preprocess HWC numpy images into a BCHW float16 GPU tensor.

        Uses pre-allocated pinned + GPU buffers so the hot path triggers
        no CUDA mallocs.  Resize and pad happen entirely on the GPU.

        Returns a *view* into the internal ``_gpu_batch`` buffer (valid
        until the next ``preprocess_batch`` call).
        """
        if self._model is None:
            raise RuntimeError("Local page_elements_v3 model was not initialized.")

        n = len(images)
        if n > self._max_batch:
            self._grow_gpu_buffer(n)

        device = self._model.device

        # Reset the pad region for this batch slice.
        self._gpu_batch[:n].fill_(_PAD_VALUE)

        for i, arr in enumerate(images):
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim != 3:
                raise ValueError(f"Expected 3D image array, got shape {arr.shape}")

            # HWC -> CHW
            if arr.shape[-1] == 3 and arr.shape[0] != 3:
                arr = np.ascontiguousarray(arr.transpose(2, 0, 1))
            else:
                arr = np.ascontiguousarray(arr)

            c, h, w = arr.shape

            # Use the pinned staging buffer when the image fits; fall back
            # to a fresh pinned tensor for unusually large images.
            if h <= _MAX_SRC_H and w <= _MAX_SRC_W:
                stage = self._pinned_stage[:c, :h, :w]
                stage.copy_(torch.from_numpy(arr))
            else:
                stage = torch.from_numpy(arr).to(dtype=torch.float16).pin_memory()

            with torch.cuda.stream(self._copy_stream):
                gpu_chw = stage.to(device=device, dtype=torch.float16, non_blocking=True)

                scale = min(_TARGET_H / h, _TARGET_W / w)
                nh = int(h * scale)
                nw = int(w * scale)

                resized = (
                    F.interpolate(
                        gpu_chw.unsqueeze(0),
                        size=(nh, nw),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .clamp_(0, 255)
                )
                self._gpu_batch[i, :, :nh, :nw] = resized

        # Ensure copies are done before the batch is consumed by the model.
        self._copy_stream.synchronize()

        return self._gpu_batch[:n]

    # ------------------------------------------------------------------
    # Single-image preprocessing (kept for backward compatibility)
    # ------------------------------------------------------------------

    def preprocess(self, tensor: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Preprocess a single image.  Prefer ``preprocess_batch`` for throughput."""
        if isinstance(tensor, np.ndarray):
            return self.preprocess_batch([tensor])
        if isinstance(tensor, torch.Tensor):
            if tensor.ndim == 3:
                return self.preprocess_batch([tensor.cpu().numpy()])
            if tensor.ndim == 4:
                return self.preprocess_batch([tensor[i].cpu().numpy() for i in range(tensor.shape[0])])
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(tensor)!r}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def __call__(
        self,
        input_data: torch.Tensor,
        orig_shape: Union[Tuple[int, int], Sequence[Tuple[int, int]]],
    ) -> List[Dict[str, torch.Tensor]]:
        return self.invoke(input_data, orig_shape)

    def invoke(
        self,
        input_data: torch.Tensor,
        orig_shape: Union[Tuple[int, int], Sequence[Tuple[int, int]]],
    ) -> List[Dict[str, torch.Tensor]]:
        if self._model is None:
            raise RuntimeError("Local page_elements_v3 model was not initialized.")

        with torch.inference_mode():
            return self._model(input_data, orig_shape)

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------

    def postprocess(
        self,
        preds: Union[Dict[str, torch.Tensor], Sequence[Dict[str, torch.Tensor]]],
    ) -> Tuple[
        Union[torch.Tensor, List[torch.Tensor]],
        Union[torch.Tensor, List[torch.Tensor]],
        Union[torch.Tensor, List[torch.Tensor]],
    ]:
        if self._model is None:
            raise RuntimeError("Local page_elements_v3 model was not initialized.")

        thresholds = self._model.thresholds_per_class
        labels = self._model.labels

        def _one(p: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            b_np, l_np, s_np = postprocess_preds_page_element(p, thresholds, labels)
            return (
                torch.as_tensor(b_np, dtype=torch.float32),
                torch.as_tensor(l_np, dtype=torch.int64),
                torch.as_tensor(s_np, dtype=torch.float32),
            )

        if isinstance(preds, dict):
            return _one(preds)

        boxes_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        scores_list: List[torch.Tensor] = []
        for p in preds:
            b, l, s = _one(p)  # noqa: E741
            boxes_list.append(b)
            labels_list.append(l)
            scores_list.append(s)

        return boxes_list, labels_list, scores_list

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _grow_gpu_buffer(self, needed: int) -> None:
        """Expand the pre-allocated GPU buffer when a batch exceeds capacity."""
        self._max_batch = needed
        device = self._model.device
        self._gpu_batch = torch.full(
            (needed, 3, _TARGET_H, _TARGET_W),
            _PAD_VALUE,
            dtype=torch.float16,
            device=device,
        )

    # ------------------------------------------------------------------
    # Metadata properties (unchanged public contract)
    # ------------------------------------------------------------------

    @property
    def model_dir(self) -> str:
        return self.model_name

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            raise RuntimeError("Local page_elements_v3 model was not initialized.")
        return self._model

    @property
    def model_name(self) -> str:
        return "page_element_v3"

    @property
    def model_type(self) -> str:
        return "object-detection"

    @property
    def model_runmode(self) -> RunMode:
        return "local"

    @property
    def input(self) -> Any:
        return {
            "type": "image",
            "format": "RGB",
            "dimensions": "(1024, 1024)",
            "description": "Document page image in RGB format, resized to 1024x1024",
        }

    @property
    def output(self) -> Any:
        return {
            "type": "detection",
            "format": "dict",
            "structure": {
                "boxes": "np.ndarray[N, 4] - normalized (x_min, y_min, x_max, y_max)",
                "labels": "List[str] - class names",
                "scores": "np.ndarray[N] - confidence scores",
            },
            "classes": ["table", "chart", "infographic", "title", "text", "header_footer"],
            "post_processing": {"conf_thresh": 0.01, "iou_thresh": 0.5},
        }

    @property
    def input_batch_size(self) -> int:
        return 1

    @property
    def input_shape(self) -> Tuple[int, int]:
        return self._page_elements_input_shape
