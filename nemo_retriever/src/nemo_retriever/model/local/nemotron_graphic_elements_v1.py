# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from nemo_retriever.utils.hf_cache import configure_global_hf_cache_base
from ..model import BaseModel, RunMode

from nemotron_graphic_elements_v1.model import define_model as define_model_graphic_elements
from nemotron_graphic_elements_v1.model import resize_pad as resize_pad_graphic_elements


class NemotronGraphicElementsV1(BaseModel):
    """
    Nemotron Graphic Elements v1 model for detecting chart and graph elements.

    Detects and extracts key elements from charts and graphs including:
    - Chart titles
    - X/Y axis titles and labels
    - Legend titles and labels
    - Marker labels
    - Value labels
    - Other text components

    Note: Input should be a cropped chart image, typically obtained using
    Nemotron Page Elements v3 model.
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()
        configure_global_hf_cache_base()
        self._model = define_model_graphic_elements(self.model_name)
        self._graphic_elements_input_shape = (1024, 1024)

    def preprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess the input tensor."""
        # Upstream `nemotron_graphic_elements_v1.model.resize_pad` expects CHW.
        # Our pipeline helpers often pass BCHW (typically B=1), so normalize here.
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)!r}")

        x = tensor
        if x.ndim == 4:
            b = int(x.shape[0])
            if b == 1:
                y = resize_pad_graphic_elements(x[0], self.input_shape)
                if not isinstance(y, torch.Tensor) or y.ndim != 3:
                    raise ValueError(f"resize_pad produced unexpected output: {type(y)!r}")
                return y.unsqueeze(0)
            outs = []
            for i in range(b):
                y = resize_pad_graphic_elements(x[i], self.input_shape)
                if not isinstance(y, torch.Tensor) or y.ndim != 3:
                    raise ValueError(f"resize_pad produced unexpected output for batch item {i}: {type(y)!r}")
                outs.append(y)
            return torch.stack(outs, dim=0)

        if x.ndim == 3:
            y = resize_pad_graphic_elements(x, self.input_shape)
            if not isinstance(y, torch.Tensor) or y.ndim != 3:
                raise ValueError(f"resize_pad produced unexpected output: {type(y)!r}")
            return y.unsqueeze(0)

        raise ValueError(f"Expected CHW or BCHW tensor, got shape {tuple(x.shape)}")

    def preprocess_batch_gpu(
        self,
        images: List[Any],
        known_shapes: Optional[List[Optional[Tuple[int, int]]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """Letterbox a list of HWC numpy crop arrays on GPU in one batched pass."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tgt_h, tgt_w = self._graphic_elements_input_shape
        tensors: List[torch.Tensor] = []
        orig_shapes: List[Tuple[int, int]] = []

        for idx, img in enumerate(images):
            if isinstance(img, np.ndarray):
                arr = img
                if arr.ndim == 4:
                    arr = arr[0]
                if int(arr.shape[-1]) == 3 and int(arr.shape[0]) != 3:
                    t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
                else:
                    t = torch.from_numpy(np.ascontiguousarray(arr))
            elif isinstance(img, torch.Tensor):
                t = img
                if t.ndim == 4:
                    t = t[0]
                if t.ndim == 3 and int(t.shape[-1]) == 3 and int(t.shape[0]) != 3:
                    t = t.permute(2, 0, 1)
            else:
                raise TypeError(f"Unsupported image type: {type(img)!r}")

            C, H, W = int(t.shape[0]), int(t.shape[1]), int(t.shape[2])
            orig_shapes.append((H, W))

            d_img = t.float().to(device=device)
            ratio = min(tgt_h / H, tgt_w / W)
            new_h, new_w = int(round(H * ratio)), int(round(W * ratio))

            d_resized = F.interpolate(
                d_img.unsqueeze(0), size=(new_h, new_w),
                mode="bilinear", align_corners=False,
            )[0]
            d_out = torch.full((C, tgt_h, tgt_w), 114.0,
                               dtype=torch.float32, device=device)
            d_out[:C, :new_h, :new_w] = d_resized
            tensors.append(d_out)

        return torch.stack(tensors, dim=0).contiguous(), orig_shapes

    def invoke(
        self, input_data: torch.Tensor, orig_shape: Tuple[int, int]
    ) -> Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        if self._model is None:
            raise RuntimeError("Local graphic_elements_v1 model was not initialized.")

        results = self._model(input_data, orig_shape)
        if len(results) == 1:
            return results[0]
        return results

    @property
    def model_name(self) -> str:
        """Human-readable model name."""
        return "Nemotron Graphic Elements v1"

    @property
    def model_type(self) -> str:
        """Model category/type."""
        return "object-detection"

    @property
    def model_runmode(self) -> RunMode:
        """Execution mode: local, NIM, or build-endpoint."""
        return "local"

    @property
    def input(self) -> Any:
        """
        Input schema for the model.

        Returns:
            dict: Schema describing RGB chart image input
        """
        return {
            "type": "image",
            "format": "RGB",
            "dimensions": "(1024, 1024)",
            "expected_content": "chart_or_graph",
            "description": "Chart/graph image in RGB format, resized to 1024x1024. Should be cropped from document using page element detection.",  # noqa: E501
            "preprocessing_recommendation": "Use Nemotron Page Elements v3 to detect and crop chart regions",
        }

    @property
    def output(self) -> Any:
        """
        Output schema for the model.

        Returns:
            dict: Schema describing detection output format
        """
        return {
            "type": "detection",
            "format": "dict",
            "structure": {
                "boxes": "np.ndarray[N, 4] - normalized (x_min, y_min, x_max, y_max)",
                "labels": "List[str] - class names",
                "scores": "np.ndarray[N] - confidence scores",
            },
            "classes": [
                "chart_title",
                "x_title",
                "y_title",
                "xlabel",
                "ylabel",
                "legend_title",
                "legend_label",
                "mark_label",
                "value_label",
                "other",
            ],
            "post_processing": {"conf_thresh": 0.01, "iou_thresh": 0.25},
            "use_cases": [
                "Chart-to-text conversion",
                "Data extraction from visualizations",
                "Automated chart understanding",
                "Multimodal RAG workflows",
            ],
            "supported_chart_types": ["bar_charts", "line_charts", "pie_charts", "scientific_plots"],
        }

    @property
    def input_batch_size(self) -> int:
        """Maximum or default input batch size."""
        return 1

    @property
    def input_shape(self) -> Tuple[int, int]:
        """Input shape."""
        return self._graphic_elements_input_shape
