# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.gpu_operator import GPUOperator
from nemo_retriever.nim.nim import NIMClient
from nemo_retriever.params import RemoteRetryParams
from nemo_retriever.table.shared import table_structure_ocr_page_elements

logger = logging.getLogger(__name__)


class TableStructureActor(AbstractOperator, GPUOperator):
    """
    Ray-friendly callable that initializes both table-structure and OCR
    models once per actor and runs the combined stage.
    """

    def __init__(
        self,
        *,
        table_structure_invoke_url: Optional[str] = None,
        ocr_invoke_url: Optional[str] = None,
        invoke_url: Optional[str] = None,
        api_key: Optional[str] = None,
        table_output_format: Optional[str] = None,
        request_timeout_s: float = 120.0,
        table_structure_trt_engine_path: Optional[str] = None,
        ocr_trt_engine_path: Optional[str] = None,
        remote_max_pool_workers: int = 16,
        remote_max_retries: int = 10,
        remote_max_429_retries: int = 5,
        inference_batch_size: int = 8,
    ) -> None:
        super().__init__()
        self._table_structure_invoke_url = (table_structure_invoke_url or "").strip()
        self._ocr_invoke_url = (ocr_invoke_url or invoke_url or "").strip()
        self._api_key = api_key
        self._request_timeout_s = float(request_timeout_s)
        self._table_output_format = table_output_format
        self._inference_batch_size = int(inference_batch_size)
        self._remote_retry = RemoteRetryParams(
            remote_max_pool_workers=int(remote_max_pool_workers),
            remote_max_retries=int(remote_max_retries),
            remote_max_429_retries=int(remote_max_429_retries),
        )

        _ts_trt = (table_structure_trt_engine_path or "").strip()
        _ocr_trt = (ocr_trt_engine_path or "").strip()

        if self._table_structure_invoke_url:
            self._table_structure_model = None
            logger.info("TableStructureActor: table-structure backend=REMOTE endpoint=%s", self._table_structure_invoke_url)
        elif _ts_trt:
            from nemo_retriever.utils.hf_cache import resolve_engine_path

            resolved_engine: str | None = None
            try:
                resolved_engine = resolve_engine_path(_ts_trt, model_type="table_structure")
            except FileNotFoundError:
                pass

            if resolved_engine is not None:
                try:
                    from nemo_retriever.model.local.trt_engine import TRTYoloxEngine

                    self._table_structure_model = TRTYoloxEngine(
                        resolved_engine, labels=["cell", "row", "column"],
                    )
                    logger.info("TableStructureActor: table-structure backend=TRT engine=%s", resolved_engine)
                except ImportError:
                    logger.warning(
                        "TableStructureActor: tensorrt not available, falling back to HUGGINGFACE "
                        "(path=%s)", _ts_trt,
                    )
                    from nemo_retriever.model.local import NemotronTableStructureV1

                    self._table_structure_model = NemotronTableStructureV1()
            else:
                from nemo_retriever.model.local import NemotronTableStructureV1

                self._table_structure_model = NemotronTableStructureV1()
                logger.info("TableStructureActor: table-structure backend=HUGGINGFACE (path=%s)", _ts_trt)
        else:
            from nemo_retriever.model.local import NemotronTableStructureV1

            self._table_structure_model = NemotronTableStructureV1()
            logger.info("TableStructureActor: table-structure backend=HUGGINGFACE model=NemotronTableStructureV1")

        if self._ocr_invoke_url:
            self._ocr_model = None
            logger.info("TableStructureActor: ocr backend=REMOTE endpoint=%s", self._ocr_invoke_url)
        elif _ocr_trt:
            from nemo_retriever.model.local import NemotronOCRV1
            from nemo_retriever.utils.hf_cache import resolve_model_dir

            ocr_model_dir: str | None = None
            if Path(_ocr_trt).is_dir():
                ocr_model_dir = resolve_model_dir(_ocr_trt, model_type="ocr")
            if ocr_model_dir:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
            self._ocr_model = NemotronOCRV1(model_dir=ocr_model_dir)
            logger.info("TableStructureActor: ocr backend=HUGGINGFACE+TRT model_dir=%s", ocr_model_dir)
        else:
            from nemo_retriever.model.local import NemotronOCRV1

            self._ocr_model = NemotronOCRV1()
            logger.info("TableStructureActor: ocr backend=HUGGINGFACE model=NemotronOCRV1")

        if self._table_structure_invoke_url or self._ocr_invoke_url:
            self._nim_client = NIMClient(
                max_pool_workers=int(remote_max_pool_workers),
            )
        else:
            self._nim_client = None

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return table_structure_ocr_page_elements(
            data,
            table_structure_model=self._table_structure_model,
            ocr_model=self._ocr_model,
            table_structure_invoke_url=self._table_structure_invoke_url,
            ocr_invoke_url=self._ocr_invoke_url,
            api_key=self._api_key,
            table_output_format=self._table_output_format,
            request_timeout_s=self._request_timeout_s,
            inference_batch_size=self._inference_batch_size,
            remote_retry=self._remote_retry,
            nim_client=self._nim_client,
            **kwargs,
        )

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def __call__(self, batch_df: Any, **override_kwargs: Any) -> Any:
        try:
            return self.run(batch_df, **override_kwargs)
        except BaseException as exc:
            if isinstance(batch_df, pd.DataFrame):
                out = batch_df.copy()
                payload = {
                    "timing": None,
                    "error": {
                        "stage": "table_structure_actor_call",
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                }
                n = len(out.index)
                out["table"] = [[] for _ in range(n)]
                out["table_structure_ocr_v1"] = [payload for _ in range(n)]
                return out
            return [
                {
                    "table_structure_ocr_v1": {
                        "timing": None,
                        "error": {
                            "stage": "table_structure_actor_call",
                            "type": exc.__class__.__name__,
                            "message": str(exc),
                        },
                    }
                }
            ]
