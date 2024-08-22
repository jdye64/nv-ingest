# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import logging

from pydantic import BaseModel

from nv_ingest.util.logging.configuration import LogLevel

logger = logging.getLogger(__name__)


class EmbedExtractionsSchema(BaseModel):
    api_key: str
    embedding_nim_endpoint: str
    embedding_model: str = "nvidia/nv-embedqa-e5-v5"
    encoding_format: str = "float"
    input_type: str = "passage"
    truncate: str = "END"
    batch_size: int = 100
    httpx_log_level: LogLevel = LogLevel.WARNING
    raise_on_failure: bool = False

    class Config:
        extra = "forbid"
