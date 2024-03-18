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
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import Field

logger = logging.getLogger(__name__)


class FileSourcePipeSchema(BaseModel):
    batch_size: int = 1024
    chunk_overlap: int = 51
    chunk_size: int = 512
    converters_meta: Optional[
        Dict[Any, Any]
    ] = {}  # Flexible dictionary for converters metadata
    enable_monitor: bool = False
    extractor_config: Optional[
        Dict[Any, Any]
    ] = {}  # Flexible dictionary for extractor configuration
    filenames: List[str] = Field(default_factory=list)  # List of file paths
    num_threads: int = 1  # Number of threads for processing
    vdb_resource_name: str
    watch: bool = False  # Flag to watch file changes
    watch_interval: float = -5.0  # Interval to watch file changes

    class Config:
        extra = "forbid"
