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

logger = logging.getLogger(__name__)


class PDFExtractorSchema(BaseModel):
    identify_nearby_objects: bool = True
    max_queue_size: int = 1
    n_workers: int = 16
    raise_on_failure: bool = False

    # TODO: Add additional sub config sections for each extraction method type.

    class Config:
        extra = "forbid"
