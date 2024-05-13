# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments

import logging
from typing import Dict

from pydantic import BaseModel

from .task_base import Task

logger = logging.getLogger(__name__)


class CaptionTaskSchema(BaseModel):
    class Config:
        extra = "forbid"


class CaptionTask(Task):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def __str__(self) -> str:
        """
        Returns a string with the object's config and run time state
        """
        info = ""
        return info

    def to_dict(self) -> Dict:
        """
        Convert to a dict for submission to redis
        """
        params = {}

        return {"type": "caption", "task_properties": params}
