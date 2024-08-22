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


class EmbedTaskSchema(BaseModel):
    text: bool = True
    tables: bool = True

    class Config:
        extra = "forbid"


class EmbedTask(Task):
    """
    Object for document embedding task
    """

    def __init__(self, text: bool = True, tables: bool = True, max_seq_len: int = 512) -> None:
        """
        Setup Split Task Config
        """
        super().__init__()
        self._text = text
        self._tables = tables

    def __str__(self) -> str:
        """
        Returns a string with the object's config and run time state
        """
        info = ""
        info += "Embed Task:\n"
        info += f"  text: {self._text}\n"
        info += f"  tables: {self._tables}\n"
        return info

    def to_dict(self) -> Dict:
        """
        Convert to a dict for submission to redis
        """

        task_properties = {
            "text": self._text,
            "tables": self._tables,
        }

        return {"type": "embed", "task_properties": task_properties}
