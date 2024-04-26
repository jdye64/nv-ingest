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
from typing import Literal

from pydantic import BaseModel
from pydantic import root_validator

from .task_base import Task

logger = logging.getLogger(__name__)

_DEFAULT_CONTENT_TYPE = "image"
_DEFAULT_STORE_METHOD = "minio"


class StoreTaskSchema(BaseModel):
    content_type: str = None
    store_method: str = None

    @root_validator(pre=True)
    def set_default_store_method(cls, values):
        content_type = values.get("content_type")
        store_method = values.get("store_method")

        if content_type is None:
            values["content_type"] = _DEFAULT_CONTENT_TYPE
        if store_method is None:
            values["store_method"] = _DEFAULT_STORE_METHOD
        return values

    class Config:
        extra = "allow"


class StoreTask(Task):
    """
    Object for image storage task.
    """

    _Type_Content_Type = Literal["image",]

    _Type_Store_Method = Literal["minio",]

    def __init__(
        self,
        content_type: _Type_Content_Type = None,
        store_method: _Type_Store_Method = None,
        **extra_params,
    ) -> None:
        """
        Setup Store Task Config
        """
        super().__init__()

        self._content_type = content_type or "image"
        self._store_method = store_method or "minio"
        self._extra_params = extra_params

    def __str__(self) -> str:
        """
        Returns a string with the object's config and run time state
        """
        info = ""
        info += "Store Task:\n"
        info += f"  content type: {self._content_type}\n"
        info += f"  store method: {self._store_method}\n"
        for key, value in self._extra_params.items():
            info += f"  {key}: {value}\n"
        return info

    def to_dict(self) -> Dict:
        """
        Convert to a dict for submission to redis (fixme)
        """
        task_properties = {
            "method": self._store_method,
            "content_type": self._content_type,
            "params": self._extra_params,
        }

        return {"type": "store", "task_properties": task_properties}
