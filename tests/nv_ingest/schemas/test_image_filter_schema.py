# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import pytest
from pydantic import ValidationError

from nv_ingest.schemas.image_filter_schema import ImageFilterSchema


def valid_module_config():
    """Returns a valid job payload for testing purposes."""
    return {
        "raise_on_failure": True,
    }


def test_task_type_str_bool():
    img_filter_module_config = valid_module_config()
    img_filter_module_config["raise_on_failure"] = bool(img_filter_module_config["raise_on_failure"])
    _ = ImageFilterSchema(**img_filter_module_config)


@pytest.mark.parametrize("dtype", [int, float, str])
def test_task_type_str_bool_sensitivity(dtype):
    img_filter_module_config = valid_module_config()
    img_filter_module_config["raise_on_failure"] = dtype(img_filter_module_config["raise_on_failure"])

    with pytest.raises(ValidationError):
        _ = ImageFilterSchema(**img_filter_module_config)
