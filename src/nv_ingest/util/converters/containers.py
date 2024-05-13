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

logger = logging.getLogger(__name__)


def merge_dict(defaults, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and value:
            defaults[key] = merge_dict(defaults.get(key, {}), value)
        else:
            defaults[key] = overrides[key]
    return defaults
