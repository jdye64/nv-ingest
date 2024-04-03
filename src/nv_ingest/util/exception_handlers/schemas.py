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

from pydantic import ValidationError

logger = logging.getLogger(__name__)


def schema_exception_handler(func, **kwargs):
    def inner_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            error_messages = "; ".join([f"{error['loc'][0]}: {error['msg']}" for error in e.errors()])
            log_error_message = f"Invalid configuration: {error_messages}"
            logger.error(log_error_message)
            raise ValueError(log_error_message)

    return inner_function
