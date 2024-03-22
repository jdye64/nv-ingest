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
from datetime import datetime
from datetime import timezone

from nv_ingest.util.converters import datetools

logger = logging.getLogger(__name__)


def datetools_exception_handler(func, **kwargs):
    def inner_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log_error_message = f"Invalid date format: {e}"
            logger.warning(log_error_message)
            return datetools.remove_tz(datetime.now(timezone.utc)).isoformat()

    return inner_function
