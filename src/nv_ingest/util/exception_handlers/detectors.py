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

from langdetect.lang_detect_exception import LangDetectException

from nv_ingest.schemas.metadata import LanguageEnum

logger = logging.getLogger(__name__)


def langdetect_exception_handler(func, **kwargs):
    def inner_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except LangDetectException as e:
            log_error_message = f"LangDetectException:{e}"
            logger.warn(log_error_message)
            return LanguageEnum.UNKNOWN

    return inner_function
