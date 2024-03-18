# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import langdetect

from nv_ingest.schemas.metadata import LanguageEnum
from nv_ingest.util.exception_handlers.detectors import langdetect_exception_handler


@langdetect_exception_handler
def detect_language(text):
    """
    Detect spoken language from a string of text.

    Parameters
    ----------
    text : str
        A string of text.

    Returns
    -------
    LanguageEnum
        A value from `LanguageEnum` detected language code.
    """

    language = langdetect.detect(text)

    if LanguageEnum.has_value(language):
        language = LanguageEnum[language.upper().replace("-", "_")]
    else:
        language = LanguageEnum.UNKNOWN

    return language
