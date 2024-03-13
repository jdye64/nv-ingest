# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
