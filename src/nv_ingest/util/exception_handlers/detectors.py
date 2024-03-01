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

import logging

from nv_ingest.schemas.metadata import LanguageEnum

from langdetect.lang_detect_exception import LangDetectException

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
