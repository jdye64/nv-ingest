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

from pydantic import ValidationError

logger = logging.getLogger(__name__)


def schema_exception_handler(func, **kwargs):
    def inner_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            error_messages = '; '.join([f"{error['loc'][0]}: {error['msg']}" for error in e.errors()])
            log_error_message = f"Invalid configuration: {error_messages}"
            logger.error(log_error_message)
            raise ValueError(log_error_message) 

    return inner_function
