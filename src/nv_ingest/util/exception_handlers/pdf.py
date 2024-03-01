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

from nv_ingest.schemas.metadata import  StatusEnum
from nv_ingest.schemas.metadata import TaskTypeEnum
from nv_ingest.schemas.pdf_extractor_schema import PDFExtractorSchema 
from nv_ingest.util.schema.schema_validator import validate_schema

import pandas as pd

logger = logging.getLogger(__name__)


def pymupdf_exception_handler(descriptor):
    def outer_function(func):
        def inner_function(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log_error_message = f"{descriptor}:{func.__name__} error:{e}"
                logger.warn(log_error_message)
                return []

        return inner_function
    
    return outer_function


def create_exception_tag(error_message, source_id=None):

    unified_metadata = {}

    error_metadata = {
        "task": TaskTypeEnum.EXTRACT,
        "status": StatusEnum.ERROR,
        "source_id": source_id,
        "error_msg": error_message
        }

    unified_metadata["error_metadata"] = error_metadata
    
    validated_unified_metadata = validate_schema(
        unified_metadata, PDFExtractorSchema)
    
    return [[None, validated_unified_metadata.dict()]]
