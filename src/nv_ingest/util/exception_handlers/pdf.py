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

from nv_ingest.schemas.metadata_schema import StatusEnum
from nv_ingest.schemas.metadata_schema import TaskTypeEnum
from nv_ingest.schemas.pdf_extractor_schema import PDFExtractorSchema
from nv_ingest.util.schema.schema_validator import validate_schema

logger = logging.getLogger(__name__)


def pymupdf_exception_handler(descriptor):
    def outer_function(func):
        def inner_function(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log_error_message = f"{descriptor}:{func.__name__} error:{e}"
                logger.warning(log_error_message)
                return []

        return inner_function

    return outer_function


def create_exception_tag(error_message, source_id=None):
    unified_metadata = {}

    error_metadata = {
        "task": TaskTypeEnum.EXTRACT,
        "status": StatusEnum.ERROR,
        "source_id": source_id,
        "error_msg": error_message,
    }

    unified_metadata["error_metadata"] = error_metadata

    validated_unified_metadata = validate_schema(unified_metadata, PDFExtractorSchema)

    return [[None, validated_unified_metadata.dict()]]
