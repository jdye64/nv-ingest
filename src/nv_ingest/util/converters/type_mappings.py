# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from nv_ingest.schemas.ingest_job_schema import DocumentTypeEnum
from nv_ingest.schemas.metadata_schema import ContentTypeEnum

DOC_TO_CONTENT_MAP = {
    DocumentTypeEnum.bmp: ContentTypeEnum.IMAGE,
    DocumentTypeEnum.docx: ContentTypeEnum.STRUCTURED,
    DocumentTypeEnum.html: ContentTypeEnum.STRUCTURED,
    DocumentTypeEnum.jpeg: ContentTypeEnum.IMAGE,
    DocumentTypeEnum.pdf: ContentTypeEnum.STRUCTURED,
    DocumentTypeEnum.png: ContentTypeEnum.IMAGE,
    DocumentTypeEnum.pptx: ContentTypeEnum.STRUCTURED,
    DocumentTypeEnum.svg: ContentTypeEnum.IMAGE,
    DocumentTypeEnum.txt: ContentTypeEnum.TEXT,
}


def doc_type_to_content_type(doc_type: DocumentTypeEnum) -> ContentTypeEnum:
    """
    Convert DocumentTypeEnum to ContentTypeEnum
    """
    return DOC_TO_CONTENT_MAP[doc_type]
