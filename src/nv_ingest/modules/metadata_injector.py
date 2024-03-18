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

import mrc
import pandas as pd
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module

import cudf

from nv_ingest.schemas.ingest_job import DocumentTypeEnum
from nv_ingest.schemas.metadata import ContentTypeEnum
from nv_ingest.util.converters.type_mappings import doc_type_to_content_type

logger = logging.getLogger(__name__)

MODULE_NAME = "metadata_injection"
MODULE_NAMESPACE = "nv_ingest"

MetadataInjectorLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE)


def _on_data(message: ControlMessage):
    with message.payload().mutable_dataframe() as mdf:
        df = mdf.to_pandas()

    update_required = False
    rows = []
    for _, row in df.iterrows():
        content_type = doc_type_to_content_type(DocumentTypeEnum(row["document_type"]))
        if "metadata" not in row or "content" not in row["metadata"]:
            update_required = True
            row["metadata"] = {
                "content": row["content"],
                "content_metadata": {
                    "type": content_type.name.lower(),
                },
                "error_metadata": None,
                "image_metadata": (
                    None
                    if content_type != ContentTypeEnum.IMAGE
                    else {"image_type": row["document_type"]}
                ),
                "source_metadata": {
                    "source_id": row["source_id"],
                    "source_name": row["source_name"],
                    "source_type": row["document_type"],
                },
                "text_metadata": (
                    None
                    if (content_type != ContentTypeEnum.TEXT)
                    else {"text_type": "document"}
                ),
            }

        rows.append(row)
        # validate_metadata(row['metadata'])

    if update_required:
        docs = pd.DataFrame(rows)

        message_meta = MessageMeta(df=cudf.from_pandas(docs))
        message.payload(message_meta)

    return message


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _metadata_injection(builder: mrc.Builder):
    node = builder.make_node("metadata_injector", _on_data)

    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
