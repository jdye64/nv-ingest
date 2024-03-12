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

import cudf
import mrc
import pandas as pd
from morpheus.messages import MessageMeta, ControlMessage
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module

from nv_ingest.schemas.metadata import validate_metadata

logger = logging.getLogger(__name__)

MODULE_NAME = "metadata_injection"
MODULE_NAMESPACE = "nv_ingest"

MetadataInjectorLoaderFactory = ModuleLoaderFactory(MODULE_NAME,
                                                    MODULE_NAMESPACE)


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _metadata_injection(builder: mrc.Builder):
    def on_data(message: ControlMessage):
        # TODO(Devin): Implement task injection logic here, and remove ad hoc versions from nemo splitter
        # Buggy because of StructDtype information loss

        #with message.payload().mutable_dataframe() as mdf:
        #    df = mdf.to_pandas()

        #update_required = False
        #rows = []
        #for _, row in df.iterrows():
        #    if 'metadata' not in row or 'content' not in row['metadata']:
        #        update_required = True
        #        row['metadata'] = {
        #            'content': row['content'],
        #            "content_metadata": {
        #                "type": row["document_type"] if row["document_type"] in ["text", "image"] else "structured",
        #            },
        #            "error_metadata": None,
        #            "image_metadata": None if row["document_type"] != "image" else {
        #                "image_type": "default"  # TODO(Devin): Add image type forwarding
        #            },
        #            "source_metadata": {
        #                "source_id": row["source_id"],
        #                "source_name": row["source_name"],
        #                "source_type": row["document_type"],
        #            },
        #            "text_metadata": None if row["document_type"] != "text" else {
        #                "text_type": "document"
        #            },
        #        }

        #    rows.append(row)
        #    validate_metadata(row['metadata'])

        #if (update_required):
        #    docs = pd.DataFrame(rows)
        #    message_meta = MessageMeta(df=cudf.from_pandas(docs))
        #    message.payload(message_meta)

        return message

    node = builder.make_node("metadata_injector", on_data)

    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
