# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import base64
import logging
import traceback
from io import BytesIO
from typing import Any
from typing import Dict

import mrc
import mrc.core.operators as ops
import pandas as pd
from minio import Minio
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module

import cudf

from nv_ingest.schemas.image_storage_schema import ImageStorageModuleSchema
from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.util.exception_handlers.decorators import nv_ingest_node_failure_context_manager
from nv_ingest.util.flow_control import filter_by_task
from nv_ingest.util.modules.config_validator import fetch_and_validate_module_config
from nv_ingest.util.tracing import traceable

logger = logging.getLogger(__name__)

MODULE_NAME = "image_storage"
MODULE_NAMESPACE = "nv_ingest"

_DEFAULT_ENDPOINT = "localhost:9000"
_DEFAULT_BUCKET_NAME = "nv-ingest"

ImageStorageLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE, ImageStorageModuleSchema)


def upload_images(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Identify contents (e.g., images) within a dataframe and uploads the data to MinIO.
    The image metadata in the metadata column is updated with the URL of the uploaded data.
    """
    content_type = params.get("content_type", ContentTypeEnum.IMAGE)
    endpoint = params.get("endpoint", _DEFAULT_ENDPOINT)
    bucket_name = params.get("bucket_name", _DEFAULT_BUCKET_NAME)

    client = Minio(
        endpoint,
        access_key=params.get("access_key", None),
        secret_key=params.get("secret_key", None),
        session_token=params.get("session_token", None),
        secure=params.get("secure", False),
        region=params.get("region", None),
    )

    bucket_found = client.bucket_exists(bucket_name)
    if not bucket_found:
        client.make_bucket(bucket_name)
        logger.debug("Created bucket %s", bucket_name)
    else:
        logger.debug("Bucket %s already exists", bucket_name)

    for idx, row in df.iterrows():
        if row["document_type"] != content_type:
            continue

        metadata = row["metadata"]

        content = base64.b64decode(metadata["content"].encode())

        source_id = metadata["source_metadata"]["source_id"]
        image_type = metadata["image_metadata"]["image_type"]

        destination_file = f"{source_id}/{idx}.{image_type}"

        source_file = BytesIO(content)
        client.put_object(
            bucket_name,
            destination_file,
            source_file,
            length=len(content),
        )

        metadata["image_metadata"]["uploaded_image_url"] = f"http://{endpoint}/{bucket_name}/{destination_file}"

        # TODO: validate metadata before putting it back in.
        df.iloc[idx]["metadata"] = metadata

    return df


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _storage_images(builder: mrc.Builder):
    validated_config = fetch_and_validate_module_config(builder, ImageStorageModuleSchema)

    @filter_by_task(["store"])
    @traceable(MODULE_NAME)
    @nv_ingest_node_failure_context_manager(
        annotation_id=MODULE_NAME,
        raise_on_failure=validated_config.raise_on_failure,
    )
    def on_data(ctrl_msg: ControlMessage):
        try:
            task_props = ctrl_msg.get_tasks().get("store").pop()
            params = task_props.get("params", {})
            # TODO(Matt) validate this resolves to the right filter criteria....
            content_type = params.get("content_type", ContentTypeEnum.IMAGE)

            with ctrl_msg.payload().mutable_dataframe() as mdf:
                # df = dftools.cudf_to_pandas(mdf, deserialize_cols=["document_type", "metadata"])
                df = mdf.to_pandas()

            image_mask = df["document_type"] == content_type
            if (~image_mask).all():  # if there are no images, return immediately.
                return ctrl_msg

            df = upload_images(df, params)

            # Update control message with new payload
            gdf = cudf.from_pandas(df)
            msg_meta = MessageMeta(df=gdf)
            ctrl_msg.payload(msg_meta)
        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"Failed to split documents: {e}")

        return ctrl_msg

    input_node = builder.make_node("image_storage", ops.map(on_data))

    builder.register_module_input("input", input_node)
    builder.register_module_output("output", input_node)
