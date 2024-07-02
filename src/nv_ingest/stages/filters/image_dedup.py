# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import hashlib
import logging
from functools import partial
from typing import Any
from typing import Dict

import pandas as pd
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.module_utils import ModuleLoaderFactory

import cudf

from nv_ingest.modules.filters.image_filter import add_info_message
from nv_ingest.schemas.image_dedup_schema import ImageDedupSchema
from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.schemas.metadata_schema import InfoMessageMetadataSchema
from nv_ingest.schemas.metadata_schema import StatusEnum
from nv_ingest.schemas.metadata_schema import TaskTypeEnum
from nv_ingest.stages.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest.util.schema.schema_validator import validate_schema

logger = logging.getLogger(__name__)

MODULE_NAME = "dedup_images"
MODULE_NAMESPACE = "nv-ingest"
ImageDedupLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE, ImageDedupSchema)


def hash_content(x, algorithm="md5"):
    return hashlib.md5(x["content"].encode()).digest()


def _cpu_only_apply_dedup_filter(df: pd.DataFrame, filter_flag: bool):
    # return if no images
    image_mask = df["document_type"] == ContentTypeEnum.IMAGE
    if not image_mask.any():
        return df[image_mask]

    base_cols = df.columns
    df_images = df.loc[image_mask].copy()
    content_hash_sr = df_images["metadata"].apply(hash_content, args=("md5",))
    df_images.loc[content_hash_sr.index, "_image_content_hash"] = content_hash_sr
    df_images_deduped = df_images.drop_duplicates(subset="_image_content_hash")
    deduped_indices = df_images_deduped.index
    duplicate_indices = df_images.loc[~df_images.index.isin(deduped_indices)].index

    if filter_flag:
        df_result = pd.concat(
            [
                df_images.loc[deduped_indices][df.columns.difference(["_image_content_hash"])],
                df.loc[~image_mask],
            ],
            axis=0,
        )

        return df_result

    duplicate_images_df = df_images.loc[duplicate_indices]

    # define and validate `info_message_metadata`
    info_msg = {
        "task": TaskTypeEnum.FILTER.value,
        "status": StatusEnum.SUCCESS.value,
        "message": "Filtered duplicate image.",
        "filter": True,
    }

    # update payload with `info_message_metadata` and `document_type`
    validated_info_msg = validate_schema(info_msg, InfoMessageMetadataSchema).dict()

    duplicate_images_df["info_message_metadata"] = [validated_info_msg] * duplicate_images_df.shape[0]
    duplicate_images_df["metadata"] = duplicate_images_df["metadata"].apply(add_info_message, args=(info_msg,))

    df.loc[duplicate_images_df["document_type"].index, "document_type"] = ContentTypeEnum.INFO_MSG
    df.drop(labels=df.columns.difference(base_cols), inplace=True, axis=1)

    return df


def _apply_dedup_filter(ctrl_msg: ControlMessage, filter_flag):
    with ctrl_msg.payload().mutable_dataframe() as mdf:
        # return if no images
        image_mask = mdf["document_type"] == ContentTypeEnum.IMAGE.value
        if not image_mask.any():
            return
        gdf = mdf.copy()

    base_cols = gdf.columns
    gdf_images = gdf.loc[image_mask]
    content_sr = gdf_images["metadata"].struct.field("content")
    content_hash_sr = content_sr.hash_values(method="md5", seed=None)
    gdf_images.loc[content_hash_sr.index, "_image_content_hash"] = content_hash_sr
    gdf_images_deduped = gdf_images.drop_duplicates(subset="_image_content_hash")
    deduped_indices = gdf_images_deduped.index
    duplicate_indices = gdf_images.loc[~gdf_images.index.isin(deduped_indices)].index

    if filter_flag:
        gdf_result = cudf.concat(
            [
                gdf_images.loc[deduped_indices][gdf.columns.difference(["_image_content_hash"])],
                gdf.loc[~image_mask],
            ],
            axis=0,
        )

        message_meta = MessageMeta(df=gdf_result)
        ctrl_msg.payload(message_meta)

        return

    # explode to extract individual metadata structs
    gdf_temp = gdf["metadata"].struct.explode()
    exploded_metadata_cols = list(gdf_temp.columns)
    gdf[exploded_metadata_cols] = gdf_temp
    duplicate_images_gdf = gdf_images.loc[duplicate_indices]

    # define and validate `info_message_metadata`
    info_msg = {
        "task": TaskTypeEnum.FILTER.value,
        "status": StatusEnum.SUCCESS.value,
        "message": "Filtered duplicate image.",
        "filter": True,
    }

    # update payload with `info_message_metadata` and `document_type`
    validated_info_msg = validate_schema(info_msg, InfoMessageMetadataSchema).dict()
    duplicate_images_gdf["info_message_metadata"] = [validated_info_msg] * duplicate_images_gdf.shape[0]
    gdf.drop(labels=["info_message_metadata", "metadata"], inplace=True, axis=1)
    gdf["info_message_metadata"] = duplicate_images_gdf["info_message_metadata"]
    gdf.loc[duplicate_images_gdf["document_type"].index, "document_type"] = ContentTypeEnum.INFO_MSG.value
    gdf["metadata"] = gdf[exploded_metadata_cols + ["info_message_metadata"]].to_struct()
    gdf.drop(labels=gdf.columns.difference(base_cols), inplace=True, axis=1)

    message_meta = MessageMeta(df=gdf)
    ctrl_msg.payload(message_meta)

    return


def dedup_image_stage(df, task_props, validated_config) -> pd.DataFrame:
    task_props.get("content_type")
    task_params = task_props.get("params", {})
    filter_flag = task_params.get("filter", True)

    logger.debug(f"De-duplicating images with filter_flag={filter_flag}")

    df_result = _cpu_only_apply_dedup_filter(df, filter_flag)

    return df_result


def generate_dedup_stage(
    c: Config,
    dedup_config: Dict[str, Any],
    task: str = "dedup",
    task_desc: str = "dedup_images",
    pe_count: int = 8,
):
    validated_config = ImageDedupSchema(**dedup_config)
    _wrapped_dedup_image_stage = partial(dedup_image_stage, validated_config=validated_config)

    logger.debug(f"Generating deduplication stage with config: {validated_config}")
    return MultiProcessingBaseStage(
        c=c,
        pe_count=pe_count,
        task=task,
        task_desc=task_desc,
        process_fn=_wrapped_dedup_image_stage,
        filter_properties={"content_type": ContentTypeEnum.IMAGE.value},
    )
