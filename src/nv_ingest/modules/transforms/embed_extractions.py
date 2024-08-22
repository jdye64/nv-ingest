# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import asyncio
import logging
import traceback
from typing import List

import mrc
import pandas as pd
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.control_message_utils import cm_skip_processing_if_failed
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module
from mrc.core import operators as ops
from openai import AsyncOpenAI

import cudf

from nv_ingest.schemas.embed_extractions_schema import EmbedExtractionsSchema
from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.schemas.metadata_schema import StatusEnum
from nv_ingest.schemas.metadata_schema import TaskTypeEnum
from nv_ingest.util.exception_handlers.decorators import nv_ingest_node_failure_context_manager
from nv_ingest.util.flow_control import filter_by_task
from nv_ingest.util.modules.config_validator import fetch_and_validate_module_config
from nv_ingest.util.tracing import traceable

logger = logging.getLogger(__name__)

MODULE_NAME = "embed_extractions"
MODULE_NAMESPACE = "nv_ingest"

EmbedExtractionsLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE, EmbedExtractionsSchema)


async def _make_async_request(
    prompts: List[str],
    api_key: str,
    embedding_nim_endpoint: str,
    embedding_model: str,
    encoding_format: str,
    input_type: str,
    truncate: str,
) -> list:
    response = {"embedding": None, "info_msg": None}

    try:
        response["info_msg"] = [None] * len(prompts)

        async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=embedding_nim_endpoint,
        )

        resp = await async_client.embeddings.create(
            input=prompts,
            model=embedding_model,
            encoding_format=encoding_format,
            extra_body={"input_type": input_type, "truncate": truncate},
        )

        response["embedding"] = resp.data

    except Exception as e:
        info_msg = {
            "task": TaskTypeEnum.EMBED,
            "status": StatusEnum.ERROR,
            "message": e,
            "filter": True,
        }

        response["embedding"] = [[]] * len(prompts)
        response["info_msg"] = [info_msg] * len(prompts)

    return response


async def _async_request_handler(
    prompts: List[str],
    api_key: str,
    embedding_nim_endpoint: str,
    embedding_model: str,
    encoding_format: str,
    input_type: str,
    truncate: str,
):
    res = await asyncio.gather(
        *(
            (
                _make_async_request(
                    prompts=prompt_batch,
                    api_key=api_key,
                    embedding_nim_endpoint=embedding_nim_endpoint,
                    embedding_model=embedding_model,
                    encoding_format=encoding_format,
                    input_type=input_type,
                    truncate=truncate,
                )
            )
            for prompt_batch in prompts
        )
    )

    return res


def _async_runner(
    prompts: List[str],
    api_key: str,
    embedding_nim_endpoint: str,
    embedding_model: str,
    encoding_format: str,
    input_type: str,
    truncate: str,
    event_loop: asyncio.SelectorEventLoop,
):
    results = event_loop.run_until_complete(
        _async_request_handler(
            prompts,
            api_key,
            embedding_nim_endpoint,
            embedding_model,
            encoding_format,
            input_type,
            truncate,
        )
    )

    flat_results = {"embeddings": [], "info_msgs": []}
    for batch_dict in results:
        info_msg = batch_dict["info_msg"]
        for embedding in batch_dict["embedding"]:
            if not isinstance(embedding, list):
                flat_results["embeddings"].append(embedding.embedding)
            else:
                flat_results["embeddings"].append(embedding)
            flat_results["info_msgs"].append(info_msg)

    return flat_results


def _add_embeddings(row, embeddings, info_msgs):
    row["metadata"]["embedding"] = embeddings[row.name]
    row["metadata"]["info_message_metadata"] = info_msgs[row.name]

    return row


def _get_text_content(row):
    return row["content"]


def _get_table_content(row):
    return row["table_metadata"]["table_content"]


def _batch_generator(iterable, batch_size=10):
    iter_len = len(iterable)
    for idx in range(0, iter_len, batch_size):
        yield iterable[idx : min(idx + batch_size, iter_len)]  # noqa: E203


def _generate_batches(prompts, batch_size=100):
    return [x for x in _batch_generator(prompts, batch_size)]


def _generate_embeddings(
    ctrl_msg: ControlMessage,
    content_type: ContentTypeEnum,
    event_loop: asyncio.SelectorEventLoop,
    batch_size: int,
    api_key: str,
    embedding_nim_endpoint: str,
    embedding_model: str,
    encoding_format: str,
    input_type: str,
    truncate: str,
):
    # TODO: status message if we truncate, need to figure out max length for embedder...
    with ctrl_msg.payload().mutable_dataframe() as mdf:
        # generate table text mask
        if content_type == ContentTypeEnum.TEXT:
            content_mask = (mdf["document_type"] == content_type.value) & (
                mdf["metadata"].struct.field("content") != ""
            )
            content_getter = _get_text_content
        elif content_type == ContentTypeEnum.STRUCTURED:
            table_mask = mdf["document_type"] == content_type.value
            if not table_mask.any():
                return None, None
            content_mask = table_mask & (
                mdf["metadata"].struct.field("table_metadata").struct.field("table_content") != ""
            )
            content_getter = _get_table_content

        # exit if matches found
        if not content_mask.any():
            return None, None

        df_text = mdf.loc[content_mask].to_pandas().reset_index(drop=True)
        # get text list
        filtered_text = df_text["metadata"].apply(content_getter)
        # calculate embeddings
        filtered_text_batches = _generate_batches(filtered_text.tolist(), batch_size)
        text_embeddings = _async_runner(
            filtered_text_batches,
            api_key,
            embedding_nim_endpoint,
            embedding_model,
            encoding_format,
            input_type,
            truncate,
            event_loop,
        )
        # update embeddings in metadata
        df_text["metadata"] = df_text[["metadata"]].apply(_add_embeddings, **text_embeddings, axis=1)["metadata"]
        df_text["_contains_embeddings"] = True
        df_text["_content"] = filtered_text

    return df_text, content_mask


def _concatenate_extractions(ctrl_msg, dataframes, masks):
    # build unified mask
    for idx, mask in enumerate(masks):
        if idx == 0:
            unified_mask = mask
        else:
            unified_mask = unified_mask & mask

    with ctrl_msg.payload().mutable_dataframe() as mdf:
        df_no_text = mdf.loc[~unified_mask].to_pandas()
        df_no_text["_contains_embeddings"] = False

    dataframes.append(df_no_text)

    df = pd.concat(dataframes, axis=0, ignore_index=True).reset_index(drop=True)

    gdf = cudf.from_pandas(df)
    meta = MessageMeta(df=gdf)
    ctrl_msg.payload(meta)

    return ctrl_msg


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _embed_extractions(builder: mrc.Builder):
    """
    A pipeline module that splits documents into smaller parts based on the specified criteria.
    """

    validated_config = fetch_and_validate_module_config(builder, EmbedExtractionsSchema)
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(validated_config.httpx_log_level.value)
    event_loop = asyncio.new_event_loop()

    @filter_by_task(["embed"])
    @traceable(MODULE_NAME)
    @cm_skip_processing_if_failed
    @nv_ingest_node_failure_context_manager(
        annotation_id=MODULE_NAME,
        raise_on_failure=validated_config.raise_on_failure,
    )
    def embed_extractions_fn(message: ControlMessage):
        try:
            task_props = message.remove_task("embed")
            embed_text = task_props.get("text")
            embed_tables = task_props.get("tables")

            logger.debug(f"Generating embeddings: text={embed_text}, tables={embed_tables}")
            embedding_dataframes = []
            content_masks = []

            if embed_text:
                df_text, content_mask = _generate_embeddings(
                    message,
                    ContentTypeEnum.TEXT,
                    event_loop,
                    validated_config.batch_size,
                    validated_config.api_key,
                    validated_config.embedding_nim_endpoint,
                    validated_config.embedding_model,
                    validated_config.encoding_format,
                    validated_config.input_type,
                    validated_config.truncate,
                )
                if df_text is not None:
                    embedding_dataframes.append(df_text)
                    content_masks.append(content_mask)

            if embed_tables:
                df_tables, table_mask = _generate_embeddings(
                    message,
                    ContentTypeEnum.STRUCTURED,
                    event_loop,
                    validated_config.batch_size,
                    validated_config.api_key,
                    validated_config.embedding_nim_endpoint,
                    validated_config.embedding_model,
                    validated_config.encoding_format,
                    validated_config.input_type,
                    validated_config.truncate,
                )
                if df_tables is not None:
                    embedding_dataframes.append(df_tables)
                    content_masks.append(table_mask)

            message = _concatenate_extractions(message, embedding_dataframes, content_masks)

            return message

        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"Failed to generate embeddings: {e}")

    embedding_node = builder.make_node("embed_extractions", ops.map(embed_extractions_fn))

    # Register the input and output of the module
    builder.register_module_input("input", embedding_node)
    builder.register_module_output("output", embedding_node)
