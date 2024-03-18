# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import json
import logging

import mrc
from morpheus.messages import ControlMessage
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module
from mrc.core import operators as ops
from pydantic import ValidationError
from redis import RedisError

from nv_ingest.schemas.redis_task_sink_schema import RedisTaskSinkSchema
from nv_ingest.util.redis import RedisClient
from nv_ingest.util.tracing import traceable

logger = logging.getLogger(__name__)

MODULE_NAME = "redis_task_sink"
RedisTaskSinkLoaderFactory = ModuleLoaderFactory("redis_task_sink", "nv_ingest")


@register_module(MODULE_NAME, "nv_ingest")
def _redis_task_sink(builder: mrc.Builder):
    module_config = builder.get_current_module_config()
    try:
        validated_config = RedisTaskSinkSchema(**module_config)
    except ValidationError as e:
        error_messages = "; ".join(
            [f"{error['loc'][0]}: {error['msg']}" for error in e.errors()]
        )
        log_error_message = f"Invalid Redis Task Sink configuration: {error_messages}"
        logger.error(log_error_message)
        raise ValueError(log_error_message)

    # Initialize RedisClient with the validated configuration
    redis_client = RedisClient(
        host=validated_config.redis_client.host,
        port=validated_config.redis_client.port,
        db=0,  # Assuming DB is always 0 for simplicity, make configurable if needed
        max_retries=validated_config.redis_client.max_retries,
        max_backoff=validated_config.redis_client.max_backoff,
        connection_timeout=validated_config.redis_client.connection_timeout,
        use_ssl=validated_config.redis_client.use_ssl,
    )

    @traceable(MODULE_NAME)
    def process_and_forward(message: ControlMessage):
        with message.payload().mutable_dataframe() as mdf:
            logger.info(f"Received DataFrame with {len(mdf)} rows.")
            df_json = mdf.to_json(orient="records")

        ret_val_json = {
            "data": df_json,
        }

        do_trace_tagging = message.get_metadata("add_trace_tagging", True)
        if do_trace_tagging:
            traces = {}
            meta_list = message.list_metadata()
            for key in meta_list:
                if key.startswith("trace::"):
                    traces[key] = message.get_metadata(key)
            ret_val_json["trace"] = traces

        response_channel = message.get_metadata("response_channel")
        try:
            redis_client.get_client().rpush(response_channel, json.dumps(ret_val_json))
            logger.info(f"Forwarded message to Redis channel '{response_channel}'.")
        except RedisError as e:
            logger.error(f"Failed to forward message to Redis: {e}")

        return message

    process_node = builder.make_node(
        "process_and_forward", ops.map(process_and_forward)
    )
    process_node.launch_options.engines_per_pe = 6

    # Register the final output of the module
    builder.register_module_input("input", process_node)
    builder.register_module_output("output", process_node)
